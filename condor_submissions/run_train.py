import subprocess
import os
import datetime

submit_file = f"""
executable = {{0}}
output = {{1}}
error = {{2}}
request_memory = {{3}}

# Grabs the current environment to send with this job
getenv = True

+JobFlavor = {{4}}

# When job finishes, if not terminated by a signal (e.g. condor_rm), then it must fail gracefully otherwise it's readded to the queue. Counts towards retries
on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
max_retries = 1

# Use this just in case the last machine had an issue with your code/environment so it doesn't grab the same machine.
requirements = Machine =!= LastRemoteHost

queue 1
"""

sh_file = f"""
#!/bin/sh
unset PYTHONPATH
unset PERL5LIB

which python
python {{0}} {{1}}
"""

now = datetime.datetime.now().strftime('%Y%m%d_%H%M')

if __name__ == "__main__":

    # Store as int - formatting comes in writing the sub file (in GB)
    memory_reqested = 16

    # Job flavor for Condor - see https://batchdocs.web.cern.ch/local/submit.html
    job_flavor = "longlunch"

    # Get directory of this file
    cwd = os.path.dirname(os.path.abspath(__file__))

    working_dir = os.path.join(cwd, now)
    os.makedirs(working_dir, exist_ok=False)

    # Specify logs directory
    logs = os.path.join(working_dir, "logs")
    os.makedirs(logs, exist_ok=False)

    # Make shell script for running on Condor machines
    sh_path = os.path.join(working_dir, f"condor_run_{now}.sh")
    with open(sh_path, "w") as f:
        f.write(
            sh_file.format(
                os.path.abspath("/users/hnelson2/dctr/analysis/train.py"),
                f"--outdir {working_dir}",
            )
        )

    # Make submission script
    submit_path = os.path.join(working_dir, f"condor_submit_{now}.sub")
    print(f"submissions script path: {submit_path}")
    print(f"path exists? {os.path.exists(submit_path)}")
    with open(submit_path, "w") as f:
        f.write(
            submit_file.format(
                sh_path,
                os.path.join(logs, f"log_{now}.out"),
                os.path.join(logs, f"log_{now}.err"),
                f"{memory_reqested}GB",
                job_flavor,
            )
        )

    # Do checks and submit Condor jobs
    assert os.path.exists(sh_path)
    assert os.path.exists(submit_path)

    print(f"created sh file: {sh_path}")
    print(f"created submit file: {submit_path}")

    # Allow Condor to execute the shell script
    os.system(f"chmod 775 {sh_path}")
    subprocess.run(["condor_submit", submit_path])
