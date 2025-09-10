import subprocess
import os
import shutil
import datetime
import argparse

sh_file = f"""
#!/bin/sh
unset PYTHONPATH
unset PERL5LIB

which python
python {{0}} {{1}}
"""


submit_file = f"""
executable = {{0}}
log = {{1}}
output = {{2}}
error = {{3}}
request_memory = {{4}}
request_cpus = {{5}}

# Grabs the current environment to send with this job
getenv = True

+JobFlavor = {{6}}

# When job finishes, if not terminated by a signal (e.g. condor_rm), then it must fail gracefully otherwise it's readded to the queue. Counts towards retries
on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
max_retries = 1

# Use this just in case the last machine had an issue with your code/environment so it doesn't grab the same machine.
requirements = Machine =!= LastRemoteHost

notify_user = hnelson2@nd.edu
notification = always

queue 1
"""


now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('--train', default='/users/hnelson2/dctr/analysis/train.py', help='python trainingi script')
    parser.add_argument('--config', default='/users/hnelson2/dctr/analysis/config.yaml', help='yaml config for train.py')
    parser.add_argument('--outdir', default=f"{now}", help='name of output directory in dctr/condor_submissions')

    args = parser.parse_args()
    train_path = args.train
    config_path = args.config
    outdir = args.outdir

    # Store as int - formatting comes in writing the sub file (in GB)
    memory_reqested = 32
    cores_requested = 8

    path_to_script = os.path.abspath(train_path)
    path_to_config = os.path.abspath(config_path)

    # Job flavor for Condor - see https://batchdocs.web.cern.ch/local/submit.html
    job_flavor = "longlunch"

    # Get directory of this file
    cwd = os.path.dirname(os.path.abspath(__file__))

    working_dir = os.path.join(cwd, outdir)
    os.makedirs(working_dir, exist_ok=False)

    # Specify logs directory
    logs = os.path.join(working_dir, "condor_logs")
    os.makedirs(logs, exist_ok=False)

    # Save config file and training scipt to working directory for tracking changes
    shutil.copy(path_to_script, os.path.join(working_dir, "train.py"))
    shutil.copy(path_to_config, os.path.join(working_dir, "config.yaml"))

    # Make shell script for running on Condor machines
    sh_path = os.path.join(working_dir, f"condor_run.sh")
    with open(sh_path, "w") as f:
        f.write(
            sh_file.format(
                os.path.abspath(path_to_script),
                f"--config {path_to_config} --outdir {working_dir} --cores {cores_requested}"
            )
        )

    # Make submission script
    submit_path = os.path.join(working_dir, f"condor_submit.sub")
    with open(submit_path, "w") as f:
        f.write(
            submit_file.format(
                sh_path,
                os.path.join(logs, f"condor.log"),
                os.path.join(logs, f"condor.out"),
                os.path.join(logs, f"condor.err"),
                f"{memory_reqested}GB",
                f"{cores_requested}",
                job_flavor,
            )
        )

    # Do checks and submit Condor jobs
    assert os.path.exists(sh_path)
    assert os.path.exists(submit_path)

    print(f"working directory: {working_dir} \n")
    print(f"created sh file: {sh_path}")
    print(f"created submit file: {submit_path}")

    print(f"saved config.yaml to {os.path.join(working_dir, 'config.yaml')}")
    print(f"saved train.py to {os.path.join(working_dir, 'train.py')}")

    # Allow Condor to execute the shell script
    os.system(f"chmod 775 {sh_path}")
    subprocess.run(["condor_submit", submit_path])
