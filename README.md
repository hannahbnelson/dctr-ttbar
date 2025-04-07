# dctr-ttbar
Modified DCTR approach to find reweighting between MadGraph EFT LO+jet sample and the central NLO Powheg sample. 

# Setup 
If conda or micromamba are not already available, choose one and install: 
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
```
Or,
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Next, clone this repository: 
```bash
git clone git@github.com:hannahbnelson/dctr-ttbar.git dctr
cd dctr
```

Then, setup the micromamba environment: 
```bash
unset PYTHONPATH 
unset PERL5LIB
micromamba env create -f pytorch_environment.yml
micromamba activate torch
pip install -e .
```

The `pytorch_environment.yml` sets up an environment with pytorch. 
Alternatively, `tensorflow_environment.yml` is available to use, and sets up an environment with tensorflow. 

The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall dctr`. 

This repository also depends on the `topcoffea` package, which is not yet available on `PyPI`, so we need to clone the `topcoffea` repo and install it ourselves.

```bash
cd 
git clone https://github.com/TopEFT/topcoffea.git
cd topcoffea
pip install -e .  
```

Now all of the dependencies have been installed and the `dctr` repository is ready.
In the future, just activate the environment: 
```bash
unset PYTHONPATH
unset PERL5LIB
micromamba activate torch
```

# Run a job with work\_queue

## Submit workers on glados
The workers must be submitted from the same environment that you are running the run script from so open a new ssh session to `glados` and run these commands: 
```bash
unset PYTHONPATH
unset PERL5LIB
micromamba activate torch
condor_submit_workers -M ${USER}-workqueue-coffea -t 900 --cores <Ncores> --memory <Nmemory(MB)> --disk 100000 <Nworkers>
```
The workers will terminate themselves after 15 minutes of inactivity.

## Submit workers on the CRC opportunistic resources
First, login to the ND CRC condor node and activate the conda environment: 
```bash
ssh glados
ssh condorfe.crc.nd.edu
unset PYTHONPATH
unset PERL5LIB
micromamba activate torch
```

To normally submit workers, use the same `condor_submit_wokers` command above. 
Currently, there is an authentication issues that causes an error when trying to run condor jobs on condorfe that access files via xrootd. Temporarily, use the script `condor_custom` to submit workers using the following command: 
```bash
cd dctr/analysis
python condor_custom  --manager ${USER}-workqueue-coffea --cores <Ncores> --memory <Nmemory (MG)> --disk 100000 --num-workers <Nworkers> 
```

