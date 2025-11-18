# dctr-ttbar
This module contains a modified DCTR approach to find reweighting between MadGraph EFT LO+jet sample and the central NLO Powheg sample. 

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
micromamba env create -f environment.yml
micromamba activate dctr
pip install -e .
```


The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall dctr`. 
This repository also depends on the `topcoffea` package, which is not yet available on `PyPI`, so clone the `topcoffea` repo and install it.

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
micromamba activate dctr
```

# Run a job with work\_queue

## Submit workers on glados
The workers must be submitted from the same environment that you are running the run script from so open a new ssh session to `glados` and run these commands: 
```bash
unset PYTHONPATH
unset PERL5LIB
micromamba activate dctr
condor_submit_workers -M ${USER}-workqueue-coffea -t 900 --cores <Ncores> --memory <Nmemory(MB)> --disk 100000 <Nworkers>
```
The workers will terminate themselves after 15 minutes of inactivity.

## Submit workers on the ND CRC opportunistic resources
First, login to the ND CRC condor node and activate the conda environment.
Normally, the same `condor_submit_wokers` command from above works to submit workers. 
But currently, there is an authentication issue that causes an error when trying to run condor jobs on condorfe while accessing files via xrootd. Temporarily, use the script `condor_custom` to submit workers using the following command: 
```bash
ssh glados
ssh condorfe.crc.nd.edu
unset PYTHONPATH
unset PERL5LIB
micromamba activate dctr

python condor_custom  --manager ${USER}-workqueue-coffea --cores <Ncores> --memory <Nmemory (MG)> --disk 100000 --num-workers <Nworkers> 
```

# Scripts

- `analysis/data_processor.py`: 
	- creates the pandas dataframes containing the variables used for neural network training
	- applies top mass cuts removing all events with top mass < 150 GeV or > 195 GeV
- `analysis/run_NN_processor.py`:
	- run script for `analysis/data_processor.py` and other coffea processors
	- can run with `futures` for small local tests, or `work_queue` for full runs
- `analysis/df_accumulator`:
	- accumulator class for pandas dataframes
	- used by `analysis/data_processor.py`
- `analysis/make_pytorch_datasets.py`:
	- takes pkl files containing dataframes produced by `analysis/data_processor.py` shuffles, and separates the events into train/validation/testing files
	- produces new pkl files containing subsets of the original ones to use as the inputs for `analysis/train.py`
- `analysis/config.yaml`
	- config file for train.py 
	- needs to contain these entries at minimum: inputs, model, params, monitoring
- `analysis/train.py`:
	- trains neural network based on inputs from config.yaml
	- can be run locally for a quick test using command line options (`--config`, `--outdir`, `--cores`)
	- submit as a condor job using `condor_submssions/run_train.py`
- `condor_submissions/run_train.py`:
	- run with command line options to optionally specify the training script, config file, output directory, cores/memory per job

