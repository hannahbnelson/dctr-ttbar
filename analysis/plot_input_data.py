import datetime
import pickle
import gzip
import shutil
import os
import sys
from pathlib import Path
import argparse 
import logging
import yaml

import hist
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

import mplhep as hep
import matplotlib.pyplot as plt

def make_standardization_df(df, outdir):

    # make a copy as to not change original df
    norm_df = df.copy()

    # select only numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns
    means = df.mean()
    stdvs = df.std()

    means.to_csv(os.path.join(outdir, "standardization_means.csv"), index=True)
    stdvs.to_csv(os.path.join(outdir, "standardization_stds.csv"), index=True)

    return means, stdvs


def standardize_df(df, means, stdvs):
    # means and stdvs are separately computed on the whole dataset

    # make a copy as to not change original df
    norm_df = df.copy()

    # if stdv is 0, set to 0 
    # if stdv is not 0, normalized = (orig - mean)/stdv
    for col in norm_df: 
        if stdvs[col] != 0:
            norm_df[col] = (df[col] - means[col])/(stdvs[col])
        else: 
            norm_df[col] = 0.0

    return norm_df


def plot_inputs(smeft_data, powheg_data, name, title, outdir):
    hep.style.use("CMS")
    fig, ax = plt.subplots()

    bins=np.linspace(-2, 7, 90) 

    ax.hist(smeft_data, bins=bins, label='smeft', histtype='step')   
    ax.hist(powheg_data, bins=bins, label='powheg', histtype='step')
    ax.set_xlabel(name)
    ax.set_ylabel('Events')
    ax.legend(loc='upper right')

    outname = os.path.join(outdir, title)
    fig.savefig(f"{outname}.png")
    print(f"figure saved in {outname}.png") 


def main():

    rando = 1234
    torch.manual_seed(rando)

    # create training datasets
    train_smeft = pickle.load(gzip.open("/users/hnelson2/dctr/analysis/smeft_training.pkl.gz")).drop(['weights'], axis=1)
    train_powheg = pickle.load(gzip.open("/users/hnelson2/dctr/analysis/powheg_training.pkl.gz")).drop(['weights'], axis=1)

    weights_train_smeft = np.ones_like(train_smeft['mtt'])
    weights_train_powheg = np.ones_like(train_powheg['mtt'])

    truth_train_smeft = np.ones_like(train_smeft['mtt'])
    truth_train_powheg = np.zeros_like(train_powheg['mtt'])

    # create validation datasets
    validation_smeft = pickle.load(gzip.open("/users/hnelson2/dctr/analysis/smeft_validation.pkl.gz")).drop(['weights'], axis=1)
    validation_powheg = pickle.load(gzip.open("/users/hnelson2/dctr/analysis/powheg_validation.pkl.gz")).drop(['weights'], axis=1)
    weights_validation_smeft = np.ones_like(validation_smeft['mtt'])
    weights_validation_powheg = np.ones_like(validation_powheg['mtt'])
    truth_validation_smeft = np.ones_like(validation_smeft['mtt'])
    truth_validation_powheg = np.zeros_like(validation_powheg['mtt'])

    ### standardize inputs
    # find means and stdvs for each variable using all of the data
    means, stdvs = make_standardization_df(pd.concat([train_smeft, train_powheg, validation_smeft, validation_powheg]), outdir="/users/hnelson2/dctr/analysis/")

    # use that mean, stdv to standardize all datasets
    norm_train_smeft = standardize_df(train_smeft, means, stdvs)
    norm_train_powheg = standardize_df(train_powheg, means, stdvs)

    norm_val_smeft = standardize_df(validation_smeft, means, stdvs)
    norm_val_powheg = standardize_df(validation_powheg, means, stdvs)

    plotdir = "/users/hnelson2/dctr/analysis/input_data_plots/"

    numerical_cols = train_smeft.select_dtypes(include=np.number).columns
    
    for col in numerical_cols:
        smeft_data = norm_train_smeft[col].to_numpy()
        powheg_data = norm_train_powheg[col].to_numpy()

        plot_inputs(smeft_data, powheg_data, name=col, title=f"training_{col}", outdir=plotdir)

    for col in numerical_cols:
        smeft_data = norm_val_smeft[col].to_numpy()
        powheg_data = norm_val_powheg[col].to_numpy()

        plot_inputs(smeft_data, powheg_data, name=col, title=f"validation_{col}", outdir=plotdir)


if __name__=="__main__":
    main()