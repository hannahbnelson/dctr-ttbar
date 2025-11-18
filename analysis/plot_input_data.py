import pickle
import gzip
import shutil
import os
import sys
import argparse 
import yaml

import numpy as np
import pandas as pd
import awkward as ak

import hist
import mplhep as hep
import matplotlib.pyplot as plt


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

    outname = os.path.join(outdir, f"{title}_{name}")
    fig.savefig(f"{outname}.png")
    print(f"figure saved in {outname}.png") 


def main(fsmeft, fpowheg, means, stdvs, outdir, title):


    smeft_df = pickle.load(gzip.open(fsmeft))
    powheg_df = pickle.load(gzip.open(fpowheg))

    smeft_df = standardize_df(smeft_df, means, stdvs)
    powheg_df = standardize_df(powheg_df, means, stdvs)

    for col in smeft_df.columns:
        plot_inputs(smeft_df[col], powheg_df[col], name=col, title=title, outdir=outdir)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='command line arguments')
    parser.add_argument('--fsmeft', required=True, help="path to smeft dataframe pkl")
    parser.add_argument('--fpowheg', required=True, help="path to powheg dataframe pkl")
    parser.add_argument('--outdir', '-o', default='.', help='output directory')
    parser.add_argument('--title', default='', help='title for plots')
    parser.add_argument('--config', required=True, help='path to yaml that contains means and stdvs')

    args = parser.parse_args()
    fsmeft = args.fsmeft
    fpowheg = args.fpowheg
    outdir = args.outdir
    title = args.title

    os.makedirs(outdir, exist_ok=True)

    with open(args.config, 'r') as f: 
        config_dict = yaml.safe_load(f)

    means = config_dict['means']
    stdvs = config_dict['stdvs']

    main(fsmeft=fsmeft, fpowheg=fpowheg, means=means, stdvs=stdvs, outdir=outdir, title=title)
