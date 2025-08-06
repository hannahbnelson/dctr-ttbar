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

class WeightedDataset(Dataset):
    def __init__(self, data, weights, targets):
        self.data = data
        self.weights = weights
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        weights = self.weights[idx]
        target = self.targets[idx]
        return sample, weights, target

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()

        layers = []
        current_input_dim = input_dim

        for layer in config: 
            layer_type = layer['type']
            if layer_type == 'Linear':
                layers.append(nn.Linear(current_input_dim, layer['out_dim']))
                current_input_dim = layer['out_dim']
            elif layer_type == 'Activation':
                name = layer['name']
                if name == 'LeakyReLU': 
                    layers.append(nn.LeakyReLU()) 
                elif name == 'Sigmoid': 
                    layers.append(nn.Sigmoid())
                else: 
                    raise ValueError(f"Unknown Activation layer name: {name}")
            elif layer_type == 'Dropout':
                layers.append(nn.Dropout(layer['p']))
            else: 
                raise ValueError(f"Unknown layer type: {layer_type}")

        self.main_module = nn.Sequential(*layers) 

    def forward(self, x):
        return self.main_module(x)


def standardize_df(df, means, stdvs):
    # means and stdvs are separately computed on the whole dataset
    # means and stdvs are also dataframes

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


def make_DNN_ouptuts_plot(smeft_predictions, powheg_predictions, outdir, nepoch):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 100)

    ax.hist(smeft_predictions, bins=bins, histtype='step', label="smeft")
    ax.hist(powheg_predictions, bins=bins, histtype='step', label="powheg")

    ax.set_xlabel("NN output")
    ax.set_ylabel("Events")
    ax.legend(loc='best')

    outname = os.path.join(outdir, f"NNoutputs_epoch{nepoch}")
    fig.savefig(f"{outname}.png")
    print(f"figure saved in {outname}.png")

# Get predictions from loaded model
def get_predictions(model, data_input):
    all_probabilities = []

    model.eval() # set model into eval mode

    # 5. Disable gradient calculation during inference
    with torch.no_grad():
        inputs = data_input
        outputs = model(inputs)

        # For binary classification, outputs are probabilities (single value per sample)
        # Flatten to 1D array if outputs are (batch_size, 1)
        probabilities = outputs.squeeze(1).cpu().numpy()

        all_probabilities.extend(probabilities)

    return np.array(all_probabilities)


def main(indir, outdir, chkpt, config):

    rando = 1234
    torch.manual_seed(rando)
    params = config['params']

    # create validation datasets
    validation_smeft = pickle.load(gzip.open("/users/hnelson2/dctr/analysis/smeft_validation.pkl.gz")).drop(['weights'], axis=1)
    validation_powheg = pickle.load(gzip.open("/users/hnelson2/dctr/analysis/powheg_validation.pkl.gz")).drop(['weights'], axis=1)
    weights_validation_smeft = np.ones_like(validation_smeft['mtt'])
    weights_validation_powheg = np.ones_like(validation_powheg['mtt'])
    truth_validation_smeft = np.ones_like(validation_smeft['mtt'])
    truth_validation_powheg = np.zeros_like(validation_powheg['mtt'])

    # standardize inputs
    # find means and stdvs for each variable using all of the data
    # print(f"using input directory: {indir}")
    # means = pd.read_csv(os.path.join(indir, "standardization_means.csv")) 
    # stdvs = pd.read_csv(os.path.join(indir, "standardization_stds.csv"))
    # means, stdvs = make_standardization_df(pd.concat([train_smeft, train_powheg, validation_smeft, validation_powheg]), outdir=base_path)

    means = {'avg_top_pt': 34.263557, 
            'mtt': 522.141900,
            'top1pt': 126.859184,
            'top1eta': -0.257265,
            'top1phi': -0.000021,
            'top1mass': 172.253560,
            'top2pt': 124.636566,
            'top2eta': 0.257370,
            'top2phi': -0.000686,
            'top2mass': 172.265670,
    }

    stdvs = {'avg_top_pt': 38.252880, 
            'mtt': 175.306980,
            'top1pt': 84.604750,
            'top1eta': 1.823326,
            'top1phi': 1.813635,
            'top1mass': 5.346320,
            'top2pt': 82.644310,
            'top2eta': 1.829129,
            'top2phi': 1.813916,
            'top2mass': 5.329451,
    }

    norm_val_smeft = standardize_df(validation_smeft, means, stdvs)
    norm_val_powheg = standardize_df(validation_powheg, means, stdvs)

    z_val = torch.from_numpy(np.concatenate([norm_val_smeft, norm_val_powheg], axis=0).astype(np.float32))
    w_val = torch.from_numpy(np.concatenate([weights_validation_smeft, weights_validation_powheg], axis=0).astype(np.float32))
    y_val = torch.from_numpy(np.concatenate([truth_validation_smeft, truth_validation_powheg], axis=0).astype(np.float32))

    validation_dataset = WeightedDataset(z_val, w_val, y_val)
    validation_dataloader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=True)    

    model_architecture = config['model']
    input_dim = z_val.shape[1]

    for num in chkpt: 
        # load in model
        model = NeuralNetwork(input_dim, model_architecture)
        model_path = os.path.join(indir, f"training_outputs/model_epoch_{num}.pt")
        model.load_state_dict(torch.load(model_path))

        model.eval()

        #get predictions from last model epoch and make plots
        print(f"getting predictions for epoch {num}")
        smeft_predictions = get_predictions(model, torch.from_numpy(norm_val_smeft.to_numpy()))
        powheg_predictions = get_predictions(model, torch.from_numpy(norm_val_powheg.to_numpy()))
        print(f"making DNN outputs plot...")
        make_DNN_ouptuts_plot(smeft_predictions, powheg_predictions, outdir, num)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description = 'Customize inputs')
    parser.add_argument('--indir', required=True, help='input directory that has standardization csv and model checkpoints')
    parser.add_argument('--outdir', required=True, help='output directory absolute path')

    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir

    # make output directory if it doesn't already exist (for running locally)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    config_path = os.path.join(indir, "config.yaml")
    with open(config_path, 'r') as f: 
        config_dict = yaml.safe_load(f)

    nepochs = config_dict['params']['nepochs']
    checkpoint_frequency = config_dict['monitoring']['checkpoint_frequency']

    list_of_checkpoints = np.arange(start=0, stop=nepochs+1, step=checkpoint_frequency)
    # list_of_checkpoints = np.arange(start=0, stop=121, step=checkpoint_frequency)
    list_of_checkpoints=list_of_checkpoints[1:]

    main(indir=indir, outdir=outdir, chkpt=list_of_checkpoints, config=config_dict)


