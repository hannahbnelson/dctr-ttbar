import numpy as np
import awkward as ak
import pandas as pd
import uproot
import os

import hist

import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle
import gzip
import logging
import datetime

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score


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
    def __init__(self, input_dim):
        super().__init__()
        self.main_module= nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            # nn.Linear(128,128),
            # nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main_module(x)


# Get predictions from loaded model
def get_predictions(model, data_input)
    all_probabilities = []

    model.eval() # set model into eval mode

    # 5. Disable gradient calculation during inference
    with torch.no_grad():
        inputs = data_input
        outputs = model(inputs)

        # For binary classification, outputs are probabilities (single value per sample)
        # Flatten to 1D array if outputs are (batch_size, 1)
        probabilities = outputs.squeeze(1).numpy()

        all_probabilities.extend(probabilities)

    return np.array(all_probabilities)


def make_basic_plots(metrics, outdir):
    
    basic_plots = {"training_loss": metrics['train_loss'].to_numpy(), 
                    "validation_loss": metrics['val_loss'].to_numpy(), 
                    "validation_accuracy": metrics['val_accuracy'].to_numpy(),
                    "validation_precision": metrics['val_precision'].to_numpy(), 
                    "validation_recall": metrics['val_recall'].to_numpy(),
                    }

    for item in basic_plots.keys(): 
        fig, ax = plt.subplots()
        ax.plot(metrics['epoch'], basic_plots[item])    
        ax.set_xlabel("epoch")
        ax.set_ylabel(item)
        ax.set_title(item)

        outname = os.path.join(outdir, item)
        fig.savefig(f"{outname}.png")
        print(f"figure saved in {outname}.png")


def make_DNN_ouptuts_plot(smeft_predictions, powheg_predictions, outdir):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 100)

    ax.hist(smeft_predictions, bins=bins, histtype='step', label="smeft")
    ax.hist(powheg_predictions, bins=bins, histtype='step', label="powheg")

    ax.set_xlabel("NN output")
    ax.set_ylabel("Events")
    ax.legend(loc='best')

    outname = os.path.join(outdir, "NNoutputs")
    fig.savefig(f"{outname}.png")
    print(f"figure saved in {outname}.png")


def make_roc_plot(true_labels, probabilities, outdir):
    hep.style.use("CMS")
    fpr, tpr, threshold = roc_curve(true_labels, probabilities)
    roc_auc = roc_auc_score(true_labels, probabilities)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve')
    ax.grid(True)

    outname = os.path.join(outdir, "ROC")
    fig.savefig(f"{outname}.png")
    print(f"figure saved in {outname}.png")

def normalize_df(df):

    # make a copy as to not change original df
    norm_df = df.copy()

    # select only numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns
    means = df.mean()
    stdvs = df.std()

    # if stdv is 0, set to 0
    # if stdv is not 0, normalized = (orig - mean)/stdv
    for col in norm_df:
        if stdvs[col] != 0:
            norm_df[col] = (df[col] - means[col])/(stdvs[col])
        else:
            norm_df[col] = 0.0

    return norm_df


def compute_r(model, z):

    model.eval() # set model in eval mode 
    with torch.no_grad(): # disable gradient calculation 
        f_z = model(z)

    return (f_z / (1-f_z + 1e-8)).numpy()


def main(outdir):

    rando = 1234
    torch.manual_seed(0)
    base_path = outdir
    plotting_dir = os.path.join(base_path, "plots")
    outputs_dir = os.path.join(base_path, "training_outputs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create training datsets
    train_smeft = pickle.load(gzip.open("/users/hnelson2/dctr/analysis/smeft_training.pkl.gz")).drop(['weights'], axis=1)
    train_powheg = pickle.load(gzip.open("/users/hnelson2/dctr/analysis/powheg_training.pkl.gz")).drop(['weights'], axis=1)
    z = torch.from_numpy(np.concatenate([train_smeft, train_powheg], axis=0).astype(np.float32))

    # create validation datasets
    validation_smeft = pickle.load(gzip.open("/users/hnelson2/dctr/analysis/smeft_validation.pkl.gz")).drop(['weights'], axis=1)
    validation_powheg = pickle.load(gzip.open("/users/hnelson2/dctr/analysis/powheg_validation.pkl.gz")).drop(['weights'], axis=1)
    weights_validation_smeft = np.ones_like(validation_smeft['mtt'])
    weights_validation_powheg = np.ones_like(validation_powheg['mtt'])
    truth_validation_smeft = np.ones_like(validation_smeft['mtt'])
    truth_validation_powheg = np.zeros_like(validation_powheg['mtt'])

    # normalize smeft inputs
    numerical_cols = validation_smeft.select_dtypes(include=np.number).columns

    norm_val_smeft = normalize_df(validation_smeft)
    norm_val_powheg = normalize_df(validation_powheg)
    
    z_val = torch.from_numpy(np.concatenate([norm_val_smeft, norm_val_powheg], axis=0).astype(np.float32))
    w_val = torch.from_numpy(np.concatenate([weights_validation_smeft, weights_validation_powheg], axis=0).astype(np.float32))
    y_val = torch.from_numpy(np.concatenate([truth_validation_smeft, truth_validation_powheg], axis=0).astype(np.float32))

    validation_dataset = WeightedDataset(z_val, w_val, y_val)
    validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=True)

    input_dim = z.shape[1]
    
    model = NeuralNetwork(input_dim)

    model_path = os.path.join(outputs_dir, "final_model.pth")
    model.load_state_dict(torch.load(model_path))

    model.eval()

    #get predictions from last model epoch and make plots
    print(f"getting predictions for smeft and powheg validation datasets")
    smeft_predictions = get_predictions(model, torch.from_numpy(norm_val_smeft.to_numpy()))
    powheg_predictions = get_predictions(model, torch.from_numpy(norm_val_powheg.to_numpy()))
    print(f"making DNN outputs plot...")
    make_DNN_ouptuts_plot(smeft_predictions, powheg_predictions, plotting_dir)
    print(f"done making DNN ouptuts plot.")

    print("\nGetting predictions on validation set...")
    validation_predictions = get_predictions(model, z_val, y_val)
    print("Prediction collection complete.")
    make_roc_plot(y_val, validation_predictions, plotting_dir) 


if __name__=="__main__":

    parser = argparse.ArgumentParser(description = 'Customize inputs')
    parser.add_argument('--outdir', required="True", help='output directory absolute path')

    args = parser.parse_args()
    out = args.outdir

    main(outdir=out)