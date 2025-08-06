import pickle
import gzip

import numpy as np
import pandas as pd
import awkward as ak

import mplhep as hep
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score


def make_basic_plots(metrics, outdir):
    hep.style.use("CMS")
    for name, vals in metrics.items(): 
        fig, ax = plt.subplots()
        ax.plot(metrics['epoch'], vals)    
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.set_title(name)

        outname = os.path.join(outdir, name)
        fig.savefig(f"{outname}.png")
        logging.info(f"figure saved in {outname}.png")


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
    logging.info(f"figure saved in {outname}.png")


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
    logging.info(f"figure saved in {outname}.png")

