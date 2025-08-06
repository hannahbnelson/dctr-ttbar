import pickle
import gzip
import yaml

import numpy as np
import pandas as pd
import awkward as ak

import mplhep as hep
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

# means and stdv to standardize pd df for input into trained model
# these are calcualted using the smeft/powheg train and validation datasets 
# negative weights and top mass cuts applied before creating the datasets
# see dctr/analysis/make_pytorch_dataset.py
means = {'avg_top_pt': 34.37594,
        'mtt': 522.8204,
        'top1pt': 145.54019,
        'top1eta': -0.00080191356,
        'top1phi': 0.0002472976,
        'top1mass': 172.4881,
        'top2pt': 106.316284,
        'top2eta': 0.0002492254,
        'top2phi': -0.00088213355,
        'top2mass': 172.44499,}
stdvs = {'avg_top_pt': 38.34243,
        'mtt': 175.4714,
        'top1pt': 88.27853,
        'top1eta': 1.6881704,
        'top1phi': 1.8136373,
        'top1mass': 3.0224695,
        'top2pt': 73.93133,
        'top2eta': 1.9869514,
        'top2phi': 1.8138397,
        'top2mass': 3.067736,}

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

def compute_reweights(predictions): 

    f_z = predictions
    weights = np.divide(f_z, (1-f_z + 1e-8))

    return weights

def load_saved_model(config_path, model_path, indim): 

    with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

    model_architecture = config_dict['model']
    # input_dim = norm_NN_inputs.shape[1]
    model = NeuralNetwork(indim, model_architecture)

    model.load_state_dict(torch.load(model_path))

    return model


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


def make_df_for_DNN(genpart):

    is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])
    gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)
    gen_top = gen_top[ak.argsort(gen_top.pt, axis=1, ascending=False)]

    ### Fill df with inputs to run through trained model
    variables_to_fill_df = {
        "avg_top_pt": np.divide(gen_top.sum().pt, 2.0),
        "mtt"       : (gen_top[:,0] + gen_top[:,1]).mass,
        "top1pt"    : gen_top.pt[:,0],
        "top1eta"   : gen_top.eta[:,0],
        "top1phi"   : gen_top.phi[:,0],
        "top1mass"  : gen_top.mass[:,0],
        "top2pt"    : gen_top.pt[:,1],
        "top2eta"   : gen_top.eta[:,1],
        "top2phi"   : gen_top.phi[:,1],
        "top2mass"  : gen_top.mass[:,1],
    }

    norm_NN_inputs = standardize_df(pd.DataFrame.from_dict(variables_to_fill_df), means, stdvs)

    return norm_NN_inputs


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
    # means and stdvs are also pandas dataframes

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


