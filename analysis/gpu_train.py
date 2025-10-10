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
import json

import hist
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import optim

import mplhep as hep
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score

class WeightedDataset(Dataset):
    def __init__(self, data, weights, targets):
        # Convert all data to PyTorch Tensors immediately in __init__ 
        # This prevents the slow, per-item conversion inside __getitem__

        # Data (features) - Convert DataFrame to tensor
        if isinstance(data, pd.DataFrame):
            self.data = torch.from_numpy(data.to_numpy().astype(np.float32))
        else:
            self.data = data
        
        # Weights - Convert 1D NumPy array to tensor
        if isinstance(weights, np.ndarray):
             # Ensure weights are a single column tensor
             self.weights = torch.from_numpy(weights.astype(np.float32)).unsqueeze(1)
        else:
             self.weights = weights
        
        # Targets - Convert 1D NumPy array to tensor
        if isinstance(targets, np.ndarray):
             # Ensure targets are a single column tensor
             self.targets = torch.from_numpy(targets.astype(np.float32)).unsqueeze(1)
        else:
             self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Now __getitem__ just returns pre-converted Tensors, which is very fast
        sample = self.data[idx]
        weights = self.weights[idx]
        target = self.targets[idx]
        return sample, weights.squeeze(), target.squeeze() # Squeeze to match expected 1D output


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


def make_basic_plots(metrics, outdir):
    hep.style.use("CMS")
    basic_plots = {"training_loss": metrics['train_loss'], 
                    "validation_loss": metrics['val_loss'], 
                    "validation_accuracy": metrics['val_accuracy'],
                    "validation_precision": metrics['val_precision'], 
                    "validation_recall": metrics['val_recall'],
                    }

    for item in basic_plots.keys(): 
        fig, ax = plt.subplots()
        ax.plot(metrics['epoch'], basic_plots[item])    
        ax.set_xlabel("epoch")
        ax.set_ylabel(item)
        ax.set_title(item)

        outname = os.path.join(outdir, item)
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


def main(outdir, config, cores=1):

    rando = 1234
    torch.manual_seed(rando)
    current_path = Path(__file__)
    base_path = outdir

    logger_path = os.path.join(base_path, "output.log")
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(filename=logger_path, encoding='utf-8', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%d-%m-%Y %H:%M:%S')

    logger.info(f"base_path: {base_path}")
    logger.info(f"cores: {cores}")

    # make output subdirectories
    output_dir = os.path.join(base_path, "training_outputs")
    os.makedirs(output_dir, exist_ok=True)
    plotting_dir = os.path.join(base_path, "plots")
    os.makedirs(plotting_dir, exist_ok=True)

    # set device and load in hyperparameters from config file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using device: {device}")

    params = config['params']
    inputs = config['inputs']

    means = config['standardization']['means']
    stdvs = config['standardization']['stdvs']

    ### create training datasets
    train_smeft = pickle.load(gzip.open(inputs['train_smeft'])).drop(['weights'], axis=1)
    norm_train_smeft = standardize_df(train_smeft, means, stdvs)
    del train_smeft

    input_dim = norm_train_smeft.shape[1] 
    weights_train_smeft = np.ones_like(norm_train_smeft['mtt'], dtype=np.float32)
    truth_train_smeft = np.zeros_like(norm_train_smeft['mtt'], dtype=np.float32)

    train_smeft_np = norm_train_smeft.to_numpy().astype(np.float32)
    del norm_train_smeft

    smeft_dataset = WeightedDataset(
        data=train_smeft_np, 
        weights=weights_train_smeft, 
        targets=truth_train_smeft
    )

    logging.info(f"created SMEFT dataset")
    del train_smeft_np
    del weights_train_smeft
    del truth_train_smeft

    train_powheg = pickle.load(gzip.open(inputs['train_powheg'])).drop(['weights'], axis=1)
    norm_train_powheg = standardize_df(train_powheg, means, stdvs)
    del train_powheg

    weights_train_powheg = np.ones_like(norm_train_powheg['mtt'], dtype=np.float32)
    truth_train_powheg = np.ones_like(norm_train_powheg['mtt'], dtype=np.float32)

    train_powheg_np = norm_train_powheg.to_numpy().astype(np.float32)
    del norm_train_powheg

    powheg_dataset = WeightedDataset(
        data=train_powheg_np, 
        weights=weights_train_powheg, 
        targets=truth_train_powheg
    )
    logging.info(f"created POWHEG dataset")
    del train_powheg_np
    del weights_train_powheg
    del truth_train_powheg

    train_dataloader = DataLoader(ConcatDataset([smeft_dataset, powheg_dataset]), batch_size=params['batch_size'], shuffle=True, num_workers=cores)
    logging.info(f"created training dataloader")

    ### create validation datasets 
    val_smeft = pickle.load(gzip.open(inputs['validation_smeft'])).drop(['weights'], axis=1)
    norm_val_smeft = standardize_df(val_smeft, means, stdvs)
    del val_smeft

    weights_val_smeft = np.ones_like(norm_val_smeft['mtt'], dtype=np.float32)
    truth_val_smeft = np.ones_like(norm_val_smeft['mtt'], dtype=np.float32)

    val_smeft_np = norm_val_smeft.to_numpy().astype(np.float32)
    del norm_val_smeft

    val_smeft_dataset = WeightedDataset(
        data=val_smeft_np, 
        weights=weights_val_smeft, 
        targets=truth_val_smeft
    )

    logging.info(f"created SMEFT validation dataset")
    del val_smeft_np
    del weights_val_smeft
    del truth_val_smeft

    val_powheg = pickle.load(gzip.open(inputs['validation_powheg'])).drop(['weights'], axis=1)
    norm_val_powheg = standardize_df(val_powheg, means, stdvs)
    del val_powheg

    weights_val_powheg = np.ones_like(norm_val_powheg['mtt'], dtype=np.float32)
    truth_val_powheg = np.zeros_like(norm_val_powheg['mtt'], dtype=np.float32)

    val_powheg_np = norm_val_powheg.to_numpy().astype(np.float32)
    del norm_val_powheg

    val_powheg_dataset = WeightedDataset(
        data=val_powheg_np, 
        weights=weights_val_powheg, 
        targets=truth_val_powheg
    )

    logging.info(f"created POWHEG validation dataset")
    del val_powheg_np
    del weights_val_powheg
    del truth_val_powheg

    validation_dataloader = DataLoader(ConcatDataset([val_smeft_dataset, val_powheg_dataset]), batch_size=params['batch_size'], shuffle=True, num_workers=0)
    logging.info(f"created validation dataloader")

    ### initialize model 
    model_architecture = config['model']
    model = NeuralNetwork(input_dim, model_architecture)
    model.to(device)
    logging.info(f"created model")

    loss_fn = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                    factor=params['sched_factor'], patience=params['sched_patience'], 
                    threshold=params['sched_threshold'], threshold_mode='abs')

    logging.info(" -------- Model Architecture -------- ")
    logging.info(str(model))

    training_outputs = {
        'epoch': [],
        'train_loss': [],
    }

    validation_outputs = {
        'epoch': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': [],
        'val_roc_auc': [],
        'val_true_pos': [],
        'val_true_neg': [],
        'val_false_pos': [],
        'val_false_neg': [],
    }

    best_val_accuracy=0.0
    best_epoch = -1

    ### training loop 
    nepochs = params['nepochs']
    logging.info(f"\n\n -------- TRAINING LOOP: {nepochs} epochs total --------")
    for epoch in range(nepochs):
        ### model training
        # epoch_loss = 0.0
        epoch_loss_gpu = torch.tensor(0.0, device=device) # GPU tensor
        model.train()   # sets the model in training mode. Crucial for layers that behave differently during training vs evaluation (e.g. dropout, mean, variance)
        for batch_samples, batch_weights, batch_targets in train_dataloader:

            batch_samples = batch_samples.to(device, dtype=torch.float32)
            batch_weights = batch_weights.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()                       # clear the gradients from the previous batch
            # forward pass
            outputs = model(batch_samples).squeeze(1)   # perform the forward pass to get the model's predictions
            loss = loss_fn(outputs, batch_targets)      # calculate the loss ###loss = (loss_fn(outputs, batch_targets) * batch_weights).mean()
            # backward pass
            loss.backward()                             # calculate the gradients of the loss w.r.t. the model's parameters
            optimizer.step()                            # update the model's parameters using the calculated gradients

            epoch_loss_gpu += loss.detach() # Keep loss on GPU and accumulate

        train_loss_epoch = epoch_loss_gpu.item() / len(train_dataloader) #once per epoch, the loss is moved from gpu to cpu
        training_outputs['train_loss'].append(train_loss_epoch)
        training_outputs['epoch'].append(epoch+1)

        all_val_outputs = []
        all_val_targets = []
        epoch_val_loss = torch.tensor(0.0, device=device)
        model.eval()    # sets the model in evaluation mode
        with torch.no_grad(): # disable gradient calculations during validation
            for batch_val_samples, batch_val_weights, batch_val_targets in validation_dataloader: 

                batch_val_samples = batch_val_samples.to(device, dtype=torch.float32)
                batch_val_weights = batch_val_weights.to(device)
                batch_val_targets = batch_val_targets.to(device) 

                batch_val_outputs = model(batch_val_samples).squeeze(1)
                batch_val_loss = loss_fn(batch_val_outputs, batch_val_targets)

                epoch_val_loss += batch_val_loss.detach() # Keep loss on GPU and accumulate
                all_val_targets.append(batch_val_targets.detach())
                all_val_outputs.append(batch_val_outputs.detach())


        val_loss_epoch = epoch_val_loss.item()/len(validation_dataloader)
        validation_outputs['val_loss'].append(val_loss_epoch)
        validation_outputs['epoch'].append(epoch+1)    

        val_targets = torch.cat(all_val_targets).cpu().numpy()
        val_outputs_all = torch.cat(all_val_outputs).cpu().numpy()
        val_predictions = (val_outputs_all > 0.5).astype(int) # creates boolean tensor from outputs (0 to 1)

        val_roc_auc = roc_auc_score(val_targets, val_predictions) 
        validation_outputs['val_roc_auc'].append(val_roc_auc)

        scheduler.step(val_loss_epoch)

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch+1}/{nepochs}, "
                     f"    Training Loss: {train_loss_epoch:.6f}, " 
                     f"    Validation Loss: {val_loss_epoch:.6f}, "
                     f"    Current LR: {current_lr:.10f}")

    logging.info(f"-------- TRAINING LOOP FINISHED ({nepochs} completed) -------- \n\n")

    ### Save Training/Validation Metrics to yaml ###
    training_outputs_path = os.path.join(output_dir, "training_metrics.yaml")
    with open(training_outputs_path, 'w') as f:
        yaml.safe_dump(training_outputs, f)
    logging.info(f"training metrics saved to {training_outputs_path}")

    validation_outputs_path = os.path.join(output_dir, "validation_metrics.yaml")
    with open(validation_outputs_path, 'w') as f:
        yaml.safe_dump(validation_outputs, f)
    logging.info(f"validation metrics saved to {validation_outputs_path}")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description = 'Customize inputs')
    parser.add_argument('--config', required=True, help='configuration yml containing hyperparameters')
    parser.add_argument('--outdir', required=True, help='output directory absolute path')
    parser.add_argument('--cores', required=False, type=int, default=1, help='number of cores to run on')

    args = parser.parse_args()
    out = args.outdir
    ncores = args.cores
    config = args.config

    with open(args.config, 'r') as f: 
        config_dict = yaml.safe_load(f)

    # make output directory if it doesn't already exist (for running locally)
    if not os.path.exists(out):
        os.makedirs(out, exist_ok=False)

    main(outdir=out, config=config_dict, cores = ncores)