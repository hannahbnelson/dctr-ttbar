import datetime
import pickle
import gzip
import shutil
import os
import sys
from pathlib import Path
import argparse 
import logging

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
            nn.Linear(input_dim, 300),
            nn.LeakyReLU(),
            # nn.Linear(128,128),
            # nn.LeakyReLU(),
            nn.Linear(300, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main_module(x)

# # not using this now, probably could convert my loop below to work for this to make it easier to read
# def train(model, dataloader, loss_fn, optimizer, device="cpu"):
#     size = len(dataloader.dataset)
#     model.train()
#     total_loss = 0.0
    
#     for batch_samples, batch_weights, batch_targets in dataloader:
#         batch_samples, batch_weights, batch_targets = (
#             batch_samples.to(device),
#             batch_weights.to(device),
#             batch_targets.to(device)
#         )

#         optimizer.zero_grad()
#         outputs = model(batch_samples).squeeze(1)
#         loss = loss_fn(outputs, batch_targets)
#         loss.backward()
#         optimizer.step()

#         total_loss += final_loss.item()

#     avg_loss = total_loss / len(dataloader)
#     print(f"Average Training Loss: {avg_loss:.4f}")


# this function comes from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# in section https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# probably I need something similar 
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch_samples, batch_weights, batch_targets in dataloader:
            pred = model(batch_samples).squeeze(1)
            test_loss += loss_fun(outputs, batch_targets)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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


def main(outdir):

    rando = 1234
    torch.manual_seed(0)
    current_path = Path(__file__)
    base_path = outdir
    # base_path = "/users/hnelson2/dctr/training_outputs"
    # now = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    # make output subdirectories
    output_dir = os.path.join(base_path, "training_outputs")
    os.makedirs(output_dir, exist_ok=True)
    plotting_dir = os.path.join(base_path, "plots")
    os.makedirs(plotting_dir, exist_ok=True)

    logger_path = os.path.join(base_path, "output.log")
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(filename=logger_path, encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%d-%m-%Y %H:%M:%S')

    # copy run script to workign directory
    shutil.copy(current_path, os.path.join(base_path, "training.py"))
    # print(f"training script {current_path} copied to {os.path.join(base_path, 'training.py')}")
    logging.info(f"training script {current_path} copied to {os.path.join(base_path, 'training.py')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using device: {device}")

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
    means, stdvs = make_standardization_df(pd.concat([train_smeft, train_powheg, validation_smeft, validation_powheg]), outdir=base_path)

    # use that mean, stdv to standardize all datasets
    norm_train_smeft = standardize_df(train_smeft, means, stdvs)
    norm_train_powheg = standardize_df(train_powheg, means, stdvs)

    norm_val_smeft = standardize_df(validation_smeft, means, stdvs)
    norm_val_powheg = standardize_df(validation_powheg, means, stdvs)

    # create datasets 
    z = torch.from_numpy(np.concatenate([norm_train_smeft, norm_train_powheg], axis=0).astype(np.float32)).to(device)
    w = torch.from_numpy(np.concatenate([weights_train_smeft, weights_train_powheg], axis=0).astype(np.float32)).to(device)
    y = torch.from_numpy(np.concatenate([truth_train_smeft, truth_train_powheg], axis=0).astype(np.float32)).to(device)

    train_dataset = WeightedDataset(z, w, y)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    z_val = torch.from_numpy(np.concatenate([norm_val_smeft, norm_val_powheg], axis=0).astype(np.float32)).to(device)
    w_val = torch.from_numpy(np.concatenate([weights_validation_smeft, weights_validation_powheg], axis=0).astype(np.float32)).to(device)
    y_val = torch.from_numpy(np.concatenate([truth_validation_smeft, truth_validation_powheg], axis=0).astype(np.float32)).to(device)

    validation_dataset = WeightedDataset(z_val, w_val, y_val)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1024, shuffle=True)

    ### initialize model 
    input_dim = z.shape[1]
    model = NeuralNetwork(input_dim)
    model.to(device)

    loss_fn = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    training_outputs = {
        'epoch': [],
        'train_loss': [],
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
    logging.info("Starting training loop...")
    nepochs = 100
    for epoch in range(nepochs):
        ### model training
        epoch_loss = 0.0
        model.train()   # sets the model in training mode. Crucial for layers that behave differently during training vs evaluation (e.g. dropout, mean, variance)
        for batch_samples, batch_weights, batch_targets in train_dataloader:
            # only need these lines if I end up using a GPU 
            # batch_samples = batch_samples.to(device)
            # batch_weights = batch_weights.to(device)
            # batch_targets = batch_targets.to(device)

            optimizer.zero_grad()                       # clear the gradients from the previous batch
            # forward pass
            outputs = model(batch_samples).squeeze(1)   # perform the forward pass to get the model's predictions
            loss = loss_fn(outputs, batch_targets)      # calculate the loss ###loss = (loss_fn(outputs, batch_targets) * batch_weights).mean()
            # backward pass
            loss.backward()                             # calculate the gradients of the loss w.r.t. the model's parameters
            optimizer.step()                            # update the model's parameters using the calculated gradients

            epoch_loss += loss.item()

        train_loss_epoch = epoch_loss / len(train_dataloader)
        training_outputs['train_loss'].append(train_loss_epoch)

        logging.info(f"Epoch {epoch+1} training complete. Moving to validation")

        ### model validation
        model.eval() # sets the model in evaulation mode
        all_val_outputs = []
        all_val_targets = []
        total_val_loss = 0.0

        logging.info(f"...starting validation loop...")
        
        with torch.no_grad(): # disable gradient calculations during validation
            
            for batch_val_samples, batch_val_weights, batch_val_targets in validation_dataloader:
                
                # batch_val_samples = batch_val_samples.to(device)
                # batch_val_weights = batch_val_weights.to(device)
                # batch_val_targets = batch_val_targets.to(device)

                batch_val_outputs = model(batch_val_samples).squeeze(1)
                batch_val_loss = loss_fn(batch_val_outputs, batch_val_targets)
                
                total_val_loss += batch_val_loss.item()
                all_val_outputs.append(batch_val_outputs.cpu())
                all_val_targets.append(batch_val_targets.cpu())

            logging.info("...validation loop done, doing validation calculations")

            # add together outputs and targets from all batches
            val_outputs_all = torch.cat(all_val_outputs)
            val_targets_all = torch.cat(all_val_targets) 

            val_loss = total_val_loss / len(validation_dataloader)
            training_outputs['val_loss'].append(val_loss)

            val_predictions = (val_outputs_all > 0.5).float() # creates boolean tensor from outputs (0 to 1)
            
            val_true_labels_np = val_targets_all.numpy()
            val_predictions_np = val_predictions.numpy()
            val_probabilities_np = val_outputs_all.numpy() 

            # use sklearn to calculate metrics 
            acc = accuracy_score(val_true_labels_np, val_predictions_np)
            prec = precision_score(val_true_labels_np, val_predictions_np, zero_division=0) # zero_division=0 to handle cases with no positive predictions
            rec = recall_score(val_true_labels_np, val_predictions_np, zero_division=0)
            f1 = f1_score(val_true_labels_np, val_predictions_np, zero_division=0)
            roc_auc = roc_auc_score(val_true_labels_np, val_probabilities_np)
           
            tn, fp, fn, tp = confusion_matrix(val_true_labels_np, val_predictions_np, labels=[0, 1]).ravel()

            training_outputs['val_accuracy'].append(acc)
            training_outputs['val_precision'].append(prec)
            training_outputs['val_recall'].append(rec)
            training_outputs['val_f1_score'].append(f1)
            training_outputs['val_roc_auc'].append(roc_auc)

            training_outputs['val_true_pos'].append(tp)
            training_outputs['val_true_neg'].append(tn)
            training_outputs['val_false_pos'].append(fp)
            training_outputs['val_false_neg'].append(fn)

            training_outputs['epoch'].append(epoch+1)

        logging.info(f"Epoch: {epoch+1}/{nepochs},  Train Loss: {train_loss_epoch:.4f},  Validation Loss: {val_loss:.4f},  Validation Accuracy: {acc:.4f}")

        # Save Model Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            # print(f"  --> Model checkpoint saved at epoch {epoch+1}")
            logging.info(f"    --> Model checkpoint saved at epoch {epoch+1}")

    ### Save Training History to a File ###
    training_outputs_df = pd.DataFrame(training_outputs)
    training_outputs_path = os.path.join(output_dir, "training_outputs.csv")
    training_outputs_df.to_csv(training_outputs_path, index=False)
    # print(f"Training history saved to {training_outputs_path}")
    logging.info(f"Training history saved to {training_outputs_path}")

    # Save the final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    # print(f"  --> Final model saved to {final_model_ipath}")
    logging.info(f"    --> Final model saved to {final_model_path}")
    make_basic_plots(training_outputs, plotting_dir)

    #get predictions from last model epoch and make plots
    logging.info(f"Getting predictions on smeft and powheg validation datasets")
    smeft_predictions = get_predictions(model, torch.from_numpy(norm_val_smeft.to_numpy()))
    powheg_predictions = get_predictions(model, torch.from_numpy(norm_val_powheg.to_numpy()))
    logging.info(f"Making DNN ouptuts plot...")
    make_DNN_ouptuts_plot(smeft_predictions, powheg_predictions, plotting_dir)

    logging.info("Getting predictions on validation set...")
    validation_predictions = get_predictions(model, z_val.cpu())
    logging.info("Making ROC curve plot...")
    make_roc_plot(y_val.cpu().numpy(), validation_predictions, plotting_dir)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description = 'Customize inputs')
    parser.add_argument('--outdir', required="True", help='output directory absolute path')

    args = parser.parse_args()
    out = args.outdir

    main(outdir=out)


