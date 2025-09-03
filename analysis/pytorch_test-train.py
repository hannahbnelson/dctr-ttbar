import numpy as np
import pandas as pd
import awkward as ak

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

import pickle
import gzip
import logging
import datetime

import os
import matplotlib.pyplot as plt
# import topcoffea.modules.utils as utils

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score

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
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
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


def main():

    outdir = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(outdir, exist_ok=False)

    rando = 1234

    np.random.seed(rando)
    N_0, N_1 = 50000, 50000
    lambda_0, lambda_1 = 1.0, 0.5

    inputs_smeft = pd.DataFrame(np.random.exponential(scale=1/lambda_0, size=N_0).reshape(-1,1).astype(np.float32))
    inputs_powheg = pd.DataFrame(np.random.exponential(scale=1/lambda_1, size=N_1).reshape(-1,1).astype(np.float32))

    smeft_train = inputs_smeft.sample(frac=0.7, random_state=rando)
    smeft_test = inputs_smeft.drop(smeft_train.index)
    powheg_train = inputs_powheg.sample(frac=0.7, random_state=rando)
    powheg_test = inputs_powheg.drop(powheg_train.index)
    truth_smeft = np.ones_like(smeft_train[0])
    truth_powheg = np.zeros_like(powheg_train[0])

    weights_smeft = np.ones_like(smeft_train[0])
    weights_powheg = np.ones_like(powheg_train[0])

    z = torch.from_numpy(np.concatenate([smeft_train, powheg_train], axis=0).astype(np.float32))
    w = torch.from_numpy(np.concatenate([weights_smeft, weights_powheg], axis=0).astype(np.float32))
    y = torch.from_numpy(np.concatenate([truth_smeft, truth_powheg], axis=0).astype(np.float32))

    train_dataset = WeightedDataset(z, w, y)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_z = torch.from_numpy(np.concatenate([smeft_test, powheg_test], axis=0).astype(np.float32))
    test_y = torch.from_numpy(np.concatenate([np.ones_like(smeft_test[0]), np.zeros_like(powheg_test[0])], axis=0).astype(np.float32))

    input_dim = z.shape[1]
    model = NeuralNetwork(input_dim)

    loss_fn = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

    nepochs = 200
    for epoch in range(nepochs):
        epoch_loss = 0.0
        model.train()   # sets the model in training mode. Crucial for layers that behave differently during training vs evaluation (e.g. dropout, mean, variance)
        for batch_samples, batch_weights, batch_targets in train_dataloader:
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

        model.eval() # sets the model in evaulation mode
        with torch.no_grad(): # disable gradient calculations during validation
            val_outputs = model(test_z).squeeze(1) 
            val_loss = loss_fn(val_outputs, test_y).item()
            training_outputs['val_loss'].append(val_loss)

            val_predictions = (val_outputs > 0.5).float()
            
            val_true_labels_np = test_y.numpy()
            val_predictions_np = val_predictions.numpy()
            val_probabilities_np = val_outputs.numpy() 

            # use sklearn to calculate metrics 
            acc = accuracy_score(val_true_labels_np, val_predictions_np)
            prec = precision_score(val_true_labels_np, val_predictions_np)
            rec = recall_score(val_true_labels_np, val_predictions_np)
            f1 = f1_score(val_true_labels_np, val_predictions_np)
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

        print(f"Epoch: {epoch+1}/{nepochs}, "
                f"Train Loss: {train_loss_epoch:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {acc:.4f}"
            )

        # Model Checkpoint
        # Save the model if validation accuracy improves
        if acc > best_val_accuracy:
            best_val_accuracy = acc
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(outdir, f"best_model_epoch_{best_epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> New best model saved at epoch {best_epoch} with Val Acc: {best_val_accuracy}:.4f")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(outdir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Model checkpoint saved at epoch {epoch+1}")

    print(f"\nTraining complete. Best Validation Accuracy: {best_val_accuracy:.4f} at Epoch {best_epoch}")

    ### Save Training History to a File ###
    training_outputs_df = pd.DataFrame(training_outputs)
    training_outputs_path = os.path.join(outdir, "training_outputs.csv")
    training_outputs_df.to_csv(training_outputs_path, index=False)
    print(f"Training history saved to {training_outputs_path}")

    # Save the final model
    final_model_path = os.path.join(outdir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"  --> Final model saved to {final_model_path}")

if __name__=="__main__":
    main()
