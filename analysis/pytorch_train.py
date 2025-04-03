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
import time

import matplotlib.pyplot as plt
import topcoffea.modules.utils as utils

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

# not using this now, probably could convert my loop below to work for this to make it easier to read
def train(model, dataloader, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    
    for batch_samples, batch_weights, batch_targets in dataloader:
        batch_samples, batch_weights, batch_targets = (
            batch_samples.to(device),
            batch_weights.to(device),
            batch_targets.to(device)
        )

        optimizer.zero_grad()
        outputs = model(batch_samples).squeeze(1)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += final_loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average Training Loss: {avg_loss:.4f}")


# this function comes from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# in section https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# probably I need something similar 
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def main():
	fSMEFT = "/afs/crc.nd.edu/user/h/hnelson2/dctr/analysis/dctr_SMEFT.pkl.gz"
	fpowheg = "/afs/crc.nd.edu/user/h/hnelson2/dctr/analysis/dctr_powheg.pkl.gz" 

	rando = 1234

	inputs_smeft= pickle.load(gzip.open(fSMEFT)).get()
	smeft_nevents = inputs_smeft.shape[0]

	# load the fpowheg file, use .query to only select events with positive weights, shuffle remaining events, then select the same number of events as the smeft sample
	inputs_powheg = (((pickle.load(gzip.open(fpowheg)).get()).query('weights>0')).sample(frac=1, random_state=rando).reset_index(drop=True)).iloc[:smeft_nevents]

	assert inputs_smeft.shape == inputs_powheg.shape, f"SMEFT and Powheg inputs are not the same shape.\n SMEFT shape: {inputs_smeft.shape} \n Powheg shape:{inputs_powheg.shape}"

	smeft_train = inputs_smeft.sample(frac=0.7, random_state=rando)
	smeft_test = inputs_smeft.drop(smeft_train.index)
	powheg_train = inputs_powheg.sample(frac=0.7, random_state=rando)
	powheg_test = inputs_powheg.drop(powheg_train.index)
	truth_smeft = np.ones_like(smeft_train['weights'])
	truth_powheg = np.zeros_like(powheg_train['weights'])

	weights_smeft = np.ones_like(smeft_train['weights'])
	weights_powheg = np.ones_like(powheg_train['weights'])

	z = torch.from_numpy(np.concatenate([smeft_train, powheg_train], axis=0).astype(np.float32))
	w = torch.from_numpy(np.concatenate([weights_smeft, weights_powheg], axis=0).astype(np.float32))
	y = torch.from_numpy(np.concatenate([truth_smeft, truth_powheg], axis=0).astype(np.float32))

	train_dataset = WeightedDataset(z, w, y)
	train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

	input_dim = z.shape[1]
	model = NeuralNetwork(input_dim)

	loss_fn = nn.BCELoss(reduction='mean')
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	nepochs = 50
	for epoch in range(nepochs):
		epoch_loss = 0.0
		model.train()	# sets the model in training mode. Crucial for layers that behave differently during training vs evaluation (e.g. dropout, mean, variance)
		for batch_samples, batch_weights, batch_targets in train_dataloader:
			optimizer.zero_grad()						# clear the gradients from the previous batch
			# forward pass
			outputs = model(batch_samples).squeeze(1)	# perform the forward pass to get the model's predictions
			loss = loss_fun(outputs, batch_targets)		# calculate the loss
			# backward pass
			loss.backward()								# calculate the gradients of the loss w.r.t. the model's parameters
			optimizer.step()							# update the model's parameters using the calculated gradients

			epoch_loss += loss.item()

		final_loss = epoch_loss / len(train_dataloader)
		print(f"Epoch {epoch+1}/{num_epochs}, Loss: {final_loss}")

if __name__=="__main__":
    main()