import numpy as np
import pandas as pd
import awkward as ak

import os
import pickle
import gzip

from sklearn.model_selection import train_test_split

def main():
    # fSMEFT = "/users/hnelson2/dctr/analysis/1807_dctr_SMEFT.pkl.gz"
    # fpowheg = "/users/hnelson2/dctr/analysis/1807_dctr_powheg.pkl.gz" 

    rando = 1234

    inputs_smeft = pickle.load(gzip.open(fSMEFT)).get()
    inputs_powheg = pickle.load(gzip.open(fpowheg)).get().query('weights>0') # only work with non negative powheg events (~0.4% events are negative)

    # inputs_powheg = inputs_powheg.query('(top1mass > 150.0) and (top2mass > 150.0) and (top1mass < 195.0) and (top2mass < 195.0)')

    # # could also separate after making a tensor but I think this is not as convenient to save right now
    # #  train, validate, test = torch.utils.data.random_split(Dataset(), [0.7, 0.15, 0.15], generator=Generator().manual_seed(rando))

    # split the smeft dataset into training and validation (no need for test set, that will be from centrally produced mtt samples)
    smeft_training, smeft_validation = train_test_split(inputs_smeft, test_size=0.3, random_state=rando)
    
    # get the number of events in each smeft dataset 
    num_smeft_train = smeft_training.shape[0]
    num_smeft_val = smeft_validation.shape[0] 
    total_smeft = num_smeft_train + num_smeft_val

    # split the powheg into training, validation, testing
    # where training and validation sets have the same number of events as smeft training/validation
    powheg_working, powheg_test = train_test_split(inputs_powheg, train_size=total_smeft, random_state=rando)
    powheg_training, powheg_validation = train_test_split(powheg_working, test_size=0.3, random_state=rando)

    # save new datasets so that they can be loaded in individually as needed 
    outdir = "/users/hnelson2/dctr/analysis"

    smeft_training.to_pickle(os.path.join(outdir, "torch_inputs_smeft_training.pkl.gz"), compression='gzip')
    print(f"smeft training dataset saved to: torch_inputs_smeft_training.pkl.gz")

    smeft_validation.to_pickle(os.path.join(outdir, "torch_inputs_smeft_validation.pkl.gz"), compression='gzip')
    print(f"smeft validation dataset saved to: torch_inputs_smeft_validation.pkl.gz")

    powheg_training.to_pickle(os.path.join(outdir, "torch_inputs_powheg_training.pkl.gz"), compression='gzip')
    print(f"powheg training dataset saved to: torch_inputs_powheg_training.pkl.gz")

    powheg_validation.to_pickle(os.path.join(outdir, "torch_inputs_powheg_validation.pkl.gz"), compression='gzip')
    print(f"powheg validation dataset saved to: torch_inputs_powheg_validation.pkl.gz")

    powheg_test.to_pickle(os.path.join(outdir, "torch_inputs_powheg_test.pkl.gz"), compression='gzip')
    print(f"powheg test dataset saved to: torch_inputs_powheg_test.pkl.gz")


if __name__=="__main__":
    main()
