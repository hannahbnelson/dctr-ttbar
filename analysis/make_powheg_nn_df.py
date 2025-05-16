import numpy as np
import pandas as pd
import awkward as ak

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch import optim

import pickle
import gzip
# import logging
# import time

# import matplotlib.pyplot as plt
# import topcoffea.modules.utils as utils


def main():
    fSMEFT = "/afs/crc.nd.edu/user/h/hnelson2/dctr/analysis/dctr_SMEFT.pkl.gz"
    fpowheg = "/afs/crc.nd.edu/user/h/hnelson2/dctr/analysis/dctr_powheg.pkl.gz" 

    rando = 1234

    inputs_smeft= pickle.load(gzip.open(fSMEFT)).get()
    smeft_nevents = inputs_smeft.shape[0]

    # load the fpowheg file, use .query to only select events with positive weights, shuffle remaining events, then select the same number of events as the smeft sample
    inputs_powheg = (((pickle.load(gzip.open(fpowheg)).get()).query('weights>0')).sample(frac=1, random_state=rando).reset_index(drop=True)).iloc[:smeft_nevents]

    assert inputs_smeft.shape == inputs_powheg.shape, f"SMEFT and Powheg inputs are not the same shape.\n SMEFT shape: {inputs_smeft.shape} \n Powheg shape:{inputs_powheg.shape}"

    inputs_powheg.to_pickle('dctr_powheg_skimmed.pkl.gz', compression='gzip')
    print(f"skimmed powheg dataframe saved to: dctr_powheg_skimmed.pkl.gz")


if __name__=="__main__":
    main()