import numpy as np
import pandas as pd
import awkward as ak

import os
import pickle
import gzip
import argparse
import yaml

from sklearn.model_selection import train_test_split
# from dctr.modules import DNN_tools 

def main(fsmeft_train, fpowheg_train, fsmeft_val, fpowheg_val, outdir, title):
    rando = 1234

    inputs_smeft_train = pickle.load(gzip.open(fsmeft_train)).get()
    inputs_smeft_val = pickle.load(gzip.open(fsmeft_val)).get()

    inputs_powheg_train = pickle.load(gzip.open(fpowheg_train)).get().query('weights>0') # only work with non negative powheg events (~0.4% events are negative)
    inputs_powheg_val = pickle.load(gzip.open(fpowheg_val)).get().query('weights>0') # only work with non negative powheg events (~0.4% events are negative)

    # split the smeft dataset into training and validation (no need for test set, that will be from centrally produced mtt samples)
    # smeft_training, smeft_validation = train_test_split(inputs_smeft, test_size=0.3, random_state=rando)

    inputs_smeft_train = inputs_smeft_train.head(4000000)
    
    # get the number of events in each smeft dataset 
    num_smeft_train = inputs_smeft_train.shape[0]
    num_smeft_val = inputs_smeft_val.shape[0] 
    total_smeft_events = num_smeft_train + num_smeft_val

    powheg_training, temp = train_test_split(inputs_powheg_train, train_size=num_smeft_train, random_state = rando)
    powheg_validation, temp = train_test_split(inputs_powheg_val, train_size=num_smeft_val, random_state=rando)

    # save new datasets so that they can be loaded in individually as needed 
    inputs_smeft_train.to_pickle(os.path.join(outdir, f"{title}_inputs_smeft_training.pkl.gz"), compression='gzip')
    print(f"smeft training dataset saved to: {title}_inputs_smeft_training.pkl.gz")

    inputs_smeft_val.to_pickle(os.path.join(outdir, f"{title}_inputs_smeft_validation.pkl.gz"), compression='gzip')
    print(f"smeft validation dataset saved to: {title}_inputs_smeft_validation.pkl.gz")

    powheg_training.to_pickle(os.path.join(outdir, f"{title}_inputs_powheg_training.pkl.gz"), compression='gzip')
    print(f"powheg training dataset saved to: {title}_inputs_powheg_training.pkl.gz")

    powheg_validation.to_pickle(os.path.join(outdir, f"{title}_inputs_powheg_validation.pkl.gz"), compression='gzip')
    print(f"powheg validation dataset saved to: {title}_inputs_powheg_validation.pkl.gz")

    # powheg_test.to_pickle(os.path.join(outdir, f"{title}_inputs_powheg_test.pkl.gz"), compression='gzip')
    # print(f"powheg test dataset saved to: {title}_inputs_powheg_test.pkl.gz")

    ### Make standardization dataframe and save as a yaml
    total_df_standardizing = pd.concat([inputs_smeft_train, inputs_smeft_val, powheg_training, powheg_validation])

    means = total_df_standardizing.mean()
    stdvs = total_df_standardizing.std()

    standardizations = {
        'means': means.to_dict(),
        'stdvs': stdvs.to_dict(),
    }

    yaml_path = os.path.join(outdir, f"{title}_standardization.yaml")
    with open(yaml_path, 'w') as file: 
        yaml.safe_dump(standardizations, file)
    print(f"standardization constants saved to {yaml_path}")


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('--fsmeft_train', required=True, help='pkl file for smeft dataframe')
    parser.add_argument('--fsmeft_val', required=True, help='pkl file for smeft dataframe')
    parser.add_argument('--fpowheg_train', required=True, help='pkl file for powheg dataframe')
    parser.add_argument('--fpowheg_val', required=True, help='pkl file for powheg dataframe')
    parser.add_argument('--outdir', '-o', default='.', help='output directory')
    parser.add_argument('--title', default='pytorch', help='text to add at the beginning of pkl files')

    args = parser.parse_args()
    fsmeft_train = args.fsmeft_train
    fpowheg_train = args.fpowheg_train
    fsmeft_val = args.fsmeft_val
    fpowheg_val = args.fpowheg_val
    outdir = args.outdir
    title = args.title

    main(fsmeft_train, fpowheg_train, fsmeft_val, fpowheg_val, outdir, title)
