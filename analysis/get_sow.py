import os
import argparse

# import pickle
# import gzip
import numpy as np
import awkward as ak

import hist
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.utils as utils

import mplhep as hep
import matplotlib.pyplot as plt

def get_sow(hists):
	h_EFT = hists['sow'].as_hist({})
	sow = h_EFT.values()
	return sow, h_EFT.axes[0]

def get_values_SM(hists, name):
    h_EFT = hists[name].as_hist({})
    vals = h_EFT.values()
    return vals, h_EFT.axes[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Customize inputs')
    parser.add_argument('--files', '-f', action='extend', nargs='+', required = True, help = "Specify a list of pkl.gz to run over.")
    parser.add_argument('--hists', action='extend', nargs='+', help="Specify list of histograms to read")

    args = parser.parse_args()

    # Set variables from command line arguments
    files = args.files
    hist_list = args.hists
        
    for file in files: 
        print("filename: ", file)
        hists = utils.get_hist_from_pkl(file, allow_empty=False)
    	# sow, hist_names = get_sow(hists)
        if hist_list is not None:
            for name in hist_list:
                vals, hist_names = get_values_SM(hists, name)
                print("hist name: ", name)
                print("values: ", vals, '\n')
        else:
            print("no hist names provided - all histograms in the file will be looped over")
            for name in hists.keys():
                vals, hist_names = get_values_SM(hists, name)
                print("hist name: ", name)
                print("values: ", vals, '\n')
