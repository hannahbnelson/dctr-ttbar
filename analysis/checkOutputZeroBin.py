import yaml
import os 

import awkward as ak
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep

import hist
from hist import Hist

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from topcoffea.modules.histEFT import HistEFT
NanoAODSchema.warn_missing_crossrefs = False

from coffea.analysis_tools import PackedSelection
from topcoffea.modules import utils
import topcoffea.modules.eft_helper as efth

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import dctr.modules.plotting_tools as plt_tools
import dctr.modules.DNN_tools as DNN_tools


# means and stdv to standardize pd df for input into trained model
means = {'avg_top_pt': 34.263557,
        'mtt': 522.141900,
        'top1pt': 126.859184,
        'top1eta': -0.257265,
        'top1phi': -0.000021,
        'top1mass': 172.253560,
        'top2pt': 124.636566,
        'top2eta': 0.257370,
        'top2phi': -0.000686,
        'top2mass': 172.265670,}
stdvs = {'avg_top_pt': 38.252880,
        'mtt': 175.306980,
        'top1pt': 84.604750,
        'top1eta': 1.823326,
        'top1phi': 1.813635,
        'top1mass': 5.346320,
        'top2pt': 82.644310,
        'top2eta': 1.829129,
        'top2phi': 1.813916,
        'top2mass': 5.329451,}

axes = {
    "outputs": {
        "regular": (100, 0, 1),
        "label": "outputs",},
    "lep1pt": {
        "regular": (40, 0, 400),
        "label": "lep1 pt",},
    "lep2pt": {
        "regular": (40, 0, 400),
        "label": "lep2 pt",},
    "lpluspt": {
        "regular": (40, 0, 400),
        "label": "lplus pt",},
    "lminuspt": {
        "regular": (40, 0, 400),
        "label": "lminus pt",},
    "top1pt": {
        "regular": (35, 0, 700),
        "label": "top1 pt",},
    "top2pt": {
        "regular": (35, 0, 700),
        "label": "top2 pt",},
    "toppt": {
        "regular": (35, 0, 700),
        "label": "top pt",},
    "antitoppt": {
        "regular": (35, 0, 700),
        "label": "antitop pt",},
    "lep1eta": {
        "regular": (50, -5, 5),
        "label": "lep1 eta",},
    "lep2eta": {
        "regular": (50, -5, 5),
        "label": "lep2 eta",},
    "lpluseta": {
        "regular": (50, -5, 5),
        "label": "lplus eta",},
    "lminuseta": {
        "regular": (50, -5, 5),
        "label": "lminus eta",},
    "top1eta": {
        "regular": (50, -5, 5),
        "label": "top1 eta",},
    "top2eta": {
        "regular": (50, -5, 5),
        "label": "top2 eta",},
    "topeta": {
        "regular": (50, -5, 5),
        "label": "top eta",},
    "antitopeta": {
        "regular": (50, -5, 5),
        "label": "antitop eta",},
    "lep1phi": {
        "regular": (40, -4, 4),
        "label": "lep1 phi",},
    "lep2phi": {
        "regular": (40, -4, 4),
        "label": "lep2 phi",},
    "lplusphi": {
        "regular": (40, -4, 4),
        "label": "lplus phi",},
    "lminusphi": {
        "regular": (40, -4, 4),
        "label": "lminus phi",},
    "top1phi": {
        "regular": (40, -4, 4),
        "label": "top1 phi",},
    "top2phi": {
        "regular": (40, -4, 4),
        "label": "top2 phi",},
    "topphi": {
        "regular": (40, -4, 4),
        "label": "top phi",},
    "antitopphi": {
        "regular": (40, -4, 4),
        "label": "antitop phi",},
    "lep1mass": {
        "regular": (20, 0, 20),
        "label": "lep1 mass", },
    "lep2mass": {
        "regular": (20, 0, 20),
        "label": "lep2 mass", },
    "lplusmass": {
        "regular": (20, 0, 20),
        "label": "lplus mass", },
    "lminusmass": {
        "regular": (20, 0, 20),
        "label": "lminus mass", },  
    "top1mass": {
        "regular": (16, 130, 210),
        "label": "top1 mass", },
    "top2mass": {
        "regular": (16, 130, 210),
        "label": "top2 mass", },
    "topmass": {
        "regular": (16, 130, 210),
        "label": "top mass", },
    "antitopmass": {
        "regular": (16, 130, 210),
        "label": "antitop mass", }, 
    "j0pt": {
        "regular": (100, 0, 500),
        "label": "j0pt",},
    "j0eta": {
        "regular": (50, -5, 5), 
        "label": "j0eta",},
    "j0phi": {
        "regular": (40, -4, 4),
        "label": "j0phi",},
    "j0mass": {
        "regular": (10, 0, 100),
        "label": "j0mass"},
    "njets": {
        "regular": (10, 0, 10),
        "label": "njets",
    },
}

def main(file):

    # Load in events from root file
    events = NanoEventsFactory.from_root(
        file,
        schemaclass=NanoAODSchema.v6,
        metadata={"dataset": "TTto2L2Nu"},
    ).events()


    ### Create objects for plotting/selections
    print(f"creating objects from root file")

    genpart = events.GenPart
    is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])

    gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)
    gen_top = gen_top[ak.argsort(gen_top.pt, axis=1, ascending=False)]

    top = genpart[is_final_mask & (genpart.pdgId == 6)]
    antitop = genpart[is_final_mask & (genpart.pdgId == -6)]

    ele  = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
    mu   = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
    tau = genpart[is_final_mask & (abs(genpart.pdgId) == 15)]

    leps = ak.concatenate([ele, mu, tau],axis=1)
    leps = leps[ak.argsort(leps.pt, axis=-1, ascending=False)]

    lplus = leps[leps.pdgId < 0] #negative pdgId corresponds with antielectorn, antimuon, antitau
    lminus = lminus = leps[leps.pdgId > 0] #positive pdgId corresponds with electron, muon, tau 

    jets = events.GenJet
    # jets = jets[ak.argsort(jets.pt, axis=-1, ascending=False)]
    j0 = jets[ak.argmax(jets.pt, axis=-1, keepdims=True)]
    njets = ak.num(jets)
    # print(f"number of events with njets < 1: {ak.sum(njets<1)}")

    # print(f"shape of njets: {np.shape(njets)} ")
    # print(f"shape of j0: {np.shape(j0)}")
    # print(f"shape of top: {np.shape(top)}")

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

    # NN_inputs = pd.DataFrame.from_dict(variables_to_fill_df)
    norm_NN_inputs = DNN_tools.standardize_df(pd.DataFrame.from_dict(variables_to_fill_df), means, stdvs)

    ### load in trained network 
    print(f"loading in trained network")

    config_path = "/users/hnelson2/dctr/condor_submissions/20250721_1722/config.yaml"
    with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

    model_architecture = config_dict['model']
    input_dim = norm_NN_inputs.shape[1]
    model = DNN_tools.NeuralNetwork(input_dim, model_architecture)

    model_path = "/users/hnelson2/dctr/condor_submissions/20250721_1722/training_outputs/final_model.pt"
    model.load_state_dict(torch.load(model_path))

    ### Evaluate the trained model with my random powheg sample
    print(f"evaluating network with the standardized inputs")
    model.eval()
    predictions = DNN_tools.get_predictions(model, torch.from_numpy(norm_NN_inputs.to_numpy()))
    print(f"predictions from model retrieved")

    # the first bin where I want to look is from 0 - 0.1 
    selections = PackedSelection()

    at_least_one_jet = ak.fill_none(njets>0, False)
    first_bin_prediction = ak.fill_none(predictions<0.1, False)

    selections.add('1j', at_least_one_jet)
    selections.add('pred', first_bin_prediction)

    output_mask = selections.all('1j', 'pred')
    basic_selec = selections.all('1j')

    print(f"initializing empty histograms")

    histos = {}
    histos_cuts = {}

    for name, info in axes.items():
        dense_axis = hist.axis.Regular(*info['regular'], name=name, label=info['label'])
        histos[name] = Hist(dense_axis, storage='weight')
        histos_cuts[name] = Hist(dense_axis, storage='weight')

    variables_to_fill_cuts = {
        "outputs":      predictions, 
        "lep1pt":       leps.pt[:,0][output_mask],
        "lep1eta":      leps.eta[:,0][output_mask],
        "lep1phi":      leps.phi[:,0][output_mask],
        "lep1mass":     leps.mass[:,0][output_mask],
        "lep2pt":       leps.pt[:,1][output_mask],
        "lep2eta":      leps.eta[:,1][output_mask],
        "lep2phi":      leps.phi[:,1][output_mask],
        "lep2mass":     leps.mass[:,1][output_mask],
        "lpluspt":      ak.flatten(lplus.pt)[output_mask],
        "lpluseta":     ak.flatten(lplus.eta)[output_mask],
        "lplusphi":     ak.flatten(lplus.phi)[output_mask],
        "lplusmass":    ak.flatten(lplus.mass)[output_mask],
        "lminuspt":     ak.flatten(lminus.pt)[output_mask],
        "lminuseta":    ak.flatten(lminus.eta)[output_mask],
        "lminusphi":    ak.flatten(lminus.phi)[output_mask],
        "lminusmass":   ak.flatten(lminus.mass)[output_mask],
        "top1pt":       gen_top[:,0].pt[output_mask],
        "top2pt":       gen_top[:,1].pt[output_mask], 
        "toppt":        ak.flatten(top.pt)[output_mask],
        "antitoppt":    ak.flatten(antitop.pt)[output_mask],
        "top1eta":      gen_top[:,0].eta[output_mask],
        "top2eta":      gen_top[:,1].eta[output_mask],
        "topeta":       ak.flatten(top.eta)[output_mask],
        "antitopeta":   ak.flatten(antitop.eta)[output_mask],
        "top1phi":      gen_top[:,0].phi[output_mask],
        "top2phi":      gen_top[:,1].phi[output_mask],
        "topphi":       ak.flatten(top.phi)[output_mask],
        "antitopphi":   ak.flatten(antitop.phi)[output_mask],
        "top1mass":     gen_top[:,0].mass[output_mask],
        "top2mass":     gen_top[:,1].mass[output_mask],
        "topmass":      ak.flatten(top.mass)[output_mask],
        "antitopmass":  ak.flatten(antitop.mass)[output_mask],
        "njets":        njets[output_mask],
        "j0pt":         ak.flatten(j0.pt)[output_mask],
        "j0eta":        ak.flatten(j0.eta)[output_mask],
        "j0phi":        ak.flatten(j0.phi)[output_mask],
        "j0mass":       ak.flatten(j0.mass)[output_mask],
        }

    variables_to_fill = {
        "outputs":      predictions[basic_selec], 
        "lep1pt":       leps.pt[:,0][basic_selec],
        "lep1eta":      leps.eta[:,0][basic_selec],
        "lep1phi":      leps.phi[:,0][basic_selec],
        "lep1mass":     leps.mass[:,0][basic_selec],
        "lep2pt":       leps.pt[:,1][basic_selec],
        "lep2eta":      leps.eta[:,1][basic_selec],
        "lep2phi":      leps.phi[:,1][basic_selec],
        "lep2mass":     leps.mass[:,1][basic_selec],
        "lpluspt":      ak.flatten(lplus.pt)[basic_selec],
        "lpluseta":     ak.flatten(lplus.eta)[basic_selec],
        "lplusphi":     ak.flatten(lplus.phi)[basic_selec],
        "lplusmass":    ak.flatten(lplus.mass)[basic_selec],
        "lminuspt":     ak.flatten(lminus.pt)[basic_selec],
        "lminuseta":    ak.flatten(lminus.eta)[basic_selec],
        "lminusphi":    ak.flatten(lminus.phi)[basic_selec],
        "lminusmass":   ak.flatten(lminus.mass)[basic_selec],
        "top1pt":       gen_top[:,0].pt[basic_selec],
        "top2pt":       gen_top[:,1].pt[basic_selec], 
        "toppt":        ak.flatten(top.pt)[basic_selec],
        "antitoppt":    ak.flatten(antitop.pt)[basic_selec],
        "top1eta":      gen_top[:,0].eta[basic_selec],
        "top2eta":      gen_top[:,1].eta[basic_selec],
        "topeta":       ak.flatten(top.eta)[basic_selec],
        "antitopeta":   ak.flatten(antitop.eta)[basic_selec],
        "top1phi":      gen_top[:,0].phi[basic_selec],
        "top2phi":      gen_top[:,1].phi[basic_selec],
        "topphi":       ak.flatten(top.phi)[basic_selec],
        "antitopphi":   ak.flatten(antitop.phi)[basic_selec],
        "top1mass":     gen_top[:,0].mass[basic_selec],
        "top2mass":     gen_top[:,1].mass[basic_selec],
        "topmass":      ak.flatten(top.mass)[basic_selec],
        "antitopmass":  ak.flatten(antitop.mass)[basic_selec],
        "njets":        njets[basic_selec],
        "j0pt":         ak.flatten(j0.pt)[basic_selec],
        "j0eta":        ak.flatten(j0.eta)[basic_selec],
        "j0phi":        ak.flatten(j0.phi)[basic_selec],
        "j0mass":       ak.flatten(j0.mass)[basic_selec],
        }

    print(f"filling histograms...")
    for var_name, var_val in variables_to_fill_cuts.items():
        histos_cuts[var_name].fill(**{var_name: var_val})

    for var_name, var_val in variables_to_fill.items():
        # print(f"--> {var_name}")
        histos[var_name].fill(**{var_name: var_val})
    print(f"done filling histograms")

    return histos, histos_cuts

def make_plots(hists_powheg, hists_powheg_cuts, hists_smeft, name, outdir):

    hep.style.use("CMS")
    fig, ax = plt.subplots()
    hep.histplot(hists_powheg[name], density=True, yerr=True, label='powheg')
    hep.histplot(hists_powheg_cuts[name], density=True, yerr=True, label='powheg w/output<0.1')
    hep.histplot(hists_smeft[name], density=True, yerr=True, label="smeft")
    ax.legend(loc='best')
    ax.set_title("Normalized to 1.0")

    # if name == 'j0pt': 
        # ax.set_xlim([0, 50])

    outname = os.path.join(outdir, f"{name}")
    fig.savefig(f"{outname}.png")
    print(f"figure saved in {outname}.png") 

    plt.close()

if __name__ == "__main__":

    fpowheg = "/cms/cephfs/data/store/mc/RunIISummer20UL17NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/2510000/74C36AED-4CB9-1A4D-A9E6-90278C68131C.root"
    fsmeft  = "/cms/cephfs/data/store/user/hnelson2/noEFT/nanoGen/TT01j2l_SM/NanoGen_TT01j2l_SM/nanoGen_10016.root"

    outdir = "/users/hnelson2/dctr/analysis/plots_zeroBin"

    powheg_hists, powheg_hists_cut = main(fpowheg)
    smeft_hists, smeft_hists_cut = main(fsmeft)

    print(f"contents of j0pt histograms")
    print(f"powheg_hists: \n {powheg_hists['j0pt'].values()}")
    print(f"powheg_hists_cut: \n {powheg_hists_cut['j0pt'].values()}")
    print(f"smeft_hists: \n {smeft_hists['j0pt'].values()}")
    print(f"smeft_hists_cut: \n {smeft_hists_cut['j0pt'].values()}")

    for name in powheg_hists: 
        make_plots(powheg_hists, powheg_hists_cut, smeft_hists, name, outdir)
