import awkward as ak
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection

import hist
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.eft_helper as efth

from dctr.modules.df_accumulator import DataframeAccumulator
import dctr.modules.DNN_tools as DNN_tools

NanoAODSchema.warn_missing_crossrefs = False
np.seterr(divide='ignore', invalid='ignore', over='ignore')


# Get the lumi for the given year
def get_lumi(year):
    lumi_dict = {
        "2016APV": 19.52,
        "2016": 16.81,
        "2017": 41.48,
        "2018": 59.83
    }
    if year not in lumi_dict.keys():
        raise Exception(f"(ERROR: Unknown year \"{year}\".")
    else:
        return(lumi_dict[year])


def is_clean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)


class AnalysisProcessor(processor.ProcessorABC):
    
    def __init__(self, samples, DNNyaml=None, DNNmodel=None, wc_names_lst=[], hist_lst = None, dtype=np.float32, do_errors=False):
        self._samples = samples
        self._dtype = dtype
        self._accumulator = DataframeAccumulator(pd.DataFrame())

        if (DNNyaml is not None) and (DNNmodel is not None): 
            self._doDNN = True 
            self._DNNyaml = DNNyaml
            self._DNNmodel = DNNmodel
        else:
            self._doDNN = False

        proc_axis = hist.axis.StrCategory([], name="process", growth=True)

        axes = {
            "NNoutput": {
                "regular": (100, 0, 1),
                "label": "NN output",},
            "reweights": {
                "regular": (25, 0, 5),
                "label": "reweight values",},
            "sow": {
                "regular": (1, 0, 2),
                "label": "sum of weights",},
            "ptll": {
                "regular": (25, 0, 500),
                "label": "$p_T(ll)$ [GeV]"},
            "pttt": {
                "regular": (35, 0, 700),
                "label": "$p_T(tt)$ [GeV]"},
            "mtt": {
                "regular": (75, 0, 1500),
                "label": "mtt"},
            "lep1pt": {
                "regular": (40, 0, 400),
                "label": "lep1 pt",},
            "lep2pt": {
                "regular": (40, 0, 400),
                "label": "lep2 pt",},
            "top1pt": {
                "regular": (35, 0, 700),
                "label": "top1 pt",},
            "top2pt": {
                "regular": (35, 0, 700),
                "label": "top2 pt",},
            "lep1eta": {
                "regular": (50, -5, 5),
                "label": "lep1 eta",},
            "lep2eta": {
                "regular": (50, -5, 5),
                "label": "lep2 eta",},
            "top1eta": {
                "regular": (50, -5, 5),
                "label": "top1 eta",},
            "top2eta": {
                "regular": (50, -5, 5),
                "label": "top2 eta",},
            "lep1phi": {
                "regular": (40, -4, 4),
                "label": "lep1 phi",},
            "lep2phi": {
                "regular": (40, -4, 4),
                "label": "lep2 phi",},
            "top1phi": {
                "regular": (40, -4, 4),
                "label": "top1 phi",},
            "top2phi": {
                "regular": (40, -4, 4),
                "label": "top2 phi",},
            "top1mass": {
                "regular": (34, 80, 250),
                "label": "top1 mass", },
            "top2mass": {
                "regular": (34, 80, 250),
                "label": "top2 mass", },
            "j0pt": {
                "regular": (100, 0, 500),
                "label": "j0pt",},
            "j0eta": {
                "regular": (50, -5, 5), 
                "label": "j0eta",},
            "j0phi": {
                "regular": (40, -4, 4),
                "label": "j0phi",},
            "njets": {
                "regular": (10, 0, 10),
                "label": "njets",},
        }

        histograms = {}

        for name, info in axes.items():
            if 'variable' in info: 
                dense_axis = hist.axis.Variable(info['variable'], name=name, label=info['label'])
            else: 
                dense_axis = hist.axis.Regular(*info['regular'], name=name, label=info['label'])

            histograms[name]=HistEFT(
                proc_axis,
                dense_axis,
                wc_names = wc_names_lst,
                label=r'Events',
            )

        self._accumulator = histograms 

        # set the list of hists to fill
        if hist_lst is None:
            self._hist_lst = list(self._accumulator.keys()) #fill all hists if not specified
        else:
            for hist_to_include in hist_lst:
                if hist_to_include not in self._accumulator.keys():
                    raise Exception(f"Error: Cannot specify hist \'{hist_to_include}\', it is not defined in the processor.")
            self._hist_lst = hist_lst 


    @property
    def accumulator(self):
        return self._accumulator


    @property
    def columns(self):
        return self._columns

        
    def process(self, events):     

        ######### Dataset parameters ##########

        dataset         = events.metadata['dataset']
        isEFT           = hasattr(events, 'EFTfitCoefficients')  
        # isEFT           = self._samples[dataset]["WCnames"] != []
        isData          = self._samples[dataset]['isData']
        hist_axis_name  = self._samples[dataset]['histAxisName']
        year            = self._samples[dataset]['year']
        xsec            = self._samples[dataset]['xsec']
        sow             = self._samples[dataset]['nSumOfWeights']


        ######### EFT coefficients ##########

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None


        ######## Initialize Objects  ########

        genpart = events.GenPart
        is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])


        ######## Object selections ########

        gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)
        gen_top = gen_top[ak.argsort(gen_top.pt, axis=1, ascending=False)]
        
        # ele  = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        # mu   = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        # nu_ele = genpart[is_final_mask & (abs(genpart.pdgId) == 12)]
        # nu_mu = genpart[is_final_mask & (abs(genpart.pdgId) == 14)]
        # nu = ak.concatenate([nu_ele,nu_mu],axis=1)
        # e_selec = ((ele.pt>20) & (abs(ele.eta)<2.5))
        # m_selec = ((mu.pt>20) & (abs(mu.eta)<2.5))

        # leps = ak.concatenate([ele[e_selec], mu[m_selec]],axis=1)
        # leps = leps[ak.argsort(leps.pt, axis=-1, ascending=False)]
        # nleps = ak.num(leps)

        jets = events.GenJet
        njets = ak.num(jets)
        # jets = jets[(jets.pt>30) & (abs(jets.eta)<2.5)]
        # # jets_clean = jets[is_clean(jets, leps, drmin=0.4)]
        # jets_clean = jets[is_clean(jets, leps, drmin=0.4) & is_clean(jets, nu, drmin=0.4)]
         
        # jets = jets_clean[ak.argsort(jets_clean.pt, axis=-1, ascending=False)]
        # j0 = jets_clean[ak.argmax(jets_clean.pt, axis=-1, keepdims=True)]
        # njets = ak.num(jets_clean)


        ######## Event selections ########

        # selections = PackedSelection()

        # at_least_two_leps = ak.fill_none(nleps>=2, False)
        # at_least_two_jets = ak.fill_none(njets>=2, False)

        # selections.add('2l', at_least_two_leps)
        # selections.add('2j', at_least_two_jets)

        # event_selection_mask = selections.all('2l', '2j')


        ######## Get NN Predictions ########

        if self._doDNN == True: 
            df_inputs = DNN_tools.make_df_for_DNN(genpart)
            input_dim = df_inputs.shape[1]

            model = DNN_tools.load_saved_model(self._DNNyaml, self._DNNmodel, input_dim)
            predictions = DNN_tools.get_predictions(model, torch.from_numpy(df_inputs.to_numpy()))

            reweights = DNN_tools.compute_reweights(predictions)


        ######## Variables for Plotting ########

        # leps = ak.pad_none(leps, 2)
        # l0 = leps[:,0]
        # l1 = leps[:,1]

        # ptll = (l0+l1).pt
        
        ######## Normalizations ########

        lumi = 1000.0*get_lumi(year)

        norm = (xsec/sow)*lumi

        if eft_coeffs is None:
            genw = events["genWeight"]
        else:
            genw = np.ones_like(events['event'])

        if self._doDNN == True: 
            # scaling = 1/(reweights + 1e-8)
            # event_weights = norm*genw*scaling
            event_weights = norm*genw*reweights
        else: 
            event_weights = norm*genw

        ######## Fill Histograms ########

        hout = self.accumulator

        variables_to_fill = {
            # "NNoutput"  : predictions,
            "sow"       : np.ones_like(events['event']),
            # "ptll"      : ptll,
            "pttt"      : (gen_top[:,0] + gen_top[:,1]).pt,
            "mtt"       : (gen_top[:,0] + gen_top[:,1]).mass,
            "top1pt"    : gen_top.pt[:,0],
            "top1eta"   : gen_top.eta[:,0],
            "top1phi"   : gen_top.phi[:,0],
            "top1mass"  : gen_top.mass[:,0],
            "top2pt"    : gen_top.pt[:,1],
            "top2eta"   : gen_top.eta[:,1],
            "top2phi"   : gen_top.phi[:,1],
            "top2mass"  : gen_top.mass[:,1],
            # "lep1pt"    : l0.pt, 
            # "lep1eta"   : l0.eta,
            # "lep1phi"   : l0.phi,
            # "lep2pt"    : l1.pt, 
            # "lep2eta"   : l1.eta,
            # "lep2phi"   : l1.phi,
            # "j0pt"      : ak.flatten(j0.pt),
            # "j0eta"     : ak.flatten(j0.eta),
            # "j0phi"     : ak.flatten(j0.phi),
            "njets"     : njets,
        }

        if self._doDNN == True: 
            variables_to_fill['NNoutput'] = predictions
            variables_to_fill['reweights'] = reweights

        # eft_coeffs_cut = eft_coeffs[event_selection_mask] if eft_coeffs is not None else None

        for var_name, var_values in variables_to_fill.items():
            if var_name not in self._hist_lst:
                print(f"Skipping \"{var_name}\", it is not in the list of hists to include")
                continue

            fill_info = {
                var_name    : var_values,
                "process"   : hist_axis_name,
                "weight"    : event_weights,
                "eft_coeff" : eft_coeffs,
            }

            # fill_info = {
            #     var_name    : var_values,
            #     "process"   : hist_axis_name,
            #     "weight"    : event_weights,
            #     "eft_coeff" : eft_coeffs_cut,
            # }

            # print(f"\n filling histogram: {var_name} \n")
            hout[var_name].fill(**fill_info)

        return hout

    def postprocess(self, accumulator):
        return accumulator
