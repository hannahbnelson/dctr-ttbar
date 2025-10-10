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
                "regular": (200, 0, 100),
                "label": "reweight values",},
            "sow": {
                "regular": (1, 0, 2),
                "label": "sum of weights",},
            "pttt": {
                "regular": (35, 0, 700),
                "label": "$p_T(tt)$ [GeV]"},
            "mtt": {
                "regular": (75, 0, 1500),
                "label": "mtt"},
            "top1pt": {
                "regular": (35, 0, 700),
                "label": "top1 pt",},
            "top2pt": {
                "regular": (35, 0, 700),
                "label": "top2 pt",},
            "top1eta": {
                "regular": (50, -5, 5),
                "label": "top1 eta",},
            "top2eta": {
                "regular": (50, -5, 5),
                "label": "top2 eta",},
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
        
        ele  = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        mu   = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        tau  = genpart[is_final_mask & (abs(genpart.pdgId) == 15)]
        nu_ele = genpart[is_final_mask & (abs(genpart.pdgId) == 12)]
        nu_mu = genpart[is_final_mask & (abs(genpart.pdgId) == 14)]
        nu_tau = genpart[is_final_mask & (abs(genpart.pdgId) == 16)]
        nu = ak.concatenate([nu_ele,nu_mu, nu_tau],axis=1)
        e_selec = ((ele.pt>20) & (abs(ele.eta)<2.5))
        m_selec = ((mu.pt>20) & (abs(mu.eta)<2.5))
        t_selec = ((tau.pt>20) & (abs(tau.eta)< 2.5))

        leps = ak.concatenate([ele[e_selec], mu[m_selec], tau[t_selec]],axis=1)
        leps = leps[ak.argsort(leps.pt, axis=-1, ascending=False)]
        nleps = ak.num(leps)

        jets = events.GenJet
        jets = jets[(jets.pt>30) & (abs(jets.eta)<2.5)]
        jets_clean = jets[is_clean(jets, leps, drmin=0.4) & is_clean(jets, nu, drmin=0.4)]
         
        njets = ak.num(jets_clean)

        ######## Get NN Predictions ########

        if self._doDNN == True: 
            df_inputs = DNN_tools.make_df_for_DNN(genpart, events.GenJet)
            input_dim = df_inputs.shape[1]

            model = DNN_tools.load_saved_model(self._DNNyaml, self._DNNmodel, input_dim)
            predictions = DNN_tools.get_predictions(model, torch.from_numpy(df_inputs.to_numpy()).float())
            reweights = DNN_tools.compute_reweights(predictions)

            ######## Event selections ########

            # selections = PackedSelection()
            # predictions_less98 = ak.fill_none(predictions<0.98, False)
            # selections.add('pred', predictions_less98)
            # event_selection_mask = selections.all('pred')

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
            "sow"       : np.ones_like(events['event']),
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
            "njets"     : njets,
        }

        eft_coeffs_cut = eft_coeffs[event_selection_mask] if eft_coeffs is not None else None

        for var_name, var_values in variables_to_fill.items():
            if var_name not in self._hist_lst:
                print(f"Skipping \"{var_name}\", it is not in the list of hists to include")
                continue

            fill_info = {
                var_name    : var_values,
                "process"   : hist_axis_name,
                "weight"    : event_weights,
                "eft_coeff" : eft_coeffs_cut,
            }

            hout[var_name].fill(**fill_info)

        if self._doDNN == True: 
            DNN_variables = {
                'NNoutput': predictions,
                'reweights': reweights,
            }

            for var_name, var_values in DNN_variables.items():
                fill_info = {
                    var_name : var_values,
                    'process': hist_axis_name,
                    'weights': norm*genw,
                    'eft_coeff': None,
                }

                hout[var_name].fill(**fill_info)

        return hout

    def postprocess(self, accumulator):
        return accumulator
