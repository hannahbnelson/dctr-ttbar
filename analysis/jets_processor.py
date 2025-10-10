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
            "sow": {
                "regular": (1, 0, 2),
                "label": "sum of weights",},
            "j0pt": {
                "regular": (500, 0, 500),
                "label": "j0pt",},
            "j0eta": {
                "regular": (50, -5, 5), 
                "label": "j0eta",},
            "j0phi": {
                "regular": (40, -4, 4),
                "label": "j0phi",},
            "j0mass": {
                "regular": (60, 0, 60),
                "label": "j0mass",},
            "j1pt": {
                "regular": (500, 0, 500),
                "label": "j1pt",},
            "j1eta": {
                "regular": (50, -5, 5), 
                "label": "j1eta",},
            "j1phi": {
                "regular": (40, -4, 4),
                "label": "j1phi",},
            "j1mass": {
                "regular": (60, 0, 60),
                "label": "j1mass",},
            "j2pt": {
                "regular": (500, 0, 500),
                "label": "j2pt",},
            "j2eta": {
                "regular": (50, -5, 5), 
                "label": "j2eta",},
            "j2phi": {
                "regular": (40, -4, 4),
                "label": "j2phi",},
            "j2mass": {
                "regular":(60, 0, 60),
                "label": "j3mass",},
            "j3pt": {
                "regular": (500, 0, 500),
                "label": "j3pt",},
            "j3eta": {
                "regular": (50, -5, 5), 
                "label": "j3eta",},
            "j3phi": {
                "regular": (40, -4, 4),
                "label": "j3phi",},
            "j3mass": {
                "regular": (60, 0, 60),
                "label": "j3mass",},
            "njets": {
                "regular": (10, 0, 10),
                "label": "njets",},
            "jet_flav":{
                "regular": (20, -10, 10),
                "label": "jet flavor",},
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
        # e_selec = ((ele.pt>20) & (abs(ele.eta)<2.5))
        # m_selec = ((mu.pt>20) & (abs(mu.eta)<2.5))
        # t_selec = ((tau.pt>20) & (abs(tau.eta)< 2.5))

        # leps = ak.concatenate([ele[e_selec], mu[m_selec], tau[t_selec]],axis=1)
        leps = ak.concatenate([ele, mu, tau],axis=1)
        leps = leps[ak.argsort(leps.pt, axis=-1, ascending=False)]

        jets = events.GenJet
        # jets = jets[abs(jets.partonFlavour) == 5] # look only at bjets
        jets = jets[abs(jets.partonFlavour) != 5] # look only at NON-bjets

        # jets = jets[(jets.pt>30) & (abs(jets.eta)<2.5)]
        jets_clean = jets[is_clean(jets, leps, drmin=0.4) & is_clean(jets, nu, drmin=0.4)]
        jets_clean = jets_clean[ak.argsort(jets_clean.pt, axis=-1, ascending=False)]

        njets = ak.num(jets_clean)
        jet_flav = ak.flatten(jets_clean.partonFlavour)

        jets_clean = ak.pad_none(jets_clean, 4)
        
        j0 = jets_clean[ak.argmax(jets_clean.pt, axis=-1, keepdims=True)]
        j1 = jets_clean[:,1]
        j2 = jets_clean[:,2]
        j3 = jets_clean[:,3]
        
        ######## Event selections ########

        selections = PackedSelection()

        exactly_one_jet = ak.fill_none(njets==1, False)
        exactly_two_jets = ak.fill_none(njets==2, False)
        exactly_three_jets = ak.fill_none(njets==3, False)
        at_least_four_jets = ak.fill_none(njets>=4, False)

        selections.add('exactly_1j', exactly_one_jet)
        selections.add('exactly_2j', exactly_two_jets)
        selections.add('exactly_3j', exactly_three_jets)
        selections.add('atleast_4j', at_least_four_jets)

        var_to_skip = {
            'exactly_1j': ['j1mass', 'j1pt', 'j1eta', 'j1phi','j2mass', 'j2pt', 'j2eta', 'j2phi', 'j3mass', 'j3pt', 'j3eta', 'j3phi'], 
            'exactly_2j': ['j2mass', 'j2pt', 'j2eta', 'j2phi', 'j3mass', 'j3pt', 'j3eta', 'j3phi'], 
            'exactly_3j': ['j3mass', 'j3pt', 'j3eta', 'j3phi'], 
            'atleast_4j': [],
        }

        # event_selection_mask = selections.all('2l', '2j')

        # ######## Get NN Predictions ########

        # if self._doDNN == True: 
        #     df_inputs = DNN_tools.make_df_for_DNN(genpart)
        #     input_dim = df_inputs.shape[1]

        #     model = DNN_tools.load_saved_model(self._DNNyaml, self._DNNmodel, input_dim)
        #     predictions = DNN_tools.get_predictions(model, torch.from_numpy(df_inputs.to_numpy()))

        #     reweights = DNN_tools.compute_reweights(predictions)
        
        ######## Normalizations ########

        lumi = 1000.0*get_lumi(year)

        norm = (xsec/sow)*lumi

        if eft_coeffs is None:
            genw = events["genWeight"]
        else:
            genw = np.ones_like(events['event'])

        # if self._doDNN == True: 
        #     # scaling = 1/(reweights + 1e-8)
        #     # event_weights = norm*genw*scaling
        #     event_weights = norm*genw*reweights
        # else: 
        #     event_weights = norm*genw

        event_weights = norm*genw

        ######## Fill Histograms ########

        hout = self.accumulator

        variables_to_fill = {
            # "sow"       : np.ones_like(events['event']),
            # "njets"     : njets,
            "j0mass"    : j0.mass,
            "j0pt"      : j0.pt,
            "j0eta"     : j0.eta,
            "j0phi"     : j0.phi,
            "j1mass"    : j1.mass,
            "j1pt"      : j1.pt,
            "j1eta"     : j1.eta,
            "j1phi"     : j1.phi,
            "j2mass"    : j2.mass,
            "j2pt"      : j2.pt,
            "j2eta"     : j2.eta,
            "j2phi"     : j2.phi,
            "j3mass"    : j3.mass,
            "j3pt"      : j3.pt,
            "j3eta"     : j3.eta,
            "j3phi"     : j3.phi,
        }

        # if self._doDNN == True: 
        #     variables_to_fill['NNoutput'] = predictions
        #     variables_to_fill['reweights'] = reweights

        njets_info = {
            'njets': njets,
            'process': hist_axis_name,
            'weight' : event_weights,
            'eft_coeff' : None,
        }

        jet_flav_info = {
            'jet_flav': jet_flav,
            'process': hist_axis_name,
            'weight': np.ones_like(jet_flav),
            'eft_coeff': None,
        }

        hout['njets'].fill(**njets_info)
        hout['jet_flav'].fill(**jet_flav_info)

        for ch in var_to_skip.keys(): 
            event_selection_mask = selections.all(ch)
            eft_coeffs_cut = eft_coeffs[event_selection_mask] if eft_coeffs is not None else None

            for var_name, var_values in variables_to_fill.items():
                if var_name not in self._hist_lst:
                    print(f"Skipping \"{var_name}\", it is not in the list of hists to include")
                    continue

                if var_name in var_to_skip[ch]:
                    print(f"Skipping \"{var_name}\" for channel {ch}")
                    continue

                fill_info = {
                    var_name    : var_values[event_selection_mask],
                    "process"   : hist_axis_name,
                    "weight"    : event_weights[event_selection_mask],
                    "eft_coeff" : eft_coeffs_cut,
                }

                # print(f"\n filling histogram: {var_name} \n")
                hout[var_name].fill(**fill_info)

        print("\n\n")

        return hout

    def postprocess(self, accumulator):
        return accumulator
