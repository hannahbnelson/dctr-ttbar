import awkward as ak
import numpy as np
import pandas as pd

from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection

from dctr.modules.df_accumulator import DataframeAccumulator

NanoAODSchema.warn_missing_crossrefs = False
np.seterr(divide='ignore', invalid='ignore', over='ignore')

def is_clean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], hist_lst = None, dtype=np.float32, do_errors=False):
        self._samples = samples
        self._dtype = dtype
        self._accumulator = DataframeAccumulator(pd.DataFrame())

    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns
        
    def process(self, events):     

        df = pd.DataFrame()

        ######## Initialize Objects  ########

        genpart = events.GenPart
        jets = events.GenJet
        is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])

        ######## Top selection ########

        gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)
        gen_top = gen_top[ak.argsort(gen_top.pt, axis=1, ascending=False)]

        ######## Jet Cleaning & Selection ######## 
        ele = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        mu = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        tau = genpart[is_final_mask & (abs(genpart.pdgId) == 15)]

        nu_ele = genpart[is_final_mask & (abs(genpart.pdgId) == 12)]
        nu_mu = genpart[is_final_mask & (abs(genpart.pdgId) == 14)]
        nu_tau = genpart[is_final_mask & (abs(genpart.pdgId) == 16)]

        leps = ak.concatenate([ele[e_selec], mu[m_selec], tau[t_selec]],axis=1)
        nu = ak.concatenate([nu_ele,nu_mu, nu_tau],axis=1)       

        jets_clean = jets[is_clean(jets, leps, drmin=0.4) & is_clean(jets, nu, drmin=0.4)]

        ######## Event selections ########

        selections = PackedSelection()
        mass_more_150 = ak.fill_none(gen_top.mass > 150, False)
        mass_less_195 = ak.fill_none(gen_top.mass < 195, False)
        selections.add('mass150', mass_more_150)
        selections.add('mass195', mass_less_195)
        event_selection_mask = selections.all('mass150', 'mass195')

        weights = events["genWeight"]


        ######## Fill pandas dataframe ########

        variables_to_fill = {
            "weights"   : weights[event_selection_mask],
            "avg_top_pt": np.divide(gen_top.sum().pt, 2.0)[event_selection_mask],
            "mtt"       : (gen_top[:,0] + gen_top[:,1]).mass[event_selection_mask],
            "top1pt"    : gen_top.pt[:,0][event_selection_mask],
            "top1eta"   : gen_top.eta[:,0][event_selection_mask],
            "top1phi"   : gen_top.phi[:,0][event_selection_mask],
            "top1mass"  : gen_top.mass[:,0][event_selection_mask],
            "top2pt"    : gen_top.pt[:,1][event_selection_mask],
            "top2eta"   : gen_top.eta[:,1][event_selection_mask],
            "top2phi"   : gen_top.phi[:,1][event_selection_mask],
            "top2mass"  : gen_top.mass[:,1][event_selection_mask],
            "njets"     : ak.num(jets_clean)[event_selection_mask],
        }

        outputs = []
        for var_name, var_values in variables_to_fill.items():
            outputs.append(pd.Series(ak.to_numpy(var_values), name=var_name))

        return DataframeAccumulator(pd.concat(outputs, axis=1))
    
    def postprocess(self, accumulator):
        return accumulator
