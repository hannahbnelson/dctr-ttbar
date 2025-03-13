import awkward as ak
import numpy as np
import pandas as pd

from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection

from df_accumulator import DataframeAccumulator

NanoAODSchema.warn_missing_crossrefs = False
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# Clean the objects
def is_clean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], hist_lst = None, dtype=np.float32, do_errors=False):
        self._samples = samples
        self._wc_names_lst = wc_names_lst

        self._dtype = dtype
        self._do_errors = do_errors

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
        is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])
        ele  = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        mu   = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        nu_ele = genpart[is_final_mask & (abs(genpart.pdgId) == 12)]
        nu_mu = genpart[is_final_mask & (abs(genpart.pdgId) == 14)]
        nu = ak.concatenate([nu_ele,nu_mu],axis=1)

        ######## Lep selection  ########

        e_selec = ((ele.pt>20) & (abs(ele.eta)<2.5))
        m_selec = ((mu.pt>20) & (abs(mu.eta)<2.5))
        leps = ak.concatenate([ele[e_selec],mu[m_selec]],axis=1)
        leps = leps[ak.argsort(leps.pt, axis=-1, ascending=False)]
        l0 = leps[ak.argmax(leps.pt, axis=-1, keepdims=True)]


        ######## Jet selection  ########

        jets = events.GenJet
        jets = jets[(jets.pt>30) & (abs(jets.eta)<2.5)]
        jets_clean = jets[is_clean(jets, leps, drmin=0.4) & is_clean(jets, nu, drmin=0.4)]
        # ht = ak.sum(jets_clean.pt, axis=-1)
        j0 = jets_clean[ak.argmax(jets_clean.pt, axis=-1, keepdims=True)]


        ######## Top selection ########

        gen_top = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)

        
        ######## Event selections ########
        nleps = ak.num(leps)
        njets = ak.num(jets)

        selections = PackedSelection()
        at_least_two_leps = ak.fill_none(nleps>=2,False)
        at_least_two_jets = ak.fill_none(njets>=2,False)

        selections.add('2l', at_least_two_leps)
        selections.add('2j', at_least_two_jets)
        event_selection_mask = selections.all('2l', '2j')

        ######## Apply selections ########
        leps  = leps[event_selection_mask]
        jets  = jets[event_selection_mask]
        tops = gen_top[event_selection_mask]


        ######## Create Variables ########
        mtt = (tops[:,0] + tops[:,1]).mass
        mll = (leps[:,0] + leps[:,1]).mass
        tops_pt = tops.sum().pt
        avg_top_pt = np.divide(tops_pt, 2.0)
        njets = ak.num(jets)

        # set all weights to one                                                            
        weights = np.ones_like(events['event'])[event_selection_mask]

        ######## Fill pandas dataframe ########

        df['weights']=weights
        df['mtt']=mtt
        df['mll']=mll
        df['tops_pt']=tops_pt
        df['avg_top_pt']=avg_top_pt
        df['njets']=njets

        df['top1pt']=tops.pt[:,0]
        df['top1eta']=tops.eta[:,0]
        df['top1phi']=tops.phi[:,0]
        df['top1mass']=tops.mass[:,0]

        df['top2pt']=tops.pt[:,1]
        df['top2eta']=tops.eta[:,1]
        df['top2phi']=tops.phi[:,1]
        df['top2mass']=tops.mass[:,1]

        df['l0pt']=leps.pt[:,0]
        df['l0eta']=leps.eta[:,0]
        df['l0phi']=leps.phi[:,0]
        df['l0mass']=leps.mass[:,0]
        
        df['l1pt']=leps.pt[:,1]
        df['l1eta']=leps.eta[:,1]
        df['l1phi']=leps.phi[:,1]
        df['l1mass']=leps.mass[:,1]

        dfa = DataframeAccumulator(df)
        return dfa
    
    def postprocess(self, accumulator):
        return accumulator
