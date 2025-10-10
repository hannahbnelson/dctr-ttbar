import awkward as ak
import numpy as np
import pandas as pd

import hist
from hist import Hist

from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection

from topcoffea.modules import utils
import topcoffea.modules.eft_helper as efth

NanoAODSchema.warn_missing_crossrefs = False
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# # Clean the objects
# def is_clean(obj_A, obj_B, drmin=0.4):
#     objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
#     mask = ak.fill_none(objB_DR > drmin, True)
#     return (mask)

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], hist_lst = None, dtype=np.float32, do_errors=False):
        self._samples = samples
        self._wc_names_lst = wc_names_lst

        self._dtype = dtype
        self._do_errors = do_errors

        proc_axis = hist.axis.StrCategory([], name="process", growth=True)

        # Create the histograms with new scikit hist
        self._histo_dict = {
            "avg_top_pt"        : Hist(proc_axis, hist.axis.Regular(bins=20, start=0,  stop=400, name='avg_top_pt', label='avg_top_pt'), storage="weight"),
            "mtt"               : Hist(proc_axis, hist.axis.Regular(bins=30, start=0,  stop=1500,name='mtt',      label='mtt'), storage="weight"),
            "top1_pt"           : Hist(proc_axis, hist.axis.Regular(bins=14, start=0,  stop=700, name='top1_pt',  label='top1_pt'), storage="weight"),
            "top2_pt"           : Hist(proc_axis, hist.axis.Regular(bins=14, start=0,  stop=700, name='top2_pt',  label='top2_pt'), storage="weight"),
            "top_pt"            : Hist(proc_axis, hist.axis.Regular(bins=14, start=0,  stop=700, name='top_pt',   label='top_pt'), storage="weight"),
            "antitop_pt"        : Hist(proc_axis, hist.axis.Regular(bins=14, start=0,  stop=700, name='antitop_pt',  label='antitop_pt'), storage="weight"),
            "lep1_pt"           : Hist(proc_axis, hist.axis.Regular(bins=20, start=0,  stop=400, name='lep1_pt',  label='lep1_pt'), storage="weight"),
            "lep2_pt"           : Hist(proc_axis, hist.axis.Regular(bins=20, start=0,  stop=400, name='lep2_pt',  label='lep2_pt'), storage="weight"),
            "lplus_pt"          : Hist(proc_axis, hist.axis.Regular(bins=20, start=0,  stop=400, name='lplus_pt',   label='lplus_pt'), storage="weight"),
            "lminus_pt"         : Hist(proc_axis, hist.axis.Regular(bins=20, start=0,  stop=400, name='lminus_pt',  label='lminus_pt'), storage="weight"),
            "top1_eta"          : Hist(proc_axis, hist.axis.Regular(bins=50, start=-5, stop=5,   name='top1_eta', label='top1_eta'), storage="weight"),
            "top2_eta"          : Hist(proc_axis, hist.axis.Regular(bins=50, start=-5, stop=5,   name='top2_eta', label='top2_eta'), storage="weight"),
            "top_eta"           : Hist(proc_axis, hist.axis.Regular(bins=50, start=-5, stop=5,   name='top_eta',  label='top_eta'), storage="weight"),
            "antitop_eta"       : Hist(proc_axis, hist.axis.Regular(bins=50, start=-5, stop=5,   name='antitop_eta', label='antitop_eta'), storage="weight"),
            "lep1_eta"          : Hist(proc_axis, hist.axis.Regular(bins=50, start=-5, stop=5,   name='lep1_eta', label='lep1_eta'), storage="weight"),
            "lep2_eta"          : Hist(proc_axis, hist.axis.Regular(bins=50, start=-5, stop=5,   name='lep2_eta', label='lep2_eta'), storage="weight"),
            "lplus_eta"         : Hist(proc_axis, hist.axis.Regular(bins=50, start=-5, stop=5,   name='lplus_eta',  label='lplus_eta'), storage="weight"),
            "lminus_eta"        : Hist(proc_axis, hist.axis.Regular(bins=50, start=-5, stop=5,   name='lminus_eta', label='lminus_eta'), storage="weight"),
            "top1_phi"          : Hist(proc_axis, hist.axis.Regular(bins=40, start=-4, stop=4,   name='top1_phi', label="top1_phi"), storage="weight"),
            "top2_phi"          : Hist(proc_axis, hist.axis.Regular(bins=40, start=-4, stop=4,   name='top2_phi', label="top2_phi"), storage="weight"),
            "top_phi"           : Hist(proc_axis, hist.axis.Regular(bins=40, start=-4, stop=4,   name='top_phi',  label="top_phi"), storage="weight"),
            "antitop_phi"       : Hist(proc_axis, hist.axis.Regular(bins=40, start=-4, stop=4,   name='antitop_phi', label="antitop_phi"), storage="weight"),
            "lep1_phi"          : Hist(proc_axis, hist.axis.Regular(bins=40, start=-4, stop=4,   name='lep1_phi', label="lep1_phi"), storage="weight"),
            "lep2_phi"          : Hist(proc_axis, hist.axis.Regular(bins=40, start=-4, stop=4,   name='lep2_phi', label="lep2_phi"), storage="weight"),
            "lplus_phi"         : Hist(proc_axis, hist.axis.Regular(bins=40, start=-4, stop=4,   name='lplus_phi',  label="lplus_phi"), storage="weight"),
            "lminus_phi"        : Hist(proc_axis, hist.axis.Regular(bins=40, start=-4, stop=4,   name='lminus_phi', label="lminus_phi"), storage="weight"),
            "top1_mass"         : Hist(proc_axis, hist.axis.Regular(bins=34, start=80,  stop=250, name='top1_mass',label="top1_mass"), storage="weight"),
            "top2_mass"         : Hist(proc_axis, hist.axis.Regular(bins=34, start=80,  stop=250, name='top2_mass',label="top2_mass"), storage="weight"),
            "top_mass"          : Hist(proc_axis, hist.axis.Regular(bins=34, start=80,  stop=250, name='top_mass', label="top_mass"), storage="weight"),
            "antitop_mass"      : Hist(proc_axis, hist.axis.Regular(bins=34, start=80,  stop=250, name='antitop_mass',label="antitop_mass"), storage="weight"),
            "lep1_mass"         : Hist(proc_axis, hist.axis.Regular(bins=20, start=0,  stop=20, name='lep1_mass',label="lep1_mass"), storage="weight"),
            "lep2_mass"         : Hist(proc_axis, hist.axis.Regular(bins=20, start=0,  stop=20, name='lep2_mass',label="lep2_mass"), storage="weight"),
            "lplus_mass"        : Hist(proc_axis, hist.axis.Regular(bins=20, start=0,  stop=20, name='lplus_mass', label="lplus_mass"), storage="weight"),
            "lminus_mass"       : Hist(proc_axis, hist.axis.Regular(bins=20, start=0,  stop=20, name='lminus_mass',label="lminus_mass"), storage="weight"), 
        }

        # Set the list of hists to to fill
        if hist_lst is None:
            self._hist_lst = list(self._histo_dict.keys())
        else:
            for h in hist_lst:
                if h not in self._histo_dict.keys():
                    raise Exception(f"Error: Cannot specify hist \"{h}\", it is not defined in self._histo_dict")
            self._hist_lst = hist_lst

        print("hist_lst: ", self._hist_lst)


    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns
        
    def process(self, events):     

        # Dataset parameters
        dataset = events.metadata['dataset']
        hist_axis_name = self._samples[dataset]["histAxisName"]

        year   = self._samples[dataset]['year']
        xsec   = self._samples[dataset]['xsec']
        sow    = self._samples[dataset]['nSumOfWeights']

        ######## Initialize Objects  ########
        genpart = events.GenPart
        is_final_mask = genpart.hasFlags(["fromHardProcess","isLastCopy"])
        ele  = genpart[is_final_mask & (abs(genpart.pdgId) == 11)]
        mu   = genpart[is_final_mask & (abs(genpart.pdgId) == 13)]
        tau = genpart[is_final_mask & (abs(genpart.pdgId) == 15)]
        # nu_ele = genpart[is_final_mask & (abs(genpart.pdgId) == 12)]
        # nu_mu = genpart[is_final_mask & (abs(genpart.pdgId) == 14)]
        # nu = ak.concatenate([nu_ele,nu_mu],axis=1)

        ######## Lep selection  ########

        # e_selec = ((ele.pt>20) & (abs(ele.eta)<2.5))
        # m_selec = ((mu.pt>20) & (abs(mu.eta)<2.5))
        leps = ak.concatenate([ele, mu, tau],axis=1)
        leps = leps[ak.argsort(leps.pt, axis=-1, ascending=False)]
        # l0 = leps[ak.argmax(leps.pt, axis=-1, keepdims=True)]

        lplus = leps[leps.pdgId < 0] #negative pdgId corresponds with antielectorn, antimuon, antitau
        lminus = lminus = leps[leps.pdgId > 0] #positive pdgId corresponds with electron, muon, tau 


        ######## Jet selection  ########

        jets = events.GenJet
        # jets = jets[(jets.pt>30) & (abs(jets.eta)<2.5)]
        # jets_clean = jets[is_clean(jets, leps, drmin=0.4) & is_clean(jets, nu, drmin=0.4)]
        # # ht = ak.sum(jets_clean.pt, axis=-1)
        # j0 = jets_clean[ak.argmax(jets_clean.pt, axis=-1, keepdims=True)]


        ######## Top selection ########

        gen_tops = ak.pad_none(genpart[is_final_mask & (abs(genpart.pdgId) == 6)],2)
        gen_tops = gen_tops[ak.argsort(gen_tops.pt, axis=1, ascending=False)]

        top = genpart[is_final_mask & (genpart.pdgId == 6)]
        antitop = genpart[is_final_mask & (genpart.pdgId == -6)]

        ######## Event selections ########
        # nleps = ak.num(leps)
        # njets = ak.num(jets)

        # selections = PackedSelection()
        # at_least_two_leps = ak.fill_none(nleps>=2,False)
        # at_least_two_jets = ak.fill_none(njets>=2,False)

        # selections.add('2l', at_least_two_leps)
        # selections.add('2j', at_least_two_jets)
        # event_selection_mask = selections.all('2l', '2j')

        ######## Apply selections ########
        # leps  = leps[event_selection_mask]
        # jets  = jets[event_selection_mask]
        # tops = gen_top[event_selection_mask]


        ######## Create Variables ########
        # mtt = (gen_top[:,0] + gen_top[:,1]).mass
        # mll = (leps[:,0] + leps[:,1]).mass
        # tops_pt = gen_top.sum().pt
        # avg_top_pt = np.divide(tops_pt, 2.0)
        # njets = ak.num(jets)

        # set all weights to one                                                            
        weights = events["genWeight"]
        # weights = np.ones_like(events['event'])[event_selection_mask]

        ######## Fill histos ########
        hout = self._histo_dict

        variables_to_fill = {
            "avg_top_pt"    : np.divide(gen_tops.sum().pt, 2.0),
            "mtt"           : (gen_tops[:,0] + gen_tops[:,1]).mass,
            "top1_pt"       : gen_tops.pt[:,0],
            "top1_eta"      : gen_tops.eta[:,0],
            "top1_phi"      : gen_tops.phi[:,0],
            "top1_mass"     : gen_tops.mass[:,0],
            "top2_pt"       : gen_tops.pt[:,1],
            "top2_eta"      : gen_tops.eta[:,1],
            "top2_phi"      : gen_tops.phi[:,1],
            "top2_mass"     : gen_tops.mass[:,1],
            "top_pt"        : ak.flatten(top.pt),
            "top_eta"       : ak.flatten(top.eta),
            "top_phi"       : ak.flatten(top.phi),
            "top_mass"      : ak.flatten(top.mass),
            "antitop_pt"    : ak.flatten(antitop.pt),
            "antitop_eta"   : ak.flatten(antitop.eta),
            "antitop_phi"   : ak.flatten(antitop.phi),
            "antitop_mass"  : ak.flatten(antitop.mass),
            # "lep1_pt"       : leps.pt[:,0],
            # "lep1_eta"      : leps.eta[:,0],
            # "lep1_phi"      : leps.phi[:,0],
            # "lep1_mass"     : leps.mass[:,0],
            # "lep2_pt"       : leps.pt[:,1],
            # "lep2_eta"      : leps.eta[:,1],
            # "lep2_phi"      : leps.phi[:,1],
            # "lep2_mass"     : leps.mass[:,1],
            # "lplus_pt"      : ak.flatten(lplus.pt),
            # "lplus_eta"     : ak.flatten(lplus.eta), 
            # "lplus_phi"     : ak.flatten(lplus.phi),
            # "lplus_mass"    : ak.flatten(lplus.mass),
            # "lminus_pt"     : ak.flatten(lminus.pt),
            # "lminus_eta"    : ak.flatten(lminus.eta),
            # "lminus_phi"    : ak.flatten(lminus.phi),
            # "lminus_mass"   : ak.flatten(lminus.mass),
        }

        for var_name, var_values in variables_to_fill.items():
            # print(f"\n\n filling {var_name} \n\n")
            if var_name not in self._hist_lst:
                print(f"Skipping {var_name}, it is not in the list of hists to include")
                continue

            hout[var_name].fill(process=hist_axis_name, **{var_name: var_values}, weight=weights)

        return hout

    
    def postprocess(self, accumulator):
        return accumulator
