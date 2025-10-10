'''
This processor processes a root file and returns a 1 bin histogram that just holds one entry per 
event weighted by the EFTFitCoeff. No cuts or event selections are made.
Taking this histogram, do histEFT.as_hist({}) to get the sow of all events at the SM for the sample. 
This is equivalent to summing EFTFitCoeff[0] for all events for an EFT sample. 
'''

#!/usr/bin/env python
import numpy as np
import awkward as ak

np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import processor

# silence warnings due to using NanoGEN instead of full NanoAOD
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False

import hist
from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.eft_helper as efth

# Main analysis processor
class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], hist_lst = None, dtype=np.float32, do_errors=False):
        self._samples = samples
        self._wc_names_lst = wc_names_lst

        self._dtype = dtype
        self._do_errors = do_errors

        print("\n\n")
        print("self._samples", self._samples)
        print("self._wc_names_lst", self._wc_names_lst)
        print("\n\n")

        proc_axis = hist.axis.StrCategory([], name="process", growth=True)
        chan_axis = hist.axis.StrCategory([], name="channel", growth=True)

        self._histo_dict = {
            "sow" :     HistEFT(
                            proc_axis,
                            hist.axis.Regular(bins=1, start=0, stop=2, name="sow", label="sum of weights for all events"), 
                            wc_names=wc_names_lst, 
                            label="Events"),
            "sow_norm": HistEFT(
                            proc_axis,
                            hist.axis.Regular(bins=1, start=0, stop=2, name="sow_norm", label="normalized sum of weights for all events"), 
                            wc_names=wc_names_lst, 
                            label="Events"), 
            "nEvents":  HistEFT(
                            proc_axis,
                            hist.axis.Regular(bins=1, start=0, stop=2, name="nEvents", label="number of events"), 
                            wc_names=wc_names_lst, 
                            label="Events"),
        }

    @property
    def columns(self):
        return self._columns

    def process(self, events):

        # Dataset parameters
        dataset = events.metadata['dataset']
        isData  = self._samples[dataset]["isData"]
        if isData: raise Exception("Why are you running this over data?")

        hist_axis_name = self._samples[dataset]["histAxisName"]
        sow = self._samples[dataset]['nSumOfWeights']
        xsec = self._samples[dataset]['xsec']

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        eft_w2_coeffs = efth.calc_w2_coeffs(eft_coeffs,self._dtype) if (self._do_errors and eft_coeffs is not None) else None

        jets = events.GenJet
        njets = ak.num(jets)

        # Get nominal wgt
        counts = np.ones_like(events['event'])
        wgts = np.ones_like(events['event'])
        if eft_coeffs is None:
            # Basically any central MC samples
            wgts = events["genWeight"]
        norm = xsec/sow


        ####### Fill Histogram #######
        hout = self._histo_dict

        sow_fill_info = {
            "sow"       : counts, 
            "process"   : hist_axis_name, 
            "weight"    : wgts, 
            "eft_coeff" : eft_coeffs,
        }

        sow_norm_fill_info = {
            "sow_norm"  : counts, 
            "process"   : hist_axis_name, 
            "weight"    : wgts*norm, 
            "eft_coeff" : eft_coeffs,
        }

        # Here, weight = counts instead of wgts because this hist is just counting the number
        # of raw events in the file, not effected by the weighting of the event
        nevents_fill_info = {
            "nEvents"   : counts, 
            "process"   : hist_axis_name, 
            "weight"    : counts,
            "eft_coeff" : eft_coeffs,
        }

        hout["sow"].fill(**sow_fill_info)
        hout["sow_norm"].fill(**sow_norm_fill_info)
        hout["nEvents"].fill(**nevents_fill_info)

        return hout


    def postprocess(self, accumulator):
        return accumulator


