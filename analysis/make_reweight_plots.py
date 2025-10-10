import argparse
import os

import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep

import topcoffea.modules.utils as utils


def make_hist_ratio_plot(h_powheg, h_smeft, h_smeft_rwgt, xlabel, title, ratio_hlines=[1,5], ratio_ylim=[0,2]):
    print(f"h_powheg: {h_powheg}")
    # get ratio info
    centers = h_powheg.axes.centers[0]
    # edges = h_powheg.axes[0].edges

    smeft_ratio = h_smeft.values()/h_powheg.values()
    smeft_rwgt_ratio = h_smeft_rwgt.values()/h_powheg.values()

    hep.style.use("CMS")
    # hep.cms.label(label='Work in progress', com=13)

    # Initialize figure and axes
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10,12),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.1)

    # Plot histograms
    hep.histplot(h_powheg, ax=ax, stack=False, yerr=False, linewidth=2, label='Powheg', color='black')
    # hep.histplot(h_smeft, ax=ax, stack=False, yerr=False, linewidth=2, label='SMEFTsim')
    # hep.histplot(h_smeft_rwgt, ax=ax, stack=False, yerr=False, linewidth=2, label='reweighted SMEFTsim')
    # # Plot ratio 
    # rax.scatter(centers, smeft_ratio)
    # rax.scatter(centers, smeft_rwgt_ratio)

    # ## Formatting
    # ax.legend(loc = 'upper right')
    # ax.set_ylabel("Events", fontsize='medium')
    # ax.set_xlabel("")
    # rax.set_ylabel("Ratio", fontsize='medium')
    # rax.set_xlabel(xlabel, fontsize="medium")
    # rax.set_ylim(ratio_ylim)
    # for line in ratio_hlines:
    #     rax.axhline(y=line, color='gray', linestyle='--')
    # plt.figtext(0.13, 0.9, title, fontsize='medium')

    return fig, ax, rax


def normalize_hist(h):
    
    counts = h.values()
    bin_edges = h.axes[0].edges
    bin_widths = np.diff(bin_edges)
    area = np.sum(counts * bin_widths)
    
    return counts/area


def save_figure(fig, figname, outdir=''):
    outname=os.path.join(outdir, f"{figname}.png")
    fig.savefig(outname, bbox_inches='tight')
    print(f'plot saved to {outname}')
    plt.close(fig)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description = 'Customize inputs')
    parser.add_argument('--fpowheg', required=True, help='pkl file with powheg histograms')
    parser.add_argument('--fsmeft', required=True, help='pkl file with smeft histograms')
    parser.add_argument('--fsmeft_rwgt', required=True, help='pkl file with reweighted smeft histograms')
    parser.add_argument('--outdir', default='.', help='output directory path')
    parser.add_argument('--hist', default=None, help='single hist to plot')

    args = parser.parse_args()
    fpowheg = args.fpowheg
    fsmeft = args.fsmeft
    fsmeft_rwgt = args.fsmeft_rwgt
    outdir = args.outdir
    hist_name = args.hist 

    fpowheg_hists = utils.get_hist_from_pkl(fpowheg, allow_empty=False)
    fsmeft_hists = utils.get_hist_from_pkl(fsmeft, allow_empty=False)
    fsmeft_rwgt_hists = utils.get_hist_from_pkl(fsmeft_rwgt, allow_empty=False)

    if hist_name is None: 
        hist_list = fpowheg_hists.keys()
    else:
        hist_list = [args.hist]

    for name in hist_list:
        h_powheg = fpowheg_hists[name].as_hist({})
        h_smeft = fsmeft_hists[name].as_hist({})
        h_smeft_rwgt = fsmeft_rwgt_hists[name].as_hist({})

        fig, ax, rax = make_hist_ratio_plot(h_powheg, h_smeft, h_smeft_rwgt, name, "With Selections")