# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v1.0.0, 11/15/2022

# utilities for plotting Triton profile outputs

# TODO: make below individual-ready (list of genes/regions to plot)
# TODO: based on Griffin also make composite line plots, composite grouped plots, etc.

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_all_profiles(name, data, sample, stacked=False):
    """
    One-hot encode a nucleotide sequence (as a binary numpy array)
        Parameters:
            seq (string): string of nucleotides to one-hot encode
        Returns:
            numpy array: one-hot encoded nucleotide sequence of size 5xN
    """
    # 1: GC-corrected (if provided by Griffin) depth
    # 2: Nucleosome-level phased profile
    # 3: Nucleosome center profile
    # 4: Mean fragment size
    # 5: Fragment size Shannon entropy
    # 6: Region fragment profile normalized Shannon entropy
    # 7: Fragment heterogeneity (unique fragment lengths / total fragments)
    # 8: Fragment MAD (Mean Absolute Deviation)
    # 9: Short:long ratio (x <= 120 / 140 <= x <= 250)
    # 10: A (Adenine) frequency
    # 11: C (Cytosine) frequency
    # 12: G (Guanine) frequency
    # 13: T (Tyrosine) frequency
    # axes are depth+phasing, shannon, norm-shannon, heterogeneity, and short:long ratio
    if stacked:
        roi_length = data.shape[0]
        x_vals = list(range(int(-roi_length / 2), int(roi_length/2)))
        comp = ' Composite'
    else:
        x_vals = list(range(data.shape[0]))
        comp = ''
    # shannon profile (non-normalized) is excluded in favor of normalized
    depth, phased_signal, center_profile, mean_profile, norm_profile, hetero_profile, mad_profile,\
        ratio_profile, a_freq, c_freq, g_freq, t_freq =\
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 5], data[:, 6], data[:, 7], data[:, 8],\
            data[:, 9], data[:, 10], data[:, 11], data[:, 12]

    fig, (ax1, ax2, ax3, ax5, ax6, ax7, ax8, ax9) = plt.subplots(8, figsize=(20, 10))
    # 1: depth+phasing
    ax1.fill_between(x_vals, 0, depth, label='GC-Corrected Coverage', color='silver')
    # ax1.plot(x_vals, phased_signal, color='darkviolet', label='Phased Component', lw=3, alpha=0.7)
    ax1.set_xlim(x_vals[0], x_vals[-1] + 1)
    ax1.set_ylim(min(depth), max(depth) + (max(depth) - min(depth))/20)
    ax1.set_ylabel('Depth', fontsize=12)
    # 2: nucleosome centers
    ax2.fill_between(x_vals, 0, center_profile, label='P.N.C. Coverage (GC-C)', color='gray')
    ax2.plot(x_vals, phased_signal, color='darkviolet', label='Phased Component', lw=3, alpha=0.7)
    ax2.set_xlim(x_vals[0], x_vals[-1] + 1)
    ax2.set_ylim(min(phased_signal), max(phased_signal) + (max(phased_signal) - min(phased_signal))/10)
    ax2.set_ylabel('Depth', fontsize=12)
    # 3: fragment means
    ax3.plot(x_vals, mean_profile, color='green', label='Fragment Mean')
    ax3.fill_between(x_vals, 0, mean_profile, color='green', alpha=0.5)
    ax3.set_xlim(x_vals[0], x_vals[-1] + 1)
    ax3.set_ylim(min(mean_profile), max(mean_profile) + (max(mean_profile) - min(mean_profile))/20)
    ax3.set_ylabel('Length', fontsize=12)
    # 4: shannon entropy
    # ax4.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    # ax4.plot(x_vals, shan_profile, color='red', label='Shannon Entropy')
    # ax4.fill_between(x_vals, 0, shan_profile, color='red', alpha=0.3)
    # ax4.set_xlim(x_vals[0], x_vals[-1] + 1)
    # ax4.set_ylim(0, 1.1)
    # ax4.set_ylabel('Entropy', fontsize=12)
    # 5: normalized shannon entropy
    ax5.plot(x_vals, norm_profile, color='red', label='Fragment Shannon Entropy (norm)')
    ax5.fill_between(x_vals, 0, norm_profile, color='red', alpha=0.5)
    ax5.set_xlim(x_vals[0], x_vals[-1] + 1)
    ax5.set_ylim(min(norm_profile), max(norm_profile) + (max(norm_profile) - min(norm_profile))/20)
    ax5.set_ylabel('bits', fontsize=12)
    # 6: fragment heterogeneity
    ax6.plot(x_vals, hetero_profile, color='blue', label='Fragment Heterogeneity (unique/total)')
    ax6.fill_between(x_vals, 0, hetero_profile, color='blue', alpha=0.5)
    ax6.set_xlim(x_vals[0], x_vals[-1] + 1)
    ax6.set_ylim(min(hetero_profile), max(hetero_profile) + (max(hetero_profile) - min(hetero_profile))/20)
    ax6.set_ylabel(' U:T Ratio', fontsize=12)
    # 7: fragment mad
    ax7.plot(x_vals, mad_profile, color='magenta', label='Fragment MAD')
    ax7.fill_between(x_vals, 0, mad_profile, color='magenta', alpha=0.5)
    ax7.set_xlim(x_vals[0], x_vals[-1] + 1)
    ax7.set_ylim(min(mad_profile), max(mad_profile) + (max(mad_profile) - min(mad_profile))/20)
    ax7.set_ylabel('MAD', fontsize=12)
    # 8: short:long ratio
    ax8.plot(x_vals, ratio_profile, color='orange', label='Fragment Short:Long Ratio')
    ax8.fill_between(x_vals, 0, ratio_profile, color='orange', alpha=0.5)
    ax8.set_xlim(x_vals[0], x_vals[-1] + 1)
    ax8.set_ylim(min(ratio_profile), max(ratio_profile) + (max(ratio_profile) - min(ratio_profile))/20)
    ax8.set_ylabel('S:L Ratio', fontsize=12)
    # 9: nt frequencies
    ax9.stackplot(x_vals, a_freq, c_freq, t_freq, g_freq, labels=['A', 'C', 'T', 'G'])
    ax9.set_xlim(x_vals[0], x_vals[-1] + 1)
    ax9.set_ylim(0, 1)
    ax9.set_ylabel('f(nt)', fontsize=12)
    lines, labels = [], []
    for ax in fig.axes:
        ax_line, ax_label = ax.get_legend_handles_labels()
        lines.extend(ax_line)
        labels.extend(ax_label)
    plt.suptitle(name + comp + ' profiles (minus background)')
    fig.legend(lines, labels, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    fig.savefig(sample + '_' + name + '_AllProfiles.pdf', bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='\n### triton_plotters.py ### plots Triton output profiles')
    parser.add_argument('-i', '--input', help='one or more TritonProfiles.npz files', nargs='*', required=True)
    parser.add_argument('-m', '--mode', help='plotting mode (all: all profiles, signal: only signal profile)',
                        required=False, default='signal')
    parser.add_argument('-c', '--categories', help='tsv file containing matched sample names (column "sample") and '
                                                   'categories (column "subtype"/"category") to color samples by '
                                                   'category rather than individually', required=False, default=None)
    parser.add_argument('-g', '--group_ci', help='if set, plot categories together with a confidence interval rather'
                                                 'than individually', action='store_true')
    parser.add_argument('-s', '--sites', help='file containing a list (row-wise) of sites to plot; defaults to plotting'
                                              'all sites', required=False, default=None)

    args = parser.parse_args()
    input_path = args.input
    plot_mode = args.mode
    categories = args.categories
    group = args.group_ci
    sites = args.sites

    ### compare results plotting mean = 1 and full scaled signals as composites of samples


    test_data = np.load(input_path)
    sample = os.path.basename(input_path).split('_TritonProfiles.npz')[0]
    for site in test_data.files:
        plot_all_profiles(site, test_data[site], sample, stacked=True)


if __name__ == "__main__":
    main()