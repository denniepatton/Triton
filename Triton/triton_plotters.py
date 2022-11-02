# TODO: make below individual-ready (list of genes/regions to plot)
# TODO: based on Griffin also make composite line plots, composite grouped plots, etc.

import numpy as np
import matplotlib.pyplot as plt
from triton_helpers import normalize_data


def plot_all_profiles(name, data):
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
    x_vals = list(range(data.shape[0]))  # currently assume centered regions
    depth, phased_signal, center_profile, mean_profile, shan_profile, norm_profile, hetero_profile, mad_profile,\
        ratio_profile, a_freq, c_freq, g_freq, t_freq =\
        data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7], data[:, 8],\
            data[:, 9], data[:, 10], data[:, 11], data[:, 12]

    depth = normalize_data(depth)
    phased_signal = normalize_data(phased_signal)
    center_profile = normalize_data(center_profile)
    mean_profile = normalize_data(mean_profile)
    shan_profile = normalize_data(shan_profile)
    norm_profile = normalize_data(norm_profile)
    hetero_profile = normalize_data(hetero_profile)
    mad_profile = normalize_data(mad_profile)
    ratio_profile = normalize_data(ratio_profile)

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, figsize=(20, 11))
    fig.suptitle(name + ' Profiles', x=0.17, y=1.1, fontsize=24)
    # 1: depth+phasing
    ax1.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax1.fill_between(x_vals, 0, depth, label='GC-Corrected Coverage', color='gray')
    ax1.plot(x_vals, phased_signal, color='#8927D6FA', label='Phased Component', lw=3, alpha=0.8)
    ax1.set_xlim(x_vals[0], x_vals[-1])
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Scaled Depth', fontsize=12)
    # 2: nucleosome centers
    ax2.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax2.plot(x_vals, center_profile, color='purple', label='Nucleosome Centers')
    ax2.fill_between(x_vals, 0, center_profile, color='purple')
    ax2.set_xlim(x_vals[0], x_vals[-1])
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Scaled Depth', fontsize=12)
    # 3: fragment means
    ax3.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax3.plot(x_vals, mean_profile, color='pink', label='Fragment Means')
    ax3.fill_between(x_vals, 0, mean_profile, color='pink', alpha=0.3)
    ax3.set_xlim(x_vals[0], x_vals[-1])
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('Scaled Entropy', fontsize=12)
    # 4: shannon entropy
    ax4.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax4.plot(x_vals, shan_profile, color='red', label='Shannon Entropy')
    ax4.fill_between(x_vals, 0, shan_profile, color='red', alpha=0.3)
    ax4.set_xlim(x_vals[0], x_vals[-1])
    ax4.set_ylim(0, 1.1)
    ax4.set_ylabel('Scaled Entropy', fontsize=12)
    # 5: normalized shannon entropy
    ax5.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax5.plot(x_vals, norm_profile, color='blue', label='Region-Fragments-Normalized Shannon Entropy')
    ax5.fill_between(x_vals, 0, norm_profile, color='blue', alpha=0.3)
    ax5.set_xlim(x_vals[0], x_vals[-1])
    ax5.set_ylim(0, 1.1)
    ax5.set_ylabel('Scaled Entropy', fontsize=12)
    # 6: fragment heterogeneity
    ax6.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax6.plot(x_vals, hetero_profile, color='green', label='Fragment Heterogeneity (unique lengths / total frags)')
    ax6.fill_between(x_vals, 0, hetero_profile, color='green', alpha=0.3)
    ax6.set_xlim(x_vals[0], x_vals[-1])
    ax6.set_ylim(0, 1.1)
    ax6.set_ylabel('Scaled Ratio', fontsize=12)
    # 7: fragment mad
    ax7.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax7.plot(x_vals, mad_profile, color='yellow', label='Fragment MAD')
    ax7.fill_between(x_vals, 0, mad_profile, color='yellow', alpha=0.3)
    ax7.set_xlim(x_vals[0], x_vals[-1])
    ax7.set_ylim(0, 1.1)
    ax7.set_ylabel('Scaled Ratio', fontsize=12)
    # 8: short:long ratio
    ax8.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax8.plot(x_vals, ratio_profile, color='orange', label='Short:Long Ratio')
    ax8.fill_between(x_vals, 0, ratio_profile, color='orange', alpha=0.3)
    ax8.set_xlim(x_vals[0], x_vals[-1])
    ax8.set_ylim(0, 1.1)
    ax8.set_ylabel('Scaled Ratio', fontsize=12)
    # 9: nt frequencies
    ax9.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax9.stackplot(x_vals, a_freq, c_freq, t_freq, g_freq, labels=['A', 'C', 'T', 'G'])
    ax9.set_xlim(x_vals[0], x_vals[-1])
    ax9.set_ylim(0, 1.1)
    ax9.set_ylabel('Scaled Ratio', fontsize=12)
    lines, labels = [], []
    for ax in fig.axes:
        ax_line, ax_label = ax.get_legend_handles_labels()
        lines.extend(ax_line)
        labels.extend(ax_label)
    fig.legend(lines, labels, title_fontsize=18, loc='lower center', bbox_to_anchor=(0.79, 0.99))
    plt.tight_layout()
    fig.savefig('/fh/fast/ha_g/user/rpatton/scripts/Triton/tests/test_comp/' + name + '.pdf', bbox_inches="tight")
    plt.close()


def main():
    test_data = np.load('/fh/fast/ha_g/user/rpatton/scripts/Triton/tests/test_comp/test_comp_TritonProfiles.npz')
    for site in test_data.files:
        plot_all_profiles(site, test_data[site])


if __name__ == "__main__":
    main()