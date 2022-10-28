# TODO: make below individual-ready (list of genes/regions to plot)
# TODO: based on Griffin also make composite line plots, composite grouped plots, etc.

if site in to_plot:  # plot selected regions
    # axes are depth+phasing, shannon, norm-shannon, heterogeneity, and short:long ratio
    x_vals = list(range(-int(roi_length / 2), int(roi_length / 2)))  # currently assume centered regions
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(20, 11))
    fig.suptitle(sample + ' ' + site + ' CNN Input Profile', x=0.17, y=1.1, fontsize=24)
    ax1.set_title(roi_sequence[:63] + ' . . .                                          . . . ' + roi_sequence[-63:])
    # 1: depth+phasing
    ax1.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax1.fill_between(x_vals, 0, depth, label='GC-Corrected Coverage', color='gray')
    ax1.plot(x_vals, phased_signal, color='#8927D6FA', label='Phased Component', lw=3, alpha=0.8)
    ax1.set_xlim(x_vals[0], x_vals[-1])
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Scaled Depth', fontsize=12)
    # 2: shannon entropy
    ax2.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax2.plot(x_vals, shan_profile, color='red', label='Shannon Entropy')
    ax2.fill_between(x_vals, 0, shan_profile, color='red', alpha=0.3)
    ax2.set_xlim(x_vals[0], x_vals[-1])
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Scaled Entropy', fontsize=12)
    # 3: normalized shannon entropy
    ax3.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax3.plot(x_vals, norm_profile, color='blue', label='Region-Fragments-Normalized Shannon Entropy')
    ax3.fill_between(x_vals, 0, norm_profile, color='blue', alpha=0.3)
    ax3.set_xlim(x_vals[0], x_vals[-1])
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('Scaled Entropy', fontsize=12)
    # 4: fragment heterogeneity
    ax4.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax4.plot(x_vals, hetero_profile, color='green', label='Fragment Heterogeneity (unique lengths / total frags)')
    ax4.fill_between(x_vals, 0, hetero_profile, color='green', alpha=0.3)
    ax4.set_xlim(x_vals[0], x_vals[-1])
    ax4.set_ylim(0, 1.1)
    ax4.set_ylabel('Scaled Ratio', fontsize=12)
    # 5: short:long ratio
    ax5.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
    ax5.plot(x_vals, ratio_profile, color='orange', label='Short:Long Ratio')
    ax5.fill_between(x_vals, 0, ratio_profile, color='orange', alpha=0.3)
    ax5.set_xlim(x_vals[0], x_vals[-1])
    ax5.set_ylim(0, 1.1)
    ax5.set_ylabel('Scaled Ratio', fontsize=12)
    lines, labels = [], []
    for ax in fig.axes:
        ax_line, ax_label = ax.get_legend_handles_labels()
        lines.extend(ax_line)
        labels.extend(ax_label)
    fig.legend(lines, labels, title_fontsize=18, loc='lower center', bbox_to_anchor=(0.79, 0.99))
    plt.tight_layout()
    fig.savefig(out_direct + '/' + sample + '_' + site + '.pdf', bbox_inches="tight")
    plt.close()