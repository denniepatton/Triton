# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v1.0.0, 11/15/2022

# utilities for plotting Triton profile outputs

# N.B. the numpy array ordering ofr profile objects:
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

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from triton_helpers import normalize_data

# TODO: add in peak locs from the composite file
# TODO: pass own palette

targets = ['Healthy', 'ARPC', 'Luminal', 'NEPC', 'Basal', 'Patient', 'NonDiff', 'Dual', 'ARlow', 'AMPC', '+', '-']
colors = ['#009988', '#0077BB', '#33BBEE', '#CC3311', '#EE7733', '#EE3377', '#BBBBBB', '#9370DB', '#77BB00', '#466F00',
          '#33BBEE', '#EE7733']
palette = {targets[i]: colors[i] for i in range(len(targets))}

cols = ['depth', 'phased-signal', 'nuc-centers', 'frag-mean', 'frag-ent', 'frag-norm',
        'frag-hetero', 'frag-mad', 'frag-ratio', 'a-freq', 'c-freq', 'g-freq', 't-freq']


def plot_all_profiles(name, data):
    """
    Plots all available profiles in one figure.
        Parameters:
            name (string): name of site being plotted
            data (pandas long df): signal profile(s) from Triton
    """
    xmin, xmax = data['loc'].min(), data['loc'].max() + 1
    if data['sample'].nunique() == 1:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, figsize=(11, 8))
        sample_name = data['sample'][0]
        # 1: GC-Corrected Coverage
        df = data[data['profile'] == 'depth']
        ax1.fill_between(data=df, x='loc', y1=0, y2='value', color='silver', label='GC-Corrected Coverage')
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax1.set_ylabel('depth', fontsize=12)
        ax1.set_xticks([])
        # 2: Phased-Nucleosomes Signal and P.N.C. Coverage
        df = data[data['profile'] == 'nuc-centers']
        ax2.fill_between(data=df, x='loc', y1=0, y2='value', color='gray', label='P.N.C. Coverage (GC-C)')
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylabel('depth', fontsize=12)
        ax2.set_xticks([])
        df = data[data['profile'] == 'phased-signal']
        sns.lineplot(data=df, x='loc', y='value', color='darkviolet', label='Phased-Nucleosome Signal',
                     lw=3, alpha=0.7, ax=ax2, legend=False)
        ax2.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        # 3: Fragment Mean
        df = data[data['profile'] == 'frag-mean']
        ax3.fill_between(data=df, x='loc', y1=0, y2='value', color='green', alpha=0.5)
        ax3.set_xlim(xmin, xmax)
        ax3.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax3.set_ylabel('length', fontsize=12)
        ax3.set_xticks([])
        sns.lineplot(data=df, x='loc', y='value', color='green', label='Fragment Mean', ax=ax3, legend=False)
        # 4: Fragment Shannon Entropy (norm)
        df = data[data['profile'] == 'frag-norm']
        ax4.fill_between(data=df, x='loc', y1=0, y2='value', color='red', alpha=0.5)
        ax4.set_xlim(xmin, xmax)
        ax4.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax4.set_ylabel('bits', fontsize=12)
        ax4.set_xticks([])
        sns.lineplot(data=df, x='loc', y='value', color='red', label='Fragment Shannon Entropy (norm)',
                     ax=ax4, legend=False)
        # 5: Fragment Heterogeneity (unique/total)
        df = data[data['profile'] == 'frag-hetero']
        ax5.fill_between(data=df, x='loc', y1=0, y2='value', color='blue', alpha=0.5)
        ax5.set_xlim(xmin, xmax)
        ax5.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax5.set_ylabel('ratio', fontsize=12)
        ax5.set_xticks([])
        sns.lineplot(data=df, x='loc', y='value', color='blue', label='Fragment Heterogeneity (unique/total)',
                     ax=ax5, legend=False)
        # 6: Fragment MAD
        df = data[data['profile'] == 'frag-mad']
        ax6.fill_between(data=df, x='loc', y1=0, y2='value', color='magenta', alpha=0.5)
        ax6.set_xlim(xmin, xmax)
        ax6.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax6.set_ylabel('bp', fontsize=12)
        ax6.set_xticks([])
        sns.lineplot(data=df, x='loc', y='value', color='magenta', label='Fragment MAD',
                     ax=ax6, legend=False)
        # 7: Fragment Short:Long Ratio
        df = data[data['profile'] == 'frag-ratio']
        ax7.fill_between(data=df, x='loc', y1=0, y2='value', color='orange', alpha=0.5)
        ax7.set_xlim(xmin, xmax)
        ax7.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax7.set_ylabel('ratio', fontsize=12)
        ax7.set_xlabel('location (bp)', fontsize=12)
        sns.lineplot(data=df, x='loc', y='value', color='orange', label='Fragment Short:Long Ratio',
                     ax=ax7, legend=False)
        lines, labels = [], []
        for ax in fig.axes:
            ax_line, ax_label = ax.get_legend_handles_labels()
            lines.extend(ax_line)
            labels.extend(ax_label)
        plt.suptitle(sample_name + ' ' + name + ' profiles (minus background)')
        fig.legend(lines, labels, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.tight_layout()
        fig.savefig(sample_name + '_' + name + '_AllProfiles.pdf', bbox_inches="tight")
        plt.close()
        return
    else:
        sample_name = 'MultiSample'
        if 'subtype' in data:
            hue = 'subtype'
        else:
            hue = 'sample'
        sea = sns.FacetGrid(data, row='profile', hue=hue, despine=False, height=1.5, aspect=5)
        # sea = sns.FacetGrid(data, row='profile', hue=hue, despine=False, height=1.5, aspect=5, palette=palette)
        sea.map(sns.lineplot, 'loc', 'value', alpha=0.5, legend=True, n_boot=100)
        sea.add_legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        # sea.set(ylim=(0, 1.1), xlim=(xmin, xmax))
        sea.set(xlim=(xmin, xmax))
        plt.suptitle(sample_name + ' ' + name + ' profiles (variance-scaled)')
        plt.tight_layout()
        plt.savefig(sample_name + '_' + name + '_AllProfiles.pdf', bbox_inches="tight")
        plt.close()
        return


def main():
    parser = argparse.ArgumentParser(description='\n### triton_plotters.py ### plots Triton output profiles')
    parser.add_argument('-i', '--input', help='one or more TritonProfiles.npz files', nargs='*', required=True)
    parser.add_argument('-m', '--mode', help='plotting mode (all: all profiles, signal: only signal profile)',
                        required=False, default='signal')
    parser.add_argument('-c', '--categories', help='tsv file containing matched sample names (column "sample") and '
                                                   'categories (column "subtype"/"category") to color samples by '
                                                   'category rather than individually', required=False, default=None)
    parser.add_argument('-s', '--sites', help='file containing a list (row-wise) of sites to plot; defaults to plotting'
                                              'all sites', required=False, default=None)
    parser.add_argument('-g', '--group_ci', help='if set, plot categories together with a confidence interval rather'
                                                 'than individually', action='store_true')
    parser.add_argument('-w', '--window', help='if set, data is "windowed" - set 0-point to the middle instead of the'
                                               '5\' end', action='store_true')

    args = parser.parse_args()
    input_path = args.input
    plot_mode = args.mode
    categories = args.categories
    sites_path = args.sites
    group = args.group_ci
    window = args.window

    if categories is not None:
        categories = pd.read_table(categories, sep='\t', header=None)
        categories = dict(zip(categories[0], categories[1]))

    if sites_path is not None:
        with open(sites_path) as f:
             sites = f.read().splitlines()

    if len(input_path) == 1:  # individual sample
        test_data = np.load(input_path[0])
        sample = os.path.basename(input_path[0]).split('_TritonProfiles.npz')[0]
        for site in test_data.files:
            if sites_path is None or site in sites:
                df = pd.DataFrame(test_data[site], columns=cols)
                df['sample'] = sample
                df['loc'] = np.arange(len(df))
                if window:
                    df['loc'] = df['loc'] - len(df)/2
                df = pd.melt(df, id_vars=['sample', 'loc'], value_vars=cols, var_name='profile')
                if plot_mode == 'all':
                    plot_all_profiles(site, df)
                else:
                    plot_signal_profile(site, test_data[site], sample, categories)
    else:  # multiple samples:
        samples = [os.path.basename(path).split('_TritonProfiles.npz')[0] for path in input_path]
        tests_data = [np.load(path) for path in input_path]
        for site in tests_data[0].files:
            if sites_path is None or site in sites:
                dfs = [pd.DataFrame(test_data[site], columns=cols) for test_data in tests_data]
                for tdf, sample in zip(dfs, samples):
                    tdf['loc'] = np.arange(len(tdf))
                    if window:
                        tdf['loc'] = tdf['loc'] - len(tdf) / 2
                    tdf['sample'] = sample
                    if categories is not None:
                        tdf['subtype'] = categories[sample]
                    ###
                    def mean_data(data):
                        if np.max(data) - np.min(data) == 0:
                            return data
                        else:
                            return data / np.mean(data)
                    ###
                    for col in cols:
                        # tdf[col] = normalize_data(tdf[col])
                        tdf[col] = mean_data(tdf[col])
                df = pd.concat(dfs)
                if categories is not None:
                    df = pd.melt(df, id_vars=['sample', 'loc', 'subtype'], value_vars=cols, var_name='profile')
                else:
                    df = pd.melt(df, id_vars=['sample', 'loc'], value_vars=cols, var_name='profile')
                df = df[~df['profile'].isin(['frag-ent', 'a-freq', 'c-freq', 'g-freq', 't-freq', 'nuc-centers',
                                             'frag-ratio', 'frag-mad', 'frag-mean'])]
                if plot_mode == 'all':
                    plot_all_profiles(site, df)
                else:
                    plot_signal_profile(site, test_data[site], sample, categories)


    ### compare results plotting mean = 1 and full scaled signals as composites of samples





if __name__ == "__main__":
    main()