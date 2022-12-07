# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v1.0.1, 12/05/2022

# utilities for plotting Triton profile outputs

# N.B. the numpy array ordering of profile objects:
# 1: Depth (GC-corrected, if provided)
# 2: Nucleosome-level phased profile
# 3: Nucleosome center profile (GC-corrected, if provided)
# 4: Mean fragment size
# 5: Fragment size Shannon entropy
# 6: Fragment heterogeneity (unique fragment lengths / total fragments)
# 7: Fragment MAD (Mean Absolute Deviation)
# 8: Short:long ratio (x <= 120 / 140 <= x <= 250)
# 9: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***
# 10: A (Adenine) frequency**
# 11: C (Cytosine) frequency**
# 12: G (Guanine) frequency**
# 13: T (Tyrosine) frequency**

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cols = ['depth', 'phased-signal', 'nuc-centers', 'frag-mean', 'frag-ent', 'frag-hetero', 'frag-mad', 'frag-ratio',
        'peaks', 'a-freq', 'c-freq', 'g-freq', 't-freq']
norm_cols = ['depth', 'phased-signal', 'nuc-centers', 'frag-mean', 'frag-ent', 'frag-hetero', 'frag-mad', 'frag-ratio']


def plot_all_profiles(name, data, palette=None):
    """
    Plots all available profiles in one figure.
        Parameters:
            name (string): name of site being plotted
            data (pandas long df): signal profile(s) from Triton
            palette (target:color dictionary): palette for subtypes if categories is passed
    """
    xmin, xmax = data['loc'].min(), data['loc'].max() + 1
    if data['sample'].nunique() == 1:
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, figsize=(11, 8))
        sample_name = data['sample'][0]
        # 1: GC-Corrected Coverage
        df = data[data['profile'] == 'depth']
        ax1.fill_between(data=df, x='loc', y1=0, y2='value', color='silver', label='GC-Corrected Coverage')
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(df['value'].min() / 1.1, df['value'].max() * 1.1)
        ax1.set_ylabel('depth', fontsize=12)
        ax1.set_xticks([])
        # 2: Phased-Nucleosomes Signal and P.N.C. Coverage
        df = data[data['profile'] == 'nuc-centers']
        ax2.fill_between(data=df, x='loc', y1=0, y2='value', color='gray', label='P.N.C. Coverage (GC-C)')
        ax2.set_xlim(xmin, xmax)
        ax2.set_xticks([])
        df = data[data['profile'] == 'phased-signal']
        signal = df['value'].tolist()
        index_shift = df['loc'].min() / -1
        sns.lineplot(data=df, x='loc', y='value', color='darkviolet', label='Phased-Nucleosome Signal',
                     lw=3, alpha=0.7, ax=ax2, legend=False)
        ymin_2, ymax_2 = df['value'].min() * 0.8, df['value'].max() * 1.2
        ax2.set_ylim(ymin_2, ymax_2)
        df_peaks = data[data['profile'] == 'peaks']
        peak_locs = df_peaks.loc[df_peaks['value'] == 1, 'loc'].tolist()
        trough_locs = df_peaks.loc[df_peaks['value'] == -1, 'loc'].tolist()
        ax2.vlines(x=peak_locs, ymin=[signal[int(i + index_shift)] for i in peak_locs], ymax=ymax_2,
                   linestyle='dashed', linewidth=1)
        ax2.vlines(x=trough_locs, ymin=ymin_2, ymax=[signal[int(i + index_shift)] for i in trough_locs],
                   linestyle='dashed', linewidth=1)
        minus_loc = int(df_peaks.loc[df_peaks['value'] == -2, 'loc'])
        plus_loc = int(df_peaks.loc[df_peaks['value'] == 2, 'loc'])
        if minus_loc and plus_loc:
            ax2.vlines(x=[minus_loc, plus_loc], ymin=[signal[int(i + index_shift)] for i in [minus_loc, plus_loc]],
                       ymax=ymax_2, color='orange', linewidth=1)
        inflection_loc = int(df_peaks.loc[df_peaks['value'] == 3, 'loc'])
        if inflection_loc:
            ax2.vlines(x=inflection_loc, ymin=ymin_2, ymax=ymax_2, color='red', linewidth=1)
        ax2.set_ylabel('signal', fontsize=12)
        # 3: Fragment Mean
        df = data[data['profile'] == 'frag-mean']
        ax3.fill_between(data=df, x='loc', y1=0, y2='value', color='green', alpha=0.5)
        ax3.set_xlim(xmin, xmax)
        ax3.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax3.set_xticks([])
        sns.lineplot(data=df, x='loc', y='value', color='green', label='Fragment Mean', ax=ax3, legend=False)
        ax3.set_ylabel('length', fontsize=12)
        # 4: Fragment Shannon Entropy
        df = data[data['profile'] == 'frag-ent']
        ax4.fill_between(data=df, x='loc', y1=0, y2='value', color='red', alpha=0.5)
        ax4.set_xlim(xmin, xmax)
        ax4.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax4.set_xticks([])
        sns.lineplot(data=df, x='loc', y='value', color='red', label='Fragment Shannon Entropy (norm)',
                     ax=ax4, legend=False)
        ax4.set_ylabel('bits', fontsize=12)
        # 5: Fragment Heterogeneity (unique/total)
        df = data[data['profile'] == 'frag-hetero']
        ax5.fill_between(data=df, x='loc', y1=0, y2='value', color='blue', alpha=0.5)
        ax5.set_xlim(xmin, xmax)
        ax5.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax5.set_xticks([])
        sns.lineplot(data=df, x='loc', y='value', color='blue', label='Fragment Heterogeneity (unique/total)',
                     ax=ax5, legend=False)
        ax5.axvspan(inflection_loc - 5, inflection_loc + 5, alpha=0.2, color='red')
        ax5.set_ylabel('ratio', fontsize=12)
        # 6: Fragment MAD
        df = data[data['profile'] == 'frag-mad']
        ax6.fill_between(data=df, x='loc', y1=0, y2='value', color='magenta', alpha=0.5)
        ax6.set_xlim(xmin, xmax)
        ax6.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax6.set_xticks([])
        sns.lineplot(data=df, x='loc', y='value', color='magenta', label='Fragment MAD',
                     ax=ax6, legend=False)
        ax6.set_ylabel('bp', fontsize=12)
        # 7: Fragment Short:Long Ratio
        df = data[data['profile'] == 'frag-ratio']
        ax7.fill_between(data=df, x='loc', y1=0, y2='value', color='orange', alpha=0.5)
        ax7.set_xlim(xmin, xmax)
        ax7.set_ylim(df['value'].min(), df['value'].max() + (df['value'].max() - df['value'].min()) / 20)
        ax7.set_xlabel('location (bp)', fontsize=12)
        sns.lineplot(data=df, x='loc', y='value', color='orange', label='Fragment Short:Long Ratio',
                     ax=ax7, legend=False)
        ax7.set_ylabel('ratio', fontsize=12)
        lines, labels = [], []
        for ax in fig.axes:
            ax_line, ax_label = ax.get_legend_handles_labels()
            lines.extend(ax_line)
            labels.extend(ax_label)
        fig.legend(lines, labels, title_fontsize=18, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        ax1.set_title(sample_name + ' ' + name + ' profiles (normalized to mean = 1)')
        plt.tight_layout()
        fig.savefig(sample_name + '_' + name + '_AllProfiles.pdf', bbox_inches="tight")
        plt.close()
        return
    else:
        df_peaks = data[data['profile'] == 'peaks']
        data = data[~data['profile'].isin(['a-freq', 'c-freq', 'g-freq', 't-freq', 'peaks', 'nuc-centers'])]
        sample_name = 'MultiSample'
        if 'subtype' in data:
            hue = 'subtype'
        else:
            hue = 'sample'
        if palette is not None:
            sea = sns.FacetGrid(data, row='profile', hue=hue, despine=False, height=1.14, aspect=11, palette=palette,
                                sharey=False, legend_out=True)
        else:
            sea = sns.FacetGrid(data, row='profile', hue=hue, despine=False, height=1.14, aspect=11, sharey=False,
                                legend_out=True)
        sea.map(sns.lineplot, 'loc', 'value', alpha=0.8, legend='full', n_boot=100)
        sea.add_legend()
        if 'subtype' in data:
            ax = sea.axes.flatten()[1]
            for subtype in data['subtype'].unique():
                sub_df = df_peaks[df_peaks['subtype'] == subtype]
                minus_locs = sub_df.loc[sub_df['value'] == -2, 'loc'].values
                plus_locs = sub_df.loc[sub_df['value'] == 2, 'loc'].values
                inflect_locs = sub_df.loc[sub_df['value'] == 3, 'loc'].values
                if minus_locs.any() and plus_locs.any() and inflect_locs.any():
                    m_mean, m_std = np.mean(minus_locs), np.std(minus_locs)
                    p_mean, p_std = np.mean(plus_locs), np.std(plus_locs)
                    i_mean, i_std = np.mean(inflect_locs), np.std(inflect_locs)
                    ax.axvspan(m_mean - m_std, m_mean + m_std, alpha=0.1, color=palette[subtype])
                    ax.axvline(x=m_mean, alpha=0.6, color=palette[subtype])
                    ax.axvspan(p_mean - p_std, p_mean + p_std, alpha=0.1, color=palette[subtype])
                    ax.axvline(x=p_mean, alpha=0.6, color=palette[subtype])
                    ax.axvspan(i_mean - i_std, i_mean + i_std, alpha=0.1, color=palette[subtype])
                    ax.axvline(x=i_mean, alpha=0.6, color=palette[subtype])
        sea.set(xlim=(xmin, xmax))
        plt.savefig(sample_name + '_' + name + '_AllProfiles.pdf', bbox_inches="tight")
        plt.close()
        return


def plot_signal_profile(name, data, palette=None, show_mpc=False):
    """
    Plots signal profile only.
        Parameters:
            name (string): name of site being plotted
            data (pandas long df): signal profile(s) from Triton
            palette (target:color dictionary): palette for subtypes if categories is passed
            show_mpc (bool): whether to plot +/- 1 stdev lines for minus-one, plus-one, and inflection locations
    """
    xmin, xmax = data['loc'].min(), data['loc'].max() + 1
    if data['sample'].nunique() == 1:
        plt.figure(figsize=(11, 4))
        sample_name = data['sample'][0]
        df = data[data['profile'] == 'nuc-centers']
        plt.fill_between(data=df, x='loc', y1=0, y2='value', color='gray', label='P.N.C. Coverage (GC-C)')
        plt.xlim(xmin, xmax)
        df = data[data['profile'] == 'phased-signal']
        signal = df['value'].tolist()
        index_shift = df['loc'].min() / -1
        sns.lineplot(data=df, x='loc', y='value', color='darkviolet', label='Phased-Nucleosome Signal',
                     lw=3, alpha=0.7, legend=False)
        ymin_2, ymax_2 = df['value'].min() * 0.8, df['value'].max() * 1.2
        plt.ylim(ymin_2, ymax_2)
        df_peaks = data[data['profile'] == 'peaks']
        peak_locs = df_peaks.loc[df_peaks['value'] == 1, 'loc'].tolist()
        trough_locs = df_peaks.loc[df_peaks['value'] == -1, 'loc'].tolist()
        plt.vlines(x=peak_locs, ymin=[signal[int(i + index_shift)] for i in peak_locs], ymax=ymax_2,
                   linestyle='dashed', linewidth=1)
        plt.vlines(x=trough_locs, ymin=ymin_2, ymax=[signal[int(i + index_shift)] for i in trough_locs],
                   linestyle='dashed', linewidth=1)
        minus_loc = int(df_peaks.loc[df_peaks['value'] == -2, 'loc'])
        plus_loc = int(df_peaks.loc[df_peaks['value'] == 2, 'loc'])
        if minus_loc and plus_loc:
            plt.vlines(x=[minus_loc, plus_loc], ymin=[signal[int(i + index_shift)] for i in [minus_loc, plus_loc]],
                       ymax=ymax_2, color='orange', linewidth=1)
        inflection_loc = int(df_peaks.loc[df_peaks['value'] == 3, 'loc'])
        if inflection_loc:
            plt.vlines(x=inflection_loc, ymin=ymin_2, ymax=ymax_2, color='red', linewidth=1)
        plt.ylabel('signal level', fontsize=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.title(sample_name + ' ' + name + ' profile (normalized to mean = 1)')
        plt.tight_layout()
        plt.savefig(sample_name + '_' + name + '_SignalProfiles.pdf', bbox_inches="tight")
        plt.close()
        return
    else:
        df_peaks = data[data['profile'] == 'peaks']
        data = data[data['profile'] == 'nuc-centers']
        sample_name = 'MultiSample'
        if 'subtype' in data:
            hue = 'subtype'
        else:
            hue = 'sample'
        plt.figure(figsize=(11, 4))
        if palette is not None:
            sns.lineplot(data=data, x='loc', y='value', hue=hue, palette=palette)
        else:
            sns.lineplot(data=data, x='loc', y='value', hue=hue)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        if 'subtype' in data and show_mpc:
            for subtype in data['subtype'].unique():
                sub_df = df_peaks[df_peaks['subtype'] == subtype]
                minus_locs = sub_df.loc[sub_df['value'] == -2, 'loc'].values
                plus_locs = sub_df.loc[sub_df['value'] == 2, 'loc'].values
                inflect_locs = sub_df.loc[sub_df['value'] == 3, 'loc'].values
                if minus_locs.any() and plus_locs.any() and inflect_locs.any():
                    m_mean, m_std = np.mean(minus_locs), np.std(minus_locs)
                    p_mean, p_std = np.mean(plus_locs), np.std(plus_locs)
                    i_mean, i_std = np.mean(inflect_locs), np.std(inflect_locs)
                    plt.axvspan(m_mean - m_std, m_mean + m_std, alpha=0.2, color=palette[subtype])
                    plt.axvline(x=m_mean, alpha=0.6, color=palette[subtype])
                    plt.axvspan(p_mean - p_std, p_mean + p_std, alpha=0.2, color=palette[subtype])
                    plt.axvline(x=p_mean, alpha=0.6, color=palette[subtype])
                    plt.axvspan(i_mean - i_std, i_mean + i_std, alpha=0.4, color=palette[subtype])
                    plt.axvline(x=i_mean, alpha=0.8, color=palette[subtype])
        plt.xlim(xmin, xmax)
        plt.ylabel('signal level')
        plt.title(sample_name + ' ' + name + ' profiles (normalized to mean = 1)')
        plt.tight_layout()
        plt.savefig(sample_name + '_' + name + '_SignalProfile.pdf', bbox_inches="tight")
        plt.close()
        return


def normalize_data(data):
    mean_val = np.mean(data)
    if mean_val == 0:
        return data
    else:
        return data / mean_val


# below is for a non-issue iterating through "files"
# noinspection PyUnresolvedReferences
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
    parser.add_argument('-w', '--window', help='if set, data is "windowed" - set 0-point to the middle instead of the'
                                               '5\' end', action='store_true')

    args = parser.parse_args()
    input_path = args.input
    plot_mode = args.mode
    categories = args.categories
    sites_path = args.sites
    window = args.window

    ####################################################################################################################
    # constants which should be passed in the future
    targets = ['HD', 'ARPC', 'NEPC', 'Basal', 'Patient']
    colors = ['#009988', '#0077BB', '#CC3311', '#EE3377', '#BBBBBB']
    palette = {targets[i]: colors[i] for i in range(len(targets))}
    ####################################################################################################################

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
                for col in norm_cols:
                    df[col] = normalize_data(df[col])
                if categories is not None:
                    print('### categories specified but running in single-sample mode - ignoring categories')
                df = pd.melt(df, id_vars=['sample', 'loc'], value_vars=cols, var_name='profile')
                if plot_mode == 'all':
                    plot_all_profiles(site, df, palette=palette)
                else:
                    plot_signal_profile(site, df, palette=palette)
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
                    for col in norm_cols:
                        tdf[col] = normalize_data(tdf[col])
                df = pd.concat(dfs)
                if categories is not None:
                    df = pd.melt(df, id_vars=['sample', 'loc', 'subtype'], value_vars=cols, var_name='profile')
                else:
                    df = pd.melt(df, id_vars=['sample', 'loc'], value_vars=cols, var_name='profile')
                if plot_mode == 'all':
                    plot_all_profiles(site, df, palette=palette)
                else:
                    plot_signal_profile(site, df, palette=palette)


if __name__ == "__main__":
    main()
