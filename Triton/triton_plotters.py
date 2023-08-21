# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v0.2.1, 06/30/2023

# Utility for plotting Triton profile outputs
# N.B. Peak locations and nucleotide-frequency plotting is not supported
# (but feel free to modify code to include them)

"""
N.B. the numpy array ordering of profile objects:
1: Depth (GC-corrected, if provided)
2: Probable nucleosome center profile (fragment length re-weighted depth)
3: Phased-nucleosome profile (Fourier filtered probable nucleosome center profile)
4: Fragment lengths' short:long ratio (x <= 150 / x > 150)
5: Fragment lengths' diversity (unique fragment lengths / total fragments)
6: Fragment lengths' Shannon Entropy (normalized by window Shannon Entropy)
7: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***
8: A (Adenine) frequency**
9: C (Cytosine) frequency**
10: G (Guanine) frequency**
11: T (Tyrosine) frequency**

eliff TritonMe (methylation mode, using Bismark-aligned outputs:
1: Depth (GC-corrected, if provided)
2: Probable nucleosome center profile (fragment length re-weighted depth)
3: Phased-nucleosome profile (Fourier filtered probable nucleosome center profile)
4: Fragment lengths' short:long ratio (x <= 150 / x > 150)
5: Fragment lengths' diversity (unique fragment lengths / total fragments)
6: Fragment lengths' Shannon Entropy (normalized by window Shannon Entropy)
7: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***
8: CpG methylation frequency (NaN if no overlapping targets)
9: CHG methylation frequency (NaN if no overlapping targets)
10: CHH methylation frequency (NaN if no overlapping targets)
11: CN/CHN methylation frequency (NaN if no overlapping targets)
12: A (Adenine) frequency**
13: C (Cytosine) frequency**
14: G (Guanine) frequency**
15: T (Tyrosine) frequency**
"""


import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cols = ['depth', 'nuc-centers', 'phased-signal', 'frag-ratio', 'frag-diversity', 'frag-entropy',
        'peaks', 'a-freq', 'c-freq', 'g-freq', 't-freq']
cols_me = ['depth', 'nuc-centers', 'phased-signal', 'frag-ratio', 'frag-diversity', 'frag-entropy', 'peaks',
           'CpG-Me-freq', 'CHG-ME-freq', 'CHH-Me-freq', 'CHN-Me-freq', 'a-freq', 'c-freq', 'g-freq', 't-freq']
# signals to normalize
norm_cols = ['depth', 'nuc-centers', 'phased-signal', 'frag-ratio', 'frag-diversity', 'frag-entropy']
smooth_cols = ['CpG-Me-freq', 'CHG-ME-freq', 'CHH-Me-freq', 'CHN-Me-freq']


def plot_profiles(name, data, plot_mode, palette=None, show_inflection=False):
    """
    Plots all profiles specified with plot_mode in one figure.
        Parameters:
            name (string): name of site being plotted
            data (pandas long df): signal profile(s) from Triton
            plot_mode (string): name of plotting mode (signal, all, DHS)
            palette (target:color dictionary): palette for subtypes if categories is passed
            show_inflection (bool): whether to plot vertical lines showing the inflection and +/-1 nucleosome positions
    """
    xmin, xmax = data['loc'].min(), data['loc'].max() + 1
    df_peaks = data[data['profile'] == 'peaks']
    if plot_mode == 'all':
        data = data[~data['profile'].isin(['a-freq', 'c-freq', 'g-freq', 't-freq', 'peaks'])]
    elif plot_mode == 'signal':
        data = data[data['profile'] == 'phased-signal']
    elif plot_mode == 'TME':
        data = data[data['profile'].isin(['depth', 'phased-signal', 'frag-diversity',
                                          'CpG-Me-freq', 'CHG-ME-freq', 'CHH-Me-freq', 'CHN-Me-freq'])]
    else:
        data = data[data['profile'].isin(['depth', 'phased-signal', 'frag-diversity'])]

    if palette is not None:
        sea = sns.FacetGrid(data, row='profile', hue='label', despine=False, height=3, aspect=3, palette=palette,
                            sharey=False, legend_out=True)
    else:
        sea = sns.FacetGrid(data, row='profile', hue='label', despine=False, height=3, aspect=3, sharey=False,
                            legend_out=True)

    sea.map(sns.lineplot, 'loc', 'value', alpha=0.8, legend='full', n_boot=100)  # n_boot effects speed substantially

    sea.add_legend()
    if show_inflection and palette is not None:
        if plot_mode == 'all':
            ax = sea.axes.flatten()[2]
        elif plot_mode == 'signal':
            ax = sea.axes.flatten()[0]
        else:
            ax = sea.axes.flatten()[1]
        for subtype in data['label'].unique():
            sub_df = df_peaks[df_peaks['label'] == subtype]
            minus_locs = sub_df.loc[sub_df['value'] == -2, 'loc'].values
            plus_locs = sub_df.loc[sub_df['value'] == 2, 'loc'].values
            inflect_locs = sub_df.loc[sub_df['value'] == 3, 'loc'].values
            if minus_locs.any() and plus_locs.any() and inflect_locs.any():
                m_mean, m_std = np.mean(minus_locs), np.std(minus_locs)
                p_mean, p_std = np.mean(plus_locs), np.std(plus_locs)
                i_mean, i_std = np.mean(inflect_locs), np.std(inflect_locs)
                ax.axvspan(m_mean - m_std, m_mean + m_std, alpha=0.05, color=palette[subtype])
                ax.axvline(x=m_mean, alpha=0.3, color=palette[subtype], ls='-')
                ax.axvspan(p_mean - p_std, p_mean + p_std, alpha=0.05, color=palette[subtype])
                ax.axvline(x=p_mean, alpha=0.3, color=palette[subtype], ls='-')
                ax.axvspan(i_mean - i_std, i_mean + i_std, alpha=0.05, color=palette[subtype])
                ax.axvline(x=i_mean, alpha=0.3, color=palette[subtype])
    if plot_mode == 'TME':
        for ax_index in range(3, 7):
            ax = sea.axes.flatten()[ax_index]
            ax.set_ylim(0, 1)
    sea.set(xlim=(xmin, xmax))
    plt.savefig(name + '-Profiles_' + plot_mode + '.pdf', bbox_inches="tight")
    plt.close()
    return


def normalize_data(data):
    """
    Normalizes signal (re-scales from 0-1)
        Parameters:
            data (vector-like array): signal profile from Triton
        Returns:
            normalized_data
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return data - min_val  # non-signal: 0s
    else:
        return (data - min_val) / (max_val - min_val)


def standardize_data(data):
    """
    Standardizes signal, AKA transforms to Z-score (centers the signal and re-scales)
        Parameters:
            data (vector-like array): signal profile from Triton
        Returns:
            standardized_data
    """
    mean_val = np.mean(data)
    stdev_val = np.std(data)
    if stdev_val == 0:
        return data - mean_val  # non-signal: 0s
    else:
        return (data - mean_val) / stdev_val


def scale_data(data):
    """
    Scales signal mean to 1 (preserves potential differences in magnitude of signal variation)
        Parameters:
            data (vector-like array): signal profile from Triton
        Returns:
            scaled_data
    """
    mean_val = np.mean(data)
    if mean_val == 0:
        return data
    else:
        return data / mean_val


def smooth_data(data, window=15):
    """
    Performs scanning window mean smoothing to replicate binned outputs
        Parameters:
            data (vector-like array): signal profile from Triton
            window (int): window-size for smoothing
        Returns:
            smoothed_data
    """
    smoothed_data = np.zeros(len(data))
    for i in range(len(data)):
        if i < int(window / 2):
            smoothed_data[i] = np.mean(data[:(window - 1)])
        elif i > (len(data) - window):
            smoothed_data[i] = np.mean(data[-window:])
        else:
            smoothed_data[i] = np.mean(data[(i - window // 2):(i + window // 2)])
    return smoothed_data


# below is for a non-issue iterating through "files"
# noinspection PyUnresolvedReferences
def main():
    parser = argparse.ArgumentParser(description='\n### triton_plotters.py ### plots Triton output profiles')
    parser.add_argument('-i', '--input', help='one or more TritonProfiles.npz files; wildcards (e.g. '
                                              'results/*.npz) are OK.', nargs='*', required=True)
    parser.add_argument('-m', '--mode', help='plotting mode (all: all profiles, signal: only phased signal profile, '
                                             'RSD: raw depth, signal, and fragment diversity profiles, TME: ONLY iff '
                                             'TritonMe was used, plots RSD + Me frequencies). DEFAULT = RSD.',
                        required=False, default='RSD')
    parser.add_argument('-c', '--categories', help='tsv file containing matched sample names (column 1) and '
                                                   'categories (column 2) to color samples by composite '
                                                   'categories rather than individually, with a .95 confidence '
                                                   'interval; sample names in column 1 MUST match the sample '
                                                   'names passed to Triton (i.e. sample1_TritonProfiles.npz). '
                                                   'Any passed inputs not present in this file will be dropped. '
                                                   'This file should NOT contain a header. DEFAULT = None (plot '
                                                   'all samples individually).',
                        required=False, default=None)
    parser.add_argument('-p', '--palette', help='tsv file containing matched categories/samples (column 1) and HEX '
                                                'color codes (column 2, e.g. #0077BB) to specify what color to use '
                                                'for samples/categories. Sample/category names must exactly match '
                                                'sample names passed to Triton, or category names passed with the '
                                                '-c/--categories option. Will error if inputs/categories include '
                                                'labels not present in palette.This file should NOT contain a header. '
                                                'DEFAULT = None (use Seaborn default colors).',
                        required=False, default=None)
    parser.add_argument('-s', '--sites', help='file containing a list (row-wise) of sites to plot. This file should '
                                              'NOT contain a header. DEFAULT = None (plotting all available sites).',
                        required=False, default=None)
    parser.add_argument('-w', '--window', help='if set, data is "windowed" - sets 0-point to the middle instead of the　'
                                               '5\' end. DEFAULT = False.', action='store_true')
    parser.add_argument('-n', '--normalization', help='signal-normalization method to use at the site + sample level: '
                                                      '"raw" (raw values), "norm" (true normalization from 0-1), '
                                                      '"stand" (standardize, i.e. z-score), "scale" (scale mean to 1). '
                                                      '"scale" is recommended, and mirrors Triton feature-extraction. '
                                                      'DEFAULT = "scale"', required=False, default='scale')

    args = parser.parse_args()
    input_path = args.input
    plot_mode = args.mode
    categories = args.categories
    palette_file = args.palette
    sites_path = args.sites
    window = args.window
    norm = args.normalization

    print('### Running triton_plotters.py in "' + plot_mode + '" mode using "' + norm + '" signal scaling.')

    if categories is not None:
        categories = pd.read_table(categories, sep='\t', header=None)
        categories = dict(zip(categories[0], categories[1]))

    if sites_path is not None:
        with open(sites_path) as f:
            sites = f.read().splitlines()

    if palette_file is not None:
        palette_file = pd.read_table(palette_file, sep='\t', header=None)
        palette = dict(palette_file.itertuples(False, None))
    else:
        palette = None

    samples = [os.path.basename(path).split('_TritonProfiles.npz')[0] for path in input_path]
    tests_data = [np.load(path) for path in input_path]
    for site in tests_data[0].files:
        if sites_path is None or site in sites:
            if plot_mode == 'TME':
                columns = cols_me
            else:
                columns = cols
            dfs = []
            for sample, test_data in zip(samples, tests_data):
                if len(test_data[site].shape) == 2:
                    tdf = pd.DataFrame(test_data[site].T, columns=columns)
                    tdf['loc'] = np.arange(len(tdf))
                    if window:
                        tdf['loc'] = tdf['loc'] - len(tdf) / 2
                    tdf['sample'] = sample
                    if categories is not None:
                        if sample in categories.keys():
                            tdf['label'] = categories[sample]
                        else:
                            tdf['label'] = np.nan
                    else:
                        tdf['label'] = sample
                    if norm == 'raw':
                        continue
                    elif norm == 'norm':
                        for col in norm_cols:
                            tdf[col] = normalize_data(tdf[col])
                    elif norm == 'stand':
                        for col in norm_cols:
                            tdf[col] = standardize_data(tdf[col])
                    else:
                        for col in norm_cols:
                            tdf[col] = scale_data(tdf[col])
                    if plot_mode == 'TME':
                        for col in smooth_cols:
                            tdf[col] = smooth_data(tdf[col])
                    dfs.append(tdf)
            if len(dfs) > 1:
                df = pd.concat(dfs)
            else:
                df = dfs[0]
            print('- Plotting ' + site)
            if categories is not None:
                df = df[df['label'].notna()]
                if len(df['label']) < 2:
                    print('No samples to plot after matching against the provided categories file. Please ensure '
                          'provided labels are an exact match! Categories provided: ')
                    print(categories.keys)
                    print('Sample names provided: ')
                    print(samples)
                    print('Exiting.')
                    quit()
            if plot_mode == 'TME':
                df = pd.melt(df, id_vars=['sample', 'loc', 'label'], value_vars=cols_me, var_name='profile')
            else:
                df = pd.melt(df, id_vars=['sample', 'loc', 'label'], value_vars=cols, var_name='profile')
            plot_profiles(site, df, plot_mode, palette=palette)


if __name__ == "__main__":
    main()
