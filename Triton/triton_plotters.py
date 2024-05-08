# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v0.3.1, 04/04/2024

# Utility for plotting Triton profile outputs
# N.B. Peak locations and nucleotide-frequency plotting is not supported
# (but feel free to modify code to include them)

"""
N.B. the numpy array ordering of profile objects:
1: Depth (GC-corrected)
2: Fragment end coverage
3: Phased-nucleosome profile (Fourier filtered probable nucleosome center profile, GC-corrected)
4: Fragment lengths' short:long ratio (x <= 150 / x > 150)
5: Fragment lengths' diversity (unique fragment lengths / total fragments)
6: Fragment lengths' Shannon Entropy
7: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***
8: A (Adenine) frequency**
9: C (Cytosine) frequency**
10: G (Guanine) frequency**
11: T (Tyrosine) frequency**
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Suppress the seaborn UserWarning about tight layout
warnings.filterwarnings('ignore', 'The figure layout has changed to tight', UserWarning)

# pretty column names
cols = ['depth', 'frag-ends', 'phased-signal', 'frag-ratio', 'frag-diversity', 'frag-entropy',
        'peaks', 'a-freq', 'c-freq', 'g-freq', 't-freq']
pretty_cols = ['Depth (GC-Corrected)', 'Fragment End Coverage', 'Phased Nucleosomal Signal', 'Fragment Short:Long Ratio',
                'Fragment Diversity Index', 'Fragment Shannon Entropy', 'Peaks', 'A Frequency', 'C Frequency', 'G Frequency', 'T Frequency']
col_map = {col: pretty_cols[i] for i, col in enumerate(cols)}
profile_labels = {'depth': 'depth',
                  'frag-ends': 'depth',
                  'phased-signal': 'depth',
                  'frag-ratio': 'ratio',
                  'frag-diversity': 'diversity',
                  'frag-entropy': 'entropy',}

# signals to normalize
norm_cols = ['depth', 'frag-ends', 'phased-signal', 'frag-ratio', 'frag-diversity', 'frag-entropy']


def normalize_data(data, region=False):
    """
    Normalizes signal (re-scales from 0-1)
        Parameters:
            data (vector-like array): signal profile from Triton
            region (bool): if False, central 500 values are ignored in normalization calculation
        Returns:
            normalized_data
    """
    if not region:
        half = len(data) // 2
        min_val = np.min(np.concatenate((data[:half-250], data[half+250:])))
        max_val = np.max(np.concatenate((data[:half-250], data[half+250:])))
    else:
        min_val = np.min(data)
        max_val = np.max(data)

    if max_val - min_val == 0:
        return data - min_val  # non-signal: 0s
    else:
        return (data - min_val) / (max_val - min_val)


def standardize_data(data, region=False):
    """
    Standardizes signal, AKA transforms to Z-score (centers the signal and re-scales)
        Parameters:
            data (vector-like array): signal profile from Triton
            region (bool): if False, central 500 values are ignored in normalization calculation
        Returns:
            standardized_data
    """
    if not region:
        half = len(data) // 2
        mean_val = np.mean(np.concatenate((data[:half-250], data[half+250:])))
        stdev_val = np.std(np.concatenate((data[:half-250], data[half+250:])))
    else:
        mean_val = np.mean(data)
        stdev_val = np.std(data)

    if stdev_val == 0:
        return data - mean_val  # non-signal: 0s
    else:
        return (data - mean_val) / stdev_val


def scale_data(data, region=False):
    """
    Scales signal mean to 1 (preserves potential differences in magnitude of signal variation)
        Parameters:
            data (vector-like array): signal profile from Triton
            region (bool): if False, central 500 values are ignored in normalization calculation
        Returns:
            scaled_data
    """
    if not region:
        half = len(data) // 2
        mean_val = np.mean(np.concatenate((data[:half-250], data[half+250:])))
    else:
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


def plot_profiles(name, data, plot_mode, norm_method, region, palette=None):
    """
    Plots all profiles specified with plot_mode in one figure.
        Parameters:
            name (string): name of site being plotted
            data (pandas long df): signal profile(s) from Triton
            plot_mode (string): name of plotting mode (signal, all, RSD, TME - see main() for details)
            norm_method (string): signal normalization method (for labels)
            region (bool): whether to plot region-based profiles (0 at left side) or not (0 in middle)
            show_inflections (bool): whether to plot vertical lines showing the inflection and +/-1 nucleosome positions
            palette (target:color dictionary): palette for subtypes if categories is passed
    """
    sns.set_context("notebook", font_scale=1.3)

    xmin, xmax = data['loc'].min(), data['loc'].max() + 1

    plot_mode_profiles = {'all': ['depth', 'frag-ends', 'phased-signal', 'frag-ratio', 'frag-diversity', 'frag-entropy'],
                          'signal': ['phased-signal'],
                          'default': ['depth', 'phased-signal', 'frag-diversity']}

    profiles = plot_mode_profiles.get(plot_mode, plot_mode_profiles['default'])
    data = data.copy()
    data = data.loc[data['profile'].isin(profiles)]
    data.loc[:, 'profile'] = data['profile'].map(col_map)
    if region:
        loc_name = 'location relative to start (bp)'
    else:
        loc_name = 'location relative to center (bp)'
    data = data.rename(columns={'profile': 'Profile', 'label': 'Sample / Group', 'loc': loc_name})

    sea = sns.FacetGrid(data, row='Profile', hue='Sample / Group', despine=False, height=3, aspect=3, palette=palette, sharey=False, sharex=True, legend_out=True)
    sea.map(sns.lineplot, loc_name, 'value', alpha=0.8, legend='full', n_boot=100)

    # Add smoothed 'frag-ends' profile if plotted
    if 'frag-ends' in profiles:
        frag_ends_data = data[data['Profile'] == 'Fragment End Coverage'].copy()  # Create a copy to avoid SettingWithCopyWarning
        frag_ends_data.loc[:, 'value'] = smooth_data(frag_ends_data['value'])  # Smooth data
        frag_ends_subplot = sea.axes[profiles.index('frag-ends'), 0]  # Get 'frag-ends' subplot
        sns.lineplot(x=loc_name, y='value', data=frag_ends_data, color='red', ax=frag_ends_subplot, legend=False)

    x_ticks = np.linspace(xmin, xmax, num=5)
    for ax in sea.axes.flat:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks.astype(int))

    prefix = {'raw': 'raw ', 'norm': 'normalized ', 'stand': '', 'default': 'scaled '}
    y_labels = {key: prefix.get(norm_method, prefix['default']) + value for key, value in profile_labels.items()}
    if norm_method == 'stand':
        y_labels = {key: value + ' Z-score' for key, value in y_labels.items()}

    for ax, profile in zip(sea.axes.flat, profiles):
        ax.set_ylabel(y_labels[profile], labelpad=10)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    sea.set_titles("{row_name}", pad=10)
    sea.set(xlim=(xmin, xmax))
    sea.add_legend()
    plt.savefig(name + '-Profiles_' + plot_mode + '.pdf')
    plt.close()
    return


def main():
    parser = argparse.ArgumentParser(description='\n### triton_plotters.py ### plots Triton output profiles')
    parser.add_argument('-i', '--input', help='one or more TritonProfiles.npz files; wildcards (e.g. '
                                              'results/*.npz) are OK.', nargs='*', required=True)
    parser.add_argument('-m', '--mode', help='plotting mode (all: all profiles, signal: only phased signal profile, '
                                             'RSD: raw depth, signal, and fragment diversity profiles. DEFAULT = RSD.',
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
                                                'labels not present in palette. This file should NOT contain a header. '
                                                'DEFAULT = None (use Seaborn default colors).',
                        required=False, default=None)
    parser.add_argument('-s', '--sites', help='file containing a list (row-wise) of sites to plot. This file should '
                                              'NOT contain a header. DEFAULT = None (plotting all available sites).',
                        required=False, default=None)
    parser.add_argument('-r', '--region_axis', help='if set, data is "region" - sets 0-point to the left side instead of the middle. '
                                               'DEFAULT = False.', action='store_true')
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
    region = args.region_axis
    norm = args.normalization

    print(f'### Running triton_plotters.py in "{plot_mode}" mode using "{norm}" signal scaling.')

    categories = pd.read_table(categories, sep='\t', header=None).set_index(0).to_dict()[1] if categories else None
    sites = open(sites_path).read().splitlines() if sites_path else None
    palette = pd.read_table(palette_file, sep='\t', header=None).set_index(0).to_dict()[1] if palette_file else None

    samples = [os.path.basename(path).split('_TritonProfiles.npz')[0] for path in input_path]
    tests_data = [np.load(path) for path in input_path]

    for site in tests_data[0].files:
        if sites_path is None or site in sites:
            columns = cols
            dfs = []
            print(f'*** Processing samples for {site}')
            for sample, test_data in zip(samples, tests_data):
                if categories is not None and sample not in categories:
                    print(f'Sample {sample} not found in categories. Skipping.')
                    continue
                if pd.isnull(test_data[site]).all():
                    print(f'No data for {site} for sample {sample}. Skipping.')
                    continue
                if len(test_data[site].shape) == 2:
                    tdf = pd.DataFrame(test_data[site].T, columns=columns)
                    tdf['loc'] = np.arange(len(tdf))
                    if not region:
                        tdf['loc'] -= len(tdf) / 2
                    tdf['sample'] = sample
                    tdf['label'] = categories.get(sample, sample) if categories else sample
                    if norm != 'raw':
                        for col in norm_cols:
                            if norm == 'norm':
                                tdf[col] = normalize_data(tdf[col], region=region)
                            elif norm == 'stand':
                                tdf[col] = standardize_data(tdf[col], region=region)
                            else:
                                tdf[col] = scale_data(tdf[col], region=region)
                    dfs.append(tdf)
            if not dfs:
                print(f'No data for {site}. Skipping.')
                continue
            df = pd.concat(dfs) if len(dfs) > 1 else dfs[0]
            print(f'*** Plotting {site}')
            if categories:
                df = df[df['label'].notna()]
                if len(df['label']) < 2:
                    print(f'No samples to plot after matching against the provided categories file. Please ensure '
                        f'provided labels are an exact match! Categories provided: {categories.keys()}')
                    print(f'Sample names provided: {samples}')
                    print('Exiting.')
                    quit()
            df = pd.melt(df, id_vars=['sample', 'loc', 'label'], value_vars=cols, var_name='profile')
            plot_profiles(site, df, plot_mode, norm, region, palette=palette)

if __name__ == "__main__":
    main()
