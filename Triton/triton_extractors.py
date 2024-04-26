# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v0.1.1, 04/04/2024

# This is a utility for extracting (modifiable) additional features from Triton signal output files.
# Use and modify at your own discretion!

"""
N.B. the numpy array ordering of profile objects:
1: Depth (GC-corrected)
2: Fragment end coverage
3: Phased-nucleosome profile (Fourier filtered probable nucleosome center profile)
4: Fragment lengths' short:long ratio (x <= 150 / x > 150)
5: Fragment lengths' diversity (unique fragment lengths / total fragments)
6: Fragment lengths' Shannon Entropy (normalized by window Shannon Entropy)
7: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***
8: A (Adenine) frequency**
9: C (Cytosine) frequency**
10: G (Guanine) frequency**
11: T (Tyrosine) frequency**
"""

import os
import argparse
import numpy as np
import pandas as pd

cols = ['depth', 'frag-ends', 'phased-signal', 'frag-mean', 'frag-stdev', 'frag-median',
        'frag-mad', 'frag-ratio', 'frag-diversity', 'frag-entropy',  'peaks']


def extract_features(data, site, features):
    """
    Extracts single-variable features from signal profiles.
        Parameters:
            data (pandas long df): signal profile(s) from Triton
            site (string): name of site
            features (list of strings): names of features to extract (much choose those implemented below)
    """
    # assume data is windowed, and take the half-way point to be 0:
    zero_loc = int((data['loc'].max() + 1) / 2)
    zero_range = list(range(zero_loc - 15, zero_loc + 15))
    dfs = []
    for sample in data['sample'].unique():
        df = data[data['sample'] == sample]
        feature_vals = []
        for feature in features:
            if feature == 'zero-depth':
                print(df[df['loc'].isin(zero_range)])
                zero_depth = np.nanmean(df[df['loc'].isin(zero_range)]['phased-signal'].values)
                feature_vals.append(zero_depth)
            elif feature == 'zero-heterogeneity':
                zero_het = np.nanmean(df[df['loc'].isin(zero_range)]['frag-hetero'].values)
                feature_vals.append(zero_het)
            elif feature == 'dip-width':
                if 2.0 in df['peaks'].values:
                    dip_width = df[df['peaks'] == 2.0]['loc'].values[0] - df[df['peaks'] == -2.0]['loc'].values[0]
                    feature_vals.append(dip_width)
                else:
                    feature_vals.append(np.nan)
            else:
                print('Feature "' + feature + '" has not been implemented; see code to implement. Skipping.')
        feature_dict = {'sample': [sample] * len(features),
                        'site': [site] * len(features),
                        'feature': features,
                        'value': feature_vals}
        dfs.append(pd.DataFrame(feature_dict))
    return pd.concat(dfs)


def normalize_data(data):
    mean_val = np.mean(data)
    if mean_val == 0:
        return data
    else:
        return data / mean_val


# below is for a non-issue iterating through "files"
# noinspection PyUnresolvedReferences
def main():
    parser = argparse.ArgumentParser(description='\n### triton_extractors.py ### extracts additional features')
    parser.add_argument('-i', '--input', help='one or more TritonProfiles.npz files; wildcards (e.g. '
                                              'results/*.npz) are OK.', nargs='*', required=True)
    parser.add_argument('-s', '--sites', help='file containing a list (row-wise) of sites to restrict to. This file'
                                              ' should NOT contain a header. DEFAULT = None (use all sites).',
                        required=False, default=None)

    args = parser.parse_args()
    input_path = args.input
    sites_path = args.sites
    
    new_feats = ['zero-depth', 'zero-heterogeneity', 'dip-width']
    # zero-depth: normalized phased profile depth at +/- 15bp of the midpoint (0, when windowed)
    # zero-heterogeneity: mean normalized heterogeneity profile depth at +/- 15bp of the midpoint (0, when windowed)
    # dip-width: distance between the minus-one and plus-one nucleosome positions

    print('### Running triton_exractors.py . . .')
    print('Features to extract:')
    print(new_feats)

    if sites_path is not None:
        with open(sites_path) as f:
            sites = f.read().splitlines()

    out_dfs = []

    # # TODO: something doesn't work when doing individual!
    # if len(input_path) == 1:  # individual sample
    #     test_data = np.load(input_path[0])
    #     sample = os.path.basename(input_path[0]).split('_TritonProfiles.npz')[0]
    #     for site in test_data.files:
    #         if sites_path is None or site in sites:
    #             df = pd.DataFrame(test_data[site], columns=cols)
    #             df['sample'] = sample
    #             df['loc'] = np.arange(len(df))
    #             for col in norm_cols:
    #                 df[col] = normalize_data(df[col])
    #             df = pd.melt(df, id_vars=['sample', 'loc'], value_vars=cols, var_name='profile')
    #             print('Extracting features for site: ' + site)
    #             out_dfs.append(extract_features(df, site, new_feats))
    # else:  # multiple samples:
    samples = [os.path.basename(path).split('_TritonProfiles.npz')[0] for path in input_path]
    tests_data = [np.load(path) for path in input_path]
    for site in tests_data[0].files:
        if sites_path is None or site in sites:
            dfs = [pd.DataFrame(test_data[site], columns=cols)
                   for test_data in tests_data if len(test_data[site].shape) == 2]
            for tdf, sample in zip(dfs, samples):
                tdf['loc'] = np.arange(len(tdf))
                tdf['sample'] = sample
                # for col in norm_cols:
                #     tdf[col] = normalize_data(tdf[col])
            if len(dfs) < 2:
                continue
            df = pd.concat(dfs)
            print('Extracting features for site: ' + site)
            out_dfs.append(extract_features(df, site, new_feats))

    print('Merging and saving results . . .')
    df_final = pd.concat(out_dfs).set_index('sample')
    out_file = 'TritonCompositeFM_EXTRACTED.tsv'
    df_final.to_csv(out_file, sep='\t')

    print('Finished')


if __name__ == "__main__":
    main()
