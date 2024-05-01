# Robert Patton, rpatton@fredhutch.org
# v1.0.0, 03/06/2024

import numpy as np
import pandas as pd

oh_mapping = {chr(i): 0 for i in range(65, 91)}  # all uppercase letters are 0 except for ACGT
oh_mapping.update(dict(zip("ACGT", range(1, 5))))  # sequence mapping for one-hot encoding nucleotides


def str2int(v):
    """
    Stand-in type to read string/int values from the command line
        Parameters:
            v (string): input int/string
        Returns:
            int or None
    """
    return int(v) if v.isdigit() else None


def get_index(header, names, default, message):
    for name in names:
        if name in header:
            return header.index(name)
    if default is not None:
        print(message.format(default, header[default]))
    else:
        print(message.format(default, 'N/A'))
    return default


def one_hot_encode(seq):
    """
    One-hot encode a nucleotide sequence (as a binary numpy array)
        Parameters:
            seq (string): string of nucleotides to one-hot encode
        Returns:
            numpy array: one-hot encoded nucleotide sequence of size 5xN
    """
    oh_seq = [oh_mapping[nt] for nt in seq]
    return np.eye(5, dtype=int)[oh_seq]


def get_gc_bias_dict(bias_path):
    """
    Modified from Griffin pipeline, credit: Anna-Lisa Doebley
    https://github.com/adoebley/Griffin
    https://doi.org/10.1038/s41467-022-35076-w
    Reads in a GC bias file and returns a dictionary of GC bias values
        Parameters:
            bias_path (string): path to GC bias output by Griffin or in Griffin format
        Returns:
            dict: GC bias in dictionary form (returns bias given [length][gc content]
    """
    bias = pd.read_csv(bias_path, sep='\t')
    bias['smoothed_GC_bias'] = bias['smoothed_GC_bias'].where(bias['smoothed_GC_bias'] >= 0.05, np.nan)
    bias = bias[['length', 'num_GC', 'smoothed_GC_bias']]
    bias = bias.set_index(['num_GC', 'length']).unstack().to_dict()
    bias_dict = {key[1]: {num_GC: bias[key][num_GC] for num_GC in range(0, key[1] + 1)} for key in bias.keys()}
    return bias_dict


def frag_metrics(frag_lengths, bins, reduce=False):
    """
    Returns the mean, standard deviation, median, median absolute deviation, ratio of short (f <= 150)
    to long (151 <= f)*, diversity, and Shannon entropy of fragment lengths given an input of fragment length counts.
    * Used to be short (f <= 120) to long (140 <= f <= 250)
        Parameters:
            frag_lengths (np.array): 1D array of fragment length counts, where the index is the fragment length
            bins (np.array): list of bin boundaries
            reduce (bool): if true, only output ratio, diversity, and entropy (save time on signal profiles)
        Returns:
            mean, stdev, MAD, ratio (float, float, int, float)
    N.B. since a histogram-array is being passed, I am doing some "fancy math" to save time and space
    """
    total_count = np.sum(frag_lengths)
    unique_count = np.count_nonzero(frag_lengths)
    frag_lengths_long = np.sum(frag_lengths[151:])
    
    if total_count < 2 or unique_count < 2 or frag_lengths_long == 0:  # BARE MINIMUM to get real metrics
        return (0.0,) * (3 if reduce else 7)

    ratio = np.sum(frag_lengths[:151]) / frag_lengths_long
    diversity = unique_count / total_count
    entropy = shannon_entropy(frag_lengths, bins)

    if reduce:
        return ratio, diversity, entropy

    mean = np.average(np.arange(len(frag_lengths)), weights=frag_lengths)
    stdev = np.sqrt(np.average((np.arange(len(frag_lengths)) - mean)**2, weights=frag_lengths))
    median = np.searchsorted(np.cumsum(frag_lengths), total_count / 2)
    mad = np.searchsorted(np.cumsum(np.bincount(np.abs(np.arange(len(frag_lengths)) - median), minlength=len(frag_lengths))), total_count / 2)

    return mean, stdev, median, mad, ratio, diversity, entropy


def shannon_entropy(counts, bins):
    """
    Returns the Shannon Entropy of a set of values; will bin with "bins" bins
        Parameters:
            counts (list/array): 1D array of fragment length counts, where the index is the fragment length
            bins (np.array): list of bin boundaries
        Returns:
            float: Shannon Entropy
    """
    # Calculate histogram using numpy's advanced indexing and binning
    histogram = np.add.reduceat(counts, bins[:-1])
    total_counts = histogram.sum()

    if total_counts < 2:
        return 0

    pdist = histogram / total_counts
    nonzero_pdist = pdist[pdist != 0.0]
    moments = nonzero_pdist * np.log(nonzero_pdist) / np.log(total_counts)

    return -moments.sum()


def local_peaks(ys, xs):
    """
    Finds LOCAL (min or max relative to neighbors) peaks given a 1D array; only appropriate for very smooth data
    (E.G. an inverse fourier transform with high frequencies removed)
        Parameters:
            ys (np.array): signal height
            xs (np.array): raw height (for insuring non-zero maxima)
        Returns:
            max_values: np.array of maxima values
            max_indices: np.array of corresponding maxima indices
            min_values: np.array of minima values
            min_indices: np.array of corresponding minima indices
    """
    # Calculate differences between consecutive elements
    diff = np.diff(ys)
    # Find indices where the difference changes sign
    sign_changes = np.where(np.diff(np.sign(diff)))[0] + 1
    # Separate into maxima and minima
    max_indices = sign_changes[np.where(diff[sign_changes - 1] > 0)]
    min_indices = sign_changes[np.where(diff[sign_changes - 1] < 0)]
    # Filter maxima to ensure non-zero maxima
    max_indices = max_indices[xs[max_indices] > 0]

    return ys[max_indices], max_indices, ys[min_indices], min_indices


def nearest_peaks(ref_point, ref_list):
    """
    Finds the nearest upstream/downstream peak to a given index/point
        Parameters:
            ref_point (int): point of interest
            ref_list (np.array): list of peak indices
        Returns:
            left_index: index of nearest upstream peak
            right_index: index of nearest downstream peak
    """
    distances = ref_point - ref_list
    positive_distances = distances[distances > 0]
    left_index = ref_point - positive_distances.min() if positive_distances.size else np.nan
    negative_distances = np.abs(distances[distances < 0])
    right_index = ref_point + negative_distances.min() if negative_distances.size else np.nan

    return left_index, right_index

def subtract_background(frag_lengths, frag_lengths_prof, frag_ends_profile, depth, site_dict, tfx):
    """
    Subtracts background from a site's fragment length profile
        Parameters:
            frag_lengths (np.array): 1D array of fragment length counts, where the index is the fragment length
            frag_length_prof (np.array): 2D array of shape (501, window) where each column represents fragment_lengths at that position
            frag_end_profile (np.array): 1D array of fragment end counts
            depth (np.array): 1D array of read counts
            site_dict (dict): dictionary of site-specific background values
                Contained are 'fragment_lengths', 'fragment_length_profile', 'fragment_end_profile', and 'depth' which have been
                normalized to sum to 1 (lengths, histogram-wise) or to have mean of 1 (depth)
            tfx (float): TFX value for site (sample)
        Returns:
            frag_lengths: np.array of fragment length counts
            frag_length_prof: np.array of fragment length profile
            frag_end_profile: np.array of fragment end counts
            depth: np.array of read counts
    """
    bg_purity = (1 - tfx)
    # Get background values for site
    frag_lengths_bg = site_dict['fragment_lengths']
    frag_lengths_prof_bg = site_dict['fragment_length_profile']
    frag_ends_profile_bg = site_dict['fragment_end_profile']
    depth_bg = site_dict['depth']

    # Subtract background from fragment length counts
    frag_lengths_counts = np.sum(frag_lengths)
    frag_lengths_sub = frag_lengths - bg_purity * frag_lengths_counts * frag_lengths_bg
    frag_lengths_sub[frag_lengths_sub < 0] = 0
    frag_lengths_sub = np.round(frag_lengths_sub)

    if frag_lengths_prof_bg is not None:
        # Subtract background from fragment length profile
        frag_lengths_prof_counts = np.sum(frag_lengths_prof, axis=0)
        frag_lengths_prof_sub = frag_lengths_prof - bg_purity * frag_lengths_prof_counts * frag_lengths_prof_bg
        frag_lengths_prof_sub[frag_lengths_prof_sub < 0] = 0
        frag_lengths_prof_sub = np.round(frag_lengths_prof_sub)
    else:
        # Skip, as this was run in region mode
        frag_lengths_prof_sub = frag_lengths_prof

    if frag_ends_profile_bg is not None:
        # Subtract background from fragment end profile
        frag_ends_profile_counts = np.sum(frag_ends_profile)
        frag_ends_profile_sub = frag_ends_profile - bg_purity * frag_ends_profile_counts * frag_ends_profile_bg
        frag_ends_profile_sub[frag_ends_profile_sub < 0] = 0
        frag_ends_profile_sub = np.round(frag_ends_profile_sub)
    else:
        # Skip, as this was run in region mode
        frag_ends_profile_sub = frag_ends_profile

    # Subtract background from depth
    mean_depth = np.mean(depth[500:-500])
    depth_sub = depth - bg_purity * mean_depth * depth_bg
    depth_sub[depth_sub < 0] = 0

    return frag_lengths_sub, frag_lengths_prof_sub, frag_ends_profile_sub, depth_sub

