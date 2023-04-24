# Robert Patton, rpatton@fredhutch.org
# v2.0.1, 04/24/2023

import numpy as np
import pandas as pd

mapping = dict(zip("NACGT", range(5)))  # sequence mapping for one-hot encoding


def one_hot_encode(seq):
    """
    One-hot encode a nucleotide sequence (as a binary numpy array)
        Parameters:
            seq (string): string of nucleotides to one-hot encode
        Returns:
            numpy array: one-hot encoded nucleotide sequence of size 5xN
    """
    seq2 = [mapping[nt] for nt in seq]
    return np.eye(5)[seq2]


def get_gc_bias_dict(bias_path):
    """
    (Modified from Griffin pipeline, credit Anna-Lisa Doebley)
    One-hot encode a nucleotide sequence (as a binary numpy array)
        Parameters:
            bias_path (string): path to GC bias output by Griffin
        Returns:
            dict: GC bias in dictionary form (returns bias given [length][gc content]
    """
    if bias_path is not None:
        bias = pd.read_csv(bias_path, sep='\t')
        bias['smoothed_GC_bias'] = np.where(bias['smoothed_GC_bias'] < 0.05, np.nan, bias['smoothed_GC_bias'])
        bias = bias[['length', 'num_GC', 'smoothed_GC_bias']]
        bias = bias.set_index(['num_GC', 'length']).unstack()
        bias = bias.to_dict()
        bias2 = {}
        for key in bias.keys():
            length = key[1]
            bias2[length] = {}
            for num_GC in range(0, length + 1):
                temp_bias = bias[key][num_GC]
                bias2[length][num_GC] = temp_bias
        bias = bias2
        del bias2
        return bias
    else:
        return None


def frag_metrics(frag_lengths, bins):
    """
    Returns the mean, standard deviation, median absolute deviation, and ratio of short (f <= 150) to long (151 <= f)*
    fragment lengths given an input of fragment length counts (see below).
    * Used to be short (f <= 120) to long (140 <= f <= 250)
        Parameters:
            frag_lengths (list/array): 1D array of fragment length counts, where the index is the fragment length
            bins (list): list of bin boundaries
        Returns:
            mean, stdev, MAD, ratio (float, float, int, float)
    """
    total_count = np.sum(frag_lengths)
    if total_count < 10:  # fewer than 10 overlapping reads - too noisy to be of use.
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    mean = sum([idx * val for idx, val in enumerate(frag_lengths)]) / total_count
    stdev = np.sqrt(sum([val * (idx - mean)**2 for idx, val in enumerate(frag_lengths)]) / total_count)
    median = 0
    while np.sum(frag_lengths[0:(median + 1)]) < (total_count / 2):
        median += 1
    ads = np.zeros(len(frag_lengths))  # ordered absolute deviation counts
    for idx, val in enumerate(frag_lengths):
        ads[np.abs(idx - median)] += val
    mad = 0
    while np.sum(ads[0:(mad + 1)]) < (total_count / 2):
        mad += 1
    ratio = np.sum(frag_lengths[:151]) / np.sum(frag_lengths[151:])
    diversity = np.count_nonzero(frag_lengths) / total_count
    entropy = shannon_entropy(frag_lengths, bins)
    return mean, stdev, median, mad, ratio, diversity, entropy


def shannon_entropy(counts, bins):
    """
    Returns the Shannon Entropy of a set of values; will bin with "bins" bins
        Parameters:
            counts (list/array): 1D array of fragment length counts, where the index is the fragment length
            bins (list): list of bin boundaries
        Returns:
            float: Shannon Entropy
    """
    histogram = [np.sum(counts[val:(val + 5)]) for val in bins]
    total_counts = sum(histogram)
    if total_counts < 2:
        return 0
    pdf = histogram / sum(histogram)
    pdf = [p for p in pdf if p != 0.0]
    # moments = [p * np.log(p) for p in pdf]  # un-normalized
    moments = [p * np.log(p) / np.log(total_counts) for p in pdf]  # normalized
    return - sum(moments)


def local_peaks(ys, xs):
    """
    Finds LOCAL (min or max relative to neighbors) given a 1D list/array; only appropriate for very smooth data
    (E.G. an inverse fourier transform with high frequencies removed)
        Parameters:
            ys (list/array): signal height
            xs (list/array): raw height (for insuring non-zero maxima)
        Returns:
            max_values: list of maxima values
            max_indices: list of corresponding maxima indices
            min_values: list of minima values
            min_indices: list of corresponding minima indices
    """
    max_values, max_indices = [], []
    min_values, min_indices = [], []
    ysl = len(ys) - 1
    for i, y in enumerate(ys):
        if 0 < i < ysl and ys[i - 1] < y and y > ys[i + 1] and xs[i] > 0:  # local maxima
            max_values.append(y)
            max_indices.append(i)
        if 0 < i < ysl and ys[i - 1] > y and y < ys[i + 1]:  # local minima
            min_values.append(y)
            min_indices.append(i)
    return max_values, max_indices, min_values, min_indices


def nearest_peaks(ref_point, ref_list):
    """
    Finds the nearest upstream/downstream peak to a given index/point
        Parameters:
            ref_point (int): point of interest
            ref_list (list/array): list of peak indices
        Returns:
            left_index: index of nearest upstream peak
            right_index: index of nearest downstream peak
    """
    distances = [ref_point - peak for peak in ref_list]
    if len([i for i in distances if i > 0]) > 0:
        left_index = ref_point - min([i for i in distances if i > 0])
    else:
        left_index = np.nan
    if len([abs(i) for i in distances if i < 0]) > 0:
        right_index = ref_point + min([abs(i) for i in distances if i < 0])
    else:
        right_index = np.nan
    return left_index, right_index


# def frag_ratio(frag_lengths):
#     """
#     Returns the ratio of short (f <= 150) to long (151 <= f) fragment lengths in a list/array
#     * Used to be short (f <= 120) to long (140 <= f <= 250)
#         Parameters:
#             frag_lengths (list/array): series of fragment lengths
#         Returns:
#             float: the short/long ratio
#     """
#     short_frags = len([x for x in frag_lengths if x <= 120])
#     long_frags = len([x for x in frag_lengths if 140 <= x <= 250])
#     if short_frags > 0 and long_frags > 0:
#         ratio = short_frags / long_frags
#         return ratio
#     else:
#         return np.nan
#
#
# def dirichlet_normalized_entropy(counts, ref_counts):
#     """
#     Given two sets of fragment lengths in a region, return the Shannon entropy normalized by the expected Shannon
#     Entropy for a given set of fragment lengths (ref_lengths).
#     N.B., if lengths==ref_lengths, normalization still occurs, using a hypothetical, Dirichlet-derived distribution
#         Parameters:
#             counts (list/array): 1D array of fragment length counts, where the index is the fragment length
#             ref_counts (list/array): same as above, but to use as a standard reference
#         Returns:
#             float: normalized entropy
#     """
#     bin_range = list(range(15, 500, 5))  # hardcoded for dict-based analyses of range 15-500
#     total_frags = np.sum(counts)
#     ref_hist = [np.sum(ref_counts[val:(val + 5)]) for val in bin_range]
#     ref_hist = [h for h in ref_hist if h != 0.0]
#     dist_pdfs = np.random.dirichlet(ref_hist, 1000)
#     norm_factor = mean_entropy(dist_pdfs, total_frags, bin_range)
#     if norm_factor > 0:
#         return shannon_entropy(counts, bin_range) / norm_factor
#     else:
#         return np.nan
#
#
# def mean_entropy(pdf_matrix, counts, bins):
#     """
#     Finds the mean (expected) entropy given probability distributions, for a number of counts and set of bins
#         Parameters:
#             pdf_matrix (ndarray): drawn PDF samples of shape (size, #)
#             counts (int): number of multinomial draws to perform
#             bins (list): list of bin boundaries
#         Returns:
#             float: expected Shannon Entropy
#     """
#     entropies = []
#     for row in pdf_matrix:
#         draws = np.random.multinomial(counts, row)
#         entropies.append(shannon_entropy(draws, bins, binned=True))
#     return np.mean(entropies)
#
#
# def point_entropy(ordered_counts, bins, background_entropy):
#     """
#     Finds the Shannon Entropy (1bp bins) at every point, scaled by some expected background
#         Parameters:
#             ordered_counts (2D array): 2D array where columns represent loci, and the rows represent fragment length
#                 counts of length row_index
#             bins (list): list of bin boundaries
#             background_entropy (float): normalization constant
#         Returns:
#             1D array: Shannon Entropies at each point (corresponds to ordering in ordered_counts)
#     """
#     point_shannon_entropies = np.apply_along_axis(shannon_entropy, axis=0, arr=ordered_counts, bins=bins)
#     return point_shannon_entropies / background_entropy


