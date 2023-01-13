# Robert Patton, rpatton@fredhutch.org
# v1.1.0, 1/11/2023

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


def frag_ratio(frag_lengths):
    """
    Returns the ratio of short (f <= 120) to long (140 <= f <= 250) fragment lengths in a list/array
        Parameters:
            frag_lengths (list/array): series ofr fragment lengths
        Returns:
            float: the short/long ratio
    """
    short_frags = len([x for x in frag_lengths if x <= 120])
    long_frags = len([x for x in frag_lengths if 140 <= x <= 250])
    if short_frags > 0 and long_frags > 0:
        ratio = short_frags / long_frags
        return ratio
    else:
        return np.nan


def shannon_entropy(values, bins, binned=False):
    """
    Returns the Shannon Entropy of a set of values; will bin with "bins" bins if a histogram is not passed
        Parameters:
            values (list/array): series of values or histogram values
            bins (list): list of bin boundaries
            binned (boolean): whether the values have been turned into a histogram already
        Returns:
            float: Shannon Entropy
    """
    if not binned:
        histogram = np.histogram(values, bins=bins)[0]
    else:
        histogram = values
    if sum(histogram) < 2:
        return 0
    pdf = histogram / sum(histogram)
    pdf = [p for p in pdf if p != 0.0]
    moments = [p * np.log(p) for p in pdf]
    return - sum(moments)


def mean_entropy(pdf_matrix, counts, bins):
    """
    Finds the mean (expected) entropy given probability distributions, for a number of counts and set of bins
        Parameters:
            pdf_matrix (ndarray): drawn PDF samples of shape (size, #)
            counts (int): number of multinomial draws to perform
            bins (list): list of bin boundaries
        Returns:
            float: expected Shannon Entropy
    """
    entropies = []
    for row in pdf_matrix:
        draws = np.random.multinomial(counts, row)
        entropies.append(shannon_entropy(draws, bins, binned=True))
    return np.mean(entropies)


def point_entropy(total_lengths, point_lengths):
    """
    Using all available fragments to inform min/max bin boundaries, find the Shannon Entropy (1bp bins) at every point
        Parameters:
            total_lengths (list): all lengths represented in a region
            point_lengths (list of lists): bp-wise list of lists containing all fragment lengths overlapping that point
        Returns:
            list: Shannon Entropies at each point (corresponds to ordering in point_lengths)
    * commented out portions are for region-level Dirichlet normalization - has little effect so dropped
    """
    min_size, max_size = min(total_lengths), max(total_lengths)
    bin_range = list(range(min_size, max_size, 1))
    # hist = np.histogram(total_lengths, bins=bin_range)[0]
    # hist = [h for h in hist if h != 0.0]
    # dist_pdfs = np.random.dirichlet(hist, 1000)
    # frag_nums = list(set([len(frags) for frags in point_lengths]))
    # norm_dict = {i: mean_entropy(dist_pdfs, i, bin_range) for i in frag_nums}
    point_shannon_entropies = [shannon_entropy(frags, bin_range) for frags in point_lengths]
    # point_norm_entropies = [(val / norm_dict[len(frags)]) if len(frags) > 2 else 0 for val, frags
    #                         in zip(point_shannon_entropies, point_lengths)]
    # return point_shannon_entropies, point_norm_entropies
    return point_shannon_entropies


def local_peaks(ys, xs):
    """
    Finds LOCAL (min or max relative to neighbors) given a 1D list/array; only appropriate for very smooth data
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


def dirichlet_normalized_entropy(lengths, ref_lengths):
    """
    Given two sets of fragment lengths in a region, return the Shannon entropy normalized by the expected Shannon
    Entropy for a given set of fragment lengths (ref_lengths).
    N.B., if lengths==ref_lengths, normalization still occurs, using a hypothetical, Dirichlet-derived distribution
        Parameters:
            lengths (list/array): list of fragment lengths to find entropy of
            ref_lengths (list/array): list of fragment lengths to use as a standard entropy reference
        Returns:
            float: normalized entropy
    """
    min_size, max_size, total_frags = min(lengths + ref_lengths), max(lengths + ref_lengths), len(lengths)
    bin_range = list(range(min_size, max_size, 1))
    hist = np.histogram(ref_lengths, bins=bin_range)[0]
    hist = [h for h in hist if h != 0.0]
    dist_pdfs = np.random.dirichlet(hist, 1000)
    norm_factor = mean_entropy(dist_pdfs, total_frags, bin_range)
    if norm_factor > 0:
        return shannon_entropy(lengths, bin_range) / norm_factor
    else:
        return np.nan
