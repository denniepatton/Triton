# Robert Patton, rpatton@fredhutch.org
# v1.0.0, 11/15/2022

# helper functions called by Triton.py

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


def frag_ratio(frag_lengths):  # compute the ratio of short to long fragments
    short_frags = len([x for x in frag_lengths if x <= 120])
    long_frags = len([x for x in frag_lengths if 140 <= x <= 250])
    if short_frags > 0 and long_frags > 0:
        ratio = short_frags / long_frags
        return ratio
    else:
        return 0


def shannon_entropy(values, bins, binned=False):
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
    entropies = []
    for row in pdf_matrix:
        draws = np.random.multinomial(counts, row)
        entropies.append(shannon_entropy(draws, bins, binned=True))
    return np.mean(entropies)


def point_entropy(total_lengths, point_lengths):
    min_size, max_size = min(total_lengths), max(total_lengths)
    bin_range = list(range(min_size, max_size, 1))
    hist = np.histogram(total_lengths, bins=bin_range)[0]
    hist = [h for h in hist if h != 0.0]
    dist_pdfs = np.random.dirichlet(hist, 1000)
    frag_nums = list(set([len(frags) for frags in point_lengths]))
    norm_dict = {i: mean_entropy(dist_pdfs, i, bin_range) for i in frag_nums}
    point_shannon_entropies = [shannon_entropy(frags, bin_range) for frags in point_lengths]
    point_norm_entropies = [(val / norm_dict[len(frags)]) if len(frags) > 2 else 0 for val, frags
                            in zip(point_shannon_entropies, point_lengths)]
    return point_shannon_entropies, point_norm_entropies


def normalize_data(data):
    if np.max(data) - np.min(data) == 0:
        return data
    else:
        return(data - np.min(data)) / (np.max(data) - np.min(data))


def local_peaks(ys, xs):
    max_values, max_indices = [], []
    ysl = len(ys) - 1
    for i, y in enumerate(ys):
        if 0 < i < ysl and ys[i - 1] <= y and y > ys[i + 1] and xs[i] > 0:  # local maxima
            max_values.append(y)
            max_indices.append(i)
    return max_values, max_indices


def nearest_peaks(ref_point, ref_list):
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


def nearest_troughs(ref_point, signal):
    left_index = ref_point - 1
    while signal[left_index] < signal[left_index + 1]:
        left_index -= 1
    right_index = ref_point + 1
    while signal[right_index] < signal[right_index - 1]:
        right_index += 1
    return left_index, right_index


def dirichlet_normalized_entropy(lengths, ref_lengths):
    min_size, max_size, total_frags = min(lengths + ref_lengths), max(lengths + ref_lengths), len(lengths)
    bin_range = list(range(min_size, max_size, 1))
    hist = np.histogram(ref_lengths, bins=bin_range)[0]
    hist = [h for h in hist if h != 0.0]
    dist_pdfs = np.random.dirichlet(hist, 1000)
    norm_factor = mean_entropy(dist_pdfs, total_frags, bin_range)
    return shannon_entropy(lengths, bin_range) / norm_factor
