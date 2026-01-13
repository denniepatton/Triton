# Robert Patton, rpatton@fredhutch.org
# v2.0.1, 10/14/2025

import numpy as np
import pandas as pd

# Pre-compute a mapping array for faster lookup
mapping_array = np.zeros(91, dtype=np.uint8)  # ASCII 'Z' is 90
mapping_array[65:91] = 0  # A-Z default to 0
mapping_array[ord('A')] = 1
mapping_array[ord('C')] = 2
mapping_array[ord('G')] = 3
mapping_array[ord('T')] = 4


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
        try:
            return header.index(name)
        except ValueError:
            continue
    print(message.format(default, header[default] if default is not None else 'N/A'))
    return default


def one_hot_encode(seq):
    """One-hot encode a nucleotide sequence"""
    # Convert sequence to ASCII codes and map
    indices = np.frombuffer(seq.encode(), dtype=np.uint8)
    return np.eye(5, dtype=np.int8)[mapping_array[indices]]


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
    # Apply filtering directly
    bias.loc[bias['smoothed_GC_bias'] < 0.10, 'smoothed_GC_bias'] = np.nan
    bias.loc[bias['smoothed_GC_bias'] > 10.0, 'smoothed_GC_bias'] = np.nan
    
    # Create dictionary directly
    bias_dict = {}
    for _, row in bias[['length', 'num_GC', 'smoothed_GC_bias']].iterrows():
        length, gc, bias_val = int(row['length']), int(row['num_GC']), row['smoothed_GC_bias']
        if length not in bias_dict:
            bias_dict[length] = {}
        bias_dict[length][gc] = bias_val
    
    return bias_dict


def frag_metrics(frag_prob_dist, reduce=False, bins=None):
    """
    Returns various metrics of fragment lengths given a probability distribution.
    
    Parameters:
        frag_prob_dist (np.array): 1D array of fragment length probabilities (sums to 1.0), where index is fragment length - 1 (1-indexed)
        reduce (bool): if true, only output ratio, entropy, and Gini-Simpson index (save time on signal profiles)
        bins (list/tuple): Pre-computed bins as (bin_start_indices, num_bins). Used only for entropy/Gini-Simpson calculation.
    
    Returns:
        Tuple containing:
            - mean (float) - computed on 1-bp resolution
            - stdev (float) - computed on 1-bp resolution
            - skew (float) - computed on 1-bp resolution
            - kurtosis (float) - computed on 1-bp resolution
            - ratio (float) - computed on 1-bp resolution
            - entropy (float) - computed on binned data
            - gini_simpson (float) - computed on binned data
    """
    # Calculate effective sample size: n_eff = 1 / sum(p_i^2)
    # This accounts for the information content in the probability distribution
    sum_p_squared = np.sum(frag_prob_dist ** 2)
    n_eff = 1.0 / sum_p_squared if sum_p_squared > 0 else 0
    if n_eff < 5:  # Minimum effective sample size for reliable metrics
        return (np.nan,) * (3 if reduce else 7)

    # Subnucleosomal ratio: computed on 1-bp resolution (threshold at 147 bp)
    frag_short = np.sum(frag_prob_dist[:146])
    frag_long = np.sum(frag_prob_dist[146:])
    ratio = np.nan if frag_long == 0 else frag_short / frag_long
    
    # Diversity metrics - always computed on binned data for smoothing
    entropy, gini_simpson = diversity_metrics(frag_prob_dist, bins=bins)
    
    # For reduce mode, skip moment calculations
    if reduce:
        return ratio, entropy, gini_simpson

    # Statistical moments - computed on 1-bp resolution for accurate distribution representation
    indices = np.arange(1, len(frag_prob_dist) + 1)
    weights = frag_prob_dist
    
    # Mean (in bp units)
    mean = np.average(indices, weights=weights)
    
    # Variance and standard deviation (in bp units)
    delta = indices - mean
    variance = np.average(delta**2, weights=weights)
    stdev = np.sqrt(variance)
    
    # Higher moments (skewness and kurtosis)
    if stdev > 1e-10:
        skew = np.average(delta**3, weights=weights) / (stdev**3)
        kurtosis = np.average(delta**4, weights=weights) / (stdev**4) - 3  # Excess kurtosis
    else:
        skew, kurtosis = 0.0, 0.0

    return mean, stdev, skew, kurtosis, ratio, entropy, gini_simpson


def diversity_metrics(prob_dist, bins=None):
    """
    Returns the max-normalized Shannon Entropy and Gini-Simpson index (1 - sum(p_i^2)) for binned probability distribution (5bp bins)
        Parameters:
            prob_dist (list/array): 1D array of fragment length probabilities (sums to 1.0), where index is fragment length -1 (1-indexed).
            bins (list/tuple): Pre-computed bins as (bin_start_indices, num_bins). If None, bins will be computed.
        Returns:
            tuple: (float: Max-normalized Shannon Entropy, float: Gini-Simpson index (max-normalized))
    """
    # If bins are not provided, compute them
    if bins is None:
        bin_size = 5
        prob_dist_len = len(prob_dist)
        num_bins = int(np.ceil(prob_dist_len / bin_size))
        
        # Early exit for empty or single bin
        if prob_dist_len == 0 or num_bins <= 1:
            return np.nan, np.nan
            
        # Faster binning using reshape and sum
        # Handle case where prob_dist is not evenly divisible by bin_size
        if prob_dist_len % bin_size != 0:
            # Pad the array with zeros to make it evenly divisible
            padding_size = bin_size - (prob_dist_len % bin_size)
            padded_prob = np.pad(prob_dist, (0, padding_size), 'constant')
            reshaped = padded_prob.reshape(-1, bin_size)
        else:
            reshaped = prob_dist.reshape(-1, bin_size)
        
        binned_probs = np.sum(reshaped, axis=1)
        
        # Remove empty bin from padding if it exists
        if prob_dist_len % bin_size != 0:
            binned_probs = binned_probs[:num_bins]
    else:
        # Use pre-computed bins
        bin_edges, num_bins = bins
        
        # Early exit for empty or single bin
        if len(prob_dist) == 0 or num_bins <= 1:
            return np.nan, np.nan
            
        # Compute binned probabilities using the provided bin edges
        binned_probs = np.zeros(num_bins)
        for i in range(num_bins):
            start_idx = bin_edges[i]
            end_idx = bin_edges[i+1] if i < num_bins-1 else len(prob_dist)
            binned_probs[i] = np.sum(prob_dist[start_idx:end_idx])
    
    # binned_probs should already sum to ~1.0 since prob_dist sums to 1.0
    # But use it directly without re-normalization to preserve the probability mass
    total = np.sum(binned_probs)
    if total <= 0:
        return np.nan, np.nan
        
    # Ensure normalization for numerical stability
    pdist = binned_probs / total if total > 0 else binned_probs
    
    # Gini-Simpson calculation (vectorized)
    gs_index = 1.0 - np.sum(pdist * pdist)
    max_gs = 1.0 - 1.0/num_bins
    gs_index = gs_index / max_gs if max_gs > 0 else np.nan
    
    # Entropy calculation - vectorized with masked operations
    mask = pdist > 0
    entropy = 0.0
    if np.any(mask):
        # Use masked array for cleaner calculation
        p_nonzero = pdist[mask]
        log2_p = np.log2(p_nonzero)
        entropy = -np.sum(p_nonzero * log2_p) / np.log2(num_bins)
    
    return entropy, gs_index


def local_peaks(ys):
    """
    Finds LOCAL (min or max relative to neighbors) peaks given a 1D array; only appropriate for very smooth data
    (E.G. an inverse fourier transform with high frequencies removed)
        Parameters:
            ys (np.array): signal height
        Returns:
            max_values: np.array of maxima values
            max_indices: np.array of corresponding maxima indices
            min_values: np.array of minima values
            min_indices: np.array of corresponding minima indices
    """
    # Check for edge cases
    n = len(ys)
    if n <= 2:
        # Not enough points for peaks - return empty arrays
        # We don't return np.nan here because these are arrays that may be used for indexing later
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)
    
    # Use NumPy's diff function for efficiency
    diff = np.diff(ys)
    
    # Find where derivative changes sign - doing this in one step
    # This pre-allocates the result arrays for better performance
    sign_changes_mask = np.empty(len(diff)-1, dtype=bool)
    np.not_equal(np.sign(diff[:-1]), np.sign(diff[1:]), out=sign_changes_mask)
    sign_changes = np.nonzero(sign_changes_mask)[0] + 1
    
    if len(sign_changes) == 0:
        # No sign changes means no peaks
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)
    
    # Calculate the directions once
    directions = diff[sign_changes-1] > 0
    
    # Get maxima and minima using boolean masking
    max_indices = sign_changes[directions]
    min_indices = sign_changes[~directions]
    
    # Get the values from original array
    max_values = ys[max_indices]
    min_values = ys[min_indices]
    
    return max_values, max_indices, min_values, min_indices

