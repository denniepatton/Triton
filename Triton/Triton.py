# Robert Patton, rpatton@fredhutch.org
# v1.0.0, 10/27/2022

# Modified from GenerateCNNInputs.py v1.1.0, GenerateFFTFeatures.py v5.0.0
# TODO: print "fingerprint" and see if any natural patterns arise
# Output structure (region-level features as a dictionary, region-level profiles as a single numpy matrix):
# ----------------------------------------------------------------------------------------------------------------------
# ### Region-level features (fragmentation):
# 1: frag-mean: fragment lengths' mean
# 2: frag-std: fragment lengths' standard deviation
# 3: frag-mad: fragment lengths' MAD (Mean Absolute Deviation)
# 4: frag-rat: fragment lengths' short:long ratio (x <= 120 / 140 <= x <= 250)
# 5: frag-ent: fragment lengths' Shannon entropy, normalized by Dirichlet expected profile in the same region
# ### Region-level features (phasing):
# 6: np-score: Nucleosome Phasing score
# 7: np-period: phased-nucleosome periodicity
# ### Region-level features (profile-based):
# 8: mean-depth: mean depth in the region (GC-corrected, if provided)
# 9: cd-shoulder*: central dip/peak value as a fraction of the mean value at the predicted +/-1 nucleosomes
# 10: cd-mean*: central dip/peak value as a fraction of the mean depth in the region
# 11: flank-diff*: height of +1 nucleosome / height of -1 nucleosome
# 12: plus-one-loc*: location relative to position of plus-one nucleosome
# 13: minus-one-loc*: location relative to position of minus-one nucleosome
# ### Region-level profiles (all un-normalized, nt-resolution)*:
# 1: GC-corrected (if provided by Griffin) depth
# 2: Nucleosome-level phased profile
# 3: Nucleosome center profile
# 4: Mean fragment size
# 4: Fragment size Shannon entropy
# 5: Region fragment profile normalized Shannon entropy
# 6: Fragment heterogeneity (unique fragment lengths / total fragments)
# 7: Fragment MAD (Mean Absolute Deviation)
# 8: Short:long ratio (x <= 120 / 140 <= x <= 250)
# 9: A (Adenine) frequency
# 10: C (Cytosine) frequency
# 11: G (Guanine) frequency
# 12: T (Tyrosine) frequency
# * these features are output as np.nan if window == False
# ----------------------------------------------------------------------------------------------------------------------
# Inputs are BAM of interest (sample), associated GC_bias, a BED-style file with equal-sized regions of interest,
# reference sequence, and optionally a list of regions (corresponding to the names used in the BED file) for which
# the output array will be printed for visualization. Outputs are numpy arrays for each sample-region combination,
# labeled as sample_region.npy (numpy format, for matching with truth values downstream). Non one-hot encoded outputs
# are all standardized so that max = 1.0, min = 0.0 (for deep learning), and 5' -> 3' directionality is enforced.

import os
import sys
import pysam
import random
import argparse
from math import floor, ceil
from functools import partial
from multiprocessing import Pool
from scipy.fft import rfft, rfftfreq, irfft
from triton_helpers import *

# default values
freq_max = 0.0068493  # theoretical nucleosome period is 146 bp -> f = 0.0068493
low_1, high_1 = 0.00556, 0.00667  # range_1: T = 150-180 bp (f = 0.00556 - 0.00667)
low_2, high_2 = 0.00476, 0.00555  # range_2: T = 180-210 bp (f = 0.00476 - 0.00555)
chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx = 0, 1, 2, 3, 5, 6  # default BED indices


def _generate_profile(region, params):
    bam_path, out_direct, frag_range, gc_bias, ref_seq_path, map_q, window, stack = params
    bam = pysam.AlignmentFile(bam_path, 'rb')
    ref_seq = pysam.FastaFile(ref_seq_path)
    roi_fragment_lengths = []
    if stack:  # assemble depth and fragment profiles for composite sites ----------------------------------------------
        site = os.path.basename(region).split('.')[0]  # use the file name as the feature name
        roi_length = window + 1000  # a 500bp buffer is added to smooth FFT boundaries
        depth = [0] * roi_length
        nc_signal = depth
        fragment_length_profile = [[] for _ in range(roi_length)]
        oh_seq = np.zeros((5, roi_length))
        with open(site.strip(), 'r') as sites_file:
            next(sites_file)  # skip header
            for entry in sites_file:  # iterate through regions in this particular BED file
                bed_tokens = entry.strip().split('\t')
                chrom, strand, center_pos =\
                    str(bed_tokens[chr_idx]), str(bed_tokens[strand_idx]), int(bed_tokens[pos_idx])
                start_pos = center_pos - (int(window/2) + 500)
                stop_pos = center_pos + (int(window/2) + 500)
                # process all fragments falling inside the ROI
                segment_reads = bam.fetch(chrom, start_pos, stop_pos)
                roi_sequence = ref_seq.fetch(chrom, start_pos, stop_pos).upper()
                if strand == '+': oh_seq = np.add(oh_seq, one_hot_encode(roi_sequence))
                else: oh_seq = np.add(oh_seq, one_hot_encode(roi_sequence[::-1]))
                for read in segment_reads:
                    fragment_length = read.template_length
                    if frag_range[0] <= np.abs(fragment_length) <= frag_range[1] and read.is_paired and read. \
                            mapping_quality >= map_q and not read.is_duplicate and not read.is_qcfail:
                        roi_fragment_lengths.append(abs(fragment_length))
                        read_start = read.reference_start - start_pos
                        if read.is_reverse and fragment_length < 0:
                            read_length = read.reference_length
                            fragment_start = read_start + read_length + fragment_length
                            fragment_end = read_start + read_length
                        elif not read.is_reverse and fragment_length > 0:
                            fragment_start = read_start
                            fragment_end = read_start + fragment_length
                        else:
                            continue
                        if gc_bias is not None:
                            fragment_seq =\
                                (ref_seq.fetch(chrom, fragment_start + start_pos, fragment_end + start_pos)).upper()
                            fragment_seq = list(fragment_seq.replace('T', '0').replace('A', '0').replace('C', '1').
                                                replace('G', '1').replace('N', str(np.random.randint(0, 2))))
                            fragment_gc_content = sum([int(m) for m in fragment_seq])
                            fragment_bias = gc_bias[np.abs(fragment_length)][fragment_gc_content]
                        else:
                            fragment_bias = 1
                        fragment_cov = np.array(range(fragment_start, fragment_end + 1))
                        nc_spread = abs(fragment_length) - 147
                        if nc_spread < 0:
                            nc_cov = None
                        elif nc_spread == 0:
                            nc_cov = np.array(fragment_start + 74)
                        elif 0 < nc_spread < 147:
                            fragment_midpoint = ceil(abs(fragment_length) / 2)
                            nc_cov = np.array(range(fragment_start + fragment_midpoint - floor(nc_spread/2),
                                                    fragment_start - fragment_midpoint + floor(nc_spread/2)))
                        else:
                            nc_cov = None
                        if strand == '+':  # positive strand:
                            for index in [val for val in fragment_cov if 0 <= val < roi_length]:
                                fragment_length_profile[index].append(abs(fragment_length))
                                if 0.05 < fragment_bias < 10:
                                    depth[index] += 1 / fragment_bias
                                    if index in nc_cov:
                                        nc_signal[index] += 1 / fragment_bias / nc_spread
                        else:  # negative strand, flip array before adding TODO: make sure this is right/needed
                            for index in [val for val in fragment_cov if 0 <= val < roi_length]:
                                fragment_length_profile[roi_length - index].append(abs(fragment_length))
                                if 0.05 < fragment_bias < 10:
                                    depth[roi_length - index] += 1 / fragment_bias
                                    if index in nc_cov:
                                        nc_signal[roi_length - index] += 1 / fragment_bias / nc_spread
        oh_seq = oh_seq/oh_seq.sum(axis=0, keepdims=True)
    else:  # assemble depth and fragment profiles for sites individually -----------------------------------------------
        bed_tokens = region.strip().split('\t')
        chrom, strand, site = str(bed_tokens[chr_idx]), str(bed_tokens[strand_idx]), str(bed_tokens[site_idx])
        if window is not None:
            roi_length = window + 1000  # a 500bp buffer is added to smooth FFT boundaries
            depth = [0] * roi_length
            center_pos = int(bed_tokens[pos_idx])
            start_pos = center_pos - (int(window / 2) + 500)
            stop_pos = center_pos + (int(window / 2) + 500)
        else:
            # a 500bp buffer is added to smooth FFT boundaries
            start_pos = int(bed_tokens[start_idx]) - 500
            stop_pos = int(bed_tokens[stop_idx]) + 500
            depth = [0] * (stop_pos - start_pos)
            roi_length = stop_pos - start_pos
        nc_signal = depth
        # process all fragments falling inside the ROI
        segment_reads = bam.fetch(bed_tokens[0], start_pos, stop_pos)
        roi_sequence = ref_seq.fetch(bed_tokens[0], start_pos, stop_pos).upper()
        if strand == '+': oh_seq = one_hot_encode(roi_sequence)
        else: oh_seq = one_hot_encode(roi_sequence[::-1])
        fragment_length_profile = [[] for _ in range(roi_length)]
        for read in segment_reads:
            fragment_length = read.template_length
            if frag_range[0] <= np.abs(fragment_length) <= frag_range[1] and read.is_paired and read.\
                    mapping_quality >= map_q and not read.is_duplicate and not read.is_qcfail:
                roi_fragment_lengths.append(abs(fragment_length))
                read_start = read.reference_start - start_pos
                if read.is_reverse and fragment_length < 0:
                    read_length = read.reference_length
                    fragment_start = read_start + read_length + fragment_length
                    fragment_end = read_start + read_length
                elif not read.is_reverse and fragment_length > 0:
                    fragment_start = read_start
                    fragment_end = read_start + fragment_length
                else:
                    continue
                if gc_bias is not None:
                    fragment_seq = (ref_seq.fetch(bed_tokens[0], fragment_start + start_pos,
                                                  fragment_end + start_pos)).upper()
                    fragment_seq = list(fragment_seq.replace('T', '0').replace('A', '0').replace('C', '1').
                                        replace('G', '1').replace('N', str(np.random.randint(0, 2))))
                    fragment_gc_content = sum([int(m) for m in fragment_seq])
                    fragment_bias = gc_bias[np.abs(fragment_length)][fragment_gc_content]
                else:
                    fragment_bias = 1
                fragment_cov = np.array(range(fragment_start, fragment_end + 1))
                nc_spread = abs(fragment_length) - 147
                if nc_spread < 0:
                    nc_cov = None
                elif nc_spread == 0:
                    nc_cov = np.array(fragment_start + 74)
                elif 0 < nc_spread < 147:
                    fragment_midpoint = ceil(abs(fragment_length) / 2)
                    nc_cov = np.array(range(fragment_start + fragment_midpoint - floor(nc_spread / 2),
                                            fragment_start - fragment_midpoint + floor(nc_spread / 2)))
                else:
                    nc_cov = None
                for index in [val for val in fragment_cov if 0 <= val < roi_length]:
                    fragment_length_profile[index].append(abs(fragment_length))
                    if 0.05 < fragment_bias < 10:
                        depth[index] += 1 / fragment_bias
                        if index in nc_cov:
                            nc_signal[index] += 1 / fragment_bias / nc_spread
        if strand == "-":
            depth = depth[::-1]
            fragment_length_profile = fragment_length_profile[::-1]
    # generate phased profile and phasing/region-level features ########################################################
    mean_depth = np.mean(depth[500:-500])  # only consider depth in the roi
    fourier = rfft(np.asarray(depth))
    freqs = rfftfreq(roi_length)
    range_1 = [idx for idx, val in enumerate(freqs) if low_1 <= val <= high_1]
    range_2 = [idx for idx, val in enumerate(freqs) if low_2 <= val <= high_2]
    primary_amp_1 = round(np.mean(np.abs(fourier[range_1])) / roi_length, 4)
    primary_amp_2 = round(np.mean(np.abs(fourier[range_2])) / roi_length, 4)
    if primary_amp_1 > 0 and primary_amp_2 > 0:
        np_score = primary_amp_2 / primary_amp_1
    else:
        np_score = np.nan
    test_freqs = [idx for idx, val in enumerate(freqs) if 0 < val <= freq_max]  # frequencies in filter
    clean_fft = [0] * len(fourier)
    try:
        clean_fft[test_freqs[0]:test_freqs[-1]] = fourier[test_freqs[0]:test_freqs[-1]]
        inverse_signal = irfft(clean_fft)  # reconstruct signal
        phased_signal = inverse_signal + len(inverse_signal) * [mean_depth]  # add in base component
    except IndexError:  # not enough data to construct a reasonable Fourier
        phased_signal = [0] * roi_length
    # Remove the buffer regions:
    depth = depth[500:-500]
    phased_signal = phased_signal[500:-500]
    max_values, peaks = local_peaks(phased_signal, depth)
    if window is not None:
        inflection = min(phased_signal[int(window/2 - 100):int(window/2 + 100)])  # assume minimum, then double-check
        inflection_loc = np.where(phased_signal == inflection)[0]
        if phased_signal[inflection_loc - 5] < inflection or phased_signal[inflection_loc + 5] < inflection:  # not min
            inflection = max(phased_signal[int(window/2 - 100):int(window/2 + 100)])
            inflection_loc = np.where(phased_signal == inflection)[0]
        left_max_loc, right_max_loc = nearest_peaks(inflection_loc, peaks)
        left_max, right_max = phased_signal[left_max_loc], phased_signal[right_max_loc]
        cd_shoulder = ((left_max + right_max) / 2 - inflection) / ((left_max + right_max) / 2)
        cd_mean = ((left_max + right_max) / 2 - inflection) / mean_depth
        flank_diff = right_max / left_max
        minus_one_loc, plus_one_loc = left_max_loc, right_max_loc
    else:
        cd_shoulder, cd_mean, flank_diff, minus_one_loc, plus_one_loc = np.nan, np.nan, np.nan, np.nan, np.nan
    if len(max_values) < 1:
        np_period = np.nan
    else:
        np_period = round(roi_length / len(peaks), 2)
    # generate fragment profiles and fragmentation features ############################################################
    frag_mean = np.mean(roi_fragment_lengths)
    frag_std = np.std(roi_fragment_lengths)
    frag_mad = np.median(np.absolute(roi_fragment_lengths - np.median(roi_fragment_lengths)))
    frag_rat = frag_ratio(roi_fragment_lengths)
    frag_ent = dirichlet_normalized_entropy(roi_fragment_lengths, roi_fragment_lengths)
    # fragment profiles
    shan_profile, norm_profile = point_entropy(roi_fragment_lengths, fragment_length_profile)
    ratio_profile = [frag_ratio(point_frags) for point_frags in fragment_length_profile]
    mf_profile = [np.mean(point_frags) for point_frags in fragment_length_profile]
    mad_profile = [np.median(np.absolute(point_frags - np.median(point_frags)))
                   for point_frags in fragment_length_profile]
    hetero_profile = [len(set(point_frags)) / len(point_frags) if len(point_frags) is not 0 else 0 for point_frags in
                      fragment_length_profile]
    # sequence profile
    seq_profile = np.delete(oh_seq, 0, 1)  # remove the N row
    # combine and save profiles ########################################################################################
    signal_array = np.column_stack((depth, phased_signal, nc_signal, mf_profile, shan_profile,
                                    norm_profile, hetero_profile, mad_profile, ratio_profile))
    out_array = np.concatenate((signal_array, seq_profile), axis=1)
    return site, frag_mean, frag_std, frag_mad, frag_rat, frag_ent, np_score, np_period, mean_depth, cd_shoulder,\
           cd_mean, flank_diff, plus_one_loc, minus_one_loc, out_array


def main():
    # parse command line arguments:
    parser = argparse.ArgumentParser(description='\n### Triton ###')
    parser.add_argument('-n', '--sample_name', help='sample identifier', required=True)
    parser.add_argument('-i', '--input', help='BAM file path', required=True)
    parser.add_argument('-b', '--bias', help='GC bias file (from Griffin)', default=None)
    parser.add_argument('-a', '--annotation', help='regions of interest in BED or list of BEDs', required=True)
    parser.add_argument('-g', '--reference_genome', help='reference genome (.fa)', required=True)
    parser.add_argument('-r', '--results_dir', help='directory for results', required=True)
    parser.add_argument('-q', '--map_quality', help='minimum mapping quality (default=20)', type=int, default=20)
    parser.add_argument('-f', '--size_range', help='fragment size range (bp; default=[15, 500])', nargs=2, type=int,
                        default=(15, 500))
    parser.add_argument('-c', '--cpus', help='number of CPUs to use for parallel processing', type=int, required=True)
    parser.add_argument('-w', '--window', help='window size (bp) for composite sites', type=int, default=None)
    parser.add_argument('-s', '--composite', help='whether to run in composite-site mode', type=bool, default=False)
    args = parser.parse_args()

    print('Loading input files . . .')

    sample_name = args.sample_name
    bam_path = args.input
    bias_path = args.bias
    sites_path = args.annotation
    ref_seq_path = args.reference_genome
    results_dir = args.results_dir
    map_q = args.map_quality
    size_range = args.size_range
    cpus = args.cpus
    window = args.window
    stack = args.composite

    print('\n### arguments provided:')
    print('\tsample_name = "' + sample_name + '"')
    print('\tbam_path = "' + bam_path + '"')
    print('\tGC_bias_path = "' + bias_path + '"')
    print('\tref_seq_path = "' + ref_seq_path + '"')
    print('\tsite_list = "' + sites_path + '"')
    print('\tresults_dir = "' + results_dir + '"')
    print('\tsize_range =', size_range)
    print('\tmap_q =', map_q)
    print('\tCPUs =', cpus)
    print('\twindow =', window)
    print('\tcomposite = ', stack)
    print('\n')
    sys.stdout.flush()

    gc_bias = get_gc_bias_dict(bias_path)
    global chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx
    if stack and window is None:
        print('ERROR: if using Triton in composite mode a window (-w) must be specified. Exiting.')
        header, sites = None, None
        exit()
    elif stack and window is not None:
        print('Running Triton in composite mode.')
        sites = [region for region in open(sites_path, 'r')]  # a list of BED-like paths
        header = [site for site in open(sites[0], 'r')].pop(0).split('\t')  # retrieve one header
    else:
        print('Running Triton in individual mode.')
        sites = [region for region in open(sites_path, 'r')]  # a list of regions
        header = sites.pop(0).split('\t')
    # Below checks for standard BED column names and the position column if window=True, updating their indices
    # if necessary. If a non-standard header format is used defaults indices will be used, which may error.
    if 'chrom' in header:
        chr_idx = header.index('chrom')
    else:
        print('No "chrom" column found in BED file(s): defaulting to index 0 (' + header[0] + ')')
    if 'chromStart' in header:
        start_idx = header.index('chromStart')
    else:
        print('No "chromStart" column found in BED file(s): defaulting to index 1 (' + header[1] + ')')
    if 'chromEnd' in header:
        stop_idx = header.index('chromEnd')
    else:
        print('No "chromEnd" column found in BED file(s): defaulting to index 2 (' + header[2] + ')')
    if 'name' in header:
        site_idx = header.index('name')
    else:
        print('No "name" column found in BED file(s): defaulting to index 3 (' + header[3] + ')')
    if 'strand' in header:
        strand_idx = header.index('strand')
    else:
        print('No "strand" column found in BED file(s): defaulting to index 5 (' + header[5] + ')')
    if stack and 'position' in header:
        pos_idx = header.index('position')
    elif stack:
        print('No "position" column found in BED file(s): defaulting to index 6 (' + header[6] + ')')
    random.shuffle(sites)  # to spread sites evenly between cores given variable size
    params = [bam_path, results_dir, size_range, gc_bias, ref_seq_path, map_q, window, stack]

    print('Running Triton on ' + str(len(sites)) + ' region sets . . .')

    with Pool(cpus) as pool:
        results = list(pool.imap_unordered(partial(_generate_profile, params=params), sites, len(sites) // cpus))

    print('Merging and saving results . . .')
    signal_dict = {}
    fm = {sample_name: {'Sample': sample_name}}
    for result in results:
        fm[sample_name][result[0] + '_frag-mean'] = result[1]
        fm[sample_name][result[0] + '_frag-std'] = result[2]
        fm[sample_name][result[0] + '_frag-mad'] = result[3]
        fm[sample_name][result[0] + '_frag-rat'] = result[4]
        fm[sample_name][result[0] + '_frag-ent'] = result[5]
        fm[sample_name][result[0] + '_np-score'] = result[6]
        fm[sample_name][result[0] + '_np-period'] = result[7]
        fm[sample_name][result[0] + '_mean-depth'] = result[8]
        fm[sample_name][result[0] + '_cd-shoulder'] = result[9]
        fm[sample_name][result[0] + '_cd-mean'] = result[10]
        fm[sample_name][result[0] + '_flank-diff'] = result[11]
        fm[sample_name][result[0] + '_plus-one-loc'] = result[12]
        fm[sample_name][result[0] + '_minus-one-loc'] = result[13]
        signal_dict[result[0]] = result[14]
    df = pd.DataFrame(fm).transpose()
    out_file = results_dir + '/' + sample_name + '_PhasingFM.tsv'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    np.savez_compressed(results_dir + '/' + sample_name, **signal_dict)
    df.to_csv(out_file, sep='\t')

    print('Finished')


if __name__ == "__main__":
    main()
