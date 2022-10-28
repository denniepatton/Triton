# Robert Patton, rpatton@fredhutch.org
# v1.0.0, 10/27/2022

# Modified from GenerateCNNInputs.py v1.1.0, GenerateFFTFeatures.py v5.0.0
# TODO: composite and individual mode
# TODO: output both feature matrix with composite features and individual profiles (a la CNNInputs)
# TODO: add additional plotting utils to be called directly
# TODO: probabilistic nucleosome positioning (area of 1 per read)
# TODO: print "fingerprint" and see if any natural patterns arise
# TODO: output full fragment profile (including distribution)
# TODO: each "site" should output a coverage profile (numpy array), fragment profile (<), dictionary of features
# TODO: add length by-pass, e.g. don't output coverage profile for long locs

# Current output structure:
# ----------------------------------------------------------------------------------------------------------------------
# 1: GC-corrected depth
# 2: (nucleosome) phased profile
# 3: Shannon entropy
# 4: Region fragment profile normalized Shannon entropy
# 5: Fragment heterogeneity (unique fragment lengths / total fragments)
# 6: Short:long ratio (x <= 120 / 140 <= x <= 250)
# 7: Adenine - one hot encoded
# 8: Cytosine - one hot encoded
# 9: Guanine - one hot encoded
# 10: Tyrosine - one hot encoded
# ----------------------------------------------------------------------------------------------------------------------
# Inputs are BAM of interest (sample), associated GC_bias, a BED file with equal-sized regions of interest,
# reference sequence, and optionally a list of regions (corresponding to the names used in the BED file) for which
# the output array will be printed for visualization. Outputs are numpy arrays for each sample-region combination,
# labeled as sample_region.npy (numpy format, for matching with truth values downstream). Non one-hot encoded outputs
# are all standardized so that max = 1.0, min = 0.0 (for deep learning), and 5' -> 3' directionality is enforced.

import os
import sys
import pysam
import random
import argparse
from functools import partial
from multiprocessing import Pool
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
from triton_helpers import *

freq_max = 0.0068493  # theoretical nucleosome period is 146 bp -> f = 0.0068493
# range_1: T = 150-180 bp (f = 0.00556 - 0.00667)
# range_2: T = 180-210 bp (f = 0.00476 - 0.00555)
low_1, high_1 = 0.00556, 0.00667
low_2, high_2 = 0.00476, 0.00555


def _generate_profile(region, params):
    bam_path, sample, out_direct, frag_range, gc_bias, ref_seq_path, map_q, window = params
    if region.endswith('.bed'):  # a list of bed files is supplied: perform stacked site analysis
        stack = True
    else:
        stack = False
    bam = pysam.AlignmentFile(bam_path, 'rb')
    ref_seq = pysam.FastaFile(ref_seq_path)
    roi_fragment_lengths = []
    if stack:
        site = os.path.basename(region).split('.')[0]
        roi_length = window + 1000  # a 500bp buffer is added to smooth FFT boundaries
        depth = [0] * roi_length
        default_pos = 3  # if 'position' not present in header, use the 4th bed column
        fragment_length_profile = [[] for _ in range(roi_length)]
        for entry in open(site.strip(), 'r'):  # really a path to each BED file when stacked
            bed_tokens = entry.strip().split('\t')
            if not bed_tokens[1].isdigit():  # header: assign position index and continue
                default_pos = bed_tokens.index('position')
                continue
            pos = default_pos
            center_pos = int(bed_tokens[pos])  # Griffin style "position"
            start_pos = center_pos - (int(window/2) + 500)
            stop_pos = center_pos + (int(window/2) + 500)
            # process all fragments falling inside the ROI
            segment_reads = bam.fetch(bed_tokens[0], start_pos, stop_pos)
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
                    # now do GC bias correction:
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
                    for index in [val for val in fragment_cov if 0 <= val < roi_length]:
                        fragment_length_profile[index].append(abs(fragment_length))
                        if 0.05 < fragment_bias < 10:
                            depth[index] += 1 / fragment_bias
    else:
        bed_tokens = region.strip().split('\t')
        # a 500bp buffer is added to smooth FFT boundaries
        start_pos = int(bed_tokens[1]) - 500
        stop_pos = int(bed_tokens[2]) + 500
        site = str(bed_tokens[3])
        depth = [0] * (stop_pos - start_pos)
        roi_length = len(depth)
        # process all fragments falling inside the ROI
        segment_reads = bam.fetch(bed_tokens[0], start_pos, stop_pos)
        fragment_length_profile = [[] for _ in range(roi_length)]
        roi_sequence = ref_seq.fetch(bed_tokens[0], start_pos, stop_pos).upper()
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
                # now do GC bias correction:
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
                for index in [val for val in fragment_cov if 0 <= val < roi_length]:
                    fragment_length_profile[index].append(abs(fragment_length))
                    if 0.05 < fragment_bias < 10:
                        depth[index] += 1 / fragment_bias
    # get phased profile ###############################################################################################
    mean_depth = np.mean(depth[500:-500])
    if mean_depth < 1.0:
        return (sample + '_' + site), np.nan  # skip regions with less than 1X mean coverage
    fourier = rfft(np.asarray(depth))
    freqs = rfftfreq(roi_length)
    range_1 = [idx for idx, val in enumerate(freqs) if low_1 <= val <= high_1]
    range_2 = [idx for idx, val in enumerate(freqs) if low_2 <= val <= high_2]
    primary_amp_1 = round(np.mean(np.abs(fourier[range_1])) / roi_length, 4)
    primary_amp_2 = round(np.mean(np.abs(fourier[range_2])) / roi_length, 4)
    if primary_amp_1 > 0 and primary_amp_2 > 0:
        amp_ratio = primary_amp_2 / primary_amp_1
    else:
        amp_ratio = np.nan
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
    # TODO: consider making some features, like central dip, stack-specific
    # TODO: output peak locations and relative heights (above mean)
    max_values, peaks = local_peaks(phased_signal, depth)
    inflection = min(phased_signal[int(window/2 - 100):int(window/2 + 100)])  # assume minimum, then double-check for inflection
    inflection_loc = np.where(phased_signal == inflection)[0]
    if phased_signal[inflection_loc - 5] < inflection or phased_signal[inflection_loc + 5] < inflection:  # not min
        inflection = max(phased_signal[int(window/2 - 100):int(window/2 + 100)])
        inflection_loc = np.where(phased_signal == inflection)[0]
    left_max_loc, right_max_loc = nearest_peaks(inflection_loc, peaks)
    left_max, right_max = phased_signal[left_max_loc], phased_signal[right_max_loc]
    central_dip_shoulder = ((left_max + right_max) / 2 - inflection) / ((left_max + right_max) / 2)
    central_dip_mean = ((left_max + right_max) / 2 - inflection) / mean_depth
    if len(max_values) < 1:
        nuc_period = np.nan
    else:
        nuc_period = round(roi_length / len(peaks), 2)


    if len(fragment_lengths) > 10:
        ratio = frag_ratio(fragment_lengths)
        point_ent, non_ent, shan_ent, point_frag = point_entropy(fragment_lengths, bam, bed_tokens[0], start_pos, stop_pos, frag_range, map_q)
        mean_point_ent = np.mean(point_ent)
        point_ent = [i / sum(point_ent) for i in point_ent]
        non_ent = [i / sum(non_ent) for i in non_ent]
        point_frag = [i / sum(point_frag) for i in point_frag]
        region_ent = dirichlet_normalized_entropy(fragment_lengths, fragment_lengths)
        flanking_lengths = get_fragment_lengths(bam, bed_tokens[0], start_pos, start_pos + 500, frag_range, map_q) +\
                           get_fragment_lengths(bam, bed_tokens[0], stop_pos - 500, stop_pos, frag_range, map_q)
        flank_ent = dirichlet_normalized_entropy(fragment_lengths, flanking_lengths)
        # global_ent = dirichlet_normalized_entropy(fragment_lengths, global_lengths)
        global_ent = np.nan
        gini = gini_index(fragment_lengths)
        std = np.std(fragment_lengths)
        mean = np.mean(fragment_lengths)
        cv = std / mean
        mad = np.median(np.absolute(fragment_lengths - np.median(fragment_lengths)))




    # get fragment profiles ############################################################################################
    shan_profile, norm_profile = point_entropy(roi_fragment_lengths, fragment_length_profile)
    ratio_profile = [frag_ratio(point_frags) for point_frags in fragment_length_profile]
    hetero_profile = [len(set(point_frags)) / len(point_frags) if len(point_frags) is not 0 else 0 for point_frags in
                      fragment_length_profile]
    # get one-hot sequence #############################################################################################
    seq_profile = np.delete(one_hot_encode(roi_sequence), 0, 1)
    # normalize profiles ###############################################################################################
    depth = normalize_data(depth)
    phased_signal = normalize_data(phased_signal)
    shan_profile = normalize_data(shan_profile)
    norm_profile = normalize_data(norm_profile)
    hetero_profile = normalize_data(hetero_profile)
    ratio_profile = normalize_data(ratio_profile)





    # combine and save profiles ########################################################################################
    signal_array = np.column_stack((depth, phased_signal, shan_profile, norm_profile, hetero_profile, ratio_profile))
    out_array = np.concatenate((signal_array, seq_profile), axis=1)
    return (sample + '_' + site), out_array  # identifier, numpy matrix


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
    parser.add_argument('-w', '--window', help='window size (bp) for composite sites', type=int, default=2000)
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
    print('\n')
    sys.stdout.flush()

    gc_bias = get_gc_bias_dict(bias_path)
    sites = [region for region in open(sites_path, 'r')]
    random.shuffle(sites)
    params = [bam_path, sample_name, results_dir, size_range, gc_bias, ref_seq_path, map_q, window]

    print('Running Triton on ' + str(len(sites)) + ' region sets . . .')

    with Pool(cpus) as pool:
        results = list(pool.imap_unordered(partial(_generate_profile, params=params), sites, len(sites) // cpus))

    print('Merging and saving results . . .')

    # results_dict = {}
    # for result in results:
    #     results_dict[result[0]] = result[1]
    # np.savez_compressed(results_dir + '/' + sample_name, **results_dict)
    #
    # fm = {sample_name: {'Sample': sample_name}}
    # for result in results:
    #     fm[sample_name][result[0] + '_peak-based-period'] = result[1]
    #     fm[sample_name][result[0] + '_amplitude-ratio'] = result[2]
    #     fm[sample_name][result[0] + '_mean-depth'] = result[3]
    #     fm[sample_name][result[0] + '_central-dip-shoulder'] = result[4]
    #     fm[sample_name][result[0] + '_central-dip-mean'] = result[5]
    # df = pd.DataFrame(fm).transpose()
    #
    # out_file = results_dir + '/' + sample_name + '_PhasingFM.tsv'
    # if not os.path.exists(results_dir):
    #     os.mkdir(results_dir)
    # df.to_csv(out_file, sep='\t')

    print('Finished')


if __name__ == "__main__":
    main()
