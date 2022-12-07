# Robert Patton, rpatton@fredhutch.org
# v1.1.0, 12/05/2022

import os
import sys
import pysam
import random
import pickle
import argparse
from functools import partial
from multiprocessing import Pool
from scipy.fft import rfft, rfftfreq, irfft
from triton_helpers import *

# constants and defaults
freq_max = 0.0068493  # theoretical minimum nucleosome period is 146 bp -> f = 0.0068493
low_1, high_1 = 0.00556, 0.00667  # range_1: T = 150-180 bp (f = 0.00556 - 0.00667)
low_2, high_2 = 0.00476, 0.00555  # range_2: T = 180-210 bp (f = 0.00476 - 0.00555)
chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx = 0, 1, 2, 3, 5, 6  # default BED indices


def generate_profile(region, params):
    """
    Generates single or composite signal profiles and extracts features (fragmentation, nucleosome phasing, and profile
    shape-based) for a single sample. Utilizes functions from triton_helpers.py.
        Parameters:
            region (string): either a file path pointing to a BED-like file (composite) or a BED-like line
            params (list): bam_path, out_direct, frag_range, gc_bias, ref_seq_path, map_q, window, stack
        Returns:
            site: annotation name if stacked, "name" from BED file for each region otherwise
                ### Region-level features (fragmentation) ###
            fragment-mean: fragment lengths' mean
            fragment-stdev: fragment lengths' standard deviation
            fragment-mad: fragment lengths' MAD (Mean Absolute Deviation)
            fragment-ratio: fragment lengths' short:long ratio (x <= 120 / 140 <= x <= 250)
            fragment-entropy: fragment lengths' Shannon entropy
                ### Region-level features (phasing) ###
            np-score: Nucleosome Phasing score
            np-period: phased-nucleosome period
            np-amplitude: phased-nucleosome mean amplitude
                ### Region-level features (profile-based) ###
            mean-depth: mean depth in the region (GC-corrected, if provided)
            var-ratio: ratio of variation to constant noise in the phased signal
            plus-one-pos*: location relative to central-loc of plus-one nucleosome
            minus-one-pos*: location relative to central-loc of minus-one nucleosome
            plus-minus-ratio*: ratio of height of +1 nucleosome to -1 nucleosome
            central-loc*: location of central inflection relative to window center (0)
            central-depth*: phased signal value at the central-loc (with mean in region set to 1)
            central-heterogeneity*: mean fragment heterogeneity value in the +/-5 bp region about the central-loc
                ### Region-level profiles (all un-normalized, nt-resolution) ###
            numpy array: shape 13xN containing:
                1: Depth (GC-corrected, if provided)
                2: Nucleosome-level phased profile
                3: Nucleosome center profile (GC-corrected, if provided)
                4: Mean fragment size
                5: Fragment size Shannon entropy
                6: Fragment heterogeneity (unique fragment lengths / total fragments)
                7: Fragment MAD (Mean Absolute Deviation)
                8: Short:long ratio (x <= 120 / 140 <= x <= 250)
                9: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***
                10: A (Adenine) frequency**
                11: C (Cytosine) frequency**
                12: G (Guanine) frequency**
                13: T (Tyrosine) frequency**
            * these features are output as np.nan if window == None
            ** sequence is based on the reference, not the reads
            *** minus-one, plus-one, and inflection locs are only called if window != None, and supersede peak/trough
    """
    bam_path, out_direct, frag_range, gc_bias, ref_seq_path, map_q, window, stack, fdict = params
    bam = pysam.AlignmentFile(bam_path, 'rb')
    ref_seq = pysam.FastaFile(ref_seq_path)
    roi_fragment_lengths = []
    if stack:  # assemble depth and fragment profiles for composite sites ----------------------------------------------
        site = os.path.basename(region).split('.')[0]  # use the file name as the feature name
        roi_length = window + 1000
        depth = [0] * roi_length
        nc_signal = [0] * roi_length
        fragment_length_profile = [[] for _ in range(window)]
        oh_seq = np.zeros((window, 5))
        with open(region.strip(), 'r') as sites_file:
            for entry in sites_file:  # iterate through regions in this particular BED file
                bed_tokens = entry.strip().split('\t')
                if not bed_tokens[pos_idx].isdigit():  # header or typo line
                    continue
                chrom, center_pos = str(bed_tokens[chr_idx]), int(bed_tokens[pos_idx])
                if strand_idx is not None:
                    strand = str(bed_tokens[strand_idx])
                else:
                    strand = '+'
                start_pos = center_pos - int(window/2) - 500
                stop_pos = center_pos + int(window/2) + 500
                # process all fragments falling inside the ROI
                if start_pos < 0: continue
                segment_reads = bam.fetch(chrom, start_pos, stop_pos)
                window_sequence = ref_seq.fetch(chrom, start_pos + 500, stop_pos - 500).upper()
                if strand == '+': oh_seq = np.add(oh_seq, one_hot_encode(window_sequence))
                else: oh_seq = np.add(oh_seq, one_hot_encode(window_sequence[::-1]))
                for read in segment_reads:
                    fragment_length = read.template_length
                    if frag_range[0] <= abs(fragment_length) <= frag_range[1] and read.is_paired and read. \
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
                        # N.B. fragment_start/end are in relative coordinates, not absolute
                        fragment_cov = np.array(range(fragment_start, fragment_end))
                        if abs(fragment_length) > 146:
                            nc_density = fdict[abs(fragment_length)]
                        else:
                            nc_density = None
                        if 0.05 < fragment_bias < 10:
                            if strand == '+':  # positive strand:
                                for place, index in enumerate(fragment_cov):
                                    if not 0 <= index < roi_length:
                                        continue
                                    depth[index] += 1 / fragment_bias
                                    if nc_density is not None:
                                        nc_signal[index] += nc_density[place] / fragment_bias
                                    if 500 <= index <= roi_length - 501:  # no buffer for frag lengths
                                        fragment_length_profile[index - 500].append(abs(fragment_length))
                            else:  # negative strand:
                                for place, index in enumerate(fragment_cov):
                                    if not 0 <= index < roi_length:
                                        continue
                                    depth[roi_length - index] += 1 / fragment_bias
                                    if nc_density is not None:
                                        nc_signal[roi_length - index] += nc_density[place] / fragment_bias
                                    if 500 <= index <= roi_length - 501:  # no buffer for frag lengths
                                        fragment_length_profile[roi_length - index - 500].append(abs(fragment_length))
        oh_seq = oh_seq/oh_seq.sum(axis=1, keepdims=True)
    else:  # assemble depth and fragment profiles for sites individually -----------------------------------------------
        bed_tokens = region.strip().split('\t')
        chrom, site = str(bed_tokens[chr_idx]), str(bed_tokens[site_idx])
        if strand_idx is not None:
            strand = str(bed_tokens[strand_idx])
        else:
            strand = '+'
        if window is not None:
            roi_length = window + 1000
            depth = [0] * roi_length
            nc_signal = [0] * roi_length
            center_pos = int(bed_tokens[pos_idx])
            start_pos = center_pos - int(window / 2) - 500
            stop_pos = center_pos + int(window / 2) + 500
        else:
            start_pos = int(bed_tokens[start_idx]) - 500
            stop_pos = int(bed_tokens[stop_idx]) + 500
            depth = [0] * (stop_pos - start_pos)
            nc_signal = [0] * (stop_pos - start_pos)
            roi_length = stop_pos - start_pos
        # process all fragments falling inside the ROI
        segment_reads = bam.fetch(bed_tokens[0], start_pos, stop_pos)
        window_sequence = ref_seq.fetch(bed_tokens[0], start_pos - 500, stop_pos - 500).upper()
        if strand == '+': oh_seq = one_hot_encode(window_sequence)
        else: oh_seq = one_hot_encode(window_sequence[::-1])
        fragment_length_profile = [[] for _ in range(window)]
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
                fragment_cov = np.array(range(fragment_start, fragment_end))
                if abs(fragment_length) > 146:
                    nc_density = fdict[abs(fragment_length)]
                else:
                    nc_density = None
                if 0.05 < fragment_bias < 10:
                    for place, index in enumerate(fragment_cov):
                        if not 0 <= index < roi_length:
                            continue
                        depth[index] += 1 / fragment_bias
                        if nc_density is not None:
                            nc_signal[index] += nc_density[place] / fragment_bias
                        if 500 <= index <= roi_length - 501:  # no buffer for frag lengths
                            fragment_length_profile[index - 500].append(abs(fragment_length))
        if strand == "-":
            depth = depth[::-1]
            fragment_length_profile = fragment_length_profile[::-1]
            nc_signal = nc_signal[::-1]
    # generate phased profile and phasing/region-level features --------------------------------------------------------
    mean_depth = np.mean(depth[500:-500])
    if mean_depth < 1.0:
        return site, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,\
               np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    fourier = rfft(np.asarray(nc_signal))
    freqs = rfftfreq(roi_length)
    range_1 = [idx for idx, val in enumerate(freqs) if low_1 <= val <= high_1]
    range_2 = [idx for idx, val in enumerate(freqs) if low_2 <= val <= high_2]
    primary_amp_1 = np.mean(np.abs(fourier[range_1])) / roi_length
    primary_amp_2 = np.mean(np.abs(fourier[range_2])) / roi_length
    if primary_amp_1 > 0 and primary_amp_2 > 0:
        np_score = primary_amp_2 / primary_amp_1
    else:
        np_score = np.nan
    test_freqs = [idx for idx, val in enumerate(freqs) if 0 < val <= freq_max]  # frequencies in filter
    clean_fft = [0] * (len(fourier) + 1)
    try:
        clean_fft[test_freqs[0]:test_freqs[-1]] = fourier[test_freqs[0]:test_freqs[-1]]
        inverse_signal = irfft(clean_fft, n=roi_length)  # reconstruct signal
        phased_signal = inverse_signal + len(inverse_signal) * [np.mean(nc_signal[500:-500])]  # add in base component
    except IndexError:  # not enough data to construct a reasonable Fourier
        phased_signal = [0] * roi_length
    # remove the 500 bp buffer from each side of the signal
    depth = depth[500:-500]
    phased_signal = phased_signal[500:-500]
    nc_signal = nc_signal[500:-500]
    signal_mean = np.mean(phased_signal)
    var_range = max(phased_signal) - min(phased_signal)
    var_rat = var_range / min(phased_signal)
    max_values, peaks, min_values, troughs = local_peaks(phased_signal, nc_signal)
    peak_profile = [0] * window
    for peak_index in range(window):
        if peak_index in peaks:
            peak_profile[peak_index] = 1
        elif peak_index in troughs:
            peak_profile[peak_index] = -1
    if window is not None:
        # assume minimum, then double-check
        inflection = min(phased_signal[int(window/2 - 73):int(window/2 + 73)])
        inflection_loc = np.where(phased_signal == inflection)
        if len(inflection_loc) > 1:
            inflection_loc = np.absolute(inflection_loc - window/2).argmin()[0]  # closest inflection to 0
        else:
            inflection_loc = inflection_loc[0][0]
        if phased_signal[inflection_loc - 3] < inflection or phased_signal[inflection_loc + 3] < inflection:  # not min
            inflection = max(phased_signal[int(window/2 - 73):int(window/2 + 73)])
            inflection_loc = np.where(phased_signal == inflection)
            if len(inflection_loc) > 1:
                inflection_loc = np.absolute(inflection_loc - window / 2).argmax()[0]
            else:
                inflection_loc = inflection_loc[0][0]
            if phased_signal[inflection_loc - 3] > inflection or phased_signal[inflection_loc + 3] > inflection:
                # saddle: inconclusive
                inflection_loc = np.nan
        if not np.isnan(inflection_loc):
            peak_profile[inflection_loc] = 3
            cd_mean = inflection / signal_mean
            left_max_loc, right_max_loc = nearest_peaks(inflection_loc, peaks)
            if not np.isnan(left_max_loc) and not np.isnan(right_max_loc):
                peak_profile[left_max_loc] = -2
                peak_profile[right_max_loc] = 2
                left_max, right_max = phased_signal[left_max_loc], phased_signal[right_max_loc]
                flank_diff = right_max / left_max
                minus_one_loc, plus_one_loc = left_max_loc - inflection_loc, right_max_loc - inflection_loc
            else:
                flank_diff, minus_one_loc, plus_one_loc = np.nan, np.nan, np.nan
        else:
            cd_mean, flank_diff, minus_one_loc, plus_one_loc, inflection_het = np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        cd_mean, flank_diff, minus_one_loc, plus_one_loc, inflection_loc, inflection_het =\
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if len(max_values) < 1:
        np_period = np.nan
        np_amp = np.nan
    else:
        np_period = (peaks[-1] - peaks[0]) / len(peaks)
        np_amp = (np.mean(max_values) - np.mean(min_values)) / signal_mean
    # generate fragment profiles and fragmentation features ------------------------------------------------------------
    frag_mean = np.mean(roi_fragment_lengths)
    frag_std = np.std(roi_fragment_lengths)
    frag_mad = np.median(np.absolute(roi_fragment_lengths - np.median(roi_fragment_lengths)))
    frag_rat = frag_ratio(roi_fragment_lengths)
    frag_ent = dirichlet_normalized_entropy(roi_fragment_lengths, roi_fragment_lengths)
    # fragment profiles
    shan_profile = point_entropy(roi_fragment_lengths, fragment_length_profile)
    ratio_profile = [frag_ratio(point_frags) for point_frags in fragment_length_profile]
    mf_profile = [np.mean(point_frags) if len(point_frags) > 0 else np.nan for point_frags in fragment_length_profile]
    mad_profile = [np.median(np.absolute(point_frags - np.median(point_frags)))
                   for point_frags in fragment_length_profile]
    hetero_profile = [len(set(point_frags)) / len(point_frags) if len(point_frags) != 0 else 0 for point_frags in
                      fragment_length_profile]
    inflection_het = np.mean(hetero_profile[(inflection_loc - 5):(inflection_loc + 5)]) / np.mean(hetero_profile)
    # sequence profile
    seq_profile = np.delete(oh_seq, 0, 1)  # remove the N row
    # combine and save profiles ----------------------------------------------------------------------------------------
    signal_array = np.column_stack((depth, phased_signal, nc_signal, mf_profile, shan_profile,
                                    hetero_profile, mad_profile, ratio_profile, peak_profile))
    out_array = np.concatenate((signal_array, seq_profile), axis=1)
    return site, frag_mean, frag_std, frag_mad, frag_rat, frag_ent, np_score, np_period, np_amp, mean_depth, var_rat,\
           cd_mean, flank_diff, inflection_loc - int(window/2), inflection_het, plus_one_loc, minus_one_loc, out_array


def main():
    """
    Takes in input parameters for a single sample Triton run, evaluates the site list format, manages breaking up
    sites into chunks to be run in parallel with generate_profile(), then combines outputs from each core and saves
    them to two files: sample_name_TritonFeatures.tsv and sample_name_TritonProfiles.npz containing region-level
    features and signal profiles, respectively.
    """
    def str2bool(v):
        """
        Stand-in type to read boolean values from the command line
            Parameters:
                v (string): input bool as string
            Returns:
                boolean
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # parse command line arguments:
    parser = argparse.ArgumentParser(description='\n### Triton.py ### main Triton pipeline')
    parser.add_argument('-n', '--sample_name', help='sample identifier', required=True)
    parser.add_argument('-i', '--input', help='input BAM file', required=True)
    parser.add_argument('-b', '--bias', help='input-matched GC bias file (from Griffin)', default=None)
    parser.add_argument('-a', '--annotation', help='regions of interest as BED or list of BEDs', required=True)
    parser.add_argument('-g', '--reference_genome', help='reference genome (.fa)', required=True)
    parser.add_argument('-r', '--results_dir', help='directory for results', required=True)
    parser.add_argument('-q', '--map_quality', help='minimum mapping quality; default=20', type=int, default=20)
    parser.add_argument('-f', '--size_range', help='fragment size range (bp); default=(15, 500)', nargs=2, type=int,
                        default=(15, 500))
    parser.add_argument('-c', '--cpus', help='number of CPUs to use for parallel processing of individual regions or'
                                             'composite regions, if -s True', type=int, required=True)
    parser.add_argument('-w', '--window', help='window size (bp) for composite sites; default=2000',
                        type=int, default=2000)
    parser.add_argument('-s', '--composite', help='whether to run in composite-site mode', type=str2bool, default=False)
    parser.add_argument('-d', '--frag_dict', help='dictionary of probable nucleosome center-frag displacements,'
                                                  'default=../nc_info/NCDict.pkl',
                        required=False, default='/fh/fast/ha_g/user/rpatton/scripts/Triton/nc_info/NCDict.pkl')
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
    dict_path = args.frag_dict

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
    print('\tfrag_dict = ', dict_path)
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
        header = [site for site in open(sites[0].strip(), 'r')].pop(0).strip().split('\t')  # retrieve one header
    else:
        print('Running Triton in individual mode.')
        sites = [region for region in open(sites_path, 'r')]  # a list of regions
        header = sites.pop(0).strip().split('\t')
    # Below checks for standard BED column names and the position column if window=True, updating their indices
    # if necessary. If a non-standard header format is used default indices will be used, which may error.
    # In composite mode, each individual BED-like file is assumed to have an identical header/header sorting;
    # will error if this is not the case.
    if 'chrom' in header:
        chr_idx = header.index('chrom')
    elif 'Chrom' in header:
        chr_idx = header.index('Chrom')
    else:
        print('No "chrom" column found in BED file(s): defaulting to index 0 (' + header[0] + ')')
    if 'chromStart' in header:
        start_idx = header.index('chromStart')
    elif 'start' in header:
        start_idx = header.index('start')
    elif 'Start' in header:
        start_idx = header.index('Start')
    else:
        print('No "chromStart" column found in BED file(s): defaulting to index 1 (' + header[1] + ')')
    if 'chromEnd' in header:
        stop_idx = header.index('chromEnd')
    elif 'end' in header:
        stop_idx = header.index('end')
    elif 'End' in header:
        stop_idx = header.index('End')
    else:
        print('No "chromEnd" column found in BED file(s): defaulting to index 2 (' + header[2] + ')')
    if 'name' in header:
        site_idx = header.index('name')
    elif 'Gene' in header:
        site_idx = header.index('Gene')
    else:
        print('No "name" column found in BED file(s): defaulting to index 3 (' + header[3] + ')\n'
              '* if running in composite mode site names will be taken from the annotation name instead')
    if 'strand' in header:
        strand_idx = header.index('strand')
    else:
        print('WARNING: No "strand" column found in BED file(s): defaulting to "+" for all sites')
        strand_idx = None
    if stack and 'position' in header:
        pos_idx = header.index('position')
    elif stack:
        print('No "position" column found in BED file(s): defaulting to index 6 (' + header[6] + ')')
    random.shuffle(sites)  # to spread sites evenly between cores given variable size

    with open(dict_path, 'rb') as f:
        frag_dict = pickle.load(f)

    params = [bam_path, results_dir, size_range, gc_bias, ref_seq_path, map_q, window, stack, frag_dict]

    print('\n### Running Triton on ' + str(len(sites)) + ' region sets ###\n')
    if len(sites) < cpus:  # more cores than we need - use at most 1 per site
        print('More CPUs passed than sites - scaling down to ' + str(len(sites)))
        cpus = len(sites)

    with Pool(cpus) as pool:
        results = list(pool.imap_unordered(partial(generate_profile, params=params), sites, len(sites) // cpus))

    print('Merging and saving results . . .')
    signal_dict = {}
    fm = {sample_name: {'sample': sample_name}}
    for result in results:
        fm[sample_name][result[0] + '_fragment-mean'] = result[1]
        fm[sample_name][result[0] + '_fragment-stdev'] = result[2]
        fm[sample_name][result[0] + '_fragment-mad'] = result[3]
        fm[sample_name][result[0] + '_fragment-ratio'] = result[4]
        fm[sample_name][result[0] + '_fragment-entropy'] = result[5]
        fm[sample_name][result[0] + '_np-score'] = result[6]
        fm[sample_name][result[0] + '_np-period'] = result[7]
        fm[sample_name][result[0] + '_np-amplitude'] = result[8]
        fm[sample_name][result[0] + '_mean-depth'] = result[9]
        fm[sample_name][result[0] + '_var-ratio'] = result[10]
        fm[sample_name][result[0] + '_central-depth'] = result[11]
        fm[sample_name][result[0] + '_plus-minus-ratio'] = result[12]
        fm[sample_name][result[0] + '_central-loc'] = result[13]
        fm[sample_name][result[0] + '_central-heterogeneity'] = result[14]
        fm[sample_name][result[0] + '_plus-one-pos'] = result[15]
        fm[sample_name][result[0] + '_minus-one-pos'] = result[16]
        signal_dict[result[0]] = result[17]
    df = pd.DataFrame(fm)
    out_file = results_dir + '/' + sample_name + '_TritonFeatures.tsv'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    np.savez_compressed(results_dir + '/' + sample_name + '_TritonProfiles', **signal_dict)
    df.to_csv(out_file, sep='\t')

    print('Finished')


if __name__ == "__main__":
    main()
