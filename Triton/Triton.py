# Robert Patton, rpatton@fredhutch.org
# v0.3.1, 04/23/2024

import os
import sys
import pysam
import pickle
import random
import argparse
import multiprocessing
from functools import partial
from scipy.sparse import csr_matrix
from scipy.fft import rfft, irfft, rfftfreq
from triton_helpers import *

# constants and defaults
iupac_trans = str.maketrans('ACGTURYSWKMBDHVNX',
                            '06600336033422430') # likelihood of GC content for each bp code, times 6 (common factor)
bin_range = np.array(range(15, 500, 5))  # hardcoded for dict-based analyses of range 15-500 > bins for Shannon Entropy
freq_max = 1.0 / 146.0  # theoretical minimum nucleosome period is 146 bp -> f = 0.0068493
low_1, high_1 = 0.00556, 0.00667  # range_1: T = 150-180 bp (f = 0.00556 - 0.00667, "small linker")
low_2, high_2 = 0.00476, 0.00555  # range_2: T = 180-210 bp (f = 0.00476 - 0.00555, "large linker")
background_dict = None  # global variable for background panel subtraction


def generate_profile(region, params):
    """
    Generates single or composite signal profiles and extracts features (fragmentation, nucleosome phasing, and profile
    shape-based) for a single sample. Utilizes functions from triton_helpers.py.
        Parameters:
            region (string): either a file path pointing to a BED-like file (composite) or a BED-like single annotation
            params (list): bam_path, frag_range, gc_bias, ref_seq_path, map_q, fdict, run_mode, chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx, subtract_panel, tfx, generate_panel
        Returns:
            site: annotation name if stacked, "name" from BED file for each region otherwise
                ### Fragmentation Features (using all fragments in passed range/bounds) ###
            fragment-mean: fragment lengths' mean
            fragment-stdev: fragment lengths' standard deviation
            fragment-median: fragment lengths' median
            fragment-mad: fragment lengths' MAD (Median Absolute Deviation)
            fragment-ratio: fragment lengths' short:long ratio (x <= 150 / x > 150)
            fragment-diversity: fragment lengths' diversity (unique fragment lengths / total fragments)
            fragment-entropy: fragment lengths' Shannon entropy
                ### Phasing Features (FFT-based, using >= 146bp fragments) ###
            np-score: Nucleosome Phasing score
            np-period: phased-nucleosome period AKA inter nucleosomeal distance
            np-amplitude: phased-nucleosome mean amplitude
                ### Profiling Features (Filtered signal-based, using >= 146bp fragments and local peak calling) ###
            mean-depth: mean depth in the region (GC-corrected)
            var-ratio: ratio of variation in total phased signal (max signal range / max signal height)
            plus-one-pos*: location relative to central-loc of the plus-one nucleosome peak
            minus-one-pos*: location relative to central-loc of the minus-one nucleosome peak
            plus-minus-ratio*: ratio of the height of the +1 nucleosome to -1 nucleosome relative to central inflection
            central-loc*: location of central inflection relative to window center (0)
            central-depth*: phased signal value at the central-loc (with mean in flanking regions set to 1)
            central-diversity*: normalized fragment diversity value in the +/-5 bp region around the central-loc (with mean in flanking regions set to 1)
                ### Region-level profiles (all are un-normalized/raw except for entropy; nt-resolution) ###
            numpy array: shape 11(15)xN containing:
                1: Depth (GC-corrected)
                2: Fragment end-point coverage
                3: Phased-nucleosome profile (Fourier filtered probable nucleosome center profile, GC-corrected)
                4: Fragment lengths' short:long ratio (x <= 150 / x > 150)
                5: Fragment lengths' diversity (unique fragment lengths / total fragments)
                6: Fragment lengths' Shannon Entropy (normalized by window Shannon Entropy)
                7: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***
                8: A (Adenine) frequency**
                9: C (Cytosine) frequency**
                10: G (Guanine) frequency**
                11: T (Tyrosine) frequency**
            skipped-sites: list of annotation lines skipped due to outlier peaks with MAD > 10 (composite only)
            * these features are centered region specific, and output as np.nan if window == None
            ** sequence report is based on the reference sequence and does not include SNVs, etc.
            *** minus-one, plus-one, and inflection locs are only called if window != None, and supersede peak/trough;
                if the inflection loc cannot be determined it will default to 0
    """
    bam_path, frag_range, gc_bias, ref_seq_path, map_q, fdict, run_mode, chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx, subtract_panel, tfx, generate_panel = params
    if run_mode == 'composite-window':
        window, stack = 2000, True
        bias_limit = 0.05
    elif run_mode == 'window':
        window, stack = 2000, False
        bias_limit = 0.05
    else:
        window, stack = None, False
        bias_limit = 0.10  # use a stricter GC-bias range for composite regions where depth is higher
    bam = pysam.AlignmentFile(bam_path, 'rb')
    ref_seq = pysam.FastaFile(ref_seq_path)
    fragment_lengths = np.zeros(frag_range[1] + 1, dtype=int)  # empty fragment length *histogram*
    skipped_sites = []  # for storing skipped sites in composite-window mode

    if stack:
    # assemble depth and fragment profiles for composite sites ---------------------------------------------------------
        site = os.path.basename(region).split('.')[0]  # use the file name as the feature name
        roi_length = window + 1000  # 500 bp buffers are added to each end for a smooth FFT at the ROI edges
        depth, nc_signal = np.zeros(roi_length), np.zeros(roi_length)
        oh_seq = np.zeros((window, 5))  # bp-frequency-encoded sequence array
        fragment_length_profile = np.zeros((frag_range[1] + 1, window), dtype=int)  # nt-level fragment histogram matrix
        fragment_end_profile = np.zeros(window)  # nt-level fragment end-points
        with open(region.strip(), 'r') as sites_file:
            for entry in sites_file:  # iterate through regions in this particular BED file
                site_depth, site_nc_signal = np.zeros(roi_length), np.zeros(roi_length)
                site_fragment_ends = np.zeros(window)
                site_fragment_lengths = np.zeros(frag_range[1] + 1, dtype=int)
                site_fragment_length_profile = np.zeros((frag_range[1] + 1, window), dtype=int)
                bed_tokens = entry.strip().split('\t')
                if not bed_tokens[pos_idx].isdigit(): continue  # header or typo line
                chrom, center_pos = str(bed_tokens[chr_idx]), int(bed_tokens[pos_idx])
                if strand_idx is not None:
                    strand = str(bed_tokens[strand_idx])
                else:  # choose +/- randomly if strand is not specified
                    """ N.B. that most TFs bind structurally, e.g. to the major groove of DNA, and/or to palindromic sequences.
                    There is a similar ambiguity for, e.g. histone modification sites. Thus, if strand context is unknown,
                    it would be implying a directionality which is not necessarily present to assume any particular strand.
                    So here strand is chosen randomly for each site, to avoid any unintended bias and double counting."""
                    strand = random.choice(['+', '-'])
                if chrom == 'chrMT':  # in future, change to "if not in list of chromosomes"
                    skipped_sites.append(entry.strip() + '\tMT_contig')
                    continue
                start_pos = center_pos - int(roi_length / 2)
                stop_pos = center_pos + int(roi_length / 2)
                # process all fragments falling inside the ROI
                if start_pos < 0:  # exclude windows on boundaries which do not fully overlap
                    skipped_sites.append(entry.strip() + '\twindow_out_of_range')
                    continue
                site_reads = bam.fetch(chrom, start_pos, stop_pos)
                window_sequence = ref_seq.fetch(chrom, start_pos, stop_pos).upper()
                if len(window_sequence) != window + 1000:  # truncated reference region: skip
                    skipped_sites.append(entry.strip() + '\ttruncated_reference')
                    continue
                for read in site_reads:
                    if not read.is_read1: continue  # only consider read 1 to avoid double counting fragments
                    fragment_length = read.template_length
                    abs_length = abs(fragment_length)
                    # fragment QC filtering
                    if frag_range[0] <= abs_length <= frag_range[1] and read.is_paired and \
                            read.mapping_quality >= map_q and not read.is_duplicate and not read.is_qcfail:
                        read_start = read.reference_start - start_pos
                        if read.is_reverse and fragment_length < 0:  # read1 is reverse
                            read_length = read.reference_length
                            fragment_start = read_start + read_length + fragment_length
                            fragment_end = read_start + read_length
                        elif not read.is_reverse and fragment_length > 0:  # read2 is not reverse
                            fragment_start = read_start
                            fragment_end = read_start + fragment_length
                        else: continue  # read2 is upstream of read1, likely an artifact or SV
                        fragment_seq = window_sequence[fragment_start:fragment_end].translate(iupac_trans)
                        fragment_gc_content = (np.array(list(fragment_seq), dtype=int).sum() / 6).round()
                        fragment_bias = gc_bias[abs_length][fragment_gc_content]
                        fragment_cov = np.arange(fragment_start, fragment_end)
                        center_in_window = 500 <= np.take(fragment_cov, abs_length // 2) <= roi_length - 501
                        if center_in_window:
                            site_fragment_lengths[abs_length] += 1
                        nc_density = fdict[abs_length] if 146 <= abs_length <= 500 else None
                        if bias_limit < fragment_bias < 10:
                            if strand == '+':  # positive strand:
                                for place, index in enumerate(fragment_cov):
                                    if not 0 <= index < roi_length: continue
                                    site_depth[index] += 1 / fragment_bias
                                    if nc_density is not None:
                                        site_nc_signal[index] += nc_density[place] / fragment_bias
                                    if 500 <= index <= roi_length - 501:
                                        site_fragment_length_profile[abs_length, index - 500] += 1
                                        if place == 0 or place == len(fragment_cov) - 1:
                                            site_fragment_ends[index - 500] += 1
                            else:  # negative strand:
                                for place, index in enumerate(fragment_cov):
                                    if not 0 <= index < roi_length: continue
                                    site_depth[roi_length - index -1] += 1 / fragment_bias
                                    if nc_density is not None:
                                        site_nc_signal[roi_length - index -1] += nc_density[place] / fragment_bias
                                    if 500 <= index <= roi_length - 501:  # no buffer for frag lengths
                                        site_fragment_length_profile[abs_length, roi_length - index - 501] += 1
                                        if place == 0 or place == len(fragment_cov) - 1:
                                            site_fragment_ends[roi_length - index - 501] += 1
                # check for egregious outliers in site, and drop site if found
                if np.sum(site_depth) == 0:
                    skipped_sites.append(entry.strip() + '\tzero_coverage')
                    continue
                raw_median = np.median(site_depth)
                raw_mad = np.median(np.abs(site_depth - raw_median))  # this IS median depth if raw_median == 0
                if raw_median > 0:  # Only perform extended site filtering for sites with minimum coverage so as to not exclude ULP sites
                    if raw_mad == 0:
                        skipped_sites.append(entry.strip() + '\tzero_MAD')
                        continue
                    raw_mads_from_median = (site_depth - raw_median) / raw_mad
                    if any(x > 10 for x in raw_mads_from_median):  # consider positive outliers only
                        skipped_sites.append(entry.strip() + '\tMAD_outlier')
                        continue
                if strand == '+':
                    oh_seq = np.add(oh_seq, one_hot_encode(window_sequence[500:-500]))
                else:
                    oh_seq = np.add(oh_seq, one_hot_encode(window_sequence[::-1][500:-500]))
                depth += site_depth
                nc_signal += site_nc_signal
                fragment_length_profile += site_fragment_length_profile
                fragment_end_profile += site_fragment_ends
                fragment_lengths += site_fragment_lengths
        oh_seq = oh_seq/oh_seq.sum(axis=1, keepdims=True)

    else:
    # assemble depth and fragment profiles for sites individually -------------------------------------------------------
        bed_tokens = region.strip().split('\t')
        chrom, site = str(bed_tokens[chr_idx]), str(bed_tokens[site_idx])
        if strand_idx is not None:
            strand = str(bed_tokens[strand_idx])
        else:
            strand = '+'
        if window is not None:
            roi_length = window + 1000
            center_pos = int(bed_tokens[pos_idx])
            start_pos = center_pos - int(roi_length / 2)
            stop_pos = center_pos + int(roi_length / 2)
            fragment_length_profile = np.zeros((frag_range[1] + 1, window), dtype=int)
            fragment_end_profile = np.zeros(window)
        else:
            start_pos = int(bed_tokens[start_idx]) - 500
            stop_pos = int(bed_tokens[stop_idx]) + 500
            roi_length = stop_pos - start_pos
            fragment_length_profile = np.zeros((frag_range[1] + 1, roi_length - 1000), dtype=int)
            fragment_end_profile = np.zeros(roi_length - 1000)
        depth, nc_signal = np.zeros(roi_length), np.zeros(roi_length)
        # process all fragments falling inside the site
        site_reads = bam.fetch(bed_tokens[0], start_pos, stop_pos)
        window_sequence = ref_seq.fetch(bed_tokens[0], start_pos, stop_pos).upper()
        if strand == '+': oh_seq = one_hot_encode(window_sequence[500:-500])
        else: oh_seq = one_hot_encode(window_sequence[::-1][500:-500])
        for read in site_reads:
            if not read.is_read1: continue # only consider read 1 to avoid double counting fragments
            fragment_length = read.template_length  # Consider this "True" and recall pysam is 0-based (like Python)
            abs_length = abs(fragment_length)
            if frag_range[0] <= abs_length <= frag_range[1] and read.is_proper_pair and \
                    read.mapping_quality >= map_q and not read.is_duplicate and not read.is_qcfail:
                read_start = read.reference_start - start_pos  # relative to the start of the ROI
                if read.is_reverse and fragment_length < 0:  # read1 is reverse 
                    read_length = read.reference_length
                    fragment_start = read_start + read_length + fragment_length
                    fragment_end = read_start + read_length
                elif not read.is_reverse and fragment_length > 0: # read1 is not reverse
                    fragment_start = read_start
                    fragment_end = read_start + fragment_length
                else: continue  # read2 is upstream of read1, likely an artifact or SV
                # Note that fragment_end - fragment_start = abs_length
                fragment_seq = window_sequence[fragment_start:fragment_end].translate(iupac_trans)
                fragment_gc_content = (np.array(list(fragment_seq), dtype=int).sum() / 6).round()
                fragment_bias = gc_bias[abs_length][fragment_gc_content]
                fragment_cov = np.arange(fragment_start, fragment_end)
                center_in_window = 500 <= np.take(fragment_cov, abs_length // 2) <= roi_length - 501
                if center_in_window:
                    fragment_lengths[abs_length] += 1
                nc_density = fdict[abs_length] if 146 <= abs_length <= 500 else None
                if bias_limit < fragment_bias < 10:
                    for place, index in enumerate(fragment_cov):
                        if not 0 <= index < roi_length:
                            continue
                        depth[index] += 1 / fragment_bias
                        if nc_density is not None:
                            nc_signal[index] += nc_density[place] / fragment_bias
                        if 500 <= index <= roi_length - 501:
                            fragment_length_profile[abs_length, index - 500] += 1
                            if place == 0 or place == len(fragment_cov) - 1:
                                fragment_end_profile[index - 500] += 1
        if strand == "-":
            depth = depth[::-1]
            nc_signal = nc_signal[::-1]
            fragment_end_profile = fragment_end_profile[::-1]
            fragment_length_profile = np.fliplr(fragment_length_profile)

    # if generating a panel, stop here and return raw matrices ---------------------------------------------------------
    if generate_panel:
        """ These panels can become exeedingly large for regions, so special handling is used here. If panel generation
        is run in region mode, the fragment-signals are not returned - in other words, *signal* subtraction is only
        available in window modes, however, all scalar features are computable without the fragment-signals.
        So, subtraction for scalar features is still conducted.
        """
        if run_mode == 'region':
            return site, fragment_lengths, None, None, depth
        else:
            return site, fragment_lengths, csr_matrix(fragment_length_profile), fragment_end_profile, depth

    # generate phased profile and phasing/region-level features --------------------------------------------------------
    if np.count_nonzero(depth) < 0.8 * roi_length:  # skip sites with less than 80% base coverage (changed from 90%)
        return (site,) + (np.nan,) * 19 + ([],)
    
    # if subtracting a background panel, do that now
    if subtract_panel is not None and subtract_panel != "None":
        fragment_lengths, fragment_length_profile, fragment_end_profile, depth = subtract_background(fragment_lengths, fragment_length_profile, fragment_end_profile, depth, background_dict[site], tfx)
    
    mean_depth = np.mean(depth[500:-500])
    fourier = rfft(nc_signal)
    freqs = rfftfreq(roi_length)
    range_1 = np.where((freqs >= low_1) & (freqs <= high_1))
    range_2 = np.where((freqs >= low_2) & (freqs <= high_2))
    primary_amp_1 = np.mean(np.abs(fourier[range_1])) / roi_length
    primary_amp_2 = np.mean(np.abs(fourier[range_2])) / roi_length
    np_score = np.nan if primary_amp_1 == 0 or primary_amp_2 == 0 else primary_amp_2 / primary_amp_1
    drop_freqs = np.where(freqs > freq_max)  # low-pass filter
    fourier[drop_freqs] = 0  # this fourier transform is now scrubbed of high frequencies
    phased_signal = irfft(fourier, n=roi_length)  # convert from filtered-freq space back to signal space
    # remove the 500 bp buffer from each side of the signal
    depth = depth[500:-500]
    phased_signal = phased_signal[500:-500]
    nc_signal = nc_signal[500:-500]
    signal_mean = np.mean(phased_signal)
    var_range = np.max(phased_signal) - np.min(phased_signal)
    if signal_mean == 0 or var_range == 0:  # skip sites with no variation in phased signal
        return (site,) + (np.nan,) * 19 + ([],)
    var_rat = var_range / np.max(phased_signal)

    # perform peak calling --------------------------------------------------------------------------------------------
    max_values, peaks, min_values, troughs = local_peaks(phased_signal, nc_signal)
    peak_profile = np.zeros(roi_length - 1000, dtype=int)
    peak_profile[peaks] = 1
    peak_profile[troughs] = -1
    if window is not None:
        # first find the peak/trough nearest to the window's center
        inflection_loc = np.argmin(np.abs(np.concatenate([peaks, troughs]) - int(window / 2))) + int(window/2)
        inflection = phased_signal[inflection_loc] # value at the inflection location
        if not np.isnan(inflection_loc):
            peak_profile[inflection_loc] = 3
            center = len(phased_signal) // 2
            inflection_depth = inflection / np.mean(np.concatenate((phased_signal[:center-250], phased_signal[center+250:]))) # inflection depth relative to the mean of the flanking regions
            left_max_loc, right_max_loc = nearest_peaks(inflection_loc, peaks)
            if not np.isnan(left_max_loc) and not np.isnan(right_max_loc):
                peak_profile[[left_max_loc, right_max_loc]] = [-2, 2]
                left_max, right_max = phased_signal[[left_max_loc, right_max_loc]]
                if right_max > inflection and left_max > inflection:
                    flank_diff = (right_max - inflection) / (left_max - inflection)
                else:
                    flank_diff = np.nan  # only consider flank_diff if central minimum found
                minus_one_loc, plus_one_loc = left_max_loc - inflection_loc, right_max_loc - inflection_loc  # relative-to-inflection-loc coordinates
            else:
                flank_diff, minus_one_loc, plus_one_loc = np.nan, np.nan, np.nan
        else:
            inflection_depth, flank_diff, minus_one_loc, plus_one_loc = np.nan, np.nan, np.nan, np.nan
    else:
        inflection_depth, flank_diff, minus_one_loc, plus_one_loc, inflection_loc = np.nan, np.nan, np.nan, np.nan, np.nan
    if len(max_values) < 1 or len(peaks) < 2:
        np_period = np.nan
        np_amp = np.nan
    else:
        np_period = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
        np_amp = (np.mean(max_values) - np.mean(min_values)) / signal_mean

    # generate fragment profiles and fragmentation features ------------------------------------------------------------
    frag_mean, frag_std, frag_med, frag_mad, frag_rat, frag_div, frag_ent = frag_metrics(fragment_lengths, bin_range)
    # fragment profiles | 3 x len(window) where the 3 ordered are: ratio, diversity, entropy
    signal_metrics = np.apply_along_axis(frag_metrics, axis=0, arr=fragment_length_profile, bins=bin_range, reduce=True)
    inflection_div = np.nan
    if not np.isnan(inflection_loc):
        inflection_div = np.mean(signal_metrics[1, (inflection_loc - 5):(inflection_loc + 5)]) / np.mean(np.concatenate((signal_metrics[1, :center-250], signal_metrics[1, center+250:])))
        inflection_loc -= int(window/2)
    # sequence profile
    seq_profile = np.delete(oh_seq, 0, 1)  # remove the N row

    # combine and save profiles ----------------------------------------------------------------------------------------
    out_array = np.nan
    if window is not None and (roi_length - 1000) <= 5000:  # do not print profiles longer than 5kb to avoid memory issues
        signal_array = np.row_stack((depth, fragment_end_profile, phased_signal,  signal_metrics, peak_profile))
        out_array = np.concatenate((signal_array, seq_profile.T), axis=0)

    return site, frag_mean, frag_std, frag_med, frag_mad, frag_rat, frag_div, frag_ent, np_score, np_period, np_amp,\
           mean_depth, var_rat, plus_one_loc, minus_one_loc, flank_diff, inflection_loc, inflection_depth,\
           inflection_div, out_array, skipped_sites


def main():
    """
    Takes in input parameters for a single sample Triton run, evaluates the site list format, manages breaking up
    sites into chunks to be run in parallel with generate_profile(), then combines outputs from each core and saves
    them to two files: sample_name_TritonFeatures.tsv and sample_name_TritonProfiles.npz containing region-level
    features and signal profiles, respectively. In composite mode, an additional file sample_name_SkippedSites.bed
    will be output with skipped sites if sites were skipped.
    """

    # parse command line arguments:
    parser = argparse.ArgumentParser(description='\n### Triton.py ### main Triton pipeline')
    parser.add_argument('-n', '--sample_name', help='sample identifier', required=True)
    parser.add_argument('-i', '--input', help='input BAM file', required=True)
    parser.add_argument('-b', '--bias', help='input-matched GC bias file (Griffin formatted)', required=True)
    parser.add_argument('-a', '--annotation', help='regions of interest as a BED or list of BEDs', required=True)
    parser.add_argument('-g', '--reference_genome', help='reference genome (.fa)', required=True)
    parser.add_argument('-r', '--results_dir', help='directory for results', required=True)
    parser.add_argument('-m', '--run_mode', help='run mode ("region", "window", or "composite-window")', type=str, required=True)
    parser.add_argument('-q', '--map_quality', help='minimum mapping quality; default=20', type=int, default=20)
    parser.add_argument('-f', '--size_range', help='fragment size range (bp); default/bounds=(15, 500)', nargs=2, type=int, default=(15, 500))
    parser.add_argument('-c', '--cpus', help='number of CPUs to use for parallel processing; default=all', type=int, required=False)
    parser.add_argument('-d', '--frag_dict', help='dictionary of probable nucleosome center-frag displacements; default=nc_info/NCDict.pkl', required=False, default='nc_info/NCDict.pkl')
    parser.add_argument('-s', '--subtract_background_panel', help='path to an annotation-matched background panel for subtraction: see documentation', type=str, required=False)
    parser.add_argument('-t', '--tumor_fraction', help='tumor fraction, required if the -s (--subtract_background_panel) option is not None: see documentation', type=float, required=False)
    parser.add_argument('-p', '--generate_panel', help='run in background panel generation mode: see documentation', action='store_true', required=False)
    args = parser.parse_args()

    print('Loading input files . . .')
    sample_name, bam_path, bias_path, sites_path, ref_seq_path, results_dir, map_q, size_range, cpus, run_mode, dict_path, sub_panel, tfx, generate_panel = \
        args.sample_name, args.input, args.bias, args.annotation, args.reference_genome, args.results_dir, args.map_quality, args.size_range, args.cpus, args.run_mode, args.frag_dict, args.subtract_background_panel, args.tumor_fraction, args.generate_panel

    if cpus is None: # If no argument was passed to cpus, use all available CPUs
        cpus = multiprocessing.cpu_count()

    print(f'\n### arguments provided:\n'
          f'\tsample name = "{sample_name}"\n'
          f'\tinput bam path = "{bam_path}"\n'
          f'\tGC bias dict path = "{bias_path}"\n'
          f'\treference genome path = "{ref_seq_path}"\n'
          f'\tsite list = "{sites_path}"\n'
          f'\tresults directory = "{results_dir}"\n'
          f'\tfragment size range = {size_range}\n'
          f'\tminimum mapping quality = {map_q}\n'
          f'\tCPUs to use = {cpus}\n'
          f'\trun mode = {run_mode}\n'
          f'\tfragment re-weighting dictionary = {dict_path}\n'
          f'\tbackground panel = {sub_panel}\n'
          f'\ttumor fraction = {tfx}\n'
          f'\tgenerate background panel = {generate_panel}\n')
    sys.stdout.flush()

    gc_bias = get_gc_bias_dict(bias_path)
    sites = [region for region in open(sites_path, 'r')]
    frag_dict = pickle.load(open(dict_path, 'rb'))
    if run_mode in ['region', 'window']:
        header = sites.pop(0).strip().split('\t')
    elif run_mode == 'composite-window':
        with open(sites[0].strip(), 'r') as f:
            header = f.readline().strip().split('\t')
            print(header)
    else:
        raise ValueError(f"Invalid run_mode ({run_mode}): must be one of 'region', 'window', or 'composite-window'")
    
    if sub_panel is not None and sub_panel != "None":
        generate_panel = False  # just making sure
        print('Loading background panel - this may take a moment . . .')
        if not os.path.exists(sub_panel):
            raise FileNotFoundError(f"Background panel file not found at {sub_panel}, please double-check the provided path. Exiting.")
        elif tfx is None:
            raise ValueError("A tumor fraction (tumor purity) must be provided when subtracting a background panel. Exiting.")
        else:
            global background_dict
            background_dict = np.load(sub_panel, allow_pickle=True)['data'].item()
            print('Background panel loaded successfully.')
        

    chr_idx = get_index(header, ['chrom', 'Chrom'], 0, 'No "chrom" column found in BED file(s): defaulting to index {} ({})')
    start_idx = get_index(header, ['chromStart', 'start', 'Start'], 1, 'No "chromStart" column found in BED file(s): defaulting to index {} ({})')
    stop_idx = get_index(header, ['chromEnd', 'end', 'End'], 2, 'No "chromEnd" column found in BED file(s): defaulting to index {} ({})')
    site_idx = get_index(header, ['name', 'Gene'], 3, 'No "name" column found in BED file(s): defaulting to index {} ({})\n* if running in composite mode site names will be taken from the annotation name instead')
    strand_idx = get_index(header, ['strand', 'Strand'], None, 'WARNING: No "strand" column found in BED file(s): defaulting to "+" for all sites')
    pos_idx = get_index(header, ['position'], 6, 'No "position" column found in BED file(s): defaulting to index {} ({})') if run_mode != 'region' else None

    params = [bam_path, size_range, gc_bias, ref_seq_path, map_q, frag_dict, run_mode,
              chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx, sub_panel, tfx, generate_panel]

    print('\n### Running Triton on {} region sets ###\n'.format(len(sites)))

    if len(sites) < cpus:
        print('More CPUs passed than sites - scaling down to {}'.format(len(sites)))
        cpus = len(sites)

    with multiprocessing.Pool(cpus) as pool:
        """ N.B. that when running in panel generation mode, the results can get very large and cause errors when passing back through
        the multiprocessing pool. To avoid this, the batch size must be reduced, e.g. for panel generation of full-length transcripts
        use a batch size of ***10*** which has a slower overhead but shouldn't fail. """
        chunksize = max(10, len(sites) // (cpus * 100))
        results = list(pool.imap_unordered(partial(generate_profile, params=params), sites, chunksize))

    # below is for running NOT in parallel (testing mode)
    # results = []
    # for site in sites:
    #     results += [generate_profile(site, params)]

    if generate_panel:
        print('Finished generating sample-level background panel. Saving . . .')
        # Get the base name of sites_path without the extension
        annotation_name = os.path.splitext(os.path.basename(sites_path))[0]
        # Create a dictionary where the keys are the site IDs and the values are dictionaries containing the other data
        data_dict = {site: {'fragment_lengths': fragment_lengths, 'fragment_length_profile': fragment_length_profile, 'fragment_end_profile': fragment_end_profile, 'depth': depth} for site, fragment_lengths, fragment_length_profile, fragment_end_profile, depth in results}
        # Save the data in compressed format
        np.savez_compressed(os.path.join(results_dir, sample_name + '_TritonRawPanel_' + annotation_name), data=data_dict)
        print('\nFinished')
        quit()

    print('Merging and saving results . . .')
    signal_dict = {}
    fm = {sample_name: {'sample': sample_name}}
    all_skipped_sites = []

    # ORDERED output feature keys
    keys = ['fragment-mean',
            'fragment-stdev',
            'fragment-median',
            'fragment-mad',
            'fragment-ratio', 
            'fragment-diversity',
            'fragment-entropy',
            'np-score',
            'np-period',
            'np-amplitude', 
            'mean-depth',
            'var-ratio',
            'plus-one-pos',
            'minus-one-pos',
            'plus-minus-ratio', 
            'central-loc',
            'central-depth',
            'central-diversity']

    for result in results:
        for i, key in enumerate(keys):
            fm[sample_name][result[0] + '_' + key] = result[i+1]
        signal_dict[result[0]] = result[19]
        if len(result) == 21 and result[20]:  # skipped sites list exists and is not empty
            all_skipped_sites.extend([line + '\t' + result[0] for line in result[20]])

    df = pd.DataFrame(fm)
    out_file = os.path.join(results_dir, sample_name + '_TritonFeatures.tsv')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    np.savez_compressed(os.path.join(results_dir, sample_name + '_TritonProfiles'), **signal_dict)
    df.to_csv(out_file, sep='\t')

    if all_skipped_sites:  # (is not empty)
        cols = header + ['reject_reason', 'site_ID']
        skipped_df = pd.DataFrame([x.strip().split('\t') for x in all_skipped_sites], columns=cols)
        skipped_df = skipped_df.sort_values(['site_ID', header[chr_idx], header[start_idx]])
        skipped_file = os.path.join(results_dir, sample_name + '_SkippedSites.bed')
        skipped_df.to_csv(skipped_file, sep='\t', index=False)
        print(f'Sites were skipped due to outliers, and can be found in: {skipped_file}')
        print('Skipped sites summary statistics:')
        print('\nsite_ID\ttotal_skipped')
        print(skipped_df['site_ID'].value_counts().to_string())

    print('\nFinished')


if __name__ == "__main__":
    main()
