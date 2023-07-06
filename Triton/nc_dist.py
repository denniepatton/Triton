# Robert Patton, rpatton@fredhutch.org
# v0.2.2, 06/29/2023

# this is a modified version of Triton, designed to output only information about where nucleosomes are located
# downstream analyses of outputs are used in nc_analyze.py

import os
import sys
import pysam
import random
import argparse
from functools import partial
from multiprocessing import Pool
from triton_helpers import *

chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx = 0, 1, 2, 3, 5, 6  # default BED indices


def generate_profile(region, params):
    """
    Generates composite signal profiles and extracts the distance from each fragment center to the composite region
    center. Utilizes functions from triton_helpers.py.
        Parameters:
            region (string): either a file path pointing to a BED-like file (composite) or a BED-like line
            params (list): bam_path, out_direct, frag_range, gc_bias, ref_seq_path, map_q, window, stack
        Returns:
            site: annotation name
            numpy array: 2D array of counts with (500 - fragment length) indexed rows, and displacement from region
                center as columns (includes negative displacements)
    """
    bam_path, _, frag_range, _, _, map_q, window, _ = params
    bam = pysam.AlignmentFile(bam_path, 'rb')
    site = os.path.basename(region).split('.')[0]  # use the file name as the feature name
    frag_cents = np.zeros((500, 1000))
    with open(region.strip(), 'r') as sites_file:
        for entry in sites_file:  # iterate through regions in this particular BED file
            bed_tokens = entry.strip().split('\t')
            if not bed_tokens[pos_idx].isdigit():  # header or typo line
                continue
            chrom, center_pos = str(bed_tokens[chr_idx]), int(bed_tokens[pos_idx])
            start_pos = center_pos - int(window/2)
            stop_pos = center_pos + int(window/2)
            # process all fragments falling inside the ROI
            if start_pos < 0: continue
            segment_reads = bam.fetch(chrom, start_pos, stop_pos)
            for read in segment_reads:
                fragment_length = read.template_length
                if frag_range[0] <= np.abs(fragment_length) <= frag_range[1] and read.is_paired and read.is_read1 and \
                        read.mapping_quality >= map_q and not read.is_duplicate and not read.is_qcfail:
                    read_start = read.reference_start
                    if read.is_reverse and fragment_length < 0:
                        read_length = read.reference_length
                        fragment_start = read_start + read_length + fragment_length
                        fragment_end = read_start + read_length
                    elif not read.is_reverse and fragment_length > 0:
                        fragment_start = read_start
                        fragment_end = read_start + fragment_length
                    else:
                        continue
                    if fragment_start <= center_pos <= fragment_end or fragment_start >= center_pos >= fragment_end:
                        frag_center = fragment_start + int((fragment_end - fragment_start) / 2)
                        center_disp = center_pos - frag_center
                        frag_cents[500 - np.abs(fragment_length), 500 - center_disp] += 1
    return site, frag_cents


def main():
    """
    Takes in input parameters for a single sample Triton run, evaluates the site list format, manages breaking up
    sites into chunks to be run in parallel with generate_profile(), then combines outputs from each core and saves
    sample_name_TritonNucPlacementProfiles.npz containing region-level displacement arrays.
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
    parser.add_argument('-q', '--map_quality', help='minimum mapping quality (default=20)', type=int, default=20)
    parser.add_argument('-f', '--size_range', help='fragment size range (bp); default=(15, 500)', nargs=2, type=int,
                        default=(15, 500))
    parser.add_argument('-c', '--cpus', help='number of CPUs to use for parallel processing', type=int, required=True)
    parser.add_argument('-w', '--window', help='window size (bp) for composite sites', type=int, default=None)
    parser.add_argument('-s', '--composite', help='whether to run in composite-site mode', type=str2bool, default=False)
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

    gc_bias = None  # not actually used, but kept in parameters for continuity
    if not stack:  # must use stacked mode
        stack = True

    global chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx
    if stack and window is None:
        print('ERROR: if using Triton in composite mode a window (-w) must be specified. Exiting.')
        header, sites = None, None
        exit()
    else:
        print('Running Triton in composite mode.')
        sites = [region for region in open(sites_path, 'r')]  # a list of BED-like paths
        header = [site for site in open(sites[0].strip(), 'r')].pop(0).strip().split('\t')  # retrieve one header

    # Below checks for standard BED column names and the position column if window=True, updating their indices
    # if necessary. If a non-standard header format is used defaults indices will be used, which may error.
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
    params = [bam_path, results_dir, size_range, gc_bias, ref_seq_path, map_q, window, stack]

    print('\n### Running Triton on ' + str(len(sites)) + ' region sets ###\n')
    if len(sites) < cpus:  # more cores than we need - use at most 1 per site
        print('More CPUs passed than sites - scaling down to ' + str(len(sites)))
        cpus = len(sites)

    with Pool(cpus) as pool:
        results = list(pool.imap_unordered(partial(generate_profile, params=params), sites, len(sites) // cpus))

    print('Merging and saving results . . .')
    signal_dict = {}
    for result in results:
        signal_dict[result[0]] = result[1]
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    np.savez_compressed(results_dir + '/' + sample_name + '_TritonNucPlacementProfiles', **signal_dict)
    print('Finished')


if __name__ == "__main__":
    main()
