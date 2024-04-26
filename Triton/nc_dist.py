# Robert Patton, rpatton@fredhutch.org
# v0.3.1, 04/04/2024

# this is a modified version of Triton, designed to output only information about where nucleosomes are located
# downstream analyses of outputs are used in nc_analyze.py, which generates a NCDict.pkl object for Triton

import os
import sys
import pysam
import argparse
from triton_helpers import *

chr_idx, start_idx, stop_idx, pos_idx = 0, 1, 2, 6  # default BED indices

def generate_profile(params):
    """
    Generates composite signal profiles and extracts the distance from each fragment center to the composite region
    center. Utilizes functions from triton_helpers.py.
        Parameters:
            params (list): bam_path, frag_range, map_q, chr_idx, pos_idx, sites
        Returns:
            numpy array: 2D array of counts with (500 - fragment length) indexed rows, and displacement from region
                center as columns (includes negative displacements)
    """
    bam_path, frag_range, map_q, chr_idx, pos_idx, sites = params

    bam = pysam.AlignmentFile(bam_path, 'rb')
    frag_cents = np.zeros((500, 1000))
    for entry in sites:  # iterate through regions in this particular BED file
        bed_tokens = entry.strip().split('\t')
        if not bed_tokens[pos_idx].isdigit():  # header or typo line
            continue
        chrom, center_pos = str(bed_tokens[chr_idx]), int(bed_tokens[pos_idx])
        start_pos = center_pos - int(500)
        stop_pos = center_pos + int(500)
        # process all fragments falling inside the ROI
        if start_pos < 0: continue
        site_reads = bam.fetch(chrom, start_pos, stop_pos)
        for read in site_reads:
            if not read.is_read1: continue  # only consider read 1 to avoid double counting fragments
            fragment_length = read.template_length
            abs_length = abs(fragment_length)
            # fragment QC filtering
            if frag_range[0] <= abs_length <= frag_range[1] and read.is_paired and \
                    read.mapping_quality >= map_q and not read.is_duplicate and not read.is_qcfail:
                read_start = read.reference_start
                if read.is_reverse and fragment_length < 0:  # read1 is reverse
                    read_length = read.reference_length
                    fragment_start = read_start + read_length + fragment_length
                    fragment_end = read_start + read_length
                elif not read.is_reverse and fragment_length > 0:  # read2 is not reverse
                    fragment_start = read_start
                    fragment_end = read_start + fragment_length
                else: continue  # read2 is upstream of read1, likely an artifact or SV
                # fill the center displacement array
                print(fragment_start, center_pos, fragment_end)
                if fragment_start <= center_pos <= fragment_end or fragment_start >= center_pos >= fragment_end:
                    frag_center = fragment_start + int((fragment_end - fragment_start) / 2)
                    center_disp = center_pos - frag_center
                    frag_cents[500 - np.abs(fragment_length), 500 - center_disp] += 1

    return frag_cents


def main():
    """
    Takes in input parameters for a single sample nc_dist run, evaluates the site list format, runs 
    generate_profile(), then saves sample_name_TritonNucPlacementProfiles.npz containing region-level displacement arrays.
    """

    # parse command line arguments:
    parser = argparse.ArgumentParser(description='\n### Triton.py ### main Triton pipeline')
    parser.add_argument('-n', '--sample_name', help='sample identifier', required=True)
    parser.add_argument('-i', '--input', help='input BAM file', required=True)
    parser.add_argument('-a', '--annotation', help='single BED file with nucleosome centerlocations ("position")', default='nc_info/hsNuc_iNPSPeak_bedops-intersect.bed')
    parser.add_argument('-r', '--results_dir', help='directory for results', default='nc_info/')
    parser.add_argument('-q', '--map_quality', help='minimum mapping quality; default=20', type=int, default=20)
    parser.add_argument('-f', '--size_range', help='fragment size range (bp); default/bounds=(15, 500)', nargs=2, type=int, default=(15, 500))
    args = parser.parse_args()

    print('Loading input files . . .')
    sample_name, bam_path, sites_path, results_dir, map_q, size_range = \
        args.sample_name, args.input, args.annotation, args.results_dir, args.map_quality, args.size_range

    print(f'\n### arguments provided:\n'
          f'\tsample name = "{sample_name}"\n'
          f'\tinput bam path = "{bam_path}"\n'
          f'\tsite list = "{sites_path}"\n'
          f'\tresults directory = "{results_dir}"\n'
          f'\tfragment size range = {size_range}\n'
          f'\tminimum mapping quality = {map_q}\n')
    sys.stdout.flush()

    sites = [region for region in open(sites_path, 'r')]
    header = sites.pop(0).strip().split('\t')
    chr_idx = get_index(header, ['chrom', 'Chrom'], 0, 'No "chrom" column found in BED file(s): defaulting to index {} ({})')
    pos_idx = get_index(header, ['position'], 6, 'No "position" column found in BED file(s): defaulting to index {} ({})')

    params = [bam_path, size_range, map_q, chr_idx, pos_idx, sites]

    print('\n### Running Triton on {} positions ###\n'.format(len(sites)))

    results = generate_profile(params)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    np.savez_compressed(results_dir + '/' + sample_name + '_TritonNucPlacementProfiles', results)
    print('Finished')


if __name__ == "__main__":
    main()
