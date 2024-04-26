# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v0.0.3, 04/22/2024

import os
import glob
import argparse
import numpy as np


def load_file(file):
    print(f"Processing file: {file}")
    data = np.load(file, allow_pickle=True)['data'].item()
    return data


def main():
    """
    Combines multiple *_TritonRawPanel_*.npz files output from Triton.py in panel generation mode (-p)
    into a single site:panel background collection using all available samples. Will combine all
    available TritonRawPanel_*.npz files in the results directory and make a site-specific
    "annotation"_BackgroundPanel.npz, for use in Triton.py in panel subtraction mode.

    The input/output data formats are as follows:
    site (str): The site name.
    fragment_lengths (np.array): a numpy array / histogram of shape (501,) (index = fragment length, value = count) representing the whole site
    fragment_length_profile (scipy.sparse.csr_matrix): a sparse matrix of shape (501, window) where each column represents fragment_lengths at that position
    fragment_end_profile (np.array): a numpy array of shape (window,) where each value represents the number of fragment-ends at that position
    depth (np.array): a numpy array of shape (window,) where each value represents the number of reads at that position
    """
    parser = argparse.ArgumentParser(description='\n### triton_panel.py ### combines _TritonRawPanel_*.npz files')
    parser.add_argument('-r', '--results_dir', help='directory for results', required=True)
    args = parser.parse_args()

    files = glob.glob(args.results_dir + "/*_TritonRawPanel_*.npz")

    # Load the first file and initialize the combined data
    combined_data = load_file(files[0])

    # Process the remaining files
    for i, file in enumerate(files[1:], 1):
        print(f"Adding data from file {i+1} of {len(files)}")
        data = load_file(file)
        for site, site_data in data.items():
            for key, value in site_data.items():
                if combined_data[site][key] is not None:
                    combined_data[site][key] += value

    # Save the combined data to a .npz file
    annotation_name = os.path.basename(files[0]).split('_TritonRawPanel_')[1].split('.npz')[0]
    np.savez_compressed(os.path.join(args.results_dir, annotation_name + '_BackgroundPanel.npz'), data=combined_data)

if __name__ == "__main__":
    main()