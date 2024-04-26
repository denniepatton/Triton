# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v0.3.1, 04/04/2024

# utility for plotting nc_dist.py outputs and generating an NCDict.pkl object, required for Triton

import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


def smooth_data(row, sigma=3):
    """
    Apply Gaussian smoothing to a row of data.

    :param row: The input data row.
    :param sigma: The standard deviation for the Gaussian kernel.
    :return: The smoothed data row.
    """
    return gaussian_filter1d(row, sigma)


def symmetric_gauss_triple(x, mu, var, var_c, theta):
    """
    Calculate a normalized signal based on three Gaussian distributions and a constant background.

    :param x: The input array.
    :param mu: The mean of the flanking Gaussians.
    :param var: The variance of the flanking Gaussians.
    :param var_c: The variance of the central Gaussian.
    :param theta: The weight of the central Gaussian.
    :return: The normalized signal.
    """
    # Precompute constants for efficiency
    sqrt_2_var_pi = np.sqrt(2 * var * np.pi)
    sqrt_2_var_c_pi = np.sqrt(2 * var_c * np.pi)
    len_x = len(x)
    len_x_half = len_x / 2
    # Calculate the flanking Gaussians
    flanking_gaussians = (1 - theta) / sqrt_2_var_pi * (np.exp(-(x - mu) ** 2 / (2 * var)) +
                                                        np.exp(-(x - (len_x - mu)) ** 2 / (2 * var)))
    # Calculate the central Gaussian
    central_gaussian = theta / sqrt_2_var_c_pi * np.exp(-(x - len_x_half) ** 2 / (2 * var))
    # Combine the Gaussians and the constant background
    signal = flanking_gaussians + central_gaussian
    # Normalize the signal
    normalized_signal = signal / np.sum(signal)

    return normalized_signal


def fit_row(row, init_params):
    """
    Fit a row of data to a triple Gaussian function.

    :param row: The input data row.
    :param init_params: Initial parameters for the curve fitting.
    :return: The fitted function and the parameters of the fit.

    N.B. if the total spread of a Gaussina distirbtuion (98% of the area) is ~ 100bp,
    then the FWHM would be ~ 100 / 2.5 = 40bp

    Hypothetical variance for a given fragment length l, assuming 146b of actual wrapping:
    variance = (l - 146 + 1)^2 - 1/12
    """
    frag_length = len(row)
    low_var = (20 + 1)**2 - 1/12  # 20bp linker of wiggle room
    high_var = (50 + 1)**2 - 1/12  # 50bp linker of wiggle room
    x_data = np.arange(frag_length)
    y_data = row / np.sum(row)
    low_theta, high_theta = 0.0, 1.0
    # was 30
    bounds = ((1, low_var, low_var, low_theta), ((frag_length / 2) - 1, high_var, high_var, high_theta))

    if init_params is None:
        # init_params = [100, (norm_const * 80)**2, (norm_const * 80)**2, 0.3]
        init_params = [100, low_var, low_var, 0.3]
    else:
        # Adjust the bounds based on init_params
        lower_bounds = [max(lb, param * 0.95) for param, lb in zip(init_params, bounds[0])]
        upper_bounds = [min(ub, param * 1.05) for param, ub in zip(init_params, bounds[1])]
        
        # For the first parameter (mu), limit the change to 2bp in either direction
        lower_bounds[0] = max(bounds[0][0], init_params[0] - 2)
        upper_bounds[0] = min(bounds[1][0], init_params[0] + 2)
        
        bounds = (lower_bounds, upper_bounds)

    # Get the lower and upper bounds
    lower_bounds, upper_bounds = bounds
    # Check if init_params are within the bounds, and adjust if necessary
    init_params = [max(min(param, ub), lb) for param, lb, ub in zip(init_params, lower_bounds, upper_bounds)]

    # Use curve_fit to fit the data to the symmetric_gauss_triple function
    params, _ = curve_fit(symmetric_gauss_triple, x_data, y_data, p0=init_params, bounds=bounds, maxfev=10000)
    mean, var, var_c, weight_c  = params
    out_params = [mean, var, var_c, weight_c]

    # Calculate the fitted function
    fit = symmetric_gauss_triple(x_data, mean, var, var_c, weight_c)
    print([frag_length, mean, var, var_c, weight_c])

    return fit, out_params


def plot_displacement_density(raw_dict, fit_dict, frag_length):
    plt.figure(figsize=(8, 4))
    x = range(frag_length)
    x = [i - frag_length // 2 for i in x]  # Adjust x values so middle point is 0
    sns.lineplot(y=raw_dict[frag_length], x=x, label='Smoothed Coverage')
    sns.lineplot(y=fit_dict[frag_length], x=x, label='Gaussian Fit')
    plt.title(f'Healthy nucleosome-overlapping cfDNA fragment center displacement density for l={frag_length}')
    plt.xlabel('Displacement of fragment center from nucleosome center (bp)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(f'NucFragDisplacements_{frag_length}bp.pdf', bbox_inches="tight")
    plt.close()


def plot_heatmap(data, filename, from_dict=False):
    if from_dict:
        df = pd.DataFrame.from_dict(data, orient='index', columns=np.arange(-250, 250)).fillna(0)
        df = df.iloc[::-1]
    else:
        # Slice data to keep rows from 145 to 500 (in reverse order) and columns from 250 to 750
        data = data[0:356, 250:750]
        df = pd.DataFrame(data, index=np.arange(500, 144, -1), columns=np.arange(-250, 250))
        df = df.iloc[::-1]
        # df = df.reindex(index=df.index[::-1])
    df = df.div(df.max(axis=1), axis=0).fillna(0)
    plt.figure(figsize=(8, 8))
    sns.heatmap(df)
    plt.axvline(x=250, color='lightblue', linestyle='dashed', lw=0.25)
    plt.axhline(y=167-146, color='lightblue', linestyle='dashed', lw=0.50)  # single nucleosome
    plt.axhline(y=314-146, color='lightblue', linestyle='dashed', lw=0.50)  # di-nucleosome
    plt.axhline(y=481-146, color='lightblue', linestyle='dashed', lw=0.50)  # tri-nucleosome
    plt.title('Healthy nucleosome-overlapping cfDNA fragment center displacement density')
    plt.xlabel('Displacement of fragment center from nucleosome center (bp)')
    plt.ylabel('Fragment length (bp)')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Nucleosome center analysis and plotting')
    parser.add_argument('-i', '--input', help='TritonNucPlacementProfiles.npz files', nargs='*', required=True)
    args = parser.parse_args()

    # Load data from input files
    input_path = args.input
    tests_data = [np.load(path)['arr_0'] for path in input_path]
    data = sum(temp for temp in tests_data)

    # Process data
    raw_dict, fit_dict, padded_dict = {}, {}, {}
    in_params = None  # Initial parameters for Gaussian fit
    for i, row in enumerate(data[:(500-145), 250:750]):
        length = 500 - i
        half = length // 2
        row_data = row[(250 - half):(250 + half + length % 2)]
        sym_row = row_data + row_data[::-1]
        sym_row = smooth_data(sym_row) ##############################
        raw_dict[length] = sym_row / sum(sym_row)
        # Apply Gaussian fit
        fit_dict[length], in_params = fit_row(sym_row, in_params)
        # Pad data for heatmap
        padding = np.zeros((500 - length) // 2)
        padded_dict[length] = np.concatenate([padding, [0] * (length % 2), fit_dict[length], padding])

    # Plot individual fragment length displacement densities
    for frag_length in [146, 150, 166, 200, 220, 246, 250, 312, 350, 400, 450, 478, 500]:
        plot_displacement_density(raw_dict, fit_dict, frag_length)

    # Plot raw and fit data as heatmaps
    plot_heatmap(data, 'NucFragDisplacements_RAW.pdf')
    plot_heatmap(padded_dict, 'NucFragDisplacements_FIT.pdf', True)

    # Save fit data
    with open('NCDict.pkl', 'wb') as f:
        pickle.dump(fit_dict, f)


if __name__ == "__main__":
    main()

