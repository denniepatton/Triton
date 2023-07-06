# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v0.2.1, 06/29/2023

# utility for plotting nc_dist.py outputs and generating an NCDict.pkl object, required for Triton

import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def symmetric_gauss_triple(x, mu, var, var_c, theta, c):  # defined to be symmetric
    """
    :param x: variable
    :param mu: distance from edges for flanking Gaussians
    :param var: flanking variance
    :param var_c: central variance
    :param theta: weight of central Gaussian
    :param c: constant background
    :return: normalized signal
    """
    signal = (1 - theta) / np.sqrt(2 * var * np.pi) * (np.exp(-(x - mu) ** 2 / (2 * var)) +
                                                       np.exp(-(x - (len(x) - mu)) ** 2 / (2 * var))) +\
           theta / np.sqrt(2 * var_c * np.pi) * np.exp(-(x - (len(x) / 2)) ** 2 / (2 * var)) + c
    return signal / sum(signal)


def fit_row(row, init_params):
    frag_length = len(row)
    norm_const = 1 / (2*np.sqrt(2*np.log(2)))  # * FWHM = Gaussian stdev
    low_var = (norm_const * 20)**2  # stdev = 20bp, small linker
    high_var = (norm_const * 100)**2  # stdev = 100bp, larger linker * 2 (50bp dinucleosomal)
    x_data = np.arange(frag_length)
    y_data = row
    low_theta, high_theta = 0.0, 1.0
    bounds = ((50, low_var, low_var, low_theta, 0), ((frag_length / 2) - 1, high_var, high_var, high_theta, 0.1))
    if init_params is not None:
        params, _ = curve_fit(symmetric_gauss_triple, x_data, y_data, p0=init_params, bounds=bounds, maxfev=10000)
    else:
        params, _ = curve_fit(symmetric_gauss_triple, x_data, y_data, bounds=bounds)
    mean, var, var_c, weight_c, const = params[0], params[1], params[2], params[3], params[4]
    out_params = [mean, var, var_c, weight_c, const]
    # theta_init = np.array([row, 50, low_var, low_var, 0.5, 0.0])
    # bounds = [(50, (frag_length / 2) - 1), (low_var, high_var), (low_var, high_var), (0.0, 1.0), (0.0, 0.1)]
    #
    # def objective(params):
    #     profile, mu, var, var_c, theta, c = params
    #     return
    #
    # weights = minimize(fun=objective, x0=theta_init, bounds=bounds)
    # _, mean, var, var_c, weight_c, const = weights.x
    fit = symmetric_gauss_triple(x_data, mean, var, var_c, weight_c, const)
    print([frag_length, mean, var, var_c, weight_c, const])
    return fit, out_params


def main():
    parser = argparse.ArgumentParser(description='\n### nc_analyze.py ### nucleosome center analysis and plotting')
    parser.add_argument('-i', '--input', help='one or more TritonNucPlacementProfiles.npz files',
                        nargs='*', required=True)

    args = parser.parse_args()
    input_path = args.input

    if len(input_path) == 1:  # individual sample
        test_data = np.load(input_path[0])
        site = test_data.files[0]
        data = test_data[site]
    else:  # multiple samples:
        tests_data = [np.load(path) for path in input_path]
        site = tests_data[0].files[0]
        tests_data = [temp[site] for temp in tests_data]
        data = sum(tests_data)

    raw_dict = {}
    data = data[:(500-145), 250:750]
    i = 0
    for row in data:
        length = 500-i
        half = int(length/2)
        if (length % 2) == 0:
            row_data = row[(250-half):(250+half)]
        else:
            row_data = row[(249 - half):(250 + half)]
        sym_row = row_data + row_data[::-1]
        raw_dict[length] = sym_row / sum(sym_row)
        i += 1

    fit_dict = {}
    padded_dict = {}
    in_params = None  # should start as a single Gaussian dominant ([mean, var, var_c, weight_c, const])
    for length in range(146, 501):
        profile = raw_dict[length]
        # profiles are first fit with a Savgol filter to smooth out noise, then fit to a symmetric triple Gaussian
        window_length = int(length / 2)  # 2 for SVG
        if (window_length % 2) == 0:
            window_length += 1
        sf_fit = savgol_filter(profile, window_length, 2)
        sf_fit /= sum(sf_fit)
        fit_dict[length], in_params = fit_row(sf_fit, in_params)
        padding = np.zeros(int((500 - length) / 2))
        if length == 500:
            padded_dict[length] = fit_dict[length]
        elif length == 499:
            padded_dict[length] = np.concatenate([fit_dict[length], [0]])
        elif (length % 2) == 0:
            padded_dict[length] = np.concatenate([padding, fit_dict[length], padding])
        else:
            padded_dict[length] = np.concatenate([padding, [0], fit_dict[length], padding])

    # for plotting individual fragment length displacement densities
    for frag_length in [146, 150, 166, 200, 220, 246, 250, 312, 350, 400, 450, 478, 500]:
        plt.figure(figsize=(8, 4))
        sns.lineplot(y=raw_dict[frag_length], x=range(frag_length), label='raw_coverage')
        window_length = int(frag_length/2)
        if (window_length % 2) == 0:
            window_length += 1
        sns.lineplot(y=savgol_filter(raw_dict[frag_length], window_length, 2), x=range(frag_length), label='savgol_fit')
        gauss_row = fit_dict[frag_length]
        sns.lineplot(y=gauss_row, x=range(frag_length), label='gauss_fit')
        plt.title('healthy nucleosome-overlapping cfDNA fragment center displacement density for l=' + str(frag_length))
        plt.xlabel('displacement of fragment center from nucleosome center (bp)')
        plt.ylabel('density')
        plt.tight_layout()
        plt.savefig('NucFragDisplacements_' + str(frag_length) + 'bp.pdf', bbox_inches="tight")
        plt.close()

    # plot raw (flipped data) as heatmap
    # data += np.fliplr(data)  # ENFORCES SYMMETRY ABOUT 0
    # df = pd.DataFrame(data, index=np.arange(500, 145, -1), columns=np.arange(-250, 250)).iloc[::-1]
    # df = df.div(df.max(axis=1), axis=0).fillna(0)
    # plt.figure(figsize=(8, 8))
    # sns.heatmap(df)
    # plt.axvline(x=250, color='lightblue', linestyle='dashed', lw=0.25)
    # plt.axhline(y=500-167, color='lightblue', linestyle='dashed', lw=0.50)
    # plt.axhline(y=500-247, color='lightblue', linestyle='dashed', lw=0.50)
    # plt.axhline(y=500-313, color='lightblue', linestyle='dashed', lw=0.50)
    # plt.axhline(y=500-373, color='lightblue', linestyle='dashed', lw=0.50)
    # plt.title('Healthy nucleosome-overlapping cfDNA fragment center displacement density')
    # plt.xlabel('displacement of fragment center from nucleosome center (bp)')
    # plt.ylabel('fragment length (bp)')
    # plt.tight_layout()
    # plt.savefig('NucFragDisplacements_RAW.pdf', bbox_inches="tight")
    # plt.close()
    #
    # # plot fit data as heatmap
    # df = pd.DataFrame.from_dict(padded_dict, orient='index', columns=np.arange(-250, 250)).fillna(0)
    # plt.figure(figsize=(8, 8))
    # sns.heatmap(df)
    # plt.axvline(x=250, color='lightblue', linestyle='dashed', lw=0.25)
    # plt.axhline(y=500-167, color='lightblue', linestyle='dashed', lw=0.50)
    # plt.axhline(y=500-246, color='lightblue', linestyle='dashed', lw=0.50)
    # plt.axhline(y=500-312, color='lightblue', linestyle='dashed', lw=0.50)
    # plt.axhline(y=500-478, color='lightblue', linestyle='dashed', lw=0.50)
    # plt.title('Healthy nucleosome-overlapping cfDNA fragment center displacement fit density')
    # plt.xlabel('displacement of fragment center from nucleosome center (bp)')
    # plt.ylabel('fragment length (bp)')
    # plt.tight_layout()
    # plt.savefig('NucFragDisplacements_FIT.pdf', bbox_inches="tight")
    # plt.close()

    for key, value in fit_dict.items():
        print(key)

    with open('NCDict.pkl', 'wb') as f:
        pickle.dump(fit_dict, f)


if __name__ == "__main__":
    main()