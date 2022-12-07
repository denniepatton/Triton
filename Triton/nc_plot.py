# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v1.0.0, 11/15/2022

# utilities for plotting Triton profile outputs

import argparse
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.stats import norm


def main():
    parser = argparse.ArgumentParser(description='\n### xxx.py ### plots Triton output profiles')
    parser.add_argument('-i', '--input', help='one or more TritonNucPlacementProfiles.npz files', nargs='*', required=True)

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

    data = data[:-14, 250:750]
    data += np.fliplr(data)  # ENFORCES SYMMETRY ABOUT 0
    df = pd.DataFrame(data, index=np.arange(500, 14, -1), columns=np.arange(-250, 250))
    df = df.div(df.max(axis=1), axis=0).fillna(0)

    # plt.figure(figsize=(8, 8))
    # sns.heatmap(df)
    # plt.axhline(y=353, color='lightgreen', linestyle='dashed', lw=0.25)
    # plt.axvline(x=250, color='lightblue', linestyle='dashed', lw=0.25)
    # plt.title('HD nucleosome-overlapping cfDNA fragment center displacement density')
    # plt.xlabel('displacement of fragment center from nucleosome center (bp)')
    # plt.ylabel('fragment length (bp)')
    # plt.tight_layout()
    # plt.savefig('NucFragDisplacements_RAW.pdf', bbox_inches="tight")
    # plt.close()

    def clean_row(row):
        frag_length = int(row.name)
        if frag_length % 2 != 0:
            frag_length += 1
        start, stop = -frag_length / 2, frag_length / 2 - 1
        row.loc[-250:start] = 0
        row.loc[stop:] = 0
        return row

    data_gf = gaussian_filter(data, sigma=[3, 8])  # y and x sigma
    dfg = pd.DataFrame(data_gf, index=np.arange(500, 14, -1), columns=np.arange(-250, 250))
    dfg = dfg.div(dfg.sum(axis=1), axis=0)
    dfg = dfg.apply(lambda row: clean_row(row), axis=1)
    # plt.figure(figsize=(8, 8))
    # sns.heatmap(dfg)
    # plt.axhline(y=353, color='lightgreen', linestyle='dashed', lw=0.25)
    # plt.axvline(x=250, color='lightblue', linestyle='dashed', lw=0.25)
    # plt.title('HD nucleosome-overlapping cfDNA fragment center displacement density')
    # plt.xlabel('displacement of fragment center from nucleosome center (bp)')
    # plt.ylabel('fragment length (bp)')
    # plt.tight_layout()
    # plt.savefig('NucFragDisplacements_GF.pdf', bbox_inches="tight")
    # plt.close()

    def dictionaryize(df_in):
        out_dict = {key: None for key in list(np.arange(15, 501))}
        for i in range(15, 501):
            if i % 2 != 0:
                j = i + 1
            else:
                j = i
            start, stop = int(-j / 2 + 250), int(-j / 2 + i + 250)
            arr = df_in.loc[i].to_numpy()
            out_dict[i] = arr[start:stop]
        return out_dict

    out_map = dictionaryize(dfg)
    with open('NCDict.pkl', 'wb') as f:
        pickle.dump(out_map, f)

    def gauss_single(x, mu, sigma):
        return 1/np.sqrt(2 * sigma * np.pi) * np.exp(-(x - mu)**2/(2 * sigma))

    def gauss_double(x, mu_1, mu_2, sigma):  # defined to be symmetric
        return 1/np.sqrt(2 * sigma * np.pi) * (np.exp(-(x - mu_1)**2/(2 * sigma)) + np.exp(-(x - mu_2)**2/(2 * sigma)))

    def fit_row(row):
        frag_length = int(row.name)
        if frag_length % 2 != 0:
            frag_length += 1
        start, stop = -frag_length / 2, frag_length / 2 - 1
        row_inset = row.loc[start:stop]
        xdata = np.arange(start, stop + 1)
        ydata = row_inset.values

        # plt.figure(figsize=(10, 4))
        # plt.plot(xdata, ydata)
        # plt.tight_layout()
        # plt.savefig(str(frag_length) + '.pdf', bbox_inches="tight")
        # plt.close()

        # fit to a single gaussian
        params_1, _ = curve_fit(gauss_single, xdata, ydata, bounds=((-5, 50), (5, 500)))
        mean_1, std_1 = params_1[0], params_1[1]
        fit_1 = gauss_single(xdata, mean_1, std_1)
        res_1 = np.linalg.norm(ydata - fit_1)
        # fit to a double gaussian
        try:
            params_2, _ = curve_fit(gauss_double, xdata, ydata, bounds=((-frag_length/2, 5, 50), (-5, frag_length/2, 500)))
            mean_2_1, mean_2_2, std_2 = params_2[0], params_2[1], params_2[2]
            fit_2 = gauss_double(xdata, mean_2_1, mean_2_2, std_2)
            res_2 = np.linalg.norm(ydata - fit_2)
        except Exception:
            res_2 = np.inf
        if res_1 < res_2:
            row.loc[start:stop] = fit_1
        else:
            row.loc[start:stop] = fit_2
        return row

    # df = df.apply(lambda row: fit_row(row), axis=1)


if __name__ == "__main__":
    main()