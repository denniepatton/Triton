# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v1.0, 11/15/2022

# combine _TritonFeatures.tsv files generated by Triton.py into a single, composite file

import argparse
import pandas as pd


def main():
    """
    Combines multiple *_TritonFeatures.tsv files output from Triton.py into a single feature matrix
    """
    parser = argparse.ArgumentParser(description='\n### triton_cleanup.py ### combines _TritonFeatures.tsv files')
    parser.add_argument('-i', '--inputs', help='_TritonFeatures.tsv files', nargs='*', required=True)
    parser.add_argument('-r', '--results_dir', help='directory for results', required=True)
    parser.add_argument('-f', '--output_format', help='output format (long or wide)', required=False, default='long')
    args = parser.parse_args()

    df_list = [pd.read_table(filename, sep='\t', index_col=0, header=0) for filename in args.inputs]
    df = pd.concat(df_list, axis=1)
    df.columns = df.loc['sample']
    df = df.drop('sample')  # already in index
    if args.output_format == 'wide':  # only for readability
        df = df.reindex(sorted(df.columns), axis=1).sort_index()
        df.to_csv(args.results_dir + '/TritonCompositeFM.tsv', sep='\t')
    else:
        df = df.transpose()
        df['sample'] = df.index
        df = pd.melt(df, id_vars='sample', var_name='temp', value_name='value')
        df[['site', 'feature']] = df.temp.str.rsplit('_', n=1, expand=True)
        df = df.drop('temp', axis=1)
        df = df[['sample', 'site', 'feature', 'value']].sort_values(by=['sample', 'site', 'feature'])
        df.to_csv(args.results_dir + '/TritonCompositeFM.tsv', sep='\t', index=False)


if __name__ == "__main__":
    main()
