# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v0.4.0, 10/02/2025 - Updated for new Triton.py output format

import os
import glob
import argparse
import pandas as pd


def main():
    """
    Combines multiple *_TritonFeatures.tsv files output from Triton.py into a single feature matrix.
    Now searches recursively in sample subdirectories for TritonFeatures.tsv files.
    Updated to handle the new wide-format output structure from Triton.py.
    
    Changes from v0.3.1:
    - Files are now in sample subdirectories: results_dir/sample_name/sample_name_TritonFeatures.tsv
    - Input format is now wide (sites as rows, features as columns) with 'site' and 'sample' columns
    - No longer needs to transpose or parse 'sample' row since it's already a column
    """
    parser = argparse.ArgumentParser(description='\n### triton_cleanup.py ### combines _TritonFeatures.tsv files')
    parser.add_argument('-r', '--results_dir', help='directory for results', required=True)
    parser.add_argument('-f', '--output_format', help='output format (long or wide)', required=False, default='long')
    args = parser.parse_args()

    # Find all files recursively in results_dir that end with "_TritonFeatures.tsv"
    # New format: results_dir/sample_name/sample_name_TritonFeatures.tsv
    inputs = glob.glob(args.results_dir + "/**/*_TritonFeatures.tsv", recursive=True)
    
    if not inputs:
        print(f"No TritonFeatures.tsv files found in {args.results_dir}")
        print("Expected format: results_dir/sample_name/sample_name_TritonFeatures.tsv")
        return
    
    print(f"Found {len(inputs)} TritonFeatures.tsv files")

    # Read all files - new format is already wide (sites as rows, features as columns)
    df_list = []
    for filename in inputs:
        try:
            df_temp = pd.read_table(filename, sep='\t', header=0)
            df_list.append(df_temp)
            print(f"Loaded: {os.path.basename(filename)} ({df_temp.shape[0]} sites)")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not df_list:
        print("No valid files could be loaded")
        return
    
    # Combine all dataframes
    df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"Combined data shape: {df.shape}")
    print(f"Samples: {df['sample'].nunique()}, Sites: {df['site'].nunique()}")
    
    if args.output_format == 'wide':
        # Wide format: each row is a site-sample combination, features as columns
        df = df.sort_values(by=['sample', 'site'])
        output_file = os.path.join(args.results_dir, 'TritonCompositeFM.tsv')
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Saved wide format to: {output_file}")
    else:
        # Long format: melt features into rows
        id_vars = ['site', 'sample']  # Keep these as identifiers
        feature_cols = [col for col in df.columns if col not in id_vars]
        
        print(f"Melting {len(feature_cols)} features into long format")
        df_long = pd.melt(df, id_vars=id_vars, value_vars=feature_cols, 
                         var_name='feature', value_name='value')
        
        output_file = os.path.join(args.results_dir, 'TritonCompositeFM.tsv')
        
        # Append to existing file if it exists
        if os.path.exists(output_file):
            print(f"Appending to existing file: {output_file}")
            old_df = pd.read_table(output_file, sep='\t', header=0)
            df_long = pd.concat([old_df, df_long], ignore_index=True)
            # Remove duplicates that might occur from multiple runs
            df_long = df_long.drop_duplicates(subset=['sample', 'site', 'feature'], keep='last')
        
        df_long = df_long[['sample', 'site', 'feature', 'value']].sort_values(by=['sample', 'site', 'feature'])
        df_long.to_csv(output_file, sep='\t', index=False)
        print(f"Saved long format to: {output_file} ({df_long.shape[0]} rows)")


if __name__ == "__main__":
    main()