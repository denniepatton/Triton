# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v2.0.0, 09/17/2025

# Utility for plotting Triton profile outputs
# N.B. Peak locations plotting is supported. Nucleotide-frequency plotting removed
# as nucleotide frequency data is no longer available in composite mode.

"""
N.B. the numpy array ordering of profile objects:
1: Depth (GC-corrected, normalized by mean flanking depth)
2: Fragment end coverage (fraction of 5'+3' fragment ends / total fragments at each position, normalized by mean flanking end coverage)
3: Fragment end orientation asymmetry ((5' - 3') / (5' + 3') at each position)
4: PN profile (normalized by mean flanking signal in NC signal, pre-FFT)
5: FL subnucleosomal ratio [log2(x < 147 / x >= 147)]
6: FL Shannon Entropy / Pielou's Evenness (robust z-score using median and MAD of flanking signal)
7: FL Gini-Simpson index (robust z-score using median and MAD of flanking signal)
8: PN peak locations (-1: trough, 1: peak)
9: A (Adenine)**
10: C (Cytosine)**
11: G (Guanine)**
12: T (Thymine)**
** Nucleotide channels are removed as they are not available in composite mode.
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid

# Suppress the seaborn UserWarning about tight layout
warnings.filterwarnings('ignore', 'The figure layout has changed to tight', UserWarning)

# pretty column names (nucleotide frequencies removed)
cols = ['raw-depth', 'frag-ends', 'frag-orient', 'pn-signal', 'fl-ratio', 'fl-entropy',
        'fl-gini-simpson', 'peaks']
pretty_cols = ['Coverage (GC-Corrected, Flank-Normalized)', 'Fragment End Coverage (Flank-Normalized, Smoothed)', 'Fragment End Orientation (Smoothed)', 'Phased Nucleosomal Signal (Flank-Normalized)',
               'FL Subnucleosomal Ratio (<147bp/>=147bp)', 'FL Shannon Entropy (Pielou\'s Evenness)', 'FL Gini-Simpson Index', 'Called Peaks']
col_map = {col: pretty_cols[i] for i, col in enumerate(cols)}
profile_labels = {'raw-depth': 'Depth',
                  'frag-ends': 'End fraction',
                  'frag-orient': '5\'/3\' asymmetry',
                  'pn-signal': 'PN depth',
                  'fl-ratio': 'Proportion (log2)',
                  'fl-entropy': 'Robust z-score',
                  'fl-gini-simpson': 'Robust z-score'}


def smooth_data(data, window=75, fill_nans=True, fill_value=0.0):
    """
    Performs Gaussian kernel smoothing with options for NaN handling
        Parameters:
            data (vector-like array): signal profile from Triton
            window (int): window-size for smoothing (determines sigma of Gaussian)
            fill_nans (bool): if True, NaN values are replaced with fill_value; if False, NaNs are preserved
            fill_value (float): value to use for filling NaNs when fill_nans=True
        Returns:
            smoothed_data
    """
    # Convert input to numpy array
    data_arr = np.asarray(data)
    n = len(data_arr)
    # Preserve the original data type
    dtype = data_arr.dtype
    
    # Create a copy of the data 
    data_filled = np.copy(data_arr)
    nan_mask = np.isnan(data_arr)
    
    # If all values are NaN, return an array of NaNs
    if np.all(nan_mask):
        return np.full_like(data_arr, np.nan, dtype=dtype)
        
    # Handle NaN values based on the fill_nans parameter
    if fill_nans:
        # Replace NaNs with the specified fill value
        data_filled[nan_mask] = fill_value
    
    # Initialize smoothed data array
    smoothed_data = np.zeros_like(data_arr, dtype=dtype)
    
    # Create Gaussian kernel
    sigma = window / 6.0  # Set sigma so ~99% of Gaussian falls within the window
    kernel_size = window * 2 + 1
    kernel_half = kernel_size // 2
    x = np.arange(-kernel_half, kernel_half + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize the kernel to sum to 1
    
    # Apply the kernel to smooth the data with edge handling
    for i in range(n):
        # If we're preserving NaNs and the current point is NaN, keep it as NaN
        if not fill_nans and np.isnan(data_arr[i]):
            smoothed_data[i] = np.nan
            continue
            
        # Calculate kernel boundaries with edge handling
        k_start = max(0, kernel_half - i)  # How much kernel to skip at start
        k_end = min(kernel_size, kernel_size - (i + kernel_half - n + 1))  # How much kernel to use at end
        # Calculate data boundaries with edge handling
        d_start = max(0, i - kernel_half)
        d_end = min(n, i + kernel_half + 1)
        # Extract appropriate portion of kernel and data
        used_kernel = kernel[k_start:k_end]
        used_data = data_filled[d_start:d_end]
        
        # If preserving NaNs, we need to adjust the kernel based on valid data points
        if not fill_nans:
            # Create a mask for non-NaN values in the data window
            valid_mask = ~np.isnan(used_data)
            # If no valid data in the window, result is NaN
            if not np.any(valid_mask):
                smoothed_data[i] = np.nan
                continue
            # Otherwise, use only valid data points and adjust kernel weights
            used_data = used_data[valid_mask]
            used_kernel = used_kernel[valid_mask]
            # Re-normalize kernel to sum to 1
            if np.sum(used_kernel) > 0:
                used_kernel = used_kernel / np.sum(used_kernel)
        
        # Apply kernel
        if len(used_kernel) > 0:
            smoothed_data[i] = np.sum(used_kernel * used_data)
        else:
            # Fall back if we have issues
            smoothed_data[i] = np.nan if not fill_nans else np.mean(used_data)
            
    return smoothed_data


# Nucleotide variation function removed - not applicable without nucleotide frequency data


def plot_profiles(name, data, region, fragmentation_data, samples, palette=None, categories=None):
    """
    Plots all profiles in a two-column layout with fragment length distributions.
        Parameters:
            name (string): name of site being plotted
            data (pandas long df): signal profile(s) from Triton
            region (bool): whether to plot region-based profiles (0 at left side) or not (0 in middle)
            fragmentation_data (list): fragmentation profile data for each sample
            samples (list): sample names corresponding to fragmentation_data
            palette (target:color dictionary): palette for subtypes if categories is passed
            categories (dict): mapping from sample names to category labels
    """
    sns.set_context("notebook", font_scale=1.1)

    # Define the profiles we want to plot and their organization (nucleotide variation removed)
    left_profiles = ['pn-signal', 'raw-depth', 'frag-ends', 'frag-orient']
    right_profiles = ['fl-ratio', 'fl-entropy', 'fl-gini-simpson']
    
    # Filter data to only include our desired profiles
    all_profiles = left_profiles + right_profiles
    data = data[data['profile'].isin(all_profiles)].copy()
    
    # First, pre-process asymmetry by filling NaNs with zeros before any other processing
    if 'frag-orient' in all_profiles:
        for sample in data[data['profile'] == 'frag-orient']['sample'].unique():
            mask = (data['profile'] == 'frag-orient') & (data['sample'] == sample)
            values = data.loc[mask, 'value'].values
            
            # Replace NaNs with zeros in asymmetry data
            nan_mask = np.isnan(values)
            if np.any(nan_mask):
                values_filled = values.copy()
                values_filled[nan_mask] = 0.0
                data.loc[mask, 'value'] = values_filled

    # No log2 scaling needed: fl-ratio is already pre-log2'd and frag-ends are fractions (0-1)
    # All signals are used as-is from Triton output
    
    # Apply smoothing based on profile-specific requirements
    if 'frag-ends' in all_profiles:
        for sample in data[data['profile'] == 'frag-ends']['sample'].unique():
            mask = (data['profile'] == 'frag-ends') & (data['sample'] == sample)
            values = data.loc[mask, 'value'].values
            original_dtype = values.dtype
            # Smooth fragment ends while preserving NaNs
            smoothed = smooth_data(values, fill_nans=False)
            if len(smoothed) == len(values):
                data.loc[mask, 'value'] = smoothed.astype(original_dtype)
    
    if 'frag-orient' in all_profiles:
        for sample in data[data['profile'] == 'frag-orient']['sample'].unique():
            mask = (data['profile'] == 'frag-orient') & (data['sample'] == sample)
            values = data.loc[mask, 'value'].values
            original_dtype = values.dtype
            # Smooth orientation asymmetry (NaNs already filled with zeros)
            smoothed = smooth_data(values, fill_nans=True, fill_value=0.0)
            if len(smoothed) == len(values):
                data.loc[mask, 'value'] = smoothed.astype(original_dtype)

    # Update column mappings (nucleotide variation removed)
    extended_col_map = col_map.copy()
    
    # Update profile labels (nucleotide variation removed)
    extended_profile_labels = profile_labels.copy()
    
    data.loc[:, 'profile'] = data['profile'].map(extended_col_map)
    loc_name = 'Location relative to start (bp)' if region else 'Location relative to center (bp)'
    data = data.rename(columns={'profile': 'Profile', 'label': 'Sample / Group', 'loc': loc_name})

    xmin, xmax = data[loc_name].min(), data[loc_name].max() + 1

    # Create custom subplot layout: 2 columns with different heights
    fig = plt.figure(figsize=(16, 12))
    
    # Left column: 4 signal plots (nucleotide variation removed)
    left_gs = fig.add_gridspec(4, 1, left=0.05, right=0.48, top=0.88, bottom=0.1, hspace=0.4)
    
    # Right column: 3 signal plots 
    right_signal_gs = fig.add_gridspec(4, 2, left=0.55, right=0.95, top=0.88, bottom=0.1, hspace=0.4, wspace=0.3)
    
    # Add column titles (closer to signals)
    fig.text(0.265, 0.93, 'Fragment Coverage Signals', ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.93, 'Fragment Length-Derived Signals', ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Plot left column profiles
    left_profile_names = [extended_col_map[p] for p in left_profiles]
    first_ax = None  # Keep reference for legend
    for i, profile_name in enumerate(left_profile_names):
        ax = fig.add_subplot(left_gs[i])
        if first_ax is None:
            first_ax = ax  # Save first axis for legend
        profile_data = data[data['Profile'] == profile_name]
        
        # Add faint vertical line at x=0 behind signals
        ax.axvline(x=0, color='lightgray', linestyle=':', alpha=0.5, zorder=0)
        
        # Plot each sample group
        for sample_group in profile_data['Sample / Group'].unique():
            sample_data = profile_data[profile_data['Sample / Group'] == sample_group]
            color = palette.get(sample_group) if palette else None
            
            if categories is not None:
                # When categories are used, plot 95% CI band instead of individual lines
                # Group by location and calculate mean and CI
                grouped = sample_data.groupby(loc_name)['value']
                x_vals = sorted(sample_data[loc_name].unique())
                means = [grouped.get_group(x).mean() if x in grouped.groups else np.nan for x in x_vals]
                stds = [grouped.get_group(x).std() if x in grouped.groups else np.nan for x in x_vals]
                counts = [len(grouped.get_group(x)) if x in grouped.groups else 0 for x in x_vals]
                
                # Calculate 95% CI
                import scipy.stats as stats
                ci_lower = []
                ci_upper = []
                for mean, std, n in zip(means, stds, counts):
                    if n > 1 and not np.isnan(std):
                        sem = std / np.sqrt(n)
                        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
                        ci_lower.append(ci[0])
                        ci_upper.append(ci[1])
                    else:
                        ci_lower.append(mean)
                        ci_upper.append(mean)
                
                # Plot mean line and CI band
                ax.plot(x_vals, means, label=sample_group, alpha=0.8, color=color, linewidth=2)
                ax.fill_between(x_vals, ci_lower, ci_upper, alpha=0.2, color=color)
            else:
                # Individual sample plotting (original behavior)
                ax.plot(sample_data[loc_name], sample_data['value'], 
                       label=sample_group, alpha=0.8, color=color, linewidth=1.5)
        
        ax.set_ylabel(extended_profile_labels[left_profiles[i]], fontsize=10)
        ax.set_title(profile_name, fontsize=11, pad=5)
        
        # Special handling for orientation asymmetry
        if left_profiles[i] == 'frag-orient':
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, zorder=0)
            ylim = ax.get_ylim()
            max_abs_y = max(abs(ylim[0]), abs(ylim[1]))
            ax.set_ylim(-max_abs_y, max_abs_y)
        
        # Only show x-axis label on bottom plot
        if i == len(left_profiles) - 1:
            ax.set_xlabel(loc_name, fontsize=10)
        
        ax.set_xlim(xmin, xmax)
    
    # Plot right column profiles (first 3 rows, spanning both columns)
    right_profile_names = [extended_col_map[p] for p in right_profiles]
    for i, profile_name in enumerate(right_profile_names):
        ax = fig.add_subplot(right_signal_gs[i, :])  # Span both columns for signals
        profile_data = data[data['Profile'] == profile_name]
        
        # Add faint vertical line at x=0 behind signals
        ax.axvline(x=0, color='lightgray', linestyle=':', alpha=0.5, zorder=0)
        
        # Plot each sample group
        for sample_group in profile_data['Sample / Group'].unique():
            sample_data = profile_data[profile_data['Sample / Group'] == sample_group]
            color = palette.get(sample_group) if palette else None
            
            if categories is not None:
                # When categories are used, plot 95% CI band instead of individual lines
                # Group by location and calculate mean and CI
                grouped = sample_data.groupby(loc_name)['value']
                x_vals = sorted(sample_data[loc_name].unique())
                means = [grouped.get_group(x).mean() if x in grouped.groups else np.nan for x in x_vals]
                stds = [grouped.get_group(x).std() if x in grouped.groups else np.nan for x in x_vals]
                counts = [len(grouped.get_group(x)) if x in grouped.groups else 0 for x in x_vals]
                
                # Calculate 95% CI using scipy.stats (already imported above)
                ci_lower = []
                ci_upper = []
                for mean, std, n in zip(means, stds, counts):
                    if n > 1 and not np.isnan(std):
                        sem = std / np.sqrt(n)
                        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
                        ci_lower.append(ci[0])
                        ci_upper.append(ci[1])
                    else:
                        ci_lower.append(mean)
                        ci_upper.append(mean)
                
                # Plot mean line and CI band
                ax.plot(x_vals, means, label=sample_group, alpha=0.8, color=color, linewidth=2)
                ax.fill_between(x_vals, ci_lower, ci_upper, alpha=0.2, color=color)
            else:
                # Individual sample plotting (original behavior)
                ax.plot(sample_data[loc_name], sample_data['value'], 
                       label=sample_group, alpha=0.8, color=color, linewidth=1.5)
        
        ax.set_ylabel(extended_profile_labels[right_profiles[i]], fontsize=10)
        ax.set_title(profile_name, fontsize=11, pad=5)
        
        # Only show x-axis label on last signal plot
        if i == len(right_profiles) - 1:
            ax.set_xlabel(loc_name, fontsize=10)
        
        ax.set_xlim(xmin, xmax)
    
    # Fragment plot should align with a single signal plot height but be 2/3 width
    # Get the position of row 3 for height alignment
    row3_subplot_spec = right_signal_gs[3, :]  # Full row 3
    row3_pos = row3_subplot_spec.get_position(fig)
    
    # Calculate 2/3 width of the signal plot area
    signal_width = row3_pos.x1 - row3_pos.x0  # Full width of signal area
    frag_width = signal_width * (2.0/3.0)  # 2/3 of signal width
    
    # Position fragment plot at left edge with 2/3 width
    frag_left = row3_pos.x0
    frag_bottom = row3_pos.y0
    frag_height = row3_pos.y1 - row3_pos.y0
    
    ax_frag = fig.add_axes([frag_left, frag_bottom, frag_width, frag_height])
    
    # Legend positioned to the right of the fragment plot (in the remaining 1/3 space)
    legend_left = frag_left + frag_width + 0.02  # Small gap after fragment plot
    legend_width = (row3_pos.x1 - legend_left) * 0.8  # Use 80% of remaining space
    legend_height = frag_height * 0.6  # 60% of fragment plot height
    legend_bottom = frag_bottom + (frag_height - legend_height) / 2  # Center vertically
    
    ax_legend = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])
    ax_legend.axis('off')  # Hide axes
    
    # Create legend using handles from the first signal plot (which has the actual colors matplotlib assigned)
    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        ax_legend.legend(handles=handles, labels=labels, loc='upper left', fontsize=10, frameon=True)
    
    # Add faint vertical line at fragment length 147
    ax_frag.axvline(x=147, color='lightgray', linestyle=':', alpha=0.5, zorder=0)
    
    # Plot fragment length distributions
    frag_lengths = np.arange(1, 501)  # 1 to 500 (as per Triton.py)
    
    # Collect fragment data by category
    frag_data_by_category = {}
    
    for sample_idx, (sample, frag_data) in enumerate(zip(samples, fragmentation_data)):
        if frag_data is None:
            print(f"No fragmentation data for {sample}")
            continue
        
        # Skip if sample not in categories (consistent with signal plotting logic)
        if categories is not None and sample not in categories:
            continue
            
        # Access the fragment_profiles data (new format only)
        if 'fragment_data' in frag_data.files and 'fragment_columns' in frag_data.files:
            fragment_data = frag_data['fragment_data']
            fragment_columns = frag_data['fragment_columns']
            
            # Reconstruct DataFrame from saved data and columns
            df = pd.DataFrame(fragment_data, columns=fragment_columns)
            
            # Find the matching site
            if 'site' in df.columns:
                site_match = df[df['site'] == name]
                if not site_match.empty:
                    # Extract fragment length data (exclude the 'site' column)
                    frag_cols = [col for col in df.columns if col.startswith('frag_len_')]
                    site_frag_data = site_match[frag_cols].iloc[0].values
                else:
                    # Try using the first available site as fallback
                    if len(df) > 0:
                        frag_cols = [col for col in df.columns if col.startswith('frag_len_')]
                        site_frag_data = df[frag_cols].iloc[0].values
                    else:
                        continue
            else:
                continue
            
            # Data is already float32 from Triton.py, convert to float64 for processing
            site_frag_data = site_frag_data.astype(np.float64)
            
            # Diagnostic: Check for shape mismatch
            if len(site_frag_data) != len(frag_lengths):
                print(f"ERROR: Shape mismatch for sample '{sample}' at site '{name}'")
                print(f"  site_frag_data shape: {site_frag_data.shape}, length: {len(site_frag_data)}")
                print(f"  frag_lengths shape: {frag_lengths.shape}, length: {len(frag_lengths)}")
                print(f"  frag_cols found: {len(frag_cols)}")
                print(f"  frag_cols: {frag_cols[:5]}..." if len(frag_cols) > 5 else f"  frag_cols: {frag_cols}")
                continue  # Skip this sample to avoid crash
            
            # Validate fragment data before processing
            # Skip samples with all NaNs, all zeros, or other invalid data
            if np.all(np.isnan(site_frag_data)) or np.all(site_frag_data == 0) or len(site_frag_data) == 0:
                print(f"Skipping sample '{sample}' at site '{name}': fragment data is all NaN, zero, or empty")
                continue
            
            # Check for too many NaNs (more than 50% of data points)
            nan_fraction = np.isnan(site_frag_data).sum() / len(site_frag_data)
            if nan_fraction > 0.5:
                print(f"Skipping sample '{sample}' at site '{name}': fragment data has {nan_fraction:.1%} NaN values")
                continue
            
            # Normalize to true density (area under curve = 1)
            # Use trapezoidal rule for integration approximation
            # Handle NaNs by using only valid data points for area calculation
            valid_mask = ~np.isnan(site_frag_data)
            if np.any(valid_mask):
                area = trapezoid(site_frag_data[valid_mask], frag_lengths[valid_mask])
            else:
                area = 0
                
            if area <= 0:
                print(f"Skipping sample '{sample}' at site '{name}': fragment data has zero or negative area under curve")
                continue
                
            # Normalize the data
            site_frag_data = site_frag_data / area
            
            # Additional validation after normalization
            if np.all(np.isnan(site_frag_data)) or not np.isfinite(site_frag_data).any():
                print(f"Skipping sample '{sample}' at site '{name}': fragment data invalid after normalization")
                continue
            
            # Get sample group label and collect data by category
            sample_group = categories.get(sample, sample) if categories else sample
            
            if sample_group not in frag_data_by_category:
                frag_data_by_category[sample_group] = []
            
            # Smooth the data before collecting (handle NaNs in smoothing)
            # Replace any remaining NaNs with zeros for smoothing
            data_for_smoothing = np.where(np.isnan(site_frag_data), 0, site_frag_data)
            smoothed_data = gaussian_filter1d(data_for_smoothing, sigma=3)
            
            # Final validation of smoothed data
            if np.all(np.isnan(smoothed_data)) or not np.isfinite(smoothed_data).any():
                print(f"Skipping sample '{sample}' at site '{name}': fragment data invalid after smoothing")
                continue
                
            frag_data_by_category[sample_group].append(smoothed_data)
    
    # Plot fragment data by category
    for sample_group, group_data in frag_data_by_category.items():
        # Skip categories with no valid data
        if not group_data or len(group_data) == 0:
            print(f"Skipping category '{sample_group}': no valid fragment data samples")
            continue
            
        sample_color = palette.get(sample_group) if palette else None
        
        if categories is not None and len(group_data) > 1:
            # Plot 95% CI band for categories with multiple samples
            try:
                group_array = np.array(group_data)
                
                # Additional validation: check if any samples in the group have all NaNs
                valid_samples = []
                for i, sample_data in enumerate(group_data):
                    if not np.all(np.isnan(sample_data)) and np.isfinite(sample_data).any():
                        valid_samples.append(sample_data)
                
                if len(valid_samples) == 0:
                    print(f"Skipping category '{sample_group}': all samples have invalid fragment data")
                    continue
                elif len(valid_samples) < len(group_data):
                    print(f"Category '{sample_group}': using {len(valid_samples)}/{len(group_data)} valid samples")
                    group_array = np.array(valid_samples)
                
                means = np.mean(group_array, axis=0)
                stds = np.std(group_array, axis=0)
                n = len(valid_samples)
                
                # Validate means and stds
                if np.all(np.isnan(means)) or not np.isfinite(means).any():
                    print(f"Skipping category '{sample_group}': computed means are invalid")
                    continue
                
                # Calculate 95% CI
                sems = stds / np.sqrt(n)
                ci_lower = means - 1.96 * sems  # Approximate 95% CI
                ci_upper = means + 1.96 * sems
                
                # Plot mean line and CI band
                ax_frag.plot(frag_lengths, means, label=f'{sample_group}', alpha=0.8, 
                            color=sample_color, linewidth=2)
                ax_frag.fill_between(frag_lengths, ci_lower, ci_upper, alpha=0.2, color=sample_color)
                
            except Exception as e:
                print(f"Error plotting category '{sample_group}': {e}")
                continue
        else:
            # Plot individual sample (original behavior) - but validate each sample
            plotted_any = False
            for data in group_data:
                # Validate individual sample data
                if np.all(np.isnan(data)) or not np.isfinite(data).any():
                    continue
                    
                ax_frag.plot(frag_lengths, data, label=f'{sample_group}', alpha=0.8, 
                           color=sample_color, linewidth=1.5)
                plotted_any = True
            
            if not plotted_any:
                print(f"Skipping sample/category '{sample_group}': no valid fragment data to plot")
    
    ax_frag.set_xlabel('Fragment Length (bp)', fontsize=10)
    ax_frag.set_ylabel('Density', fontsize=10)
    # Remove the title as requested
    ax_frag.set_xlim(1, 500)
    # The plot dimensions are already set to be square-ish by the add_axes call above
    
    # Save the figure without tight_layout to avoid warnings with manual gridspec
    plt.savefig(name + '-Profiles_all.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    return


def load_file_data(file_path, delimiter='\t', header=None):
    """
    Helper function to load data from files
    """
    if file_path:
        return pd.read_table(file_path, sep=delimiter, header=header).set_index(0).to_dict()[1]
    return None


def main():
    parser = argparse.ArgumentParser(description='\n### triton_plotters.py ### plots Triton output profiles')
    parser.add_argument('-i', '--input', help='one or more directories containing Triton output files; '
                                              'directory names will be used as sample names.', nargs='*', required=True)
    parser.add_argument('-c', '--categories', help='tsv file containing matched sample names (column 1) and '
                                                   'categories (column 2) to color samples by composite '
                                                   'categories rather than individually, with a .95 confidence '
                                                   'interval; sample names in column 1 MUST match the sample '
                                                   'names passed to Triton (i.e. sample1_TritonProfiles.npz). '
                                                   'Any passed inputs not present in this file will be dropped. '
                                                   'This file should NOT contain a header. DEFAULT = None (plot '
                                                   'all samples individually).',
                        required=False, default=None)
    parser.add_argument('-p', '--palette', help='tsv file containing matched categories/samples (column 1) and HEX '
                                                'color codes (column 2, e.g. #0077BB) to specify what color to use '
                                                'for samples/categories. Sample/category names must exactly match '
                                                'sample names passed to Triton, or category names passed with the '
                                                '-c/--categories option. Will error if inputs/categories include '
                                                'labels not present in palette. This file should NOT contain a header. '
                                                'DEFAULT = None (use Seaborn default colors).',
                        required=False, default=None)
    parser.add_argument('-s', '--sites', help='file containing a list (row-wise) of sites to plot. This file should '
                                              'NOT contain a header. DEFAULT = None (plotting all available sites).',
                        required=False, default=None)
    parser.add_argument('-r', '--region_axis', help='if set, data is "region" - sets 0-point to the left side instead of the middle. '
                                               'DEFAULT = False.', action='store_true')

    args = parser.parse_args()
    print('### Running triton_plotters.py ###')

    # Load configuration data
    categories = load_file_data(args.categories)
    palette = load_file_data(args.palette)
    sites = open(args.sites).read().splitlines() if args.sites else None

    # Extract sample names from input directories and load data
    samples = []
    tests_data = []
    fragmentation_data = []
    
    for directory in args.input:
        if not os.path.isdir(directory):
            # print(f'Warning: {directory} is not a directory. Skipping.')
            continue
            
        sample_name = os.path.basename(directory.rstrip('/'))
        signal_file = os.path.join(directory, f'{sample_name}_TritonSignalProfiles.npz')
        fragmentation_file = os.path.join(directory, f'{sample_name}_TritonFragmentationProfiles.npz')
        
        # Check for required signal file
        if not os.path.exists(signal_file):
            print(f'Warning: Missing {sample_name}_TritonSignalProfiles.npz in directory {directory}. Skipping.')
            continue
            
        # Check for fragmentation file (warn but don't skip)
        if not os.path.exists(fragmentation_file):
            print(f'Warning: Missing {sample_name}_TritonFragmentationProfiles.npz in directory {directory}.')
            fragmentation_data.append(None)
        else:
            fragmentation_data.append(np.load(fragmentation_file, allow_pickle=True))
        
        # Load signal data and add to lists
        samples.append(sample_name)
        tests_data.append(np.load(signal_file))

    # Check if we have any valid samples
    if not samples:
        print('Error: No valid samples found. Please check your input directories.')
        quit()

    print(f'Loaded {len(samples)} samples: {", ".join(samples)}')

    # Filter samples by categories and show warnings once
    if categories is not None:
        filtered_samples = []
        filtered_tests_data = []
        filtered_fragmentation_data = []
        
        for i, sample in enumerate(samples):
            if sample not in categories:
                print(f'Sample {sample} not found in categories. Skipping.')
            else:
                filtered_samples.append(sample)
                filtered_tests_data.append(tests_data[i])
                filtered_fragmentation_data.append(fragmentation_data[i])
        
        samples = filtered_samples
        tests_data = filtered_tests_data
        fragmentation_data = filtered_fragmentation_data
        
        if not samples:
            print('Error: No valid samples remain after category filtering.')
            quit()
        
        print(f'Using {len(samples)} samples after category filtering: {", ".join(samples)}')

    # Process each site
    for site in tests_data[0].files:
        if sites is None or site in sites:
            print(f'*** Processing samples for {site}')
            
            # Process each sample for the current site
            dfs = []
            for sample, test_data in zip(samples, tests_data):
                # Skip if no data
                
                if pd.isnull(test_data[site]).all():
                    print(f'No data for {site} for sample {sample}. Skipping.')
                    continue
                    
                # Debug fragment orientation data
                if site in test_data:
                    data_array = test_data[site]
                    if data_array.shape[0] > 2:  # Ensure we have the fragment orientation row (index 2)
                        frag_orient_data = data_array[2]
                        nan_count = np.isnan(frag_orient_data).sum()
                        total_count = len(frag_orient_data)
                        if nan_count == total_count:
                            print(f"Warning: All fragment orientation data for {site} in sample {sample} is NaN")
                        # elif nan_count > 0:
                        #     print(f"Info: Fragment orientation for {site} in sample {sample} has {nan_count}/{total_count} NaN values")
                
                # Process data if it's in the right shape
                if len(test_data[site].shape) == 2:
                    # Create a DataFrame for the sample - use only first 8 channels (discard nucleotide frequencies if present)
                    data_array = test_data[site]
                    original_channels = data_array.shape[0]
                    
                    # Take only the first 8 rows (channels) to match our expected cols
                    data_to_use = data_array[:8, :] if data_array.shape[0] >= 8 else data_array
                    channels_used = data_to_use.shape[0]
                    
                    # if original_channels > 8:
                    #     print(f"Info: Using first {channels_used} of {original_channels} channels for {site} (discarding nucleotide frequencies)")
                    
                    tdf = pd.DataFrame(data_to_use.T, columns=cols[:channels_used])
                    tdf['loc'] = np.arange(len(tdf))
                    if not args.region_axis:
                        tdf['loc'] -= len(tdf) / 2
                    tdf['sample'] = sample
                    tdf['label'] = categories.get(sample, sample) if categories else sample
                    dfs.append(tdf)
            
            # Skip if no data to plot
            if not dfs:
                print(f'No data for {site}. Skipping.')
                continue
                
            # Combine all sample data
            df = pd.concat(dfs) if len(dfs) > 1 else dfs[0]
            print(f'*** Plotting {site}')
            
            # Verify we have data to plot if using categories
            if categories:
                df = df[df['label'].notna()]
                if len(df['label'].unique()) < 1:
                    print(f'No samples to plot after matching against the provided categories file. Please ensure '
                        f'provided labels are an exact match! Categories provided: {categories.keys()}')
                    print(f'Sample names provided: {samples}')
                    print('Exiting.')
                    quit()
            
            # Transform to long format for plotting
            df_melted = pd.melt(df, id_vars=['sample', 'loc', 'label'], value_vars=cols, var_name='profile')
            
            # Plot the data
            plot_profiles(site, df_melted, args.region_axis, fragmentation_data, samples, palette=palette, categories=categories)

if __name__ == "__main__":
    main()
