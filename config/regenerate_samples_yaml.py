#!/usr/bin/env python3
"""
Generate samples.yaml for pairing BAM/CRAM files with GC bias files.

This script searches for BAM/CRAM files and GC_bias.txt files in specified directories,
pairs them based on sample names, and generates a samples.yaml file for pipeline usage.
(For use re-linking bias files to BAM/CRAMs after moving files around.)
"""

from __future__ import annotations

import os
import glob
import argparse
import math
from typing import Dict, List, Tuple, Any

# ===== CONFIGURATION =====
# Hardcoded directory lists - UPDATE THESE AS NEEDED

# Directories containing GC_bias subdirectories with sample.GC_bias.txt files
GC_BIAS_DIRECTS = [
    "/your/path/to/a/directory/with/GC_bias",
    # Add more directories as needed
]

# Directories containing BAM/CRAM files (may be in subdirectories)
ALIGNMENT_DIRECTS = [
    "/your/path/to/a/directory/with/BAM_or_CRAM",
    # Add more directories as needed
]

# Option to match samples based on prefix before first underscore
MATCH_BY_PREFIX: bool = False  # Set to True to enable prefix matching

# Optional substring to drop from sample names for matching/output naming.
# Use float("nan") (default) to disable.
# Example: DROP_SUBTRING = "_recalibrated" will match
#   SAMPLE_recalibrated.cram <-> SAMPLE.GC_bias.txt
DROP_SUBTRING: Any = float("nan")

# Output file name
OUTPUT_YAML = "samples.yaml"

# =========================

def find_gc_bias_files(gc_bias_directories: List[str]) -> Dict[str, str]:
    """
    Find all GC_bias.txt files in the specified directories.
    
    Args:
        gc_bias_directories: List of directories to search
        
    Returns:
        dict: {sample_name: full_path_to_gc_bias_file}
    """
    gc_bias_files: Dict[str, str] = {}
    
    for base_dir in gc_bias_directories:
        if not os.path.exists(base_dir):
            print(f"Warning: GC bias directory does not exist: {base_dir}")
            continue
            
        # Look for GC_bias subdirectories
        gc_bias_pattern = os.path.join(base_dir, "**/GC_bias/*.GC_bias.txt")
        found_files = glob.glob(gc_bias_pattern, recursive=True)
        
        # Also check direct pattern in case GC_bias is the base directory
        direct_pattern = os.path.join(base_dir, "*.GC_bias.txt")
        found_files.extend(glob.glob(direct_pattern))
        
        for filepath in found_files:
            filename = os.path.basename(filepath)
            # Extract sample name by removing .GC_bias.txt extension
            if filename.endswith('.GC_bias.txt'):
                sample_name = filename[:-len('.GC_bias.txt')]
                gc_bias_files[sample_name] = os.path.abspath(filepath)
                
    print(f"Found {len(gc_bias_files)} GC bias files")
    return gc_bias_files

def find_bam_cram_files(alignment_directories: List[str]) -> Dict[str, str]:
    """
    Find all BAM/CRAM files in the specified directories and subdirectories.
    
    Args:
        alignment_directories: List of directories to search
        
    Returns:
        dict: {sample_name: full_path_to_bam_cram_file}
    """
    alignment_files: Dict[str, str] = {}
    
    for base_dir in alignment_directories:
        if not os.path.exists(base_dir):
            print(f"Warning: Alignment directory does not exist: {base_dir}")
            continue
            
        # Search for BAM and CRAM files recursively
        for extension in ['*.bam', '*.cram']:
            pattern = os.path.join(base_dir, "**", extension)
            found_files = glob.glob(pattern, recursive=True)
            
            for filepath in found_files:
                filename = os.path.basename(filepath)
                # Extract sample name by removing extension
                sample_name = os.path.splitext(filename)[0]
                
                # Handle cases where there might be additional suffixes before extension
                # Keep the original logic but ensure we get the base sample name
                alignment_files[sample_name] = os.path.abspath(filepath)
                
    print(f"Found {len(alignment_files)} alignment files")
    return alignment_files

def extract_prefix(sample_name: str) -> str:
    """
    Extract prefix up to first underscore.
    
    Args:
        sample_name: Original sample name
        
    Returns:
        str: Prefix before first underscore, or original name if no underscore
    """
    return sample_name.split('_')[0]

def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)

def normalize_sample_name(sample_name: str, drop_substring: Any = DROP_SUBTRING) -> str:
    """Normalize a sample name for matching/output naming.

    This intentionally does NOT touch the file paths; it only affects the
    derived sample key used for matching and the sample names written to YAML.
    """
    if drop_substring is None or drop_substring == "" or _is_nan(drop_substring):
        return sample_name
    if not isinstance(drop_substring, str):
        return sample_name
    return sample_name.replace(drop_substring, "")

def sample_key(raw_name: str, match_by_prefix: bool, drop_substring: Any = DROP_SUBTRING) -> str:
    normalized = normalize_sample_name(raw_name, drop_substring=drop_substring)
    return extract_prefix(normalized) if match_by_prefix else normalized

def match_samples(
    gc_bias_files: Dict[str, str],
    alignment_files: Dict[str, str],
    match_by_prefix: bool = False,
    drop_substring: Any = DROP_SUBTRING,
) -> Dict[str, Dict[str, str]]:
    """
    Match GC bias files with alignment files based on sample names.
    
    Args:
        gc_bias_files: Dict of {sample_name: gc_bias_path}
        alignment_files: Dict of {sample_name: alignment_path}
        match_by_prefix: If True, match based on prefix before first underscore
        
    Returns:
        dict: {final_sample_name: {'bam': path, 'GC_bias': path}}
    """
    matched_samples: Dict[str, Dict[str, str]] = {}

    if match_by_prefix:
        print("Matching samples by prefix (before first underscore)")
    else:
        print("Matching samples by exact name")

    # Build key->[(raw_name, path), ...] maps
    gc_key_map: Dict[str, List[Tuple[str, str]]] = {}
    for raw_name, path in gc_bias_files.items():
        key = sample_key(raw_name, match_by_prefix=match_by_prefix, drop_substring=drop_substring)
        gc_key_map.setdefault(key, []).append((raw_name, path))

    align_key_map: Dict[str, List[Tuple[str, str]]] = {}
    for raw_name, path in alignment_files.items():
        key = sample_key(raw_name, match_by_prefix=match_by_prefix, drop_substring=drop_substring)
        align_key_map.setdefault(key, []).append((raw_name, path))

    # Match by computed key
    for key in gc_key_map:
        if key not in align_key_map:
            continue

        gc_sample, gc_path = gc_key_map[key][0]
        align_sample, align_path = align_key_map[key][0]

        matched_samples[key] = {
            'bam': align_path,
            'GC_bias': gc_path
        }

        label = "prefix" if match_by_prefix else "sample"
        print(f"Matched {label} '{key}': {os.path.basename(align_path)} <-> {os.path.basename(gc_path)}")

        if len(gc_key_map[key]) > 1:
            print(f"  Warning: Multiple GC bias files for {label} '{key}', using first: {gc_sample}")
        if len(align_key_map[key]) > 1:
            print(f"  Warning: Multiple alignment files for {label} '{key}', using first: {align_sample}")
    
    print(f"Successfully matched {len(matched_samples)} sample pairs")
    return matched_samples

def generate_yaml(matched_samples: Dict[str, Dict[str, str]], output_file: str) -> None:
    """
    Generate samples.yaml file from matched samples.
    
    Args:
        matched_samples: Dict of matched sample data
        output_file: Output YAML file path
    """
    with open(output_file, 'w') as f:
        f.write("samples:\n")
        
        # Sort samples by name for consistent output
        for sample_name in sorted(matched_samples.keys()):
            sample_data = matched_samples[sample_name]
            f.write(f"  {sample_name}:\n")
            f.write(f"    bam: {sample_data['bam']}\n")
            f.write(f"    GC_bias: {sample_data['GC_bias']}\n")
    
    print(f"Generated {output_file} with {len(matched_samples)} samples")

def main():
    """Main execution function."""
    print("=== BAM/CRAM and GC Bias File Pairing ===")
    print(f"Searching for GC bias files in: {len(GC_BIAS_DIRECTS)} directories")
    print(f"Searching for alignment files in: {len(ALIGNMENT_DIRECTS)} directories")
    print(f"Prefix matching: {'Enabled' if MATCH_BY_PREFIX else 'Disabled'}")
    if DROP_SUBTRING is None or _is_nan(DROP_SUBTRING):
        print("Drop substring: Disabled")
    else:
        print(f"Drop substring: {DROP_SUBTRING}")
    print()
    
    # Find files
    print("1. Finding GC bias files...")
    gc_bias_files = find_gc_bias_files(GC_BIAS_DIRECTS)
    
    print("\n2. Finding alignment files...")
    alignment_files = find_bam_cram_files(ALIGNMENT_DIRECTS)
    
    if not gc_bias_files:
        print("Error: No GC bias files found!")
        return
    
    if not alignment_files:
        print("Error: No alignment files found!")
        return
    
    print(f"\n3. Matching samples...")
    matched_samples = match_samples(gc_bias_files, alignment_files, MATCH_BY_PREFIX, DROP_SUBTRING)
    
    if not matched_samples:
        print("Error: No matching sample pairs found!")
        print("\nAvailable GC bias samples:")
        for sample in sorted(gc_bias_files.keys())[:10]:  # Show first 10
            print(f"  {sample}")
        if len(gc_bias_files) > 10:
            print(f"  ... and {len(gc_bias_files) - 10} more")
            
        print("\nAvailable alignment samples:")
        for sample in sorted(alignment_files.keys())[:10]:  # Show first 10
            print(f"  {sample}")
        if len(alignment_files) > 10:
            print(f"  ... and {len(alignment_files) - 10} more")
        return
    
    print(f"\n4. Generating YAML...")
    generate_yaml(matched_samples, OUTPUT_YAML)

    # Report any discovered files that were not used in a matched pair.
    matched_keys = set(matched_samples.keys())
    gc_key_map: Dict[str, List[Tuple[str, str]]] = {}
    for raw_name, path in gc_bias_files.items():
        key = sample_key(raw_name, match_by_prefix=MATCH_BY_PREFIX, drop_substring=DROP_SUBTRING)
        gc_key_map.setdefault(key, []).append((raw_name, path))

    align_key_map: Dict[str, List[Tuple[str, str]]] = {}
    for raw_name, path in alignment_files.items():
        key = sample_key(raw_name, match_by_prefix=MATCH_BY_PREFIX, drop_substring=DROP_SUBTRING)
        align_key_map.setdefault(key, []).append((raw_name, path))

    unmatched_gc_paths: List[str] = []
    unmatched_align_paths: List[str] = []

    all_keys = set(gc_key_map.keys()) | set(align_key_map.keys())
    for key in sorted(all_keys):
        gc_entries = gc_key_map.get(key, [])
        align_entries = align_key_map.get(key, [])

        if key in matched_keys:
            # Match_samples takes the first entry from each group.
            unmatched_gc_paths.extend([p for _, p in gc_entries[1:]])
            unmatched_align_paths.extend([p for _, p in align_entries[1:]])
        else:
            unmatched_gc_paths.extend([p for _, p in gc_entries])
            unmatched_align_paths.extend([p for _, p in align_entries])

    print("\n=== Unmatched Files ===")
    print(f"Unmatched GC bias files: {len(unmatched_gc_paths)}")
    for path in unmatched_gc_paths:
        print(f"  {path}")

    print(f"Unmatched alignment files (BAM/CRAM): {len(unmatched_align_paths)}")
    for path in unmatched_align_paths:
        print(f"  {path}")
    
    print(f"\n=== Complete ===")
    print(f"Output saved to: {os.path.abspath(OUTPUT_YAML)}")

def run_with_args():
    """Run the script with command line arguments."""
    global MATCH_BY_PREFIX, OUTPUT_YAML
    
    # Command line argument parsing (optional)
    parser = argparse.ArgumentParser(description="Generate samples.yaml for BAM/CRAM and GC bias file pairing")
    parser.add_argument("--prefix-match", action="store_true", 
                       help="Match samples by prefix before first underscore")
    parser.add_argument("--output", "-o", default="samples.yaml",
                       help="Output YAML file name (default: samples.yaml)")
    
    args = parser.parse_args()
    
    # Override global settings with command line arguments if provided
    if args.prefix_match:
        MATCH_BY_PREFIX = True  # pyright: ignore[reportConstantRedefinition]
    
    if args.output:
        OUTPUT_YAML = args.output  # pyright: ignore[reportConstantRedefinition]
    
    main()

if __name__ == "__main__":
    run_with_args()