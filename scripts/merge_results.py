#!/usr/bin/env python3
"""
Utility to merge per-model benchmark CSV files from the results/ directory
into a single master CSV file.
"""
import argparse
import os
import pandas as pd
from typing import List

def find_csv_files(results_dir: str) -> List[str]:
    """Finds all benchmark_*.csv files in the specified directory."""
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return []
    
    all_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.startswith("benchmark_") and f.endswith(".csv")
    ]
    # Exclude the master file itself to prevent merging it into itself
    all_files = [f for f in all_files if "master" not in os.path.basename(f)]
    return all_files

def merge_csv_files(file_paths: List[str], output_path: str):
    """Merges a list of CSV files into a single DataFrame and saves it."""
    if not file_paths:
        print("No benchmark CSV files found to merge.")
        return

    df_list = []
    for f in file_paths:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
            print(f"Read {len(df)} rows from {f}")
        except Exception as e:
            print(f"Could not read {f}: {e}")
            continue
    
    if not df_list:
        print("No valid data found in CSV files. Master file not created.")
        return

    master_df = pd.concat(df_list, ignore_index=True)
    
    # Drop duplicate rows if any, keeping the last entry
    master_df.drop_duplicates(subset=['model_name'], keep='last', inplace=True)
    
    # Sort by model name for consistency
    master_df.sort_values(by='model_name', inplace=True)
    
    # Write to output file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        master_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully merged {len(df_list)} files into {output_path}")
        print(f"Master file contains {len(master_df)} unique model results.")
    except Exception as e:
        print(f"Failed to write master file to {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Merge benchmark CSV results.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing the benchmark CSV files."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/master_benchmark_results.csv",
        help="Path to the master output CSV file."
    )
    args = parser.parse_args()

    csv_files = find_csv_files(args.results_dir)
    merge_csv_files(csv_files, args.output_file)

if __name__ == "__main__":
    main()
