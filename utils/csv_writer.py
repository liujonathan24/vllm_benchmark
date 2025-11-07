"""
CSV writing utilities for benchmark results.
"""
import csv
import os
from typing import List, Dict


def write_results_to_csv(results: List[Dict], output_file: str):
    """
    Write benchmark results to CSV file.
    
    Args:
        results: List of dictionaries containing benchmark results
        output_file: Path to output CSV file
    """
    if not results:
        print("No results to write.")
        return
    
    # Create directory for output file if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        # If no directory specified, use results/ directory
        os.makedirs("results", exist_ok=True)
        output_file = os.path.join("results", output_file)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(output_file)
    
    # Define CSV columns
    fieldnames = [
        "model_name",
        "parameters",
        "gpqa",
        "avg_hf_time",
        "avg_vllm_time",
        "ratio",
        "num_tests",
        "attention_type",
        "has_gqa",
        "is_moe",
        "num_experts"
    ]
    
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write results
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults written to: {output_file}")

