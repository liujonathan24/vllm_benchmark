#!/usr/bin/env python3
"""
Launcher to submit one sbatch per model for GPQA benchmarking.

Usage examples:
  # Dry-run (print sbatch commands):
  python scripts/benchmark_gpqa.py --dry-run

  # Submit jobs for all models (default GPQA percentage 10.0):
  python scripts/benchmark_gpqa.py --submit

  # Submit jobs for a subset:
  python scripts/benchmark_gpqa.py --models Llama-3.1-8B-Instruct Qwen2-7b --submit

This script reads the available model aliases from utils.model_metadata.LLM_MAP
and calls `sbatch` to submit `scripts/benchmark_gpqa.sbatch` once per model.
It exports MODEL and GPQA parameters so each job writes/appends to the same
shared CSV (default: results/benchmark_shared.csv).
"""
import argparse
import shlex
import subprocess
import os
import sys
import glob
import csv
from typing import List

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.model_metadata import LLM_MAP


def build_sbatch_command(
    model_alias: str,
    gpqa_percentage: float,
    gpqa_path: str,
    output_file: str,
    gpus: int,
    cpus: int,
    mem: str,
    time: str,
):
    # Build sbatch command with overrides and exported env vars
    export_vars = f"MODEL={model_alias},GPQA_PERCENTAGE={gpqa_percentage},GPQA_PATH={gpqa_path},OUTPUT_FILE={output_file}"
    cmd = [
        "sbatch",
        f"--job-name=benchmark_{model_alias}",
        f"--export={export_vars}",
        f"--gres=gpu:{gpus}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time}",
        "scripts/benchmark_gpqa.sbatch",
    ]
    return cmd


def submit_jobs(
    models: List[str],
    gpqa_percentage: float,
    gpqa_path: str,
    output_file: str, # Now a format string like "results/benchmark_{model}.csv"
    gpus: int,
    cpus: int,
    mem: str,
    time: str,
    dry_run: bool,
):
    per_model_files = []
    submitted_job_ids = []
    for alias in models:
        if alias not in LLM_MAP:
            print(f"Warning: model alias '{alias}' not found in LLM_MAP; skipping.")
            continue

        # Generate the specific output file path for this model
        model_output = output_file.format(model=alias)
        per_model_files.append(model_output)

        cmd = build_sbatch_command(
            model_alias=alias,
            gpqa_percentage=gpqa_percentage,
            gpqa_path=gpqa_path,
            output_file=model_output, # Pass the specific path to sbatch
            gpus=gpus,
            cpus=cpus,
            mem=mem,
            time=time,
        )

        readable = " ".join(shlex.quote(p) for p in cmd)
        print(f"Submitting for model: {alias}")
        print(f"Command: {readable}")

        if not dry_run:
            try:
                res = subprocess.run(cmd, check=True, capture_output=True, text=True)
                out = (res.stdout or "").strip()
                err = (res.stderr or "").strip()
                if out:
                    print(out)
                    # Try to parse job id from sbatch output
                    parts = out.split()
                    if parts and parts[-1].isdigit():
                        submitted_job_ids.append(parts[-1])
                if err:
                    print(err)
            except subprocess.CalledProcessError as e:
                print(f"sbatch failed for {alias}: returncode={e.returncode}")
                if e.stdout:
                    print(e.stdout)
                if e.stderr:
                    print(e.stderr)
            except FileNotFoundError:
                print("sbatch not found on PATH; cannot submit jobs from this machine.")
                return per_model_files, submitted_job_ids
    return per_model_files, submitted_job_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Submit one sbatch per model for GPQA benchmarking.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="List of model aliases to submit. If omitted or 'all', all models from LLM_MAP are used.",
    )
    parser.add_argument("--gpqa-percentage", type=float, default=10.0, help="GPQA percentage to sample (0-100)")
    parser.add_argument("--gpqa-path", type=str, default="data/gpqa_extended.csv", help="Path to GPQA CSV")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs per job")
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per job")
    parser.add_argument("--mem", type=str, default="80G", help="Memory per job (e.g., 60G)")
    parser.add_argument("--time", type=str, default="5:00:00", help="Time limit per job")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands but don't submit")
    parser.add_argument("--submit", action="store_true", help="Actually submit jobs (default: no)" )
    parser.add_argument("--merge-results", dest="merge_results", action="store_true", help="Merge per-model results into the shared output file (default)")
    parser.add_argument("--no-merge-results", dest="merge_results", action="store_false", help="Do not merge results")
    parser.add_argument("--wait", action="store_true", help="Wait for submitted jobs to complete before merging results")
    parser.add_argument("--wait-interval", type=int, default=30, help="Seconds between squeue polls when waiting")
    parser.add_argument("--plot", dest="plot", action="store_true", help="Generate plots after merging results (default)")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Do not generate plots")
    parser.set_defaults(merge_results=True, plot=True)
    return parser.parse_args()


from datetime import datetime
import time

def wait_for_jobs(job_ids: List[str], interval: int):
    """Polls squeue until the given job IDs are no longer in the queue."""
    if not job_ids:
        return

    print(f"\n--- Waiting for {len(job_ids)} jobs to complete ---")
    job_id_str = ",".join(job_ids)
    
    while True:
        try:
            # Use -h for no header, -j for specific jobs.
            # The output will be empty if no jobs are found.
            cmd = ["squeue", "-h", "-j", job_id_str]
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            stdout = res.stdout.strip()
            if not stdout:
                print("All jobs completed.")
                break
            
            num_remaining = len(stdout.splitlines())
            print(f"Polling squeue: {num_remaining} jobs remaining. Waiting {interval}s...")

        except FileNotFoundError:
            print("Warning: squeue not found on PATH. Cannot wait for jobs.")
            return
        except subprocess.CalledProcessError as e:
            # squeue often exits with an error if the job ID is no longer valid,
            # which is our success condition.
            if "Invalid job id specified" in e.stderr:
                print("Jobs no longer found in squeue. Assuming completion.")
                break
            else:
                print(f"An error occurred while polling squeue: {e.stderr}")
                print("Stopping wait loop.")
                return
        
        time.sleep(interval)


def main():
    args = parse_args()

    if args.models is None or (len(args.models) == 1 and args.models[0] == "all"):
        models = list(LLM_MAP.keys())
    else:
        models = args.models

    if not args.submit and not args.dry_run:
        print("Neither --submit nor --dry-run specified; defaulting to --dry-run (no jobs will be submitted). Use --submit to actually submit sbatch jobs.)")
        args.dry_run = True

    # --- Create a unique directory for this experiment run ---
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", experiment_id)
    plots_dir = os.path.join("plots", experiment_id)
    
    if not args.dry_run:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
    
    print(f"--- Starting Experiment Run: {experiment_id} ---")
    print(f"Results will be saved in: {results_dir}")
    print(f"Plots will be saved in: {plots_dir}")
    # ---

    per_model_output_format = os.path.join(results_dir, "benchmark_{model}.csv")
    master_csv_path = os.path.join(results_dir, "master_results.csv")

    per_model_files, submitted_job_ids = submit_jobs(
        models=models,
        gpqa_percentage=args.gpqa_percentage,
        gpqa_path=args.gpqa_path,
        output_file=per_model_output_format,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        time=args.time,
        dry_run=args.dry_run,
    )

    # Optionally wait for job completion
    if not args.dry_run and args.wait and submitted_job_ids:
        wait_for_jobs(submitted_job_ids, args.wait_interval)

    # Post-processing steps
    if not args.dry_run and (args.merge_results or args.plot):
        # Merge results using the standalone script
        if args.merge_results:
            print("\n--- Merging results ---")
            merge_cmd = [
                "python", "scripts/merge_results.py",
                "--results-dir", results_dir,
                "--output-file", master_csv_path
            ]
            try:
                print(f"Executing: {' '.join(merge_cmd)}")
                subprocess.run(merge_cmd, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Failed to merge results: {e}")
                return # Stop if merging fails

        # Generate plots if requested and merging was successful
        if args.plot:
            if not os.path.exists(master_csv_path):
                print("\n--- Skipping plotting: Master results file not found. ---")
                return
                
            print("\n--- Generating plots ---")
            plot_cmd = [
                "python", "utils/plotter.py",
                "--input-file", master_csv_path,
                "--output-dir", plots_dir
            ]
            try:
                print(f"Executing: {' '.join(plot_cmd)}")
                subprocess.run(plot_cmd, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Failed to generate plots: {e}")

if __name__ == "__main__":
    main()