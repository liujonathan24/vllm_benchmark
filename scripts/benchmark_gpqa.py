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
import glob
import csv
from typing import List

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
    output_file: str,
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

        # Determine per-model output file to avoid CSV write races.
        # If the provided output_file contains a '{model}' placeholder use it,
        # otherwise create a per-model file at results/benchmark_<alias>.csv
        if "{model}" in output_file:
            model_output = output_file.format(model=alias)
        else:
            model_output = f"results/benchmark_{alias}.csv"

        per_model_files.append(model_output)

        cmd = build_sbatch_command(
            model_alias=alias,
            gpqa_percentage=gpqa_percentage,
            gpqa_path=gpqa_path,
            output_file=model_output,
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
                    # Try to parse job id from sbatch output: usually 'Submitted batch job <id>'
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
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/benchmark_shared.csv",
        help=("Shared output CSV file. If you include the string '{model}' in the path, "
              "the launcher will expand it per-model (e.g. results/benchmark_{model}.csv). "
              "If not present, per-model files will be created as results/benchmark_<alias>.csv and merged into the shared file by default.")
    )
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs per job")
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per job")
    parser.add_argument("--mem", type=str, default="80G", help="Memory per job (e.g., 60G)")
    parser.add_argument("--time", type=str, default="5:00:00", help="Time limit per job")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands but don't submit")
    parser.add_argument("--submit", action="store_true", help="Actually submit jobs (default: no)" )
    # Merge results behavior: default ON. Provide --no-merge-results to disable.
    parser.add_argument("--merge-results", dest="merge_results", action="store_true", help="Merge per-model results into the shared output file (default)")
    parser.add_argument("--no-merge-results", dest="merge_results", action="store_false", help="Do not merge results")
    parser.add_argument("--wait", action="store_true", help="Wait for submitted jobs to complete before merging results")
    parser.add_argument("--wait-interval", type=int, default=30, help="Seconds between squeue polls when waiting")
    parser.set_defaults(merge_results=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.models is None or (len(args.models) == 1 and args.models[0] == "all"):
        models = list(LLM_MAP.keys())
    else:
        models = args.models

    if not args.submit and not args.dry_run:
        print("Neither --submit nor --dry-run specified; defaulting to --dry-run (no jobs will be submitted). Use --submit to actually submit sbatch jobs.)")
        args.dry_run = True

    per_model_files, submitted_job_ids = submit_jobs(
        models=models,
        gpqa_percentage=args.gpqa_percentage,
        gpqa_path=args.gpqa_path,
        output_file=args.output_file,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        time=args.time,
        dry_run=args.dry_run,
    )

    # Optionally wait for job completion before merging
    if not args.dry_run and args.wait and submitted_job_ids:
        print(f"Waiting for {len(submitted_job_ids)} submitted jobs to complete...")
        try:
            # Poll squeue for the submitted job IDs
            while True:
                jid_list = ",".join(submitted_job_ids)
                try:
                    res = subprocess.run(["squeue", "-j", jid_list], capture_output=True, text=True)
                except FileNotFoundError:
                    print("squeue not found on PATH; cannot wait for jobs. Skipping wait.")
                    break

                out = (res.stdout or "").strip()
                # If squeue returns no lines for these job IDs, they are finished
                if out == "":
                    print("All jobs finished.")
                    break
                else:
                    print(f"Jobs still running/queued. Next check in {args.wait_interval}s...")
                    time.sleep(args.wait_interval)
        except KeyboardInterrupt:
            print("Wait interrupted by user; proceeding to merge whatever results are available.")

    # Merge per-model files into the shared output file if requested.
    if not args.dry_run and args.merge_results:
        # Look for files that were intended; only merge those that exist.
        # Also check file timestamps if we just submitted jobs
        now = time.time()
        files_to_merge = []
        for p in per_model_files:
            if os.path.exists(p):
                mtime = os.path.getmtime(p)
                # If we just submitted jobs, only consider files newer than submission
                if submitted_job_ids and mtime < now:
                    print(f"Skipping {p}: file exists but is from a previous run (modified {time.ctime(mtime)})")
                    continue
                files_to_merge.append(p)
                if submitted_job_ids:
                    print(f"Found fresh result file: {p} (modified {time.ctime(mtime)})")
                else:
                    print(f"Found existing result file: {p} (modified {time.ctime(mtime)})")

        if not files_to_merge:
            if submitted_job_ids:
                print("No new result files found yet. Run again with --wait to wait for jobs to finish, or re-run later to merge results.")
            else:
                print("No result files found to merge.")
        else:
            print(f"\nMerging {len(files_to_merge)} result files into '{args.output_file}'...")
            # Merge and write to the shared output file (overwrite)
            try:
                # Read header from first file
                with open(files_to_merge[0], newline='') as f0:
                    reader0 = csv.DictReader(f0)
                    fieldnames = reader0.fieldnames or []
                    rows = list(reader0)

                # Append others (validate headers)
                for p in files_to_merge[1:]:
                    with open(p, newline='') as fp:
                        r = csv.DictReader(fp)
                        if r.fieldnames != fieldnames:
                            print(f"Warning: header mismatch in {p}; skipping this file.")
                            continue
                        rows.extend(list(r))

                # Ensure output dir exists
                out_dir = os.path.dirname(args.output_file)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)

                with open(args.output_file, 'w', newline='') as outf:
                    writer = csv.DictWriter(outf, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)

                print(f"Merged results written to: {args.output_file}")
            except Exception as e:
                print(f"Error while merging results: {e}")


if __name__ == "__main__":
    main()
