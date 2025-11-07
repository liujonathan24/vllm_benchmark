#!/bin/bash
#SBATCH --job-name=benchmark_job        # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --gres=gpu:4                    # Number of GPUs per node
#SBATCH --cpus-per-task=4               # CPU cores per task
#SBATCH --mem=32G                       # Memory per node
#SBATCH --time=01:00:00                 # Time limit (1 hour)
#SBATCH --output=logs/output_%j.log     # Output file (%j = job ID)

python benchmark.py                     # Your job command