#!/bin/bash
#SBATCH --job-name=benchmark_job        # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --gres=gpu:4                    # Number of GPUs per node
#SBATCH --cpus-per-task=4               # CPU cores per task
#SBATCH --mem=32G                       # Memory per node
#SBATCH --time=03:00:00                 # Time limit (1 hour)
#SBATCH --output=logs/output_%j.log     # Output file (%j = job ID)
#SBATCH --error=logs/error_%j.log       # Error file (%j = job ID)

cd /n/fs/vision-mix/jl0796/vllm_benchmark
source .venv/bin/activate

echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA devices: $(nvidia-smi -L 2>/dev/null || echo 'nvidia-smi not available')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"

# Run the benchmark
python benchmark.py -m all