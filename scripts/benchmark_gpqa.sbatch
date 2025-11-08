#!/bin/bash
#SBATCH --job-name=benchmark_gpqa       # Job name (can be overridden via command line)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --gres=gpu:4                    # Number of GPUs per node (can be overridden)
#SBATCH --cpus-per-task=4               # CPU cores per task (can be overridden)
#SBATCH --mem=60G                       # Memory per node (can be overridden)
#SBATCH --time=12:00:00                 # Time limit (can be overridden)
#SBATCH --output=logs/output_%x_%j.log  # Output file (%x = job name, %j = job ID)
#SBATCH --error=logs/error_%x_%j.log    # Error file (%x = job name, %j = job ID)

cd /n/fs/vision-mix/jl0796/vllm_benchmark
source .venv/bin/activate

echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA devices: $(nvidia-smi -L 2>/dev/null || echo 'nvidia-smi not available')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"

# Expect the submitting command to export MODEL (model alias) and optional GPQA_PERCENTAGE, GPQA_PATH, OUTPUT_FILE
if [ -z "${MODEL}" ]; then
	echo "Environment variable MODEL is not set. Please submit with --export=MODEL=<alias> or set MODEL in the environment."
	exit 1
fi

GPQA_PERCENTAGE=${GPQA PERCENTAGE:-2.0}
GPQA_PATH=${GPQA_PATH:-data/gpqa_extended.csv}
OUTPUT_FILE=${OUTPUT_FILE:-results/benchmark_shared.csv}

echo "Running benchmark for model: ${MODEL}"
echo "GPQA percentage: ${GPQA_PERCENTAGE}"
echo "GPQA path: ${GPQA_PATH}"
echo "Shared output file: ${OUTPUT_FILE}"

# Run the benchmark for the single model (append to shared CSV)
python benchmark.py -m "${MODEL}" --mode gpqa --gpqa-percentage "${GPQA_PERCENTAGE}" --gpqa-path "${GPQA_PATH}" --output "${OUTPUT_FILE}"