import logging, os
os.environ["VLLM_LOG_LEVEL"] = "CRITICAL"
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

os.environ["NCCL_DEBUG"] = "WARN" # Or 'ERROR' to be even more strict
os.environ["PYTHONWARNINGS"] = "ignore"

logging.disable(logging.CRITICAL)

import argparse
import statistics
import torch
from datetime import datetime
from typing import List, Dict

from utils.model_metadata import LLM_MAP, collect_model_metadata
from utils.csv_writer import write_results_to_csv
from utils.dataset_loader import load_gpqa_dataset, sample_gpqa_dataset, gpqa_to_chat_format
from testing.backend_runner import run_backend_batch

# Standardized prompt to use for generation
DEFAULT_PROMPT = "Hi, tell me a piece of useful and actionable advice on triton and GPU programming."
DEFAULT_CHAT = [
    {"role": "user", "content": DEFAULT_PROMPT}
]


def _process_results(results: List[Dict]) -> Dict:
    """Helper to calculate statistics from a list of result dicts."""
    valid_times = [r['time_taken'] for r in results if r['time_taken'] != float('inf')]
    valid_tokens = [r['tokens_generated'] for r in results if r['tokens_generated'] > 0]
    valid_memory = [r['peak_memory_gb'] for r in results if r['peak_memory_gb'] != float('inf')]

    avg_time = statistics.mean(valid_times) if valid_times else float('inf')
    avg_tokens = statistics.mean(valid_tokens) if valid_tokens else 0
    avg_memory = statistics.mean(valid_memory) if valid_memory else float('inf')

    # Throughput
    if avg_time > 0 and avg_tokens > 0:
        throughput = avg_tokens / avg_time
    else:
        throughput = 0.0
    
    return {
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "avg_memory": avg_memory,
        "throughput": throughput,
        "successful_runs": len(valid_times),
    }


def run_prompt_mode(models_to_test: List, prompt: List[Dict[str, str]], repetitions: int) -> List[Dict]:
    """
    Run benchmark in prompt repetition mode.
    """
    all_results = []
    gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    
    for alias, model_id in models_to_test:
        print("\n" + "="*50)
        print(f"--- SPEED TEST: {alias} ({model_id}) ---")
        print(f"--- Mode: Prompt Repetition ({repetitions} repetitions) ---")
        print("="*50)
        
        # Run VLLM backend
        print(f"\n--- Running VLLM Backend {repetitions} times (single load) ---")
        vllm_results = run_backend_batch("vllm", model_id, [prompt] * repetitions)
        vllm_stats = _process_results(vllm_results)
        
        # Run HF backend
        print(f"\n--- Running HF Backend {repetitions} times (single load) ---")
        hf_results = run_backend_batch("hf", model_id, [prompt] * repetitions)
        hf_stats = _process_results(hf_results)
        
        # Calculate ratio
        if vllm_stats['avg_time'] > 0 and hf_stats['avg_time'] != float('inf'):
            ratio = hf_stats['avg_time'] / vllm_stats['avg_time']
        else:
            ratio = "inf"
        
        # Print results
        print("\n" + "="*50)
        print(f"--- Comparison Results: {alias} ---")
        print(f"VLLM Average Time: {vllm_stats['avg_time']:.4f}s | Throughput: {vllm_stats['throughput']:.2f} tokens/s | Peak Memory: {vllm_stats['avg_memory']:.2f} GB ({vllm_stats['successful_runs']}/{repetitions} successful)")
        print(f"HF Average Time:   {hf_stats['avg_time']:.4f}s | Throughput: {hf_stats['throughput']:.2f} tokens/s | Peak Memory: {hf_stats['avg_memory']:.2f} GB ({hf_stats['successful_runs']}/{repetitions} successful)")
        
        if ratio != "inf" and ratio > 1:
            print(f"\nResult: VLLM was {ratio:.2f}x faster on average.")
        else:
            print(f"\nResult: HF Transformers was faster or VLLM failed.")
        print("="*50 + "\n")
        
        # Collect metadata and create result entry
        metadata = collect_model_metadata(alias, model_id)
        result_entry = {
            "model_name": metadata["model_name"],
            "parameters": metadata["parameters"],
            "gpu_type": gpu_type,
            "avg_hf_time": hf_stats['avg_time'],
            "avg_vllm_time": vllm_stats['avg_time'],
            "hf_throughput_tokens_per_sec": hf_stats['throughput'],
            "vllm_throughput_tokens_per_sec": vllm_stats['throughput'],
            "hf_peak_gpu_memory_gb": hf_stats['avg_memory'],
            "vllm_peak_gpu_memory_gb": vllm_stats['avg_memory'],
            "ratio_hf_vllm": ratio,
            "num_tests": repetitions,
            "attention_type": metadata["attention_type"],
            "has_gqa": metadata["has_gqa"],
            "is_moe": metadata["is_moe"],
            "num_experts": metadata["num_experts"],
        }
        all_results.append(result_entry)
    
    return all_results


def run_gpqa_mode(models_to_test: List, gpqa_path: str, gpqa_percentage: float) -> List[Dict]:
    """
    Run benchmark in GPQA dataset sampling mode.
    """
    all_results = []
    gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    
    # Load and sample GPQA dataset
    print(f"\n--- Loading GPQA Dataset from {gpqa_path} ---")
    try:
        df = load_gpqa_dataset(gpqa_path)
        sampled_df = sample_gpqa_dataset(df, gpqa_percentage)
        print(f"--- Sampled {len(sampled_df)} questions ({gpqa_percentage}% of {len(df)} total) ---")
    except Exception as e:
        print(f"Error loading GPQA dataset: {e}")
        return []
    
    prompts = [gpqa_to_chat_format(row["Question"]) for _, row in sampled_df.iterrows()]
    
    for alias, model_id in models_to_test:
        print("\n" + "="*50)
        print(f"--- SPEED TEST: {alias} ({model_id}) ---")
        print(f"--- Mode: GPQA Dataset ({len(prompts)} questions) ---")
        print("="*50)
        
        # Run VLLM backend
        print(f"\n--- Running VLLM Backend on {len(prompts)} questions (single load) ---")
        vllm_results = run_backend_batch("vllm", model_id, prompts)
        vllm_stats = _process_results(vllm_results)

        # Run HF backend
        print(f"\n--- Running HF Backend on {len(prompts)} questions (single load) ---")
        hf_results = run_backend_batch("hf", model_id, prompts)
        hf_stats = _process_results(hf_results)
        
        # Calculate ratio
        if vllm_stats['avg_time'] > 0 and hf_stats['avg_time'] != float('inf'):
            ratio = hf_stats['avg_time'] / vllm_stats['avg_time']
        else:
            ratio = "inf"
        
        # Print results
        print("\n" + "="*50)
        print(f"--- Comparison Results: {alias} ---")
        print(f"VLLM Average Time: {vllm_stats['avg_time']:.4f}s | Throughput: {vllm_stats['throughput']:.2f} tokens/s | Peak Memory: {vllm_stats['avg_memory']:.2f} GB ({vllm_stats['successful_runs']}/{len(prompts)} successful)")
        print(f"HF Average Time:   {hf_stats['avg_time']:.4f}s | Throughput: {hf_stats['throughput']:.2f} tokens/s | Peak Memory: {hf_stats['avg_memory']:.2f} GB ({hf_stats['successful_runs']}/{len(prompts)} successful)")
        
        if ratio != "inf" and ratio > 1:
            print(f"\nResult: VLLM was {ratio:.2f}x faster on average.")
        else:
            print(f"\nResult: HF Transformers was faster or VLLM failed.")
        print("="*50 + "\n")
        
        # Collect metadata and create result entry
        metadata = collect_model_metadata(alias, model_id)
        result_entry = {
            "model_name": metadata["model_name"],
            "parameters": metadata["parameters"],
            "gpu_type": gpu_type,
            "avg_hf_time": hf_stats['avg_time'],
            "avg_vllm_time": vllm_stats['avg_time'],
            "hf_throughput_tokens_per_sec": hf_stats['throughput'],
            "vllm_throughput_tokens_per_sec": vllm_stats['throughput'],
            "hf_peak_gpu_memory_gb": hf_stats['avg_memory'],
            "vllm_peak_gpu_memory_gb": vllm_stats['avg_memory'],
            "ratio_hf_vllm": ratio,
            "num_tests": len(prompts),
            "attention_type": metadata["attention_type"],
            "has_gqa": metadata["has_gqa"],
            "is_moe": metadata["is_moe"],
            "num_experts": metadata["num_experts"],
        }
        all_results.append(result_entry)
    
    return all_results


def main(args):
    """
    Main function to run speed tests on selected LLMs.
    """
    # Determine which models to test
    if "all" in args.models:
        models_to_test = list(LLM_MAP.items())
    else:
        models_to_test = []
        for model_alias in args.models:
            if model_alias in LLM_MAP:
                models_to_test.append((model_alias, LLM_MAP[model_alias]))
            else:
                print(f"Warning: Model alias '{model_alias}' not found in LLM_MAP. Skipping.")
    
    if not models_to_test:
        print("No models to test. Exiting.")
        return
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Create timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/benchmark_{timestamp}.csv"
    
    # Run appropriate mode
    if args.mode == "prompt":
        # Prompt repetition mode
        if args.prompt:
            # User provided a custom prompt (string)
            prompt = [{"role": "user", "content": args.prompt}]
        else:
            # Use default prompt
            prompt = DEFAULT_CHAT
        all_results = run_prompt_mode(models_to_test, prompt, args.repetitions)
    elif args.mode == "gpqa":
        # GPQA dataset sampling mode
        all_results = run_gpqa_mode(models_to_test, args.gpqa_path, args.gpqa_percentage)
    else:
        print(f"Unknown mode: {args.mode}")
        return
    
    # Write all results to CSV
    if all_results:
        write_results_to_csv(all_results, output_file)
    
    print("--- End of script ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark VLLM vs. HF Transformers backends.")
    parser.add_argument(
        "-m", "--models", 
        nargs="+", 
        default=["Llama-3.1-8B-Instruct"],
        help=f"List of model aliases to test (e.g., Llama-3.1-8B Qwen2-7b). "
             f"Use 'all' to test all models. "
             f"Available: {', '.join(LLM_MAP.keys())}"
    )
    parser.add_argument(
        "--mode",
        choices=["prompt", "gpqa"],
        default="prompt",
        help="Testing mode: 'prompt' for custom prompt repetition, 'gpqa' for GPQA dataset sampling (default: prompt)"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="A custom prompt to use for the test (only used in 'prompt' mode). "
             f"Default: '{DEFAULT_PROMPT}'"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of times to run the custom prompt (only used in 'prompt' mode, default: 5)"
    )
    parser.add_argument(
        "-g", "--gpqa-percentage",
        type=float,
        default=10.0,
        help="Percentage of GPQA dataset to sample (only used in 'gpqa' mode, default: 10.0, range: 0-100)"
    )
    parser.add_argument(
        "--gpqa-path",
        type=str,
        default="data/gpqa_extended.csv",
        help="Path to GPQA extended CSV file (only used in 'gpqa' mode, default: data/gpqa_extended.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output CSV file. If not provided, creates a timestamped file in results/ directory."
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "gpqa" and (args.gpqa_percentage <= 0 or args.gpqa_percentage > 100):
        parser.error("--gpqa-percentage must be between 0 and 100")
    
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    
    main(args)
