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
from datetime import datetime
from typing import List, Dict

from utils.model_metadata import LLM_MAP, collect_model_metadata
from utils.csv_writer import write_results_to_csv
from utils.dataset_loader import load_gpqa_dataset, sample_gpqa_dataset, gpqa_to_chat_format
from testing.backend_runner import run_backend_test

# Standardized prompt to use for generation
DEFAULT_PROMPT = "Hi, tell me a piece of useful and actionable advice on triton and GPU programming."
DEFAULT_CHAT = [
    {"role": "user", "content": DEFAULT_PROMPT}
]


def run_prompt_mode(models_to_test: List, prompt: List[Dict[str, str]], repetitions: int) -> List[Dict]:
    """
    Run benchmark in prompt repetition mode.
    
    Args:
        models_to_test: List of (alias, model_id) tuples
        prompt: Prompt in chat format
        repetitions: Number of times to run each prompt
        
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    for alias, model_id in models_to_test:
        print("\n" + "="*50)
        print(f"--- SPEED TEST: {alias} ({model_id}) ---")
        print(f"--- Mode: Prompt Repetition ({repetitions} repetitions) ---")
        print("="*50)
        
        # Collect timing results for multiple runs
        vllm_times = []
        hf_times = []
        
        # Run VLLM backend
        print(f"\n--- Running VLLM Backend {repetitions} times ---")
        for i in range(repetitions):
            print(f"\n[Repetition {i+1}/{repetitions}]")
            time_taken = run_backend_test("vllm", model_id, prompt)
            if time_taken != float('inf'):
                vllm_times.append(time_taken)
        
        # Run HF backend
        print(f"\n--- Running HF Backend {repetitions} times ---")
        for i in range(repetitions):
            print(f"\n[Repetition {i+1}/{repetitions}]")
            time_taken = run_backend_test("hf", model_id, prompt)
            if time_taken != float('inf'):
                hf_times.append(time_taken)
        
        # Calculate averages
        avg_vllm_time = statistics.mean(vllm_times) if vllm_times else float('inf')
        avg_hf_time = statistics.mean(hf_times) if hf_times else float('inf')
        
        # Calculate ratio (HF/vLLM)
        if avg_vllm_time != float('inf') and avg_vllm_time > 0 and avg_hf_time != float('inf'):
            ratio = avg_hf_time / avg_vllm_time
        else:
            ratio = "inf"
        
        # Print results
        print("\n" + "="*50)
        print(f"--- Comparison Results: {alias} ---")
        print(f"VLLM Average Time: {avg_vllm_time:.4f} seconds ({len(vllm_times)}/{repetitions} successful)")
        print(f"HF Average Time:   {avg_hf_time:.4f} seconds ({len(hf_times)}/{repetitions} successful)")
        
        if avg_vllm_time < avg_hf_time:
            speedup = avg_hf_time / avg_vllm_time if avg_vllm_time > 0 else float('inf')
            print(f"\nResult: VLLM was {speedup:.2f}x faster on average.")
        else:
            speedup = avg_vllm_time / avg_hf_time if avg_hf_time > 0 else float('inf')
            print(f"\nResult: HF Transformers was {speedup:.2f}x faster on average (or VLLM failed).")
        
        print("="*50 + "\n")
        
        # Collect metadata and create result entry
        metadata = collect_model_metadata(alias, model_id)
        result_entry = {
            "model_name": metadata["model_name"],
            "parameters": metadata["parameters"],
            "gpqa": metadata["gpqa"],
            "avg_hf_time": avg_hf_time if avg_hf_time != float('inf') else "inf",
            "avg_vllm_time": avg_vllm_time if avg_vllm_time != float('inf') else "inf",
            "ratio": ratio if isinstance(ratio, (int, float)) else "inf",
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
    
    Args:
        models_to_test: List of (alias, model_id) tuples
        gpqa_path: Path to GPQA extended CSV file
        gpqa_percentage: Percentage of dataset to sample (0-100)
        
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    # Load and sample GPQA dataset
    print(f"\n--- Loading GPQA Dataset from {gpqa_path} ---")
    try:
        df = load_gpqa_dataset(gpqa_path)
        sampled_df = sample_gpqa_dataset(df, gpqa_percentage)
        print(f"--- Sampled {len(sampled_df)} questions ({gpqa_percentage}% of {len(df)} total) ---")
    except Exception as e:
        print(f"Error loading GPQA dataset: {e}")
        return []
    
    for alias, model_id in models_to_test:
        print("\n" + "="*50)
        print(f"--- SPEED TEST: {alias} ({model_id}) ---")
        print(f"--- Mode: GPQA Dataset ({len(sampled_df)} questions) ---")
        print("="*50)
        
        # Collect timing results for all questions
        vllm_times = []
        hf_times = []
        
        # Run VLLM backend on all questions
        print(f"\n--- Running VLLM Backend on {len(sampled_df)} questions ---")
        for idx, row in sampled_df.iterrows():
            question = row["Question"]
            chat_prompt = gpqa_to_chat_format(question)
            print(f"\n[Question {idx+1}/{len(sampled_df)}]")
            time_taken = run_backend_test("vllm", model_id, chat_prompt)
            if time_taken != float('inf'):
                vllm_times.append(time_taken)
        
        # Run HF backend on all questions
        print(f"\n--- Running HF Backend on {len(sampled_df)} questions ---")
        for idx, row in sampled_df.iterrows():
            question = row["Question"]
            chat_prompt = gpqa_to_chat_format(question)
            print(f"\n[Question {idx+1}/{len(sampled_df)}]")
            time_taken = run_backend_test("hf", model_id, chat_prompt)
            if time_taken != float('inf'):
                hf_times.append(time_taken)
        
        # Calculate averages
        avg_vllm_time = statistics.mean(vllm_times) if vllm_times else float('inf')
        avg_hf_time = statistics.mean(hf_times) if hf_times else float('inf')
        
        # Calculate ratio (HF/vLLM)
        if avg_vllm_time != float('inf') and avg_vllm_time > 0 and avg_hf_time != float('inf'):
            ratio = avg_hf_time / avg_vllm_time
        else:
            ratio = "inf"
        
        # Print results
        print("\n" + "="*50)
        print(f"--- Comparison Results: {alias} ---")
        print(f"VLLM Average Time: {avg_vllm_time:.4f} seconds ({len(vllm_times)}/{len(sampled_df)} successful)")
        print(f"HF Average Time:   {avg_hf_time:.4f} seconds ({len(hf_times)}/{len(sampled_df)} successful)")
        
        if avg_vllm_time < avg_hf_time:
            speedup = avg_hf_time / avg_vllm_time if avg_vllm_time > 0 else float('inf')
            print(f"\nResult: VLLM was {speedup:.2f}x faster on average.")
        else:
            speedup = avg_vllm_time / avg_hf_time if avg_hf_time > 0 else float('inf')
            print(f"\nResult: HF Transformers was {speedup:.2f}x faster on average (or VLLM failed).")
        
        print("="*50 + "\n")
        
        # Collect metadata and create result entry
        metadata = collect_model_metadata(alias, model_id)
        result_entry = {
            "model_name": metadata["model_name"],
            "parameters": metadata["parameters"],
            "gpqa": metadata["gpqa"],
            "avg_hf_time": avg_hf_time if avg_hf_time != float('inf') else "inf",
            "avg_vllm_time": avg_vllm_time if avg_vllm_time != float('inf') else "inf",
            "ratio": ratio if isinstance(ratio, (int, float)) else "inf",
            "num_tests": len(sampled_df),
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
        "--gpqa-percentage",
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
