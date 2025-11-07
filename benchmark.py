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
import time
import csv
import re
import os
from datetime import datetime
from models import BaseModel, VLLMWrapper, HFTransformerWrapper, load_model

# --- Constants ---

# Map of model aliases to their Hugging Face IDs
LLM_MAP = {
    "Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-7B-instruct": "google/gemma-7b-it",
    "Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
    "Falcon-7B-Instruct": "tiiuae/Falcon3-7B-Instruct",
    "Qwen3-Omni-30B-A3B-Instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Phi-3-7b": "microsoft/Phi-3-small-128k-instruct",
    # "opt-125m": "facebook/opt-125m", 
}

# GPQA scores (to be filled in)
MODEL_GPQA = {
    "Llama-3.1-8B-Instruct": None,  # Fill in with actual scores
    "Mistral-7B-Instruct": None,
    "gemma-7B-instruct": None,
    "Qwen2-7B-Instruct": None,
    "Falcon-7B-Instruct": None,
    "Qwen3-Omni-30B-A3B-Instruct": None,
    "Phi-3-7b": None,
}

# Model architecture information
MODEL_ARCHITECTURE = {
    "Llama-3.1-8B-Instruct": {
        "attention_type": "MHA",
        "has_gqa": True,  # Llama 3.1 uses Grouped Query Attention
        "is_moe": False,
        "num_experts": None,
    },
    "Mistral-7B-Instruct": {
        "attention_type": "MHA",
        "has_gqa": True,  # Mistral uses GQA
        "is_moe": False,
        "num_experts": None,
    },
    "gemma-7B-instruct": {
        "attention_type": "MHA",
        "has_gqa": False,  # Gemma uses standard MHA
        "is_moe": False,
        "num_experts": None,
    },
    "Qwen2-7B-Instruct": {
        "attention_type": "MHA",
        "has_gqa": True,  # Qwen2 uses GQA
        "is_moe": False,
        "num_experts": None,
    },
    "Falcon-7B-Instruct": {
        "attention_type": "MHA",
        "has_gqa": True,  # Falcon uses GQA
        "is_moe": False,
        "num_experts": None,
    },
    "Qwen3-Omni-30B-A3B-Instruct": {
        "attention_type": "MHA",
        "has_gqa": True,  # Qwen3 uses GQA
        "is_moe": True,
        "num_experts": 8,  # A3B suggests 8 experts
    },
    "Phi-3-7b": {
        "attention_type": "MHA",
        "has_gqa": True,  # Phi-3 uses GQA
        "is_moe": False,
        "num_experts": None,
    },
}

# Standardized prompt to use for generation
DEFAULT_PROMPT = "Hi, tell me a piece of useful and actionable advice on triton and GPU programming."
DEFAULT_CHAT = [
    {"role": "user", "content": DEFAULT_PROMPT}
]

# def extract_parameters(model_name: str) -> str:
#     """
#     Extract parameter count from model name (e.g., "8B" -> "8", "30B" -> "30").
#     Returns the parameter count as a string, or "N/A" if not found.
#     """
#     match = re.search(r'(\d+(?:\.\d+)?)B', model_name, re.IGNORECASE)
#     if match:
#         return match.group(1)
#     # Try to find other patterns like "125m" for smaller models
#     match = re.search(r'(\d+(?:\.\d+)?)m', model_name, re.IGNORECASE)
#     if match:
#         return match.group(1)
#     return "N/A"

# def collect_model_metadata(alias: str, model_id: str) -> dict:
#     """
#     Collect metadata for a model including parameters, GPQA score, and architecture info.
    
#     Args:
#         alias: Model alias from LLM_MAP
#         model_id: Hugging Face model ID
        
#     Returns:
#         Dictionary with model metadata
#     """
#     parameters = extract_parameters(alias)
#     gpqa = MODEL_GPQA.get(alias, None)
#     arch = MODEL_ARCHITECTURE.get(alias, {
#         "attention_type": "N/A",
#         "has_gqa": False,
#         "is_moe": False,
#         "num_experts": None,
#     })
    
#     return {
#         "model_name": alias,
#         "parameters": parameters,
#         "gpqa": gpqa if gpqa is not None else "N/A",
#         "attention_type": arch.get("attention_type", "N/A"),
#         "has_gqa": arch.get("has_gqa", False),
#         "is_moe": arch.get("is_moe", False),
#         "num_experts": arch.get("num_experts", "") if arch.get("is_moe", False) else "",
#     }

def write_results_to_csv(results: list, output_file: str):
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
        "hf_time",
        "vllm_time",
        "ratio",
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

def run_backend_test(backend: str, model_name: str, prompt: str) -> float:
    """
    Loads, runs, and times a single model backend.

    Args:
        backend: The backend to test (e.g., "vllm" or "hf").
        model_name: The Hugging Face ID of the model.
        prompt: The prompt to send to the model.

    Returns:
        The time taken for generation in seconds, or float('inf') if it fails.
    """
    print(f"\n--- Loading {backend.upper()} Backend for {model_name} ---")
    model = load_model(backend=backend, model_name=model_name, verbose=True)
    
    time_taken = float('inf')
    
    if model:
        try:
            start_time = time.perf_counter()
            response = model.chat(prompt)
            end_time = time.perf_counter()
            
            time_taken = end_time - start_time
            print(f"\n[{backend.upper()}] Response: '{response}'")
            print(f"[{backend.upper()}] Time taken: {time_taken:.4f} seconds")
            
        except Exception as e:
            print(f"Error during {backend.upper()} generation: {e}")
            
        finally:
            print(f"--- Destroying {backend.upper()} Model ---")
            model.destroy()
    else:
        print(f"Failed to load model {model_name} with {backend.upper()} backend.")
        
    return time_taken

def main(args):
    """
    Main function to run speed tests on selected LLMs.
    """
    # Determine which models to test
    if "all" in args.models:
        models_to_test = LLM_MAP.items()
    else:
        models_to_test = []
        for model_alias in args.models:
            if model_alias in LLM_MAP:
                models_to_test.append((model_alias, LLM_MAP[model_alias]))
            else:
                print(f"Warning: Model alias '{model_alias}' not found in LLM_MAP. Skipping.")
    
    # Use the provided prompt or the default
    prompt = args.prompt if args.prompt else DEFAULT_CHAT

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Create timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/benchmark_{timestamp}.csv"
    
    # List to store all results
    all_results = []

    # Run the comparison for each selected model
    for alias, model_id in models_to_test:
        print("\n" + "="*50)
        print(f"--- SPEED TEST: {alias} ({model_id}) ---")
        print("="*50)

        vllm_time = run_backend_test("vllm", model_id, prompt)
        hf_time = run_backend_test("hf", model_id, prompt)

        # --- Final Results for this model ---
        print("\n" + "="*50)
        print(f"--- Comparison Results: {alias} ---")
        print(f"VLLM Time: {vllm_time:.4f} seconds")
        print(f"HF Time:   {hf_time:.4f} seconds")
        
        # Calculate ratio (HF/vLLM)
        if vllm_time != float('inf') and vllm_time > 0 and hf_time != float('inf'):
            ratio = hf_time / vllm_time
        else:
            ratio = "inf"
        
        if vllm_time < hf_time:
            speedup = hf_time / vllm_time if vllm_time > 0 else float('inf')
            print(f"\nResult: VLLM was {speedup:.2f}x faster.")
        else:
            speedup = vllm_time / hf_time if hf_time > 0 else float('inf')
            print(f"\nResult: HF Transformers was {speedup:.2f}x faster (or VLLM failed).")
            
        print("="*50 + "\n")
        
        # Collect metadata and create result entry
        metadata = collect_model_metadata(alias, model_id)
        result_entry = {
            "model_name": metadata["model_name"],
            "parameters": metadata["parameters"],
            "gpqa": metadata["gpqa"],
            "hf_time": hf_time if hf_time != float('inf') else "inf",
            "vllm_time": vllm_time if vllm_time != float('inf') else "inf",
            "ratio": ratio if isinstance(ratio, (int, float)) else "inf",
            "attention_type": metadata["attention_type"],
            "has_gqa": metadata["has_gqa"],
            "is_moe": metadata["is_moe"],
            "num_experts": metadata["num_experts"],
        }
        all_results.append(result_entry)
        
    # Write all results to CSV
    write_results_to_csv(all_results, output_file)
    
    print("--- End of script ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark VLLM vs. HF Transformers backends.")
    parser.add_argument(
        "-m", "--models", 
        nargs="+", 
        default=["Llama-3.1-8B-Instruct"],
        help=f"List of model aliases to test (e.g., Llama-3.1-8B Qwen2-7b). \
              Use 'all' to test all models. \
              Available: {', '.join(LLM_MAP.keys())}"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=DEFAULT_CHAT,
        help="A custom prompt to use for the test."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output CSV file. If not provided, creates a timestamped file in results/ directory."
    )
    
    args = parser.parse_args()
    main(args)
