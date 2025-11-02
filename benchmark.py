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
from models import BaseModel, VLLMWrapper, HFTransformerWrapper, load_model

# --- Constants ---

# Map of model aliases to their Hugging Face IDs
LLM_MAP = {
    "Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-7B-instruct": "google/gemma-7b-it",
    "Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
    "Falcon-7B-Instruct": "tiiuae/Falcon3-7B-Instruct",
    # "Phi-3-7b": "microsoft/Phi-3-small-128k-instruct",
    # "opt-125m": "facebook/opt-125m", 
}

# Standardized prompt to use for generation
DEFAULT_PROMPT = "Hi, tell me a piece of useful and actionable advice on triton and GPU programming."
DEFAULT_CHAT = [
    {"role": "user", "content": DEFAULT_PROMPT}
]

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
        
        if vllm_time < hf_time:
            speedup = hf_time / vllm_time if vllm_time > 0 else float('inf')
            print(f"\nResult: VLLM was {speedup:.2f}x faster.")
        else:
            speedup = vllm_time / hf_time if hf_time > 0 else float('inf')
            print(f"\nResult: HF Transformers was {speedup:.2f}x faster (or VLLM failed).")
            
        print("="*50 + "\n")
        
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
    
    args = parser.parse_args()
    main(args)
