import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import time
from models import BaseModel, VLLMWrapper, HFTransformerWrapper, load_model

# --- Constants ---

# Map of model aliases to their Hugging Face IDs
LLM_MAP = {
    "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-7B-instruct": "google/gemma-7b-it",
    "Qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "falcon-7b": "tiiuae/Falcon3-7B-Base",
    # "Phi-3-7b": "microsoft/Phi-3-small-128k-instruct",
    # "opt-125m": "facebook/opt-125m", 
}

# Default prompt to use for generation
DEFAULT_PROMPT = "Hi, tell me a piece of useful and actionable advice on triton and GPU programming."

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
    prompt = args.prompt if args.prompt else DEFAULT_PROMPT

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
        default=["Llama-3.1-8B"],
        help=f"List of model aliases to test (e.g., Llama-3.1-8B Qwen2-7b). \
              Use 'all' to test all models. \
              Available: {', '.join(LLM_MAP.keys())}"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="A custom prompt to use for the test."
    )
    
    args = parser.parse_args()
    main(args)















# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import argparse
# # from vllm import LLM, SamplingParams
# import time
# from models import BaseModel, VLLMWrapper, HFTransformerWrapper, load_model


# llm_map = {
#     "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
#     "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
#     "gemma-7B": "google/gemma-7b",
#     "Qwen2-7b": "Qwen/Qwen2-7B-Instruct",
#     "falcon-7b": "tiiuae/falcon-7b",
#     "Phi-3-7b": "microsoft/Phi-3-small-128k-instruct",
# }

# prompt = ["""Hi, tell me a piece of useful and actionable advice on triton and GPU programming."""]

# # sampling_params = SamplingParams(temperature=0.8, top_p = 0.95, max_tokens=256, min_tokens=64)


# vllm_model = load_model(backend="vllm", model_name="facebook/opt-125m")
# hf_model = load_model(backend="hf", model_name="facebook/opt-125m")

# vllm_model.generate("Hello")
# hf_model.generate("Hello")

# vllm_model.destroy()
# hf_model.destroy()







# if __name__ == "__main__":
    
#     MODEL_ID = "facebook/opt-125m"
#     PROMPT = "The capital of France is"
    
#     print("="*50)
#     print(f"--- SPEED TEST: {MODEL_ID} ---")
#     print("="*50)

#     # --- Test VLLM Backend ---
#     print("\n--- Loading VLLM Backend ---")
#     vllm_model = load_model(backend="vllm", model_name=MODEL_ID, verbose=True)
    
#     vllm_time = float('inf')
#     if vllm_model:
#         start_time = time.perf_counter()
#         response = vllm_model.generate(PROMPT)
#         end_time = time.perf_counter()
        
#         vllm_time = end_time - start_time
#         print(f"\n[VLLM] Response: '{response}'")
#         print(f"[VLLM] Time taken: {vllm_time:.4f} seconds")
        
#         print("\n--- Destroying VLLM Model ---")
#         vllm_model.destroy()
    
#     print("\n" + "="*50 + "\n")
    
#     # --- Test HF Transformers Backend ---
#     print("--- Loading HF Transformers Backend ---")
#     hf_model = load_model(backend="hf", model_name=MODEL_ID, verbose=True)

#     hf_time = float('inf')
#     if hf_model:
#         start_time = time.perf_counter()
#         response = hf_model.generate(PROMPT)
#         end_time = time.perf_counter()
        
#         hf_time = end_time - start_time
#         print(f"\n[HF] Response: '{response}'")
#         print(f"[HF] Time taken: {hf_time:.4f} seconds")
        
#         print("\n--- Destroying HF Model ---")
#         hf_model.destroy()
        
#     print("\n" + "="*50 + "\n")
    
#     # --- Final Results ---
#     print("--- Comparison Results ---")
#     print(f"VLLM Time: {vllm_time:.4f} seconds")
#     print(f"HF Time:   {hf_time:.4f} seconds")
    
#     if vllm_time < hf_time:
#         print(f"\nResult: VLLM was {hf_time / vllm_time:.2f}x faster.")
#     else:
#         print(f"\nResult: HF Transformers was {vllm_time / hf_time:.2f}x faster.")
        
#     print("\n--- End of script ---")
#     print("Check nvidia-smi. VRAM should be freed.")













# def main(args):
#     # Initialize the vLLM engine.
#     llms = args.LLMs
#     n = args.num_trials
#     if llms[0] == "all":
#         llms = llm_map.keys()

#     for llm in llms:
#         llm_name = llm_map[llm] 
#         print(f"Testing {llm_name} LLM speed. ")
#         llm = LLM(model=llm_name)
#         start_time = time.time()
#         prompts = prompt * n
#         assert len(prompts) == n
#         outputs = llm.generate(prompts, sampling_params)

#         for output in outputs:
#             ptext = output.prompt
#             generated_text = output.outputs[0].text
#             print(f"Prompt: {ptext!r}, Generated text: {generated_text!r}")
#         end_time = time.time()
#         print(f"The total time used was: {(end_time-start_time):.3f}\n\n")



# if __name__=="__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-n", "--num_trials", default=10, type=int, help="Defines the number of trials to test the LLM")
#     parser.add_argument("-l", "--LLMs", nargs="+", help="Pass in a list of LLMs to try. \"all\" will automatically run all LLMs.")
#     args = parser.parse_args()
#     main(args)


