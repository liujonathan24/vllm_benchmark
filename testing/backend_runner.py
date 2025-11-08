"""
Backend runner for benchmark testing.
Contains function to run and time model backends.
"""
import time
from typing import List, Dict
from models import load_model


import time
import torch
from typing import List, Dict, Union
from models import load_model


def run_backend_batch(backend: str, model_name: str, prompts: List[List[Dict[str, str]]]) -> List[Dict[str, Union[float, int]]]:
    """
    Loads the model once, runs multiple prompts sequentially, and returns a list of performance data.

    Args:
        backend: The backend to test (e.g., "vllm" or "hf").
        model_name: The Hugging Face ID of the model.
        prompts: A list of prompts (each prompt is a chat-format list of dicts).

    Returns:
        List of dictionaries, each containing:
        - 'time_taken': Time for generation in seconds.
        - 'tokens_generated': Number of tokens in the response.
        - 'peak_memory_gb': Peak GPU memory used during generation in GB.
        If a prompt failed, the values will be float('inf') or -1.
    """
    print(f"\n--- Loading {backend.upper()} Backend for batch run: {model_name} ---")
    model = load_model(backend=backend, model_name=model_name, verbose=True)

    results = []

    if model:
        try:
            for idx, prompt in enumerate(prompts):
                result_data = {
                    'time_taken': float('inf'),
                    'tokens_generated': -1,
                    'peak_memory_gb': float('inf')
                }
                try:
                    print(f"\n[Batch item {idx+1}/{len(prompts)}]")
                    
                    # Reset CUDA memory stats
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    start_time = time.perf_counter()
                    response_data = model.chat(prompt)  # Expecting a dict with 'text' and 'tokens'
                    end_time = time.perf_counter()

                    time_taken = end_time - start_time
                    
                    # Get peak memory
                    if torch.cuda.is_available():
                        peak_memory_bytes = torch.cuda.max_memory_allocated()
                        peak_memory_gb = peak_memory_bytes / (1024**3)
                    else:
                        peak_memory_gb = 0.0

                    result_data['time_taken'] = time_taken
                    result_data['tokens_generated'] = response_data.get('tokens', -1)
                    result_data['peak_memory_gb'] = peak_memory_gb
                    
                    print(f"\n[{backend.upper()}] Response: '{response_data.get('text', '')}'")
                    print(f"[{backend.upper()}] Time taken: {time_taken:.4f} seconds")
                    print(f"[{backend.upper()}] Tokens generated: {result_data['tokens_generated']}")
                    print(f"[{backend.upper()}] Peak Memory: {peak_memory_gb:.4f} GB")

                except Exception as e:
                    print(f"Error during {backend.upper()} generation for item {idx+1}: {e}")
                
                results.append(result_data)

        except Exception as e:
            print(f"Unexpected error during batch generation: {e}")
        finally:
            print(f"--- Destroying {backend.upper()} Model (batch) ---")
            model.destroy()
    else:
        print(f"Failed to load model {model_name} with {backend.upper()} backend for batch run.")
        # Populate with failure metrics for all prompts
        for _ in prompts:
            results.append({
                'time_taken': float('inf'),
                'tokens_generated': -1,
                'peak_memory_gb': float('inf')
            })

    return results
