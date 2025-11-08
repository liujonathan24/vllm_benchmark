"""
Backend runner for benchmark testing.
Contains function to run and time model backends.
"""
import time
from typing import List, Dict
from models import load_model


def run_backend_test(backend: str, model_name: str, prompt: List[Dict[str, str]]) -> float:
    """
    Loads, runs, and times a single model backend.

    Args:
        backend: The backend to test (e.g., "vllm" or "hf").
        model_name: The Hugging Face ID of the model.
        prompt: The prompt to send to the model (in chat format).

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


def run_backend_batch(backend: str, model_name: str, prompts: List[List[Dict[str, str]]]) -> List[float]:
    """
    Loads the model once, runs multiple prompts sequentially, and returns a list of timings.

    Args:
        backend: The backend to test (e.g., "vllm" or "hf").
        model_name: The Hugging Face ID of the model.
        prompts: A list of prompts (each prompt is a chat-format list of dicts).

    Returns:
        List of time taken for each generation in seconds. If a prompt failed, that entry will be float('inf').
    """
    print(f"\n--- Loading {backend.upper()} Backend for batch run: {model_name} ---")
    model = load_model(backend=backend, model_name=model_name, verbose=True)

    times = []

    if model:
        try:
            for idx, prompt in enumerate(prompts):
                try:
                    print(f"\n[Batch item {idx+1}/{len(prompts)}]")
                    start_time = time.perf_counter()
                    response = model.chat(prompt)
                    end_time = time.perf_counter()

                    time_taken = end_time - start_time
                    times.append(time_taken)
                    print(f"\n[{backend.upper()}] Response: '{response}'")
                    print(f"[{backend.upper()}] Time taken: {time_taken:.4f} seconds")
                except Exception as e:
                    print(f"Error during {backend.upper()} generation for item {idx+1}: {e}")
                    times.append(float('inf'))
        except Exception as e:
            print(f"Unexpected error during batch generation: {e}")
        finally:
            print(f"--- Destroying {backend.upper()} Model (batch) ---")
            model.destroy()
    else:
        print(f"Failed to load model {model_name} with {backend.upper()} backend for batch run.")
        times = [float('inf')] * len(prompts)

    return times

