# models/__init__.py
from .base import BaseModel
from .vllm import VLLMWrapper
from .hf import HFTransformerWrapper
from .factory import load_model

__all__ = [
    "BaseModel",
    "VLLMWrapper",
    "HFTransformerWrapper",
    "load_model"
]


if __name__ == "__main__":
    # Check functions used.
    MODEL_ID = "facebook/opt-125m"

    print("--- Starting wrapper and loading model ---")
    # VRAM is allocated here
    wrapper = VLLModelWrapper(model_name=MODEL_ID)

    if wrapper.model:
        print("\n--- Generating text ---")
        response = wrapper.generate("The capital of France is")
        print(f"Model response: {response}")

    print("\n--- Pausing for 10s. Check nvidia-smi now. ---")
    print("VRAM should be high.")
    # time.sleep(10) # Uncomment to pause and check

    print("\n--- Calling destroy() ---")
    # This is the crucial step to free the memory
    wrapper.destroy()

    print("\n--- Pausing for 10s. Check nvidia-smi again. ---")
    print("VRAM should be freed now.")
    # time.sleep(10) # Uncomment to pause and check
    
    print("\n--- End of script ---")
    # At this point, even if you hadn't called destroy(),
    # the wrapper object would be garbage collected,
    # triggering its __del__ and freeing the memory.
    # But calling destroy() explicitly gives you control.