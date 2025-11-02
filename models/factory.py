from typing import Optional
from .base import BaseModel
from .vllm import VLLMWrapper
from .hf import HFTransformerWrapper

def load_model(backend: str, model_name: str, **kwargs) -> Optional[BaseModel]:
    """
    Factory function to load a model with a specific backend.
    """
    if backend == "vllm":
        return VLLMWrapper(model_name, **kwargs)
    elif backend == "hf":
        return HFTransformerWrapper(model_name, **kwargs)
    else:
        print(f"Error: Unknown backend '{backend}'. Available backends are vllm and hf. ")
        return None