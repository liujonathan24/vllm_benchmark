from typing import List, Optional, Dict
import gc
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from .base import BaseModel

class VLLMWrapper(BaseModel):
    """
    VLLM model wrapper that manages initialization, generation,
    chat, and memory cleanup.
    """
    
    def __init__(self, model_name: str, verbose: bool = True, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95):
        """
        Initializes and loads the vLLM model and tokenizer.
        """
        super().__init__(model_name, verbose) # Call parent __init__
        self.tokenizer = None
        self.llm = None
        
        # Store sampling params for reuse
        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p, 
            max_tokens=max_tokens
        )
        
        if self.verbose:
            print(f"[VLLMWrapper] Loading model: {self.model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = LLM(model=model_name, disable_log_stats=True, log_level="ERROR")
            
            if self.verbose:
                print(f"[VLLMWrapper] Model {self.model_name} loaded successfully.")
                
        except Exception as e:
            print(f"[VLLMWrapper] Error loading model {self.model_name}: {e}")
            self.llm = None
            self.tokenizer = None

    def generate(self, prompt: str) -> Optional[str]:
        if self.llm is None:
            print("[VLLMWrapper] Model is not loaded.")
            return None
        
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
        except Exception as e:
            print(f"[VLLMWrapper] Error during generation: {e}")
            return None

    def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if self.llm is None or self.tokenizer is None:
            print("[VLLMWrapper] Model or tokenizer is not loaded.")
            return None
        
        try:
            prompt_str = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[VLLMWrapper] Error applying chat template: {e}")
            return None

        return self.generate(prompt_str)

    def destroy(self):
        if self.llm is None:
            if self.verbose:
                print("[VLLMWrapper] Model is already destroyed or was never loaded.")
            return

        if self.verbose:
            print(f"[VLLMWrapper] Destroying model: {self.model_name}...")
        
        del self.llm
        del self.tokenizer
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.llm = None
        self.tokenizer = None
        
        if self.verbose:
            print(f"[VLLMWrapper] Model {self.model_name} destroyed and VRAM freed.")