from typing import List, Optional, Dict
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModel

class HFTransformerWrapper(BaseModel):
    """
    Hugging Face 'transformers' model wrapper.
    This runs the model directly without vLLM.
    """
    
    def __init__(self, model_name: str, verbose: bool = True, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95):
        """
        Initializes and loads the HF model and tokenizer.
        """
        super().__init__(model_name, verbose)
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Store generation params
        self.generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": 50256 # Default for many models, like GPT-2/OPT
        }
        
        if self.verbose:
            print(f"[HFTransformerWrapper] Loading model: {self.model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            
            # Set pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id
            
            self.generation_config["pad_token_id"] = self.model.config.pad_token_id

            if self.verbose:
                print(f"[HFTransformerWrapper] Model {self.model_name} loaded successfully to {self.device}.")
                
        except Exception as e:
            print(f"[HFTransformerWrapper] Error loading model {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None

    @torch.no_grad() # Disable gradient calculations for inference
    def generate(self, prompt: str) -> Optional[str]:
        if self.model is None or self.tokenizer is None:
            print("[HFTransformerWrapper] Model is not loaded.")
            return None
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, **self.generation_config)
            
            # Decode only the newly generated tokens
            generated_text = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            return generated_text[0]
        
        except Exception as e:
            print(f"[HFTransformerWrapper] Error during generation: {e}")
            return None

    def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if self.model is None or self.tokenizer is None:
            print("[HFTransformerWrapper] Model or tokenizer is not loaded.")
            return None
        
        try:
            prompt_str = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[HFTransformerWrapper] Error applying chat template: {e}")
            return None

        return self.generate(prompt_str)

    def destroy(self):
        if self.model is None:
            if self.verbose:
                print("[HFTransformerWrapper] Model is already destroyed or was never loaded.")
            return

        if self.verbose:
            print(f"[HFTransformerWrapper] Destroying model: {self.model_name}...")
        
        del self.model
        del self.tokenizer
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model = None
        self.tokenizer = None
        
        if self.verbose:
            print(f"[HFTransformerWrapper] Model {self.model_name} destroyed and VRAM freed.")