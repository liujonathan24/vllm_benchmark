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

    def __init__(
        self,
        model_name: str,
        verbose: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95
    ):
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
            "pad_token_id": 50256  # default placeholder
        }

        if self.verbose:
            print(f"[HFTransformerWrapper] Loading model: {self.model_name}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Auto-detect number of GPUs
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            if self.verbose and num_gpus > 0:
                print(f"[HFTransformerWrapper] Detected {num_gpus} GPU(s), using device_map='auto'")
            
            # Use device_map="auto" to automatically distribute across available GPUs
            # This replaces the manual .to(device) call
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto"  # Automatically distribute across all available GPUs
            )
            # Remove: self.model.to(self.device)  # Not needed with device_map="auto"
            self.model.eval()

            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

            # Normalize token IDs (Llama 3 uses lists)
            def _normalize_token_id(token_id):
                if isinstance(token_id, list) and len(token_id) > 0:
                    return token_id[0]
                return token_id

            self.model.config.eos_token_id = _normalize_token_id(self.model.config.eos_token_id)
            self.model.config.pad_token_id = _normalize_token_id(self.model.config.pad_token_id)
            self.generation_config["pad_token_id"] = self.model.config.pad_token_id

            if self.verbose:
                print(f"[HFTransformerWrapper] Model {self.model_name} loaded successfully to {self.device}.")

        except Exception as e:
            print(f"[HFTransformerWrapper] Error loading model {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None

    @torch.no_grad()
    def generate(self, prompt: str) -> Optional[Dict[str, any]]:
        if self.model is None or self.tokenizer is None:
            print("[HFTransformerWrapper] Model is not loaded.")
            return None

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = inputs.input_ids.shape[1]

            generate_kwargs = {
                "input_ids": inputs.input_ids,
                "max_new_tokens": self.generation_config["max_new_tokens"],
                "temperature": self.generation_config["temperature"],
                "top_p": self.generation_config["top_p"],
                "pad_token_id": self.generation_config["pad_token_id"],
                "do_sample": self.generation_config["temperature"] > 0
            }

            if "attention_mask" in inputs:
                generate_kwargs["attention_mask"] = inputs.attention_mask

            outputs = self.model.generate(**generate_kwargs)

            # Decode only newly generated tokens
            generated_ids = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            return {
                "text": generated_text.strip(),
                "tokens": len(generated_ids)
            }

        except Exception as e:
            print(f"[HFTransformerWrapper] Error during generation: {e}")
            return None

    @torch.no_grad()
    def chat(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, any]]:
        """
        Applies the model's chat template to a conversation history
        and returns the next response. For llama-3-instruct models (HF only),
        this ensures the input is in the proper chat format.
        """
        if self.model is None or self.tokenizer is None:
            print("[HFTransformerWrapper] Model or tokenizer is not loaded.")
            return None

        try:
            # Apply chat template (Llama-3-style)
            prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[HFTransformerWrapper] Error applying chat template: {e}")
            return None

        response_data = self.generate(prompt_str)
        if response_data is None:
            return None

        response_text = response_data['text']
        # Try to strip out any repeated chat history or assistant markers
        if "<|start_header_id|>assistant<|end_header_id|>" in response_text:
            response_text = response_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        
        response_data['text'] = response_text.strip()
        return response_data

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


# from typing import List, Optional, Dict
# import gc
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from .base import BaseModel

# class HFTransformerWrapper(BaseModel):
#     """
#     Hugging Face 'transformers' model wrapper.
#     This runs the model directly without vLLM.
#     """
    
#     def __init__(self, model_name: str, verbose: bool = True, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95):
#         """
#         Initializes and loads the HF model and tokenizer.
#         """
#         super().__init__(model_name, verbose)
#         self.tokenizer = None
#         self.model = None
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         # Store generation params
#         self.generation_config = {
#             "max_new_tokens": max_tokens,
#             "temperature": temperature,
#             "top_p": top_p,
#             "pad_token_id": 50256 # Default for many models, like GPT-2/OPT
#         }
        
#         if self.verbose:
#             print(f"[HFTransformerWrapper] Loading model: {self.model_name}...")
        
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name, 
#                 torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
#             )
#             self.model.to(self.device)
#             self.model.eval() # Set to evaluation mode
            
#             # Set pad token if it doesn't exist
#             if self.tokenizer.pad_token is None:
#                 self.tokenizer.pad_token = self.tokenizer.eos_token
#                 self.model.config.pad_token_id = self.model.config.eos_token_id
            
#             self.generation_config["pad_token_id"] = self.model.config.pad_token_id

#             if self.verbose:
#                 print(f"[HFTransformerWrapper] Model {self.model_name} loaded successfully to {self.device}.")
                
#         except Exception as e:
#             print(f"[HFTransformerWrapper] Error loading model {self.model_name}: {e}")
#             self.model = None
#             self.tokenizer = None

#     @torch.no_grad() # Disable gradient calculations for inference
#     def generate(self, prompt: str) -> Optional[str]:
#         if self.model is None or self.tokenizer is None:
#             print("[HFTransformerWrapper] Model is not loaded.")
#             return None
        
#         try:
#             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#             input_length = inputs.input_ids.shape[1]
            
#             # Prepare generation kwargs
#             generate_kwargs = {
#                 "input_ids": inputs.input_ids,
#                 "max_new_tokens": self.generation_config["max_new_tokens"],
#                 "temperature": self.generation_config["temperature"],
#                 "top_p": self.generation_config["top_p"],
#                 "pad_token_id": self.generation_config["pad_token_id"],
#                 "do_sample": self.generation_config["temperature"] > 0
#             }
            
#             # Add attention_mask if it exists
#             if "attention_mask" in inputs:
#                 generate_kwargs["attention_mask"] = inputs.attention_mask
            
#             outputs = self.model.generate(**generate_kwargs)
            
#             # Decode only the newly generated tokens
#             generated_ids = outputs[0][input_length:]
#             generated_text = self.tokenizer.decode(
#                 generated_ids, 
#                 skip_special_tokens=True
#             )
#             return generated_text
        
#         except Exception as e:
#             print(f"[HFTransformerWrapper] Error during generation: {e}")
#             return None

#     @torch.no_grad()
#     def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
#         """
#         Applies the model's chat template to a conversation history
#         and returns the next response. For llama-3-instruct models (HF only),
#         this ensures the input is in the proper chat format.
#         """
#         if self.model is None or self.tokenizer is None:
#             print("[HFTransformerWrapper] Model or tokenizer is not loaded.")
#             return None
        
#         try:
#             # Apply chat template - for llama-3-instruct (HF only), this ensures
#             # the input is in the proper "chat" format required by the model
#             prompt_str = self.tokenizer.apply_chat_template(
#                 messages, 
#                 tokenize=False, 
#                 add_generation_prompt=True
#             )
#         except Exception as e:
#             print(f"[HFTransformerWrapper] Error applying chat template: {e}")
#             return None

#         return self.generate(prompt_str)

#     def destroy(self):
#         if self.model is None:
#             if self.verbose:
#                 print("[HFTransformerWrapper] Model is already destroyed or was never loaded.")
#             return

#         if self.verbose:
#             print(f"[HFTransformerWrapper] Destroying model: {self.model_name}...")
        
#         del self.model
#         del self.tokenizer
#         gc.collect()
        
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
            
#         self.model = None
#         self.tokenizer = None
        
#         if self.verbose:
#             print(f"[HFTransformerWrapper] Model {self.model_name} destroyed and VRAM freed.")