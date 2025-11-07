"""
Model metadata utilities for benchmark.
Contains model constants and metadata collection functions.
"""
import re
from typing import Dict

# Map of model aliases to their Hugging Face IDs
LLM_MAP = {
    "Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-7B-instruct": "google/gemma-7b-it",
    "Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
    "Falcon-7B-Instruct": "tiiuae/Falcon3-7B-Instruct",
    "Qwen3-Omni-30B-A3B-Instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
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


def extract_parameters(model_name: str) -> str:
    """
    Extract parameter count from model name (e.g., "8B" -> "8", "30B" -> "30").
    Returns the parameter count as a string, or "N/A" if not found.
    """
    match = re.search(r'(\d+(?:\.\d+)?)B', model_name, re.IGNORECASE)
    if match:
        return match.group(1)
    # Try to find other patterns like "125m" for smaller models
    match = re.search(r'(\d+(?:\.\d+)?)m', model_name, re.IGNORECASE)
    if match:
        return match.group(1)
    return "N/A"


def collect_model_metadata(alias: str, model_id: str) -> Dict:
    """
    Collect metadata for a model including parameters, GPQA score, and architecture info.
    
    Args:
        alias: Model alias from LLM_MAP
        model_id: Hugging Face model ID
        
    Returns:
        Dictionary with model metadata
    """
    parameters = extract_parameters(alias)
    gpqa = MODEL_GPQA.get(alias, None)
    arch = MODEL_ARCHITECTURE.get(alias, {
        "attention_type": "N/A",
        "has_gqa": False,
        "is_moe": False,
        "num_experts": None,
    })
    
    return {
        "model_name": alias,
        "parameters": parameters,
        "gpqa": gpqa if gpqa is not None else "N/A",
        "attention_type": arch.get("attention_type", "N/A"),
        "has_gqa": arch.get("has_gqa", False),
        "is_moe": arch.get("is_moe", False),
        "num_experts": arch.get("num_experts", "") if arch.get("is_moe", False) else "",
    }

