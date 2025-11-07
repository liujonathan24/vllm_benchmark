"""
Dataset loading utilities for GPQA dataset.
"""
import pandas as pd
from typing import List, Dict
import os


def load_gpqa_dataset(path: str) -> pd.DataFrame:
    """
    Load GPQA extended CSV dataset.
    
    Args:
        path: Path to GPQA extended CSV file
        
    Returns:
        DataFrame containing the GPQA dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file doesn't have required columns
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"GPQA dataset file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Validate that required column exists
    if "Question" not in df.columns:
        raise ValueError(f"GPQA dataset must have a 'Question' column. Found columns: {df.columns.tolist()}")
    
    return df


def sample_gpqa_dataset(df: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """
    Sample x% of the GPQA dataset.
    
    Args:
        df: DataFrame containing the GPQA dataset
        percentage: Percentage to sample (0-100)
        
    Returns:
        Sampled DataFrame
    """
    if percentage <= 0 or percentage > 100:
        raise ValueError(f"Percentage must be between 0 and 100, got {percentage}")
    
    # Calculate number of samples
    num_samples = int(len(df) * (percentage / 100.0))
    num_samples = max(1, num_samples)  # At least 1 sample
    
    # Sample without replacement
    sampled_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    
    return sampled_df


def gpqa_to_chat_format(question: str) -> List[Dict[str, str]]:
    """
    Convert GPQA question to chat format.
    
    Args:
        question: Question string from GPQA dataset
        
    Returns:
        List with chat message format: [{"role": "user", "content": question}]
    """
    return [{"role": "user", "content": str(question)}]

