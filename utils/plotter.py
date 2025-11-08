#!/usr/bin/env python3
"""
Utility to generate plots from the master benchmark results CSV file.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the master CSV file into a pandas DataFrame."""
    if not os.path.exists(file_path):
        print(f"Error: Master results file not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def plot_backend_comparison(df: pd.DataFrame, output_dir: str):
    """Plots a bar chart comparing avg generation time for each backend."""
    required_cols = ['model_name', 'avg_vllm_time', 'avg_hf_time']
    if not all(col in df.columns for col in required_cols):
        print("Skipping backend comparison plot: required columns missing.")
        return

    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='model_name', y='avg_vllm_time', color='skyblue', label='vLLM')
    sns.barplot(data=df, x='model_name', y='avg_hf_time', color='salmon', label='Hugging Face')
    plt.xticks(rotation=45, ha='right')
    plt.title('Backend Performance Comparison (Average Generation Time)')
    plt.ylabel('Average Time (seconds)')
    plt.xlabel('Model')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'backend_comparison_time.png'))
    print(f"Saved backend comparison plot to {output_dir}")

def plot_vllm_speedup(df: pd.DataFrame, output_dir: str):
    """Plots a bar chart showing the vLLM speedup ratio."""
    required_cols = ['model_name', 'avg_hf_time', 'avg_vllm_time']
    if not all(col in df.columns for col in required_cols):
        print("Skipping vLLM speedup plot: required columns missing.")
        return
    
    # Avoid division by zero
    if (df['avg_vllm_time'] == 0).any():
        print("Warning: 'avg_vllm_time' contains zero values. Speedup cannot be calculated for these entries.")
        df = df[df['avg_vllm_time'] != 0]

    if df.empty:
        print("Skipping vLLM speedup plot: no valid data after filtering zero division.")
        return

    df['speedup'] = df['avg_hf_time'] / df['avg_vllm_time']
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='model_name', y='speedup', hue='model_name', palette='viridis', legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title('vLLM Speedup vs. Hugging Face')
    plt.ylabel('Speedup Ratio (HF Time / vLLM Time)')
    plt.xlabel('Model')
    plt.axhline(1, color='grey', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vllm_speedup_ratio.png'))
    print(f"Saved vLLM speedup plot to {output_dir}")

def plot_throughput(df: pd.DataFrame, output_dir: str):
    """Plots a bar chart comparing throughput for each backend."""
    required_cols = ['model_name', 'vllm_throughput_tokens_per_sec', 'hf_throughput_tokens_per_sec']
    if not all(col in df.columns for col in required_cols):
        print("Skipping throughput plot: required columns missing.")
        return

    plt.figure(figsize=(12, 8))
    # Melt the dataframe to have backend as a variable
    throughput_df = df.melt(id_vars='model_name', 
                            value_vars=required_cols[1:], # only throughput columns
                            var_name='backend', value_name='throughput')
    throughput_df['backend'] = throughput_df['backend'].str.replace('_throughput_tokens_per_sec', '').str.upper()
    
    sns.barplot(data=throughput_df, x='model_name', y='throughput', hue='backend', palette='magma')
    plt.xticks(rotation=45, ha='right')
    plt.title('Backend Throughput Comparison')
    plt.ylabel('Throughput (tokens/second)')
    plt.xlabel('Model')
    plt.legend(title='Backend')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'backend_throughput.png'))
    print(f"Saved throughput comparison plot to {output_dir}")

def plot_memory_usage(df: pd.DataFrame, output_dir: str):
    """Plots a bar chart comparing peak GPU memory usage."""
    required_cols = ['model_name', 'vllm_peak_gpu_memory_gb', 'hf_peak_gpu_memory_gb']
    if not all(col in df.columns for col in required_cols):
        print("Skipping memory usage plot: required columns missing.")
        return

    plt.figure(figsize=(12, 8))
    memory_df = df.melt(id_vars='model_name',
                        value_vars=required_cols[1:], # only memory columns
                        var_name='backend', value_name='memory')
    memory_df['backend'] = memory_df['backend'].str.replace('_peak_gpu_memory_gb', '').str.upper()

    sns.barplot(data=memory_df, x='model_name', y='memory', hue='backend', palette='coolwarm')
    plt.xticks(rotation=45, ha='right')
    plt.title('Peak GPU Memory Usage Comparison')
    plt.ylabel('Peak Memory (GB)')
    plt.xlabel('Model')
    plt.legend(title='Backend')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'peak_gpu_memory.png'))
    print(f"Saved memory usage plot to {output_dir}")

def plot_performance_vs_size(df: pd.DataFrame, output_dir: str):
    """Plots a scatter plot of model parameters vs. generation time."""
    required_cols = ['model_name', 'parameters', 'avg_vllm_time', 'avg_hf_time']
    if not all(col in df.columns for col in required_cols):
        print("Skipping performance vs. size plot: required columns missing.")
        return

    plot_df = df.copy()
    # Convert parameters to numeric, coercing errors to NaN, then drop them
    plot_df['parameters_numeric'] = pd.to_numeric(plot_df['parameters'], errors='coerce')
    plot_df.dropna(subset=['parameters_numeric'], inplace=True)

    if plot_df.empty:
        print("Skipping performance vs. size plot: no valid parameter data found.")
        return

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=plot_df, x='parameters_numeric', y='avg_vllm_time', s=100, color='skyblue', label='vLLM')
    sns.scatterplot(data=plot_df, x='parameters_numeric', y='avg_hf_time', s=100, color='salmon', label='Hugging Face')
    
    plt.title('Performance vs. Model Size')
    plt.xlabel('Number of Parameters (Billions)')
    plt.ylabel('Average Generation Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_size.png'))
    print(f"Saved performance vs. size plot to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from benchmark results.")
    parser.add_argument(
        "--input-file",
        type=str,
        default="results/master_benchmark_results.csv",
        help="Path to the master benchmark results CSV file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save the generated plots."
    )
    args = parser.parse_args()

    df = load_data(args.input_file)
    if df is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_backend_comparison(df, args.output_dir)
        plot_vllm_speedup(df, args.output_dir)
        plot_throughput(df, args.output_dir)
        plot_memory_usage(df, args.output_dir)
        plot_performance_vs_size(df, args.output_dir)

if __name__ == "__main__":
    main()
