import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib as mpl

def get_model_display_name(filename):
    """Convert filename to display name based on specific prefixes."""
    base_name = os.path.basename(filename).split('.')[0]
    
    if base_name.startswith("bitnet"):
        return "T-MAC"
    elif base_name.startswith("ggml-model-I2_V"):
        return "Ours"
    elif base_name.startswith("ggml-model-Q2_K"):
        return "llama.cpp"
    else:
        return base_name  # Return original name if no match
    
    # if base_name.startswith("3B_"):
    #     return "T-MAC"
    # elif base_name.startswith("3B-I2S"):
    #     return "Ours I2_V"
    # elif base_name.startswith("3B-I2T"):
    #     return "Ours I2_T"
    # elif base_name.startswith("3B-Q2K"):
    #     return "llama.cpp Q2_K"
    # else:
    #     return base_name  # Return original name if no match

def read_all_results(script_dir=None):
    """Read all CSV files in the results folder and return a dictionary of dataframes."""
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the results folder path relative to the script directory
    folder_path = os.path.join(os.path.dirname(script_dir), "results")
    
    result_files = glob.glob(os.path.join(folder_path, "*.csv"))
    results = {}
    
    for file_path in result_files:
        model_name = get_model_display_name(os.path.basename(file_path))
        results[model_name] = pd.read_csv(file_path)
    
    # Sort results by model name in ASCII order
    results = dict(sorted(results.items()))
    
    return results, os.path.join(os.path.dirname(script_dir), "figures")

def setup_plot_style():
    """Set up plot style with Arial font and larger font sizes."""
    # Set font to Arial
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    
    # Set larger fonts
    mpl.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })

def plot_comparison_fixed_t(results_dict, output_dir, t_value=1, figsize=(10, 7)):
    """Plot 1: Compare avg_ts of all models with fixed t and varying p."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort the models by ASCII order
    for model_name in sorted(results_dict.keys()):
        df = results_dict[model_name]
        # Filter data for the specified t value
        t_data = df[df['t'] == t_value]
        if not t_data.empty:
            ax.plot(t_data['p'], t_data['avg_ts'], marker='o', linewidth=2, markersize=8, label=model_name)
    
    ax.set_xlabel('Prompt Length')
    ax.set_ylabel('Tokens per Sec')
    
    # Add title with proper spacing
    fig.suptitle(f'Performance Comparison of All Models (t={t_value})', y=0.95)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, f'comparison_t{t_value}.png'), dpi=300)
    plt.close()

def plot_model_fixed_model_varying_t(results_dict, output_dir, figsize=(10, 7)):
    """Plot 2: For each model, show avg_ts with different t values (different curves) and p on x-axis."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in sorted(results_dict.keys()):
        df = results_dict[model_name]
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique t values
        t_values = sorted(df['t'].unique())
        
        for t in t_values:
            t_data = df[df['t'] == t]
            ax.plot(t_data['p'], t_data['avg_ts'], marker='o', linewidth=2, markersize=8, label=f't={t}')
        
        ax.set_xlabel('Prompt Length')
        ax.set_ylabel('Tokens per Sec')
        
        # Add title with proper spacing
        fig.suptitle(f'Performance of {model_name} with Different Threads', y=0.95)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, f'{model_name}_varying_t.png'), dpi=300)
        plt.close()

def plot_model_fixed_model_varying_p(results_dict, output_dir, figsize=(10, 7)):
    """Plot 3: For each model, show avg_ts with different p values (different curves) and t on x-axis."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in sorted(results_dict.keys()):
        df = results_dict[model_name]
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique p values
        p_values = sorted(df['p'].unique())
        
        for p in p_values:
            p_data = df[df['p'] == p]
            ax.plot(p_data['t'], p_data['avg_ts'], marker='o', linewidth=2, markersize=8, label=f'p={p}')
        
        ax.set_xlabel('Threads')
        ax.set_ylabel('Tokens per Sec')
        
        # Add title with proper spacing
        fig.suptitle(f'Performance of {model_name} with Different Prompt Lengths', y=0.95)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, f'{model_name}_varying_p.png'), dpi=300)
        plt.close()

def generate_all_plots():
    """Generate all required plots."""
    # Set up plot style
    setup_plot_style()
    
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read all results
    results_dict, output_dir = read_all_results(script_dir)
    
    if not results_dict:
        print("No result files found in the 'results' folder.")
        return
    
    # Get all unique t values from all dataframes
    all_t_values = set()
    for df in results_dict.values():
        all_t_values.update(df['t'].unique())
    all_t_values = sorted(all_t_values)
    
    # 1. Compare models with fixed t
    for t in all_t_values:
        plot_comparison_fixed_t(results_dict, output_dir, t_value=t)
    
    # 2. For each model, show performance with different Threads
    plot_model_fixed_model_varying_t(results_dict, output_dir)
    
    # 3. For each model, show performance with different Prompt Lengths
    plot_model_fixed_model_varying_p(results_dict, output_dir)
    
    print(f"All plots have been generated successfully in {output_dir}!")

if __name__ == "__main__":
    generate_all_plots()