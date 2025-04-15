import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TYPE_MAP = {
    'i2_s': 'ours 2-bit',
    'i1_58_t': 'ours 1-bit',
    'q4_0': 'llama.cpp 4-bit',
    'tq1_0': 'llama.cpp 1-bit',
    'tq2_0': 'llama.cpp 2-bit',
}

ARCH_MAP = {
    'aws1': 'ARM Neoverse-V1 (SVE)',
    'aws2': 'Intel Xeon Platinum 8375C (AVX512)',
    'pc1': 'Intel Core i9-12900k (AVX2)',
    'laptop1': 'Intel Core Ultra 7 258V (AVX2)',
}

arch = 'pc1'
threads = 1

def parse_gemm_results(filepath):
    """Parse GEMM benchmark results into a pandas DataFrame."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            # Extract information using regex
            match = re.search(r'MUL_MAT\(type_a=(\w+),type_b=(\w+),m=(\d+),n=(\d+),k=(\d+).*?(\d+\.\d+) us/run.*?(\d+\.\d+) GFLOPS', line)
            if match:
                type_a, type_b, m, n, k, us_per_run, gflops = match.groups()
                data.append({
                    'type_a': TYPE_MAP[type_a] if type_a in TYPE_MAP else type_a,
                    'type_b': type_b,
                    'm': int(m),
                    'n': int(n),
                    'k': int(k),
                    'us_per_run': float(us_per_run),
                    'ms_per_run': float(us_per_run) / 1000,  # Convert to milliseconds
                    'runs_per_sec': 1000000 / float(us_per_run),
                    'GFLOPS': float(gflops)
                })
    return pd.DataFrame(data)

def plot_performance_comparison(df):
    """Create subplots comparing performance for each m,k combination."""
    # Get unique m,k combinations
    mk_pairs = df[['m', 'k']].drop_duplicates().values

    # print(mk_pairs)
    
    # Calculate subplot grid dimensions
    n_plots = len(mk_pairs)
    n_rows = 3  # Fixed number of rows per column
    n_cols = (n_plots + n_rows - 1) // n_rows  # Calculate needed columns
    
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16  # Base font size
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 15))

    if n_cols == 1:
        axes = axes.reshape(-1, 1)  # Ensure axes is 2D
    axes = axes.flatten()
    
    # Colors for different type_a values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (m, k) in enumerate(mk_pairs):
        # transpose idx
        x = idx // n_rows
        y = idx % n_rows
        # Select the corresponding subplot
        print(f"Plotting for m={m}, k={k} at position ({x}, {y})")
        ax = axes[y * n_cols + x]

        subset = df[(df['m'] == m) & (df['k'] == k)]
        
        # Plot lines for each type_a
        for i, type_a in enumerate(subset['type_a'].unique()):
            type_data = subset[subset['type_a'] == type_a]
            ax.plot(type_data['n'], type_data['runs_per_sec'], 
                   marker='o', label=type_a, color=colors[i % len(colors)])
        
        ax.set_title(f'm={m}, k={k}', fontsize=24, pad=10)
        ax.set_xlabel('n (sequence length)', fontsize=20)
        ax.set_ylabel('speed (runs/sec)', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=20)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Remove any unused subplots
    for idx in range(len(mk_pairs), len(axes)):
        fig.delaxes(axes[idx])
    
    # fig.suptitle(f'GEMM Performance Comparison on {ARCH_MAP[arch]}', fontsize=32, y=0.95)
    # Add space at the top for the title
    plt.subplots_adjust(top=0.9)
    fig.suptitle(f'GEMM benchmark on {ARCH_MAP[arch]}, {threads} cores', fontsize=32)
    plt.tight_layout()

    return fig

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), 'results_gemm')
    
    # Define combinations of arch and threads to plot
    combinations = [
        # ('pc1', 1),
        # ('pc1', 8),
        # ('aws1', 1),
        # ('aws2', 1),
        ('laptop1', 1),
    ]
    
    for arch_val, threads_val in combinations:
        # Read and process results
        results_file = os.path.join(results_dir, f'results_{arch_val}_t{threads_val}.txt')
        if not os.path.exists(results_file):
            print(f"Skipping {results_file} - file not found")
            continue
            
        global arch, threads
        arch = arch_val
        threads = threads_val
        
        df = parse_gemm_results(results_file)
        
        # Create plots
        fig = plot_performance_comparison(df)
        
        # Save plot
        output_file = os.path.join(results_dir, f'gemm_perf_{arch_val}_t{threads_val}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    main()