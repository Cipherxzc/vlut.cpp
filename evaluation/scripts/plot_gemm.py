import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.ticker import MaxNLocator
import load_tmac_gemm_log as load_tmac

TYPE_MAP = {
    'i2_s': 'ours 2-bit',
    'i1_58_m': 'ours 1.58-bit',
    'tmac': 'T-MAC 2-bit',
    'q4_0': 'llama.cpp 4-bit',
    'tq1_0': 'llama.cpp 1.58-bit',
    'tq2_0': 'llama.cpp 2-bit',
}

# Create a mapping dictionary where keys are characters and values are their positions
TYPE_ORDER_MAP = {
    TYPE_MAP['i2_s']: 0,
    TYPE_MAP['i1_58_m']: 1,
    TYPE_MAP['tmac']: 2,
    TYPE_MAP['q4_0']: 3,
    TYPE_MAP['tq1_0']: 4,
    TYPE_MAP['tq2_0']: 5,
}

ARCH_MAP = {
    'aws_arm': 'ARM Neoverse-V1 (SVE)',
    # 'aws2': 'Intel Xeon Platinum 8375C (AVX512)',
    'pc_intel': 'Intel Core i7-13700k (AVX2)',
    'laptop_amd': 'AMD Ryzen 7 5800H (AVX2)',
    # 'laptop1': 'Intel Core Ultra 7 258V (AVX2)',
    'smartphone': 'Qualcomm Snapdragon 8 Elite (NEON)'
}

MODEL_MAP = {
    'bitnet_3b': 'BitNet 3B',
    'falcon_1b': 'Falcon 1B',
    'trilm_1.5b': 'TriLM 1.5B',
    'llama3_8b': 'LLaMA 3 8B',
    # Add more models as needed
}

combinations_to_plot = [
    (3200, 3200, 256),
    (3200, 8640, 256),
    (8640, 3200, 256),
    (4096, 4096, 256),
    (4096, 14336, 256),
    (14336, 4096, 256),
]

arch = 'aws_arm'
# arch = 'pc_intel'

def load_adapt_tmac(tmac_arch: str):
    df_tmac = load_tmac.load_and_process_results(tmac_arch)
    # calculate rps with latency_s
    df_tmac['runs_per_sec'] = 1 / df_tmac['total_latency_s']
    # add a type_a column
    df_tmac['type_a'] = TYPE_MAP['tmac']
    # remove cols
    # df_tmac = df_tmac.drop(columns=['device', 'latency_s'])

    return df_tmac


def parse_gemm_results(filepath):
    """Parse GEMM benchmark results from CSV file into a pandas DataFrame."""
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_columns = ['name', 'm', 'n', 'k', 'uspr', 'rps']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing required columns in {filepath}: {missing_columns}")
            return pd.DataFrame()
        
        # Rename columns to match the expected format
        df = df.rename(columns={
            'name': 'type_a',
            'uspr': 'us_per_run',
            'rps': 'runs_per_sec'
        })
        
        # Add additional calculated columns
        df['ms_per_run'] = df['us_per_run'] / 1000  # Convert to milliseconds
        
        # Map type_a to friendly names if available
        df['type_a'] = df['type_a'].apply(lambda x: TYPE_MAP.get(x, x))
        
        return df
        
    except Exception as e:
        print(f"Error parsing file {filepath}: {str(e)}")
        return pd.DataFrame()

def extract_file_metadata(filename):
    """Extract metadata from filename pattern."""
    # Get the basename without extension
    basename = os.path.basename(filename)
    basename_no_ext = os.path.splitext(basename)[0]
    
    # Split pattern into parts based on specific markers
    # Look for thread marker 't' followed by number
    thread_pattern = re.search(r'_t(\d+)_', basename_no_ext)
    if not thread_pattern:
        return None
    
    thread_pos = thread_pattern.start()
    threads = int(thread_pattern.group(1))
    
    # Extract model name (everything before the thread marker)
    model = basename_no_ext[:thread_pos]
    
    # Extract 'ns' part for n values
    ns_pattern = re.search(r'_ns([\d-]+)_', basename_no_ext)
    if not ns_pattern:
        return None
    
    n_values_str = ns_pattern.group(1)
    n_values = [int(n) for n in n_values_str.split('-')]
    
    # Extract lut2 status
    lut2_pattern = re.search(r'_(lon|loff)_', basename_no_ext)
    if not lut2_pattern:
        return None
    
    lut2_on = lut2_pattern.group(1) == 'lon'
    
    # Extract entry size
    size_pattern = re.search(r'_s(\d+)$', basename_no_ext)
    if not size_pattern:
        return None
    
    entry_size = int(size_pattern.group(1))
    
    return {
        'model': MODEL_MAP.get(model, model),
        'threads': threads,
        'n_values': n_values,
        'lut2_on': lut2_on,
        'entry_size': entry_size
    }

def load_all_results(results_dir):
    """Load all CSV results files from the directory."""
    all_data = []
    
    for csv_file in glob.glob(os.path.join(results_dir, '*.csv')):
        # Extract metadata from filename
        metadata = extract_file_metadata(csv_file)
        if metadata is None:
            print(f"Couldn't extract metadata from {csv_file}, skipping")
            continue
            
        # Read the CSV content
        df = parse_gemm_results(csv_file)
        if df.empty:
            continue

        # Add metadata columns
        df['model'] = metadata['model']
        df['threads'] = metadata['threads']
        df['lut2_on'] = metadata['lut2_on']
        df['entry_size'] = metadata['entry_size']
        
        all_data.append(df)
    
    # Combine all dataframes
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def plot_performance_comparison(df, arch_val, threads_val, mkn_to_plot=None, lut2_on=None, entry_size=None):
    """Create subplots comparing performance for specific m,k,n combinations using bar charts."""
    # Filter data for specific configuration if provided
    if lut2_on is not None:
        df = df[df['lut2_on'] == lut2_on]
    if entry_size is not None:
        df = df[df['entry_size'] == entry_size]
    
    df = df[df['threads'] == threads_val]
    
    # Get unique m,k,n combinations
    if mkn_to_plot is None:
        # Plot all combinations if none specified
        mkn_combinations = df[['m', 'k', 'n']].drop_duplicates().values
    else:
        # Only plot specified m,k,n combinations
        mkn_combinations = []
        for m, k, n in mkn_to_plot:
            if any(((df['m'] == m) & (df['k'] == k) & (df['n'] == n)).values):
                mkn_combinations.append((m, k, n))
        
        if not mkn_combinations:
            raise ValueError("None of the specified m,k,n combinations found in the data")
    
    # Calculate subplot grid dimensions
    n_plots = len(mkn_combinations)
    n_rows = 1  # Fixed number of rows per column
    n_cols = (n_plots + n_rows - 1) // n_rows  # Calculate needed columns
    
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16  # Base font size
    
    # Create figure with extra space at the bottom for the legend
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5))

    if n_plots <= 1:
        axes = np.array([axes])  # Ensure axes is an array
    
    axes = axes.flatten()
    
    # Create a consistent color map for type_a values
    # First, collect all unique type_a values across all m,k,n combinations we'll plot
    all_type_a_values = set()
    for m, k, n in mkn_combinations:
        subset = df[(df['m'] == m) & (df['k'] == k) & (df['n'] == n)]
        for type_a in subset['type_a'].unique():
            all_type_a_values.add(type_a)
    
    # Sort the types for consistent ordering
    all_type_a_values = sorted(all_type_a_values)
    # all_type_a_values = sorted(all_type_a_values, key=lambda s: TYPE_ORDER_MAP.get(s[0], len(TYPE_ORDER_MAP)))
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_type_a_values)))
    type_color_map = {type_a: colors[i] for i, type_a in enumerate(all_type_a_values)}
    
    # Prepare legend data
    legend_handles = []
    legend_labels = []
    
    for i, type_a in enumerate(all_type_a_values):
        legend_handles.append(plt.Rectangle((0,0), 1, 1, color=type_color_map[type_a], edgecolor='black'))
        legend_labels.append(type_a)
    
    for idx, (m, k, n) in enumerate(mkn_combinations):
        # Calculate position
        x = idx // n_rows
        y = idx % n_rows
        # Select the corresponding subplot
        print(f"Plotting for m={m}, k={k}, n={n} at position ({x}, {y})")
        ax = axes[y * n_cols + x]

        # Get data for this m,k,n combination
        subset = df[(df['m'] == m) & (df['k'] == k) & (df['n'] == n)]
        subset = subset.drop_duplicates(subset=['m', 'n', 'k', 'type_a'], keep='first')  # Drop duplicates
        
        # Sort by type_a to ensure consistent ordering
        subset = subset.sort_values('type_a')
        
        # Prepare for bar chart
        type_a_values = subset['type_a'].values
        x_pos = np.arange(len(type_a_values))
        performance = subset['runs_per_sec'].values
        
        # Create bar chart with consistent colors based on type_a
        bars = ax.bar(x_pos, performance, width=0.8, edgecolor='black', zorder=3,
                     color=[type_color_map[t] for t in type_a_values])
        
        # Disable x-axis label, ticks, and tick labels
        ax.set_xlabel('')
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add padding to the left and right of the bar group
        xlim = ax.get_xlim()
        padding = 0.5
        new_xlim = (xlim[0] - padding, xlim[1] + padding)
        ax.set_xlim(new_xlim)
        
        ax.set_title(f'm={m}, k={k}, n={n}', fontsize=24, pad=10)
        ax.set_ylabel('Speed (runs/sec)', fontsize=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        ax.tick_params(axis='y', which='major', labelsize=18)
    
    # Remove any unused subplots
    for idx in range(len(mkn_combinations), len(axes)):
        fig.delaxes(axes[idx])
    
    # Add a single legend for the entire figure at the bottom
    # This replaces the previous legend code
    # fig.subplots_adjust(bottom=0.2)  # Make room for legend at the bottom
    fig.subplots_adjust(bottom=0.24, top=0.8, left=0.1, right=0.95, wspace=0.3, hspace=0.4)
    fig.legend(handles=legend_handles, labels=legend_labels, 
            loc='lower center', ncol=len(legend_labels),  # Force all items into one row
            fontsize=24, frameon=True, bbox_to_anchor=(0.5, 0.05),
            columnspacing=3.0)  # Adjust spacing between legend items
    
    # # Add information about LUT2 and entry size to title if provided
    # lut2_info = f", LUT2 {'on' if lut2_on else 'off'}" if lut2_on is not None else ""
    # entry_size_info = f", Entry size {entry_size}" if entry_size is not None else ""
    
    # plt.subplots_adjust(top=0.85)
    
    fig.suptitle(f'GeMM benchmark on {ARCH_MAP[arch_val]}, {threads_val} {"cores" if threads_val > 1 else "core"}', 
                fontsize=32)
    
    # plt.tight_layout(rect=[0, 0.1, 1, 0.88])  # Reduce the top margin 
    # plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout but leave space for legend

    return fig

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), f'results_gemm_{arch}')
    
    # Load all results
    df = load_all_results(results_dir)

    # append tmac results
    try:
        df_tmac = load_adapt_tmac(arch)
        df = pd.concat([df, df_tmac], ignore_index=True)
    except Exception as e:
        print(f"Error loading T-MAC results: {e}")
        print(f"Skip T-MAC results.")
    # df_tmac = load_adapt_tmac(arch)
    # df = pd.concat([df, df_tmac], ignore_index=True)
    
    if df.empty:
        print("No results found in the specified directory.")
        return
    
    # Get unique configurations to plot
    # configs = df[['threads', 'lut2_on', 'entry_size']].drop_duplicates().values
    configs = df['threads'].drop_duplicates().values
    
    # for threads_val, lut2_on, entry_size in configs:
    for threads_val in configs:
        # Create plot
        fig = plot_performance_comparison(
            df, 
            arch_val=arch, 
            threads_val=threads_val, 
            mkn_to_plot=combinations_to_plot,
            # lut2_on=lut2_on,
            # entry_size=entry_size
        )
        
        # Create filename
        # lut2_str = 'lon' if lut2_on else 'loff'
        output_file = os.path.join(results_dir, f'gemm_{arch}_t{threads_val}.png')
        
        # Save plot
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    main()