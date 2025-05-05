import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.ticker import MaxNLocator
import load_tmac_gemm_log as load_tmac
from plot_utils import *
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot GeMM Benchmarks')
    parser.add_argument('--multi-thread', action='store_true', default=False, 
                        help='Use multi-threaded configuration for each architecture')
    parser.add_argument('--single-thread', action='store_true', default=False, 
                        help='Use single-threaded configuration for each architecture')
    parser.add_argument('--both', action='store_true', default=True,
                       help='Plot both single-thread and multi-thread configurations separately')
    return parser.parse_args()

combinations_to_plot = [
    (3200, 3200, 256),
    (4096, 4096, 256),
    (3200, 8640, 256),
    (4096, 14336, 256),
    (8640, 3200, 256),
    (14336, 4096, 256),
]

# List of architectures to include
all_archs = [
    'aws_arm',
    'smartphone',
    'pc_intel',
    'laptop_amd',
    'orangepi'
]

# Multi-thread configuration for each architecture
MULTI_THREAD_CONFIG = {
    'aws_arm': 8,
    'smartphone': 2,
    'pc_intel': 4,
    'laptop_amd': 4,
    'orangepi': 4
}

def load_adapt_tmac(tmac_arch: str):
    df_tmac = load_tmac.load_and_process_results(tmac_arch)
    # calculate rps with latency_s
    df_tmac['runs_per_sec'] = 1 / df_tmac['total_latency_s']
    # add a type_a column
    df_tmac['type_a'] = GEMM_TYPE_MAP['tmac']
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
        df['type_a'] = df['type_a'].apply(lambda x: GEMM_TYPE_MAP.get(x, x))
        
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
        'model': GEMM_MODEL_MAP.get(model, model),
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

def load_results_for_all_archs(archs_to_load, threads_config=None):
    """Load results for multiple architectures."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dict = {}
    
    for arch in archs_to_load:
        results_dir = os.path.join(os.path.dirname(script_dir), f'results_gemm_{arch}')
        # Load results for this architecture
        df = load_all_results(results_dir)
        
        # Try to append tmac results if available
        try:
            df_tmac = load_adapt_tmac(arch)
            df = pd.concat([df, df_tmac], ignore_index=True)
        except Exception as e:
            print(f"Error loading T-MAC results for {arch}: {e}")
            print(f"Skip T-MAC results for {arch}.")
        
        if not df.empty:
            # Filter by thread value if specified in threads_config
            if threads_config and arch in threads_config:
                threads_val = threads_config[arch]
                df = df[df['threads'] == threads_val]
            
            # Store in dictionary
            results_dict[arch] = df
            print(f"Loaded results for {arch}")
        else:
            print(f"No results found for {arch}")
    
    return results_dict

def plot_multi_arch_comparison(results_dict, mkn_to_plot=None, lut2_on=None, entry_size=None, thread_mode="auto"):
    """Create a plot comparing performance across multiple architectures."""
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16  # Base font size
    
    archs = list(results_dict.keys())
    n_archs = len(archs)
    
    # Each arch gets 2 columns, with 3 rows per architecture
    n_cols_per_arch = 2
    n_rows = 3
    cols_total = n_archs * n_cols_per_arch
    
    # Create the figure
    fig = plt.figure(figsize=(6*n_archs, 10))
    
    # First, find all unique type_a values across all datasets
    all_type_a_values = set()
    for arch, df in results_dict.items():
        # Apply filters
        if lut2_on is not None:
            df = df[df['lut2_on'] == lut2_on]
        if entry_size is not None:
            df = df[df['entry_size'] == entry_size]
        
        # Collect unique type_a values
        for m, k, n in mkn_to_plot:
            subset = df[(df['m'] == m) & (df['k'] == k) & (df['n'] == n)]
            for type_a in subset['type_a'].unique():
                all_type_a_values.add(type_a)
    
    # Sort the types for consistent ordering
    all_type_a_values = sorted(all_type_a_values)

    
    # Create the subplots grid for data only
    gs = fig.add_gridspec(n_rows, cols_total)
    
    # Define subplot adjustment parameters
    left_margin = 0.1
    right_margin = 0.95
    bottom_margin = 0.1
    top_margin = 0.9
    wspace = 0.3
    hspace = 0.4
    
    # Apply subplot adjustments
    fig.subplots_adjust(left=left_margin, right=right_margin, 
                      bottom=bottom_margin, top=top_margin,
                      wspace=wspace, hspace=hspace)
    
    # Create the actual plot axes for data
    axes = []
    for row in range(n_rows):
        for arch_idx, arch in enumerate(archs):
            # Get data for this arch
            df = results_dict[arch]
            
            # Calculate column start for this architecture
            col_start = arch_idx * n_cols_per_arch
            
            # Create the two plots for this row
            for col_offset in range(n_cols_per_arch):
                # Calculate linear index into mkn_to_plot
                plot_idx = row * n_cols_per_arch + col_offset
                
                # Skip if we've run out of mkn combinations
                if plot_idx >= len(mkn_to_plot):
                    continue
                
                m, k, n = mkn_to_plot[plot_idx]
                
                # Create the axis
                ax = fig.add_subplot(gs[row, col_start+col_offset])
                axes.append(ax)
                
                # Filter data for this m,k,n combination
                subset = df[(df['m'] == m) & (df['k'] == k) & (df['n'] == n)]
                if lut2_on is not None:
                    subset = subset[subset['lut2_on'] == lut2_on]
                if entry_size is not None:
                    subset = subset[subset['entry_size'] == entry_size]
                
                subset = subset.drop_duplicates(subset=['m', 'n', 'k', 'type_a'], keep='first')
                
                # If we have data, plot it
                if not subset.empty:
                    # Sort by type_a for consistent ordering
                    subset = subset.sort_values('type_a')
                    
                    # Prepare for bar chart
                    type_a_values = subset['type_a'].values
                    x_pos = np.arange(len(type_a_values))
                    performance = subset['runs_per_sec'].values
                    
                    # Create bar chart with custom colors and patterns
                    bars = []
                    for i, type_a in enumerate(type_a_values):
                        style = GEMM_TYPE_STYLES.get(type_a, {'color': '#000000', 'hatch': ''})
                        bar = ax.bar(x_pos[i], performance[i], width=0.8, linewidth=1.5,
                                    color=style['color'], hatch=style['hatch'],
                                    edgecolor='black', zorder=3)
                        bars.append(bar)
                    
                    # Set y-axis ticks
                    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
                
                # Disable x-axis label, ticks, and tick labels for all subplots
                ax.set_xlabel('')
                ax.set_xticks([])
                ax.set_xticklabels([])
                
                # Add padding to the left and right of the bar group
                xlim = ax.get_xlim()
                padding = 0.5
                new_xlim = (xlim[0] - padding, xlim[1] + padding)
                ax.set_xlim(new_xlim)
                
                # Set title showing m×k×n combination
                ax.set_title(f'{m}×{k}×{n}', fontsize=20, pad=10)
                
                # Add y-axis label only to the leftmost subplots in each row
                if arch_idx == 0 and row == 1 and col_offset == 0:
                    ax.set_ylabel('Speed (runs/sec)', fontsize=24, fontweight='bold', labelpad=10)
                
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='y', which='major', labelsize=18)
    
    # Now add the device titles AFTER the subplots are created
    for arch_idx, arch in enumerate(archs):
        # Calculate the actual position of the first subplot for this architecture
        col_start = arch_idx * n_cols_per_arch
        
        # Get first two subplot positions for this architecture
        first_ax = fig.get_axes()[col_start]
        second_ax = fig.get_axes()[col_start + 1]
        
        # Find center position between these subplots in figure coordinates
        first_bbox = first_ax.get_position()
        second_bbox = second_ax.get_position()
        
        # Calculate center between the two axes
        center_x = (first_bbox.x0 + second_bbox.x1) / 2
        
        # Position slightly above the first row of subplots
        title_y = top_margin + 0.08  # Small gap above the plots
        
        # Create device name text
        device_name = DEVICE_MAP.get(arch, arch)
        # Add thread count if available
        if thread_mode != "auto":
            for df_arch, df in results_dict.items():
                if df_arch == arch and not df.empty:
                    threads = df['threads'].iloc[0]
                    device_name = f"{device_name}"
                    break
                    
        fig.text(center_x, title_y, device_name, fontsize=22, ha='center', va='center', color='black', fontweight='bold')
    
        
    # Replace your legend creation code with this:
    legend_handles = []
    legend_labels = []

    # Create a separate invisible axes for legend handles
    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])

    for type_a in all_type_a_values:
        style = GEMM_TYPE_STYLES.get(type_a, {'color': '#000000', 'hatch': ''})
        # Create a small bar with proper hatching
        dummy_bar = legend_ax.bar(0, 0, color=style['color'], hatch=style['hatch'], 
                                edgecolor='black')
        legend_handles.append(dummy_bar[0])
        legend_labels.append(type_a)
        
    # Hide the dummy axis
    legend_ax.set_visible(False)

    # Add a single legend for the entire figure at the bottom
    fig.legend(handles=legend_handles, labels=legend_labels, 
              loc='lower center', ncol=min(len(legend_labels), 8),
              fontsize=20, frameon=True, bbox_to_anchor=(0.5, 0.02),
              columnspacing=2.0)
    
    # Add overall title based on thread mode
    # if thread_mode == "single":
    #     fig.suptitle('GeMM Performance Comparison (Single-threaded)', fontsize=26, y=0.98)
    # elif thread_mode == "multi":
    #     fig.suptitle('GeMM Performance Comparison (Multi-threaded)', fontsize=26, y=0.98)
    
    return fig

def main():
    args = parse_arguments()
    
    # Determine thread configuration and plotting mode
    if args.both:
        # Plot both single-thread and multi-thread separately
        # First, do single-thread
        single_threads_config = {arch: 1 for arch in all_archs}
        results_dict_single = load_results_for_all_archs(all_archs, single_threads_config)
        
        fig_single = plot_multi_arch_comparison(
            results_dict_single,
            mkn_to_plot=combinations_to_plot,
            thread_mode="single"
        )
        
        # Save the single-thread plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file_single = os.path.join(os.path.dirname(script_dir), 'figures/gemm_comparison_single_thread.png')
        fig_single.savefig(output_file_single, dpi=300, bbox_inches='tight')
        print(f"Single-thread comparison plot saved to {output_file_single}")
        
        # Second, do multi-thread
        results_dict_multi = load_results_for_all_archs(all_archs, MULTI_THREAD_CONFIG)
        
        fig_multi = plot_multi_arch_comparison(
            results_dict_multi,
            mkn_to_plot=combinations_to_plot,
            thread_mode="multi"
        )
        
        # Save the multi-thread plot
        output_file_multi = os.path.join(os.path.dirname(script_dir), 'figures/gemm_comparison_multi_thread.png')
        fig_multi.savefig(output_file_multi, dpi=300, bbox_inches='tight')
        print(f"Multi-thread comparison plot saved to {output_file_multi}")
        
    else:
        # Original functionality
        if args.single_thread:
            thread_mode = "single"
            threads_config = {arch: 1 for arch in all_archs}
            title_suffix = "single_thread"
        elif args.multi_thread:
            thread_mode = "multi"
            threads_config = MULTI_THREAD_CONFIG
            title_suffix = "multi_thread"
        else:
            thread_mode = "auto"
            threads_config = None
            title_suffix = "auto_thread"
        
        # Load results for all architectures
        results_dict = load_results_for_all_archs(all_archs, threads_config)
        
        # If no thread configuration specified, find best for each architecture
        if threads_config is None:
            # For each architecture, find the maximum thread count with data
            auto_threads = {}
            for arch, df in results_dict.items():
                if not df.empty:
                    thread_counts = sorted(df['threads'].unique())
                    if thread_counts:
                        auto_threads[arch] = max(thread_counts)
                        print(f"Using {auto_threads[arch]} threads for {arch}")
                        
            # Update results_dict to use auto-selected thread counts
            if auto_threads:
                results_dict = load_results_for_all_archs(all_archs, auto_threads)
        
        # Create multi-architecture comparison plot
        fig = plot_multi_arch_comparison(
            results_dict,
            mkn_to_plot=combinations_to_plot,
            thread_mode=thread_mode
        )
        
        # Save the plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(os.path.dirname(script_dir), f'figures/gemm_comparison_{title_suffix}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Multi-architecture comparison plot saved to {output_file}")

if __name__ == '__main__':
    main()