import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.ticker import MaxNLocator
from plot_utils import *
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot End-to-End Prefill Benchmarks')
    parser.add_argument('--multi-thread', action='store_true', default=False, 
                        help='Use multi-threaded configuration for each architecture')
    parser.add_argument('--single-thread', action='store_true', default=False, 
                        help='Use single-threaded configuration for each architecture')
    parser.add_argument('--both', action='store_true', default=True,
                       help='Plot both single-thread and multi-thread configurations separately')
    return parser.parse_args()

# List of architectures to include
all_archs = [
    'aws_arm',
    'smartphone',
    'pc_intel',
    'laptop_amd',
    'orangepi'
]

# Models to include in the plots
models_to_plot = [
    'BitNet 3B', 
    'Llama3 8B', 
    'Falcon 1B'
]

# Multi-thread configuration for each architecture
MULTI_THREAD_CONFIG = {
    'aws_arm': 8,
    'smartphone': 2,
    'pc_intel': 4,
    'laptop_amd': 4,
    'orangepi': 4
}

def read_csv_files(directory):
    """Read all CSV files in directory and subdirectories into a single DataFrame."""
    all_data = []
    failed_files = []
    
    # Find all CSV files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
    print(f"Found {len(csv_files)} CSV files in {directory}")
    
    for csv_file in csv_files:
        try:
            # Get the directory name as model_name (the parent folder of the CSV file)
            parent_dir = os.path.basename(os.path.dirname(csv_file))
            model_name = parent_dir
            
            # Get the basename of the file
            basename = os.path.basename(csv_file)
            
            # Extract the part before _p as model_quant
            if '_p' in basename:
                model_quant = basename.split('_p')[0]
                if model_quant.startswith('ggml-model'):
                    model_quant = model_quant.split('-')[-1] # others
                    if model_quant == "TQ2_0" or model_quant == "TQ1_0":
                        if E2E_MODEL_MAP[model_name] == "BitNet 3B":
                            model_quant = "Q4_0"
                else:
                    model_quant = model_quant.split('.')[-1] # T-MAC
            else:
                model_quant = None
                failed_files.append(csv_file)
                continue
            
            # Extract date from filename (assuming it's always in format YYYYMMDD)
            date_pattern = r'_(\d{8})_'
            date_match = re.search(date_pattern, basename)
            date = date_match.group(1) if date_match else None
            
            # Read CSV data
            df = pd.read_csv(csv_file)
            
            # Add metadata columns
            df['model_name'] = E2E_MODEL_MAP.get(model_name, model_name)
            df['model_quant'] = E2E_TYPE_MAP.get(model_quant, model_quant)
            
            if not df.empty:
                all_data.append(df)
            else:
                print(f"DataFrame of {basename} is empty")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            failed_files.append(csv_file)
    
    # Report on parsing success/failure
    if failed_files:
        print(f"Could not process {len(failed_files)} files:")
        for f in failed_files[:5]:  # Show first 5 failed files
            print(f"  - {os.path.basename(f)}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    # Combine all dataframes
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Successfully processed {len(all_data)} files")
        return combined
    return pd.DataFrame()

def load_results_for_all_archs(archs_to_load, thread_config=None):
    """Load results for multiple architectures."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dict = {}
    
    for arch in archs_to_load:
        results_dir = os.path.join(os.path.dirname(script_dir), f'results_e2e_prefill_{arch}')
        # Load results for this architecture
        df = read_csv_files(results_dir)
        
        if not df.empty:
            # Filter by thread value if specified in thread_config
            if thread_config and arch in thread_config:
                thread_val = thread_config[arch]
                df = df[df['t'] == thread_val]
            
            # Store in dictionary
            results_dict[arch] = df
            print(f"Loaded results for {arch}")
        else:
            print(f"No results found for {arch}")
    
    return results_dict
def plot_all_archs_e2e_prefill(results_dict, model_names=None, thread_mode="auto"):
    """
    Create a plot comparing prefill performance across multiple architectures.
    Each arch gets one column, with one row per model.
    Bar plot version - groups by prompt length with proper handling of available quantizations.
    
    Parameters:
    results_dict: Dictionary of DataFrames with results for each architecture
    model_names: List of models to include in the plot
    thread_mode: String indicating thread mode for title
    """
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16  # Base font size
    
    archs = list(results_dict.keys())
    n_archs = len(archs)
    
    # Filter for models to plot
    if model_names is None:
        all_models = set()
        for df in results_dict.values():
            all_models.update(df['model_name'].unique())
        model_names = sorted(all_models)
    
    n_models = len(model_names)
    
    # Create figure with 1 column per arch, 1 row per model
    fig = plt.figure(figsize=(6*n_archs, 3.5*n_models))
    
    # Create a grid for subplots
    gs = fig.add_gridspec(n_models, n_archs)
    
    # Define subplot adjustment parameters
    left_margin = 0.1
    right_margin = 0.85  # Increase right margin to leave room for model names
    bottom_margin = 0.2
    top_margin = 0.95
    wspace = 0.3
    hspace = 0.4
    
    # Apply subplot adjustments
    fig.subplots_adjust(left=left_margin, right=right_margin, 
                      bottom=bottom_margin, top=top_margin,
                      wspace=wspace, hspace=hspace)
    
    # Find all unique quantization values across all datasets
    all_quants = set()
    for df in results_dict.values():
        if not df.empty:
            all_quants.update(df['model_quant'].unique())
    
    # Sort the quantization values for consistent ordering
    all_quants = sorted(all_quants)
    
    # Prepare legend data
    legend_handles = []
    legend_labels = []
    
    # Find all unique prompt lengths across all datasets and models
    all_prompt_lengths = set()
    for df in results_dict.values():
        if not df.empty:
            all_prompt_lengths.update(df['p'].unique())
    all_prompt_lengths = sorted(all_prompt_lengths)
    
    # Create plots for each architecture and model
    for col_idx, arch in enumerate(archs):
        df = results_dict[arch]
        
        for row_idx, model in enumerate(model_names):
            # Create subplot
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Get data for this model on this architecture
            subset = df[df['model_name'] == model]
            
            # Set up the x positions for the prompt length groups
            x_positions = np.arange(len(all_prompt_lengths))
            
            if not subset.empty:
                # For each prompt length
                for p_idx, p_length in enumerate(all_prompt_lengths):
                    # Get data for this prompt length
                    p_data = subset[subset['p'] == p_length]
                    
                    # Get the quantization types available for this model/arch/prompt length
                    available_quants = sorted(p_data['model_quant'].unique())
                    
                    if not available_quants:
                        continue  # Skip if no data for this prompt length
                    
                    # Calculate position for grouped bars
                    num_quants = len(available_quants)
                    total_width = 0.8  # Total width of the group
                    bar_width = total_width / num_quants if num_quants > 0 else total_width
                    
                    # For each available quantization
                    for q_idx, quant in enumerate(available_quants):
                        # Get data for this quantization
                        quant_data = p_data[p_data['model_quant'] == quant]
                        
                        if not quant_data.empty:
                            # Calculate position - center the group around the x position
                            if num_quants % 2 == 0:  # Even number of bars
                                start = x_positions[p_idx] - (bar_width * num_quants) / 2 + bar_width / 2
                            else:  # Odd number of bars
                                start = x_positions[p_idx] - (bar_width * (num_quants - 1)) / 2
                            
                            pos = start + q_idx * bar_width
                            
                            # Get style from E2E_TYPE_STYLES mapping
                            style = E2E_TYPE_STYLES.get(quant, {'color': '#000000', 'hatch': ''})
                            
                            # Plot the bar with appropriate style
                            ax.bar(
                                pos,
                                quant_data['avg_ts'].values[0],
                                width=bar_width,
                                color=style['color'],
                                hatch=style['hatch'],
                                edgecolor='black',
                                linewidth=1.5,
                                align='center',
                                zorder=3
                            )
            
            # Set x-ticks at the center of each prompt length group
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(p) for p in all_prompt_lengths])
            
            # Add labels
            if col_idx == 0 and row_idx == 1:
                ax.set_ylabel('Throughput (tokens/s)', fontsize=24, fontweight='bold')
            if row_idx == n_models - 1:
                ax.set_xlabel('Prompt length (tokens)', fontsize=20)
            
            # Add model name to right side of the last column's plots
            if col_idx == n_archs - 1:
                ax.text(1.05, 0.5, model, transform=ax.transAxes, 
                        fontsize=20, va='center', ha='left', fontweight='bold', rotation=270)
            
            # Grid and formatting
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.set_ylim(0)  # Start y-axis at 0
            
            # Ensure x-axis limits show all groups
            ax.set_xlim(min(x_positions) - 0.5, max(x_positions) + 0.5)
    
    # Now add the device titles at the top of each column
    for arch_idx, arch in enumerate(archs):
        # Calculate the correct index for the first subplot in this column
        first_ax_idx = row_idx * arch_idx
        
        if first_ax_idx < len(fig.get_axes()):
            first_ax = fig.get_axes()[arch_idx * n_models]  # Corrected indexing
            
            # Get the position of the first subplot in this column
            bbox = first_ax.get_position()
            
            # Calculate center of the subplot
            center_x = bbox.x0 + bbox.width/2
            
            # Position slightly above the first subplot
            center_y = bbox.y1 + 0.02
            
            # Create device name text
            device_name = DEVICE_MAP.get(arch, arch)
                        
            fig.text(center_x, center_y, device_name, fontsize=22, ha='center', va='bottom', 
                    color='black', fontweight='bold')
    
    # Create a separate invisible axes for legend handles
    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    
    for quant in all_quants:
        # Get style from E2E_TYPE_STYLES mapping (similar to GEMM_TYPE_STYLES)
        style = E2E_TYPE_STYLES.get(quant, {'color': '#000000', 'hatch': ''})
        
        # Create a small bar with proper styling
        dummy_bar = legend_ax.bar(0, 0, color=style['color'], hatch=style['hatch'], 
                                edgecolor='black')
        legend_handles.append(dummy_bar[0])
        legend_labels.append(quant)
    
    # Hide the dummy axis
    legend_ax.set_visible(False)

    # Add a single legend for the entire figure at the bottom
    fig.legend(
        handles=legend_handles, 
        labels=legend_labels, 
        loc='lower center', 
        ncol=min(len(legend_labels), 4),
        fontsize=18, 
        frameon=True, 
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.0
    )
    
    return fig

def main():
    args = parse_arguments()
    
    # Determine thread configuration and plotting mode
    if args.both:
        # Plot both single-thread and multi-thread separately
        # First, do single-thread
        single_thread_config = {arch: 1 for arch in all_archs}
        results_dict_single = load_results_for_all_archs(all_archs, single_thread_config)
        
        if results_dict_single:
            fig_single = plot_all_archs_e2e_prefill(
                results_dict_single,
                model_names=models_to_plot,
                thread_mode="single"
            )
            
            # Save the single-thread plot
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_file_single = os.path.join(os.path.dirname(script_dir), 'figures/e2e_prefill_comparison_single_thread.png')
            fig_single.savefig(output_file_single, dpi=300, bbox_inches='tight')
            print(f"Single-thread comparison plot saved to {output_file_single}")
        
        # Second, do multi-thread
        results_dict_multi = load_results_for_all_archs(all_archs, MULTI_THREAD_CONFIG)
        
        if results_dict_multi:
            fig_multi = plot_all_archs_e2e_prefill(
                results_dict_multi,
                model_names=models_to_plot,
                thread_mode="multi"
            )
            
            # Save the multi-thread plot
            output_file_multi = os.path.join(os.path.dirname(script_dir), 'figures/e2e_prefill_comparison_multi_thread.png')
            fig_multi.savefig(output_file_multi, dpi=300, bbox_inches='tight')
            print(f"Multi-thread comparison plot saved to {output_file_multi}")
        
    else:
        # Original functionality
        if args.single_thread:
            thread_mode = "single"
            thread_config = {arch: 1 for arch in all_archs}
            title_suffix = "single_thread"
        elif args.multi_thread:
            thread_mode = "multi"
            thread_config = MULTI_THREAD_CONFIG
            title_suffix = "multi_thread"
        else:
            thread_mode = "auto"
            thread_config = None
            title_suffix = "auto_thread"
        
        # Load results for all architectures
        results_dict = load_results_for_all_archs(all_archs, thread_config)
        
        if results_dict:
            # Create and save the plot
            fig = plot_all_archs_e2e_prefill(
                results_dict,
                model_names=models_to_plot,
                thread_mode=thread_mode
            )
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(os.path.dirname(script_dir), f'figures/e2e_prefill_comparison_{title_suffix}.png')
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"End-to-end prefill comparison plot saved to {output_file}")

if __name__ == '__main__':
    main()