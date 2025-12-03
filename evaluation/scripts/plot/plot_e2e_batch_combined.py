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
    parser = argparse.ArgumentParser(description='Plot GeMM Batch Benchmarks')
    parser.add_argument('--multi-thread', action='store_true', default=False, 
                        help='Use multi-threaded configuration for each architecture')
    parser.add_argument('--single-thread', action='store_true', default=False, 
                        help='Use single-threaded configuration for each architecture')
    parser.add_argument('--both', action='store_true', default=True,
                       help='Plot both single-thread and multi-thread configurations separately')
    parser.add_argument('-a', '--arch', type=str, default=None, help='Input arch name (overrides other options)')
    
    args = parser.parse_args()
    return args

# List of architectures to include
all_archs = [
    'pc_intel',
    # 'laptop_amd',
    'aws_arm',
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

def read_batch_csv_files(directory, arch):
    """Read all batch CSV files in directory and subdirectories into a single DataFrame."""
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
            
            # Extract the part before _npp as model_quant
            if '_npp' in basename:
                model_quant = basename.split('_npp')[0]
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

            # Skip if ours i1/i2 quant not in map
            if model_quant in E2E_TYPE_VARIANTS and model_quant not in E2E_TYPE_DEVICE_MAP[arch]:
                continue
            
            # Extract thread count from filename (if present)
            thread_pattern = r'_t(\d+)_'
            thread_match = re.search(thread_pattern, basename)
            threads = int(thread_match.group(1)) if thread_match else 1
            
            # Read CSV data
            df = pd.read_csv(csv_file)
            
            # Add metadata columns
            df['model_name'] = E2E_MODEL_MAP.get(model_name, model_name)
            df['model_quant'] = E2E_TYPE_MAP.get(model_quant, model_quant)
            df['threads'] = threads

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
        results_dir = os.path.join(os.path.dirname(script_dir), f'results_e2e_batch_{arch}')
        # Load results for this architecture
        df = read_batch_csv_files(results_dir, arch)
        
        if not df.empty:
            # Filter by thread value if specified in thread_config
            if thread_config and arch in thread_config:
                thread_val = thread_config[arch]
                df = df[df['threads'] == thread_val]
            
            # Store in dictionary
            results_dict[arch] = df
            print(f"Loaded results for {arch}")
        else:
            print(f"No results found for {arch}")
    
    return results_dict

def plot_all_archs_e2e_batch(results_dict, model_names=None, thread_mode="auto", tg_value=None):
    """
    Create a plot comparing batch performance across multiple architectures.
    Each arch gets one column, with one row per model.
    Bar plot version - groups by batch size with proper handling of available quantizations.
    
    Parameters:
    results_dict: Dictionary of DataFrames with results for each architecture
    model_names: List of models to include in the plot
    thread_mode: String indicating thread mode for title
    tg_value: Token generation length to filter by
    """
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16  # Base font size
    
    # Filter by TG value if specified
    if tg_value is not None:
        for arch in results_dict:
            results_dict[arch] = results_dict[arch][results_dict[arch]['TG'] == tg_value]
    else:
        # If not specified, find the most common TG value across all datasets
        all_tg_counts = {}
        for df in results_dict.values():
            if not df.empty:
                tg_counts = df['TG'].value_counts()
                for tg, count in tg_counts.items():
                    all_tg_counts[tg] = all_tg_counts.get(tg, 0) + count
        
        if all_tg_counts:
            most_common_tg = max(all_tg_counts.items(), key=lambda x: x[1])[0]
            tg_value = most_common_tg
            print(f"Automatically selecting most common TG value: {tg_value}")
            for arch in results_dict:
                results_dict[arch] = results_dict[arch][results_dict[arch]['TG'] == tg_value]
    
    archs = list(results_dict.keys())
    n_archs = len(archs)
    
    # Filter for models to plot
    if model_names is None:
        all_models = set()
        for df in results_dict.values():
            all_models.update(df['model_name'].unique())
        model_names = sorted(all_models)
    else:
        # Only include models that exist in the data
        model_names = [model for model in model_names if any(
            df['model_name'].str.contains(model).any() if not df.empty else False 
            for df in results_dict.values())]
    
    n_models = len(model_names)
    
    if n_models == 0:
        print("No models found in data matching the specified models.")
        return None
    
    # Create figure with 1 column per arch, 1 row per model
    # fig = plt.figure(figsize=(6*n_archs, 3.5*n_models))
    fig = plt.figure(figsize=(6*n_archs, 8.5)) # 3 models
    
    # Create a grid for subplots
    gs = fig.add_gridspec(n_models, n_archs)
    
    # Define subplot adjustment parameters
    left_margin = 0.1
    right_margin = 0.85  # Increase right margin to leave room for model names
    # bottom_margin = 0.2
    bottom_margin = 0.26
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
    all_quants = sorted(all_quants, key=lambda x: TYPE_ORDER.get(x, 999))
    
    # Find all unique batch sizes across all datasets and models
    all_batch_sizes = set()
    for df in results_dict.values():
        if not df.empty:
            all_batch_sizes.update(df['B'].unique())
    all_batch_sizes = sorted(all_batch_sizes)
    
    # Create plots for each architecture and model
    for col_idx, arch in enumerate(archs):
        df = results_dict[arch]
        
        for row_idx, model in enumerate(model_names):
            # Create subplot
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Get data for this model on this architecture
            subset = df[df['model_name'] == model]
            
            # Set up the x positions for the batch size groups
            x_positions = np.arange(len(all_batch_sizes))
            
            if not subset.empty:
                # For each batch size
                for b_idx, batch_size in enumerate(all_batch_sizes):
                    # Get data for this batch size
                    b_data = subset[subset['B'] == batch_size]
                    
                    # Get the quantization types available for this model/arch/batch size
                    available_quants = sorted(b_data['model_quant'].unique(), key=lambda x: TYPE_ORDER.get(x, 999))
                    
                    if not available_quants:
                        continue  # Skip if no data for this batch size
                    
                    # Calculate position for grouped bars
                    num_quants = len(available_quants)
                    total_width = 0.8  # Total width of the group
                    bar_width = total_width / num_quants if num_quants > 0 else total_width
                    
                    # For each available quantization
                    for q_idx, quant in enumerate(available_quants):
                        # Get data for this quantization
                        quant_data = b_data[b_data['model_quant'] == quant]
                        
                        if not quant_data.empty:
                            # Calculate position - center the group around the x position
                            if num_quants % 2 == 0:  # Even number of bars
                                start = x_positions[b_idx] - (bar_width * num_quants) / 2 + bar_width / 2
                            else:  # Odd number of bars
                                start = x_positions[b_idx] - (bar_width * (num_quants - 1)) / 2
                            
                            pos = start + q_idx * bar_width
                            
                            # Get style from E2E_TYPE_STYLES mapping
                            style = E2E_TYPE_STYLES.get(quant, {'color': '#000000', 'hatch': ''})
                            
                            # Plot the bar with appropriate style
                            ax.bar(
                                pos,
                                quant_data['S_TG_t/s'].values[0],
                                width=bar_width,
                                color=style['color'],
                                hatch=style['hatch'],
                                edgecolor='black',
                                linewidth=1.5,
                                align='center',
                                zorder=3
                            )
                            
            ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_zorder(10)
            
            # Set x-ticks at the center of each batch size group
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(b) for b in all_batch_sizes])
            
            # Add labels
            if col_idx == 0 and row_idx == 1:
                ax.set_ylabel('Throughput (tokens/s)', fontsize=24, fontweight='bold', labelpad=10)
            if row_idx == n_models - 1:
                ax.set_xlabel('Batch size', fontsize=20)
            
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
    
    legend_handles = []
    legend_labels = []
    
    for quant in all_quants:
        # Get style from E2E_TYPE_STYLES mapping
        style = E2E_TYPE_STYLES.get(quant, {'color': '#000000', 'hatch': ''})
        
        # Create a small bar with proper styling
        dummy_bar = legend_ax.bar(0, 0, color=style['color'], hatch=style['hatch'], 
                                edgecolor='black')
        legend_handles.append(dummy_bar[0])
        legend_labels.append(quant)
    
    # Hide the dummy axis
    legend_ax.set_visible(False)
    
    import matplotlib.font_manager as font_manager
    font_prop = font_manager.FontProperties(weight='bold', size=20)

    # Add a single legend for the entire figure at the bottom
    fig.legend(
        handles=legend_handles, 
        labels=legend_labels, 
        loc='lower center', 
        ncol=min(len(legend_labels), 3),
        prop=font_prop,
        fontsize=18, 
        frameon=False, 
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.0
    )
    
    # # Add overall title with TG value
    # thread_str = "Single-Thread" if thread_mode == "single" else "Multi-Thread" if thread_mode == "multi" else ""
    # fig.suptitle(f'End-to-End Batched Decoding {thread_str} (TG={tg_value} tokens)', fontsize=24)
    
    return fig

def generate_speedup_reports(results_dict, models_to_plot, output_dir, lut2_on=None, entry_size=None):
    """Generate CSV reports on speedups for each architecture and matrix size."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare dataframes to store average speedups across all matrix sizes
    avg_speedups_by_arch = {}
    
    for arch, df in results_dict.items():
        # Apply filters
        if lut2_on is not None:
            df = df[df['lut2_on'] == lut2_on]
        if entry_size is not None:
            df = df[df['entry_size'] == entry_size]
        
        # Create a dataframe to store speedup results for this architecture
        speedup_results = []
        
        # Create a dataframe to accumulate speedups for averaging
        all_speedups = []
        
        # Get relevant E2E types for this architecture
        relevant_types = set()
        for model in models_to_plot:
            subset = df[(df['model_name'] == model)]
            for quant in subset['model_quant'].unique():
                relevant_types.add(quant)
        
        # Filter to only use types from E2E_TYPE_DEVICE_MAP if available
        if 'E2E_TYPE_DEVICE_MAP' in globals() and arch in E2E_TYPE_DEVICE_MAP:
            comparison_types = set([E2E_TYPE_MAP[t] for t in E2E_TYPE_DEVICE_MAP[arch]]) & relevant_types
        else:
            comparison_types = relevant_types
        
        relevant_types = sorted(list(relevant_types))
        comparison_types = sorted(list(comparison_types))
        
        # For each model
        for model in models_to_plot:
            # Filter data for this model
            subset = df[(df['model_name'] == model)]
            
            # Remove duplicates
            subset = subset.drop_duplicates(subset=['model_name', 'model_quant'], keep='first')
            
            # If we have data for this size
            if not subset.empty:
                # Create a dictionary to store performance for each type
                perf_by_type = {}
                
                # Collect performance data
                for _, row in subset.iterrows():
                    perf_by_type[row['model_quant']] = row['S_TG_t/s']
                
                # Calculate speedups for each pair of types
                for comp_type in comparison_types:
                    if comp_type not in perf_by_type:
                        continue
                        
                    comp_perf = perf_by_type[comp_type]
                    
                    for baseline_type in relevant_types:
                        if baseline_type not in perf_by_type or baseline_type == comp_type:
                            continue
                            
                        baseline_perf = perf_by_type[baseline_type]
                        speedup = comp_perf / baseline_perf
                        
                        # Add to results
                        speedup_results.append({
                            'model': model,
                            'comparison_type': comp_type,
                            'baseline_type': baseline_type,
                            'speedup': speedup
                        })
                        
                        # Add to all speedups for averaging
                        all_speedups.append({
                            'comparison_type': comp_type,
                            'baseline_type': baseline_type,
                            'speedup': speedup,
                            'model': model
                        })
        
        # Convert results to dataframe and save to CSV
        if speedup_results:
            speedup_df = pd.DataFrame(speedup_results)
            output_file = os.path.join(output_dir, f'{arch}_speedup_details.csv')
            speedup_df.to_csv(output_file, index=False)
            print(f"Detailed speedup report for {arch} saved to {output_file}")
        
        # Calculate average, min, and max speedups and save to CSV
        if all_speedups:
            all_speedups_df = pd.DataFrame(all_speedups)
            
            # Group by comparison and baseline types
            grouped = all_speedups_df.groupby(['comparison_type', 'baseline_type'])
            
            # Calculate stats
            avg_speedups = grouped['speedup'].mean().reset_index()
            min_speedups = grouped['speedup'].min().reset_index().rename(columns={'speedup': 'min_speedup'})
            max_speedups = grouped['speedup'].max().reset_index().rename(columns={'speedup': 'max_speedup'})
            
            # Get model combinations for min and max speedups
            min_model = grouped.apply(lambda x: x.loc[x['speedup'].idxmin(), 'model'], include_groups=False).reset_index(name='min_speedup_model')
            max_model = grouped.apply(lambda x: x.loc[x['speedup'].idxmax(), 'model'], include_groups=False).reset_index(name='max_speedup_model')
            
            # Merge all stats
            stats_df = avg_speedups.merge(min_speedups, on=['comparison_type', 'baseline_type'])
            stats_df = stats_df.merge(max_speedups, on=['comparison_type', 'baseline_type'])
            stats_df = stats_df.merge(min_model, on=['comparison_type', 'baseline_type'])
            stats_df = stats_df.merge(max_model, on=['comparison_type', 'baseline_type'])
            
            # Add architecture column
            stats_df['architecture'] = arch
            
            # Reorder columns with comparison_type as first column
            stats_df = stats_df[['comparison_type', 'baseline_type', 'speedup', 'min_speedup', 'max_speedup', 
                                'min_speedup_model', 'max_speedup_model', 'architecture']]
            
            # Save per-architecture average speedups
            output_file = os.path.join(output_dir, f'{arch}_speedup_stats.csv')
            stats_df.to_csv(output_file, index=False)
            print(f"Speedup statistics report for {arch} saved to {output_file}")
            
            # Store for combined report
            avg_speedups_by_arch[arch] = stats_df
    
    # Combine all average speedups into a single report
    if avg_speedups_by_arch:
        combined_stats = pd.concat(avg_speedups_by_arch.values(), ignore_index=True)
        output_file = os.path.join(output_dir, 'combined_speedup_stats.csv')
        combined_stats.to_csv(output_file, index=False)
        print(f"Combined speedup statistics report saved to {output_file}")
    
    # Create a pivot table for easier comparison
    if avg_speedups_by_arch:
        # For average speedups
        avg_pivot_rows = []
        for arch, stats_df in avg_speedups_by_arch.items():
            for _, row in stats_df.iterrows():
                avg_pivot_rows.append({
                    'architecture': arch,
                    'comparison_type': row['comparison_type'],
                    'baseline_type': row['baseline_type'],
                    'avg_speedup': row['speedup']
                })
        
        if avg_pivot_rows:
            avg_pivot_df = pd.DataFrame(avg_pivot_rows)
            avg_pivot_table = avg_pivot_df.pivot_table(
                values='avg_speedup',
                index=['architecture', 'comparison_type'],
                columns=['baseline_type']
            ).reset_index()
            
            output_file = os.path.join(output_dir, 'avg_speedup_pivot.csv')
            avg_pivot_table.to_csv(output_file)
            print(f"Average speedup pivot table saved to {output_file}")
        
        # For min speedups
        min_pivot_rows = []
        for arch, stats_df in avg_speedups_by_arch.items():
            for _, row in stats_df.iterrows():
                min_pivot_rows.append({
                    'architecture': arch,
                    'comparison_type': row['comparison_type'],
                    'baseline_type': row['baseline_type'],
                    'min_speedup': row['min_speedup']
                })
        
        if min_pivot_rows:
            min_pivot_df = pd.DataFrame(min_pivot_rows)
            min_pivot_table = min_pivot_df.pivot_table(
                values='min_speedup',
                index=['architecture', 'comparison_type'],
                columns=['baseline_type']
            ).reset_index()
            
            output_file = os.path.join(output_dir, 'min_speedup_pivot.csv')
            min_pivot_table.to_csv(output_file)
            print(f"Minimum speedup pivot table saved to {output_file}")
        
        # For max speedups
        max_pivot_rows = []
        for arch, stats_df in avg_speedups_by_arch.items():
            for _, row in stats_df.iterrows():
                max_pivot_rows.append({
                    'architecture': arch,
                    'comparison_type': row['comparison_type'],
                    'baseline_type': row['baseline_type'],
                    'max_speedup': row['max_speedup']
                })
        
        if max_pivot_rows:
            max_pivot_df = pd.DataFrame(max_pivot_rows)
            max_pivot_table = max_pivot_df.pivot_table(
                values='max_speedup',
                index=['architecture', 'comparison_type'],
                columns=['baseline_type']
            ).reset_index()
            
            output_file = os.path.join(output_dir, 'max_speedup_pivot.csv')
            max_pivot_table.to_csv(output_file)
            print(f"Maximum speedup pivot table saved to {output_file}")

def main():
    args = parse_arguments()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If a specific architecture is provided, only plot for that one
    if args.arch:
        directory = f"evaluation/results_e2e_batch_{args.arch}"
        combined_df = read_batch_csv_files(directory)
        
        if not combined_df.empty:
            # Find unique TG values
            tg_values = sorted(combined_df['TG'].unique())
            
            for tg in tg_values:
                plot_df = combined_df[combined_df['TG'] == tg]
                
                # Create the old-style plot for backward compatibility
                plot_batch_throughput(plot_df, [tg])
        else:
            print("No data was loaded.")
        
        return
    
    # Determine thread configuration and plotting mode
    if args.both:
        # Plot both single-thread and multi-thread separately
        # First, do single-thread
        single_thread_config = {arch: 1 for arch in all_archs}
        results_dict_single = load_results_for_all_archs(all_archs, single_thread_config)
        
        if results_dict_single:
            # Find unique TG values across all architectures
            all_tg_values = set()
            for df in results_dict_single.values():
                if not df.empty:
                    all_tg_values.update(df['TG'].unique())
            
            # For each TG value, create a separate plot
            for tg in sorted(all_tg_values):
                fig_single = plot_all_archs_e2e_batch(
                    results_dict_single,
                    model_names=models_to_plot,
                    thread_mode="single",
                    tg_value=tg
                )
                
                # Generate speedup reports for single-thread configuration
                script_dir = os.path.dirname(os.path.abspath(__file__))
                reports_dir_single = os.path.join(os.path.dirname(script_dir), 'reports_e2e_batch/single_thread')
                generate_speedup_reports(
                    results_dict_single,
                    models_to_plot,
                    reports_dir_single
                )
                
                if fig_single:
                    # Save the single-thread plot
                    output_dir = os.path.join(os.path.dirname(script_dir), 'figures')
                    os.makedirs(output_dir, exist_ok=True)
                    # output_file_single = os.path.join(output_dir, f'e2e_batch_comparison_single_thread_TG{tg}.png')
                    output_file_single = os.path.join(output_dir, f'e2e_batch_comparison_single_thread_TG{tg}.pdf')
                    fig_single.savefig(output_file_single, dpi=300, bbox_inches='tight')
                    print(f"Single-thread comparison plot for TG={tg} saved to {output_file_single}")
        
        # Second, do multi-thread
        results_dict_multi = load_results_for_all_archs(all_archs, MULTI_THREAD_CONFIG)
        
        if results_dict_multi:
            # Find unique TG values across all architectures
            all_tg_values = set()
            for df in results_dict_multi.values():
                if not df.empty:
                    all_tg_values.update(df['TG'].unique())
            
            # For each TG value, create a separate plot
            for tg in sorted(all_tg_values):
                fig_multi = plot_all_archs_e2e_batch(
                    results_dict_multi,
                    model_names=models_to_plot,
                    thread_mode="multi",
                    tg_value=tg
                )
                
                # Generate speedup reports for single-thread configuration
                script_dir = os.path.dirname(os.path.abspath(__file__))
                reports_dir_single = os.path.join(os.path.dirname(script_dir), 'reports_e2e_batch/multi_thread')
                generate_speedup_reports(
                    results_dict_single,
                    models_to_plot,
                    reports_dir_single
                )
                
                if fig_multi:
                    # Save the multi-thread plot
                    output_dir = os.path.join(os.path.dirname(script_dir), 'figures')
                    os.makedirs(output_dir, exist_ok=True)
                    # output_file_multi = os.path.join(output_dir, f'e2e_batch_comparison_multi_thread_TG{tg}.png')
                    output_file_multi = os.path.join(output_dir, f'e2e_batch_comparison_multi_thread_TG{tg}.pdf')
                    fig_multi.savefig(output_file_multi, dpi=300, bbox_inches='tight')
                    print(f"Multi-thread comparison plot for TG={tg} saved to {output_file_multi}")
    else:
        # Original functionality with specific thread mode
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
            # Find unique TG values across all architectures
            all_tg_values = set()
            for df in results_dict.values():
                if not df.empty:
                    all_tg_values.update(df['TG'].unique())
            
            # For each TG value, create a separate plot
            for tg in sorted(all_tg_values):
                # Create and save the plot
                fig = plot_all_archs_e2e_batch(
                    results_dict,
                    model_names=models_to_plot,
                    thread_mode=thread_mode,
                    tg_value=tg
                )
                
                if fig:
                    output_dir = os.path.join(os.path.dirname(script_dir), 'figures')
                    os.makedirs(output_dir, exist_ok=True)
                    # output_file = os.path.join(output_dir, f'e2e_batch_comparison_{title_suffix}_TG{tg}.png')
                    output_file = os.path.join(output_dir, f'e2e_batch_comparison_{title_suffix}_TG{tg}.pdf')
                    fig.savefig(output_file, dpi=300, bbox_inches='tight')
                    print(f"End-to-end batch comparison plot for TG={tg} saved to {output_file}")

if __name__ == '__main__':
    args = parse_arguments()
    
    main()