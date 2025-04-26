import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.ticker import MaxNLocator
from plot_utils import *

# arch = "aws_arm"
arch = "pc_intel"

def read_batch_csv_files(directory):
    """Read all batch CSV files in directory and subdirectories into a single DataFrame."""
    all_data = []
    failed_files = []
    
    # Find all CSV files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
    print(f"Found {len(csv_files)} CSV files")
    
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
                else:
                    model_quant = model_quant.split('.')[-1] # T-MAC
            else:
                model_quant = None
                failed_files.append(csv_file)
                continue
            
            # Extract thread count from filename (if present)
            thread_pattern = r'_t(\d+)_'
            thread_match = re.search(thread_pattern, basename)
            threads = int(thread_match.group(1)) if thread_match else 1
            
            # Read CSV data
            df = pd.read_csv(csv_file)
            
            # Add metadata columns
            df['model_name'] = E2E_MODEL_MAP[model_name]
            df['model_quant'] = E2E_TYPE_MAP[model_quant]
            df['threads'] = threads
            
            all_data.append(df)
            
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

def plot_batch_throughput(df, tg_values=None, model_names=None):
    """
    Create bar plots comparing token generation throughput for different models, batch sizes,
    and quantizations.
    
    Parameters:
    df: DataFrame with columns PP, TG, B, S_TG_t/s, model_name, model_quant, threads
    tg_values: List of token generation lengths to plot
    model_names: List of model names to include
    """
    # Filter data based on parameters
    filtered_df = df.copy()
    
    if tg_values is not None:
        filtered_df = filtered_df[filtered_df['TG'].isin(tg_values)]
    else:
        tg_values = sorted(filtered_df['TG'].unique())
        
    if model_names is not None:
        filtered_df = filtered_df[filtered_df['model_name'].isin(model_names)]
    else:
        model_names = sorted(filtered_df['model_name'].unique())
    
    # Get the max thread count for each model
    max_threads = filtered_df.groupby('model_name')['threads'].max().to_dict()
    
    # Filter to only include max thread counts
    filtered_df = filtered_df[filtered_df.apply(lambda row: row['threads'] == max_threads[row['model_name']], axis=1)]
    
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16  # Base font size
    
    # Get unique batch sizes
    batch_sizes = sorted(filtered_df['B'].unique())
    
    # Create a consistent color map for quantization values
    all_quants = sorted(filtered_df['model_quant'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_quants)))
    quant_color_map = {quant: colors[i] for i, quant in enumerate(all_quants)}
    
    # Prepare legend data
    legend_handles = []
    legend_labels = []
    
    for i, quant in enumerate(all_quants):
        handle = plt.Rectangle((0, 0), 1, 1, color=quant_color_map[quant])
        legend_handles.append(handle)
        legend_labels.append(quant)
    
    # Create a figure for each token generation length
    for tg in tg_values:
        tg_df = filtered_df[filtered_df['TG'] == tg]
        
        if tg_df.empty:
            print(f"No data for TG={tg}")
            continue
        
        # Calculate subplot grid dimensions
        n_rows = len(model_names)
        n_cols = len(batch_sizes)
        
        if n_rows == 0 or n_cols == 0:
            print(f"Insufficient data for TG={tg}")
            continue
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3*n_rows))
        
        # Ensure axes is a 2D array even with single row or column
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each combination
        for row_idx, model in enumerate(model_names):
            for col_idx, batch_size in enumerate(batch_sizes):
                ax = axes[row_idx, col_idx]
                
                # Get data for this model, TG, and batch size
                subset = tg_df[(tg_df['model_name'] == model) & (tg_df['B'] == batch_size)]
                
                if subset.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    continue
                
                # Sort by quantization for consistent ordering
                subset = subset.sort_values('model_quant')
                
                # Create x positions for bars
                x_pos = np.arange(len(subset))
                
                # Plot bars for each quantization
                ax.bar(
                    x_pos,
                    subset['S_TG_t/s'],
                    width=0.7,
                    zorder=3,
                    color=[quant_color_map[q] for q in subset['model_quant']]
                )
                
                # # Add thread count annotation
                # thread_count = subset['threads'].iloc[0]
                # ax.text(0.98, 0.95, f"{thread_count} threads", 
                #         transform=ax.transAxes, ha='right', va='top', 
                #         fontsize=14, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
                
                # Set title and labels
                if row_idx == 0:
                    ax.set_title(f'Batch Size = {batch_size}', fontsize=20)
                if col_idx == 0 and row_idx == 1:
                    ax.set_ylabel('Throughput (tokens/s)', fontsize=20)
                
                # Add model name to right side of plot
                if col_idx == n_cols - 1:
                    ax.text(1.05, 0.5, model, transform=ax.transAxes, 
                            fontsize=18, va='center', rotation=-90)
                
                # Grid and formatting
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.set_ylim(0)

                # Disable x-axis label, ticks, and tick labels
                ax.set_xlabel('')
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.yaxis.set_major_locator(MaxNLocator(5))
                
                # Add padding to the left and right of the bar group
                xlim = ax.get_xlim()
                padding = 0.5
                new_xlim = (xlim[0] - padding, xlim[1] + padding)
                ax.set_xlim(new_xlim)
        
        # Adjust the layout
        # fig.subplots_adjust(bottom=0.1, right=0.9, top=0.9, left=0.1, wspace=0.3, hspace=0.2)
        fig.subplots_adjust(bottom=0.18, top=0.9, left=0.1, right=0.95, wspace=0.3, hspace=0.4)
        
        # Add a single legend for the entire figure
        if len(all_quants) > 1:
            fig.legend(
                handles=legend_handles, 
                labels=legend_labels, 
                loc='lower center', 
                # ncol=min(len(legend_labels), 5),
                ncol=min((len(legend_labels) + 1)/2, 3),
                fontsize=18, 
                frameon=True, 
                bbox_to_anchor=(0.5, 0.02),
                columnspacing=1.0
            )
        
        # Add overall title
        fig.suptitle(f'Token Generation Throughput (TG={tg} tokens)', fontsize=24)
        
        # Save the figure
        output_file = f"evaluation/results_e2e_batch_{arch}/e2e_batch_{arch}_TG{tg}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_file}")
        plt.close(fig)

if __name__ == '__main__':
    
    directory = f"evaluation/results_e2e_batch_{arch}"
    combined_df = read_batch_csv_files(directory)
    
    if not combined_df.empty:
        # Plot batch throughput for all TG values
        # print(combined_df)
        plot_batch_throughput(combined_df)
    else:
        print("No data was loaded.")