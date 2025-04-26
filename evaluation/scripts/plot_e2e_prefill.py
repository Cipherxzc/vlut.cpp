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

def read_csv_files(directory):
    """Read all CSV files in directory and subdirectories into a single DataFrame."""
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
            
            # Extract the part before _p as model_quant
            if '_p' in basename:
                model_quant = basename.split('_p')[0]
                if model_quant.startswith('ggml-model'):
                    model_quant = model_quant.split('-')[-1] # others
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
            df['model_name'] = E2E_MODEL_MAP[model_name]
            df['model_quant'] = E2E_TYPE_MAP[model_quant]
            
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

def plot_latency_curves(df, t_values=None, model_names=None):
    """
    Create line plots comparing latency for different models and quantizations.
    
    Parameters:
    df: DataFrame with columns p, t, avg_ts, stdev_ts, model_name, model_quant
    t_values: List of thread counts to plot, if None all unique t values are plotted
    model_names: List of model names to include, if None all models are included
    """
    # Filter data based on parameters
    filtered_df = df.copy()
    
    if t_values is not None:
        filtered_df = filtered_df[filtered_df['t'].isin(t_values)]
        
    if model_names is not None:
        filtered_df = filtered_df[filtered_df['model_name'].isin(model_names)]
    
    # Get unique combinations to plot
    unique_t = sorted(filtered_df['t'].unique())
    unique_models = sorted(filtered_df['model_name'].unique())
    
    # Calculate subplot grid dimensions
    n_plots = len(unique_t) * len(unique_models)
    n_rows = len(unique_models)
    n_cols = len(unique_t)
    
    # Set the font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16  # Base font size
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Ensure axes is a 2D array even with single row or column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Create a consistent color map for quantization values
    all_quants = sorted(filtered_df['model_quant'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_quants)))
    quant_color_map = {quant: colors[i] for i, quant in enumerate(all_quants)}
    
    # Prepare legend data
    legend_handles = []
    legend_labels = []
    
    for i, quant in enumerate(all_quants):
        handle = plt.Line2D([0], [0], color=quant_color_map[quant], lw=2, marker='o', markersize=6)
        legend_handles.append(handle)
        legend_labels.append(quant)
    
    # Plot each combination
    for row_idx, model in enumerate(unique_models):
        for col_idx, t in enumerate(unique_t):
            ax = axes[row_idx, col_idx]
            
            # Get data for this model and thread count
            subset = filtered_df[(filtered_df['model_name'] == model) & (filtered_df['t'] == t)]
            
            # Plot lines for each quantization
            for quant in all_quants:
                quant_data = subset[subset['model_quant'] == quant]
                if not quant_data.empty:
                    # Sort by p to ensure correct line
                    quant_data = quant_data.sort_values('p')
                    ax.errorbar(
                        quant_data['p'], 
                        quant_data['avg_ts'],
                        yerr=quant_data['stdev_ts'],
                        fmt='o-', 
                        linewidth=2,
                        color=quant_color_map[quant],
                        capsize=4,
                        label=quant
                    )
            
            # Set title and labels
            if row_idx == 0:
                ax.set_title(f'{t} {"threads" if t > 1 else "thread"}', fontsize=20)
            if col_idx == 0:
                ax.set_ylabel('Throughput (tokens/s)', fontsize=20)
            if row_idx == n_rows - 1:
                ax.set_xlabel('Prompt length (tokens)', fontsize=20)
            
            # Add model name to right side of plot
            if col_idx == n_cols - 1:
                ax.text(1.05, 0.5, model, transform=ax.transAxes, 
                        fontsize=18, va='center', rotation=-90)
            
            # Grid and formatting
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.set_ylim(0)
            ax.set_xscale('log', base=2)
            xticks = df['p'].unique()
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(val) for val in xticks])

    # Add a single legend for the entire figure at the bottom
    fig.subplots_adjust(bottom=0.15, right=0.9, top=0.92, left=0.1, wspace=0.3, hspace=0.3)
    fig.legend(
        handles=legend_handles, 
        labels=legend_labels, 
        loc='lower center', 
        ncol=min(len(legend_labels), 5),  # Adjust columns based on number of items
        fontsize=18, 
        frameon=True, 
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.0
    )
    
    # Add overall title
    fig.suptitle('Model Performance: Latency vs. Prompt Length', fontsize=28)
    
    return fig

if __name__ == '__main__':
    
    # Example usage
    directory = f"evaluation/results_e2e_prefill_{arch}"  # Path to your results folder
    combined_df = read_csv_files(directory)

    # Display the resulting dataframe
    if not combined_df.empty:
        # print("\nDataFrame preview:")
        # print(combined_df.head())
        # print(f"\nTotal rows: {len(combined_df)}")
        # print(f"Columns: {combined_df.columns.tolist()}")
        # print(combined_df)
        fig = plot_latency_curves(combined_df, t_values=[1, 4, 8], model_names=['BitNet 3B', 'Llama3 8B', 'Falcon 1B'])
        plt.savefig(f'{directory}/e2e_prefill_{arch}.png', dpi=300, bbox_inches='tight')
        # plt.show()
    else:
        print("No data was loaded.")

    # Save the combined data to a new CSV file
    # combined_df.to_csv("combined_results.csv", index=False)