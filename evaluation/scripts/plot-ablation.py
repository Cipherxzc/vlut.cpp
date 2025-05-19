import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data from local file
file_path = "evaluation/results_ablation/bitnet_ablation_results.csv"
df = pd.read_csv(file_path)

# Split data into single thread and multithread
single_thread = df[df['threads'] == 1]
multi_thread = df[df['threads'] == 4]

# Baseline (T-MAC) values (can be customized)
baseline_single = 20.6  # T-MAC for 1 thread
baseline_multi = 80.1   # Same as no_transpose for 4 threads

# Create custom colormap based on the pattern in GEMM_TYPE_STYLES
custom_colors = [
    '#7EB875',
    '#CDB236',
    '#D76B69',
    '#5DA8D4',
    '#9067A9',
    '#3274A1',
    '#32A178',
    '#999999',
]

# legend_name_map = {
#     'no_transpose': 'Naive Parallel LUT',
#     'transpose': '+ Parallelism-Aware Layout',
#     'grouped_add': '+ Hierarchical Accumulation',
#     'lut2': '+ Streamed Precomputing-Lookup',
#     'table_opt': '+ Topological Precomputing',
#     'n_tile': '+ N-Tiling',
#     'k_tile': '+ K-Tiling',
# }

# legend_name_map = {
#     'no_transpose': 'Naive Par. LUT',
#     'transpose': '+ Par-Aware Layout',
#     'grouped_add': '+ Hier. Accum.',
#     'lut2': '+ Stream. Lookup',
#     'table_opt': '+ Topo. Precomp.',
#     'n_tile': '+ N-Tile',
#     'k_tile': '+ K-Tile (Ours)',
# }

legend_name_map = {
    'no_transpose': 'Naive',
    'transpose': '+ Layout',
    'grouped_add': '+ Accum.',
    'lut2': '+ Stream.',
    'table_opt': '+ Topo.',
    'n_tile': '+ N-Tile',
    'k_tile': '+ K-Tile (Ours)',
}

# Set the font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12  # Base font size

# Create figure with two subplots (no top title)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

# Create a list to store bar objects for legend
bar_plots = []
optimization_names = list(single_thread['optimization'])
# Apply the name mapping
display_names = [legend_name_map.get(opt, opt) for opt in optimization_names]

# Increase linewidth for axis borders
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

# Plot for single thread without error bars
for i, (opt, val) in enumerate(zip(single_thread['optimization'], single_thread['avg_ts'])):
    color_idx = i % len(custom_colors)
    bar = ax1.bar(i, val, color=custom_colors[color_idx], edgecolor='black', zorder=3)
    bar_plots.append(bar[0])
    
    # Add value labels on top of bars
    ax1.annotate(f'{val:.1f}',
                xy=(i, val),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

# Calculate appropriate y-axis limit with extra space for annotations
y_max1 = max(single_thread['avg_ts']) * 1.2

ax1.set_title('Single-thread prefilling', fontsize=14, fontweight='bold')
ax1.set_ylabel('Throughput (tokens/sec)', fontsize=14, fontweight='bold')  # Only display on first subfigure
ax1.axhline(y=baseline_single, color='red', linestyle='--', 
           label=f'T-MAC: {baseline_single}', zorder=4)  # Increased zorder
ax1.set_xticks([])  # Remove x-ticks
ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
ax1.set_ylim(0, y_max1)  # Set y-axis limit with extra space
ax1.legend(loc='upper left')

# Plot for multi thread without error bars
for i, (opt, val) in enumerate(zip(multi_thread['optimization'], multi_thread['avg_ts'])):
    color_idx = i % len(custom_colors)
    ax2.bar(i, val, color=custom_colors[color_idx], edgecolor='black', zorder=3)
    
    # Add value labels on top of bars
    ax2.annotate(f'{val:.1f}',
                xy=(i, val),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12)

# Calculate appropriate y-axis limit with extra space for annotations
y_max2 = max(multi_thread['avg_ts']) * 1.2

ax2.set_title('Multi-thread prefilling', fontsize=14, fontweight='bold')
# No y-axis label for second subfigure
ax2.axhline(y=baseline_multi, color='red', linestyle='--', 
           label=f'T-MAC: {baseline_multi}', zorder=4)  # Increased zorder
ax2.set_xticks([])  # Remove x-ticks
ax2.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
ax2.set_ylim(0, y_max2)  # Set y-axis limit with extra space
ax2.legend(loc='upper left')

# Create a shared legend for optimization names with more columns
ncols = min(len(optimization_names), 4)

# Creating a custom handler map for row-major ordering
handles = bar_plots
labels = display_names

# Create bold font property
import matplotlib.font_manager as font_manager
font_prop = font_manager.FontProperties(weight='bold')

# # Reorder handles and labels to create row-major order
# if len(handles) > ncols:
#     num_rows = (len(handles) + ncols - 1) // ncols
#     row_major_handles = []
#     row_major_labels = []
    
#     for col in range(ncols):
#         for row in range(num_rows):
#             idx = row * ncols + col
#             if idx < len(handles):
#                 row_major_handles.append(handles[idx])
#                 row_major_labels.append(labels[idx])
    
#     handles = row_major_handles
#     labels = row_major_labels

fig.legend(handles, labels,
           loc='lower center', ncol=ncols, fontsize=12, prop=font_prop, alignment="right",
           bbox_to_anchor=(0.5, 0.02), frameon=False)

# Increase zorder for axis borders (spines)
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # You already have this
        spine.set_zorder(10)      # Add this to increase zorder

# Use tight_layout with proper parameters to ensure space for annotations and legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.26, top=0.92)  # Increased bottom margin for the shared legend
plt.savefig('evaluation/figures/ablation.pdf', dpi=300, bbox_inches='tight')
plt.show()