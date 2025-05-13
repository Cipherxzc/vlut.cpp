import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# Set the font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12  # Base font size

# Load the CSV data
data = pd.read_csv('evaluation/results_gemm_bits_pc_intel/llama3_8b_t1_ns256_lon_s32.csv')

# Step 1: Create a map of "name" to "bits" for each quantization
bits_map = {
    'q8_0': 8.5,
    'q6_K': 6.56,
    'q5_K': 5.68,
    'q4_K': 4.84,
    'q4_0': 4.5,
    'q3_K': 3.91,
    'q2_K': 3.35,
    'tq2_0': 2.06,
    'tq1_0': 1.69,
    'i2_s_4': 2.00,
    'i1_m_2': 1.60
}

name_map = {
    'q8_0': 'Q8_0',
    'q6_K': 'Q6_K',
    'q5_K': 'Q5_K',
    'q4_K': 'Q4_K',
    'q4_0': 'Q4_0',
    'q3_K': 'Q3_K',
    'q2_K': 'Q2_K',
    'tq2_0': 'TQ2_0',
    'tq1_0': 'TQ1_0',
    'i2_s_4': 'Ours I2',
    'i1_m_2': 'Ours I1'
}

offsets = {
    'q8_0': (-40, -10),
    'q6_K': (10, 0),
    'q5_K': (-10, 10),
    'q4_K': (10, -10),
    'q4_0': (-10, -20),
    'q3_K': (10, 0),
    'q2_K': (5, -10),
    'tq2_0': (0, 25),
    'tq1_0': (10, -10),
    'i2_s_4': (15, -10),
    'i1_m_2': (-20, -20)
}

# Add bits column to the dataframe
data['bits'] = data['name'].map(bits_map)

# Step 2: Filter for a specific GeMM shape
filtered_data = data[(data['m'] == 4096) & (data['n'] == 256) & (data['k'] == 4096)]

# Calculate latency in milliseconds from microseconds per run
filtered_data['latency_ms'] = filtered_data['uspr'] / 1000

# Create figure with a more compact size suitable for a paper
plt.figure(figsize=(7, 4))

# Define groups
traditional_quants = ['q8_0', 'q6_K', 'q5_K', 'q4_K', 'q4_0', 'q3_K', 'q2_K', 'tq2_0', 'tq1_0']
new_quants = ['i1_m_2', 'i2_s_4', 'q2_K']  # Including q2_K as the connection point

# Filter dataframes for each group
trad_df = filtered_data[filtered_data['name'].isin(traditional_quants)].sort_values('bits')
new_df = filtered_data[filtered_data['name'].isin(new_quants)].sort_values('bits')

# Get max y value for region filling
max_y = filtered_data['latency_ms'].max() * 1.1

# Line A: Connect traditional
trad_x = trad_df['bits'].values
trad_y = trad_df['latency_ms'].values
plt.plot(trad_x, trad_y, 'r-', linewidth=1.5, label='MAD-based mpGeMM (llama.cpp)', zorder=5)

# Line B: Connect i1_m_2, i2_s, and q2_K with dotted line
new_x = new_df['bits'].values
new_y = new_df['latency_ms'].values
plt.plot(new_x, new_y, 'b--', linewidth=1.5, label='LUT-based  mpGeMM (ours)', zorder=5)

# Get coordinates for Q8_0 and Ours I2 for the arrow
q8_0_point = filtered_data[filtered_data['name'] == 'q8_0'][['bits', 'latency_ms']].values[0]
i2_point = filtered_data[filtered_data['name'] == 'i2_s_4'][['bits', 'latency_ms']].values[0]

# Plot all data points
for idx, row in filtered_data.iterrows():
    plt.scatter(row['bits'], row['latency_ms'], s=40, 
                color='blue' if row['name'] in ['i1_m_2', 'i2_s_4'] else 'red',
                zorder=10)
    
    offset = offsets.get(row['name'], (10, -10))

    plt.annotate(f"{name_map[row['name']]}", 
                (row['bits'], row['latency_ms']),
                textcoords="offset points", 
                xytext=offset,
                ha='left',
                # fontsize=16,
                fontweight='bold',
                zorder=15)

# Create the red region (above traditional line)
red_vertices = list(zip(trad_x, trad_y))
red_vertices = [(trad_x[0], max_y)] + red_vertices + [(trad_x[-1] + 1, trad_y[-1])] + [(trad_x[-1] + 1, max_y)]
red_path = Path(red_vertices)
red_patch = patches.PathPatch(red_path, facecolor='red', alpha=0.15, zorder=1)
plt.gca().add_patch(red_patch)

# Create the blue region (between new line and traditional line)
# First identify the relevant points from the traditional line that overlap with the new line range
trad_points_in_range = []
for x, y in zip(trad_x, trad_y):
    if 1 <= x <= 2:
        trad_points_in_range.append((x, y))

# For each x value in the new line, find the corresponding y on the traditional line
q2k_point = filtered_data[filtered_data['name'] == 'q2_K'][['bits', 'latency_ms']].values[0]
i2s_point = filtered_data[filtered_data['name'] == 'i2_s_4'][['bits', 'latency_ms']].values[0]
i1_point = filtered_data[filtered_data['name'] == 'i1_m_2'][['bits', 'latency_ms']].values[0]

# Look up the traditional line points at x=1 and x=2
tq1_point = filtered_data[filtered_data['name'] == 'tq1_0'][['bits', 'latency_ms']].values[0]
tq2_point = None
for name in ['tq2_0', 'q2_K']:
    temp = filtered_data[filtered_data['name'] == name]
    if not temp.empty:
        tq2_point = temp[['bits', 'latency_ms']].values[0]
        break

# Build the blue region vertices
blue_vertices = [
    (i1_point[0], max_y),          # Top left corner at x=1
    (i1_point[0], i1_point[1]),    # i1_m_2 point
    (i2s_point[0], i2s_point[1]),  # i2_s point
    (q2k_point[0], q2k_point[1]),  # q2_K point
    (tq2_point[0], tq2_point[1]),  # Traditional at x=2
    (tq1_point[0], tq1_point[1]),  # Traditional at x=1
    (tq1_point[0], max_y),         # Top right corner at x=2
]

blue_path = Path(blue_vertices)
blue_patch = patches.PathPatch(blue_path, facecolor='blue', alpha=0.15, zorder=2)
plt.gca().add_patch(blue_patch)

# Add the straight arrow from Q8_0 to Ours I2, with slight offset to avoid overlapping
# Create offset points for the arrow
start_x = q8_0_point[0] - 1  # Offset to avoid overlapping with the Q8_0 point
start_y = q8_0_point[1] - 5
end_x = i2_point[0] + 1      # Offset to avoid overlapping with the I2 point
end_y = i2_point[1] + 5

# Create straight arrow
arrow = FancyArrowPatch(
    (start_x, start_y),
    (end_x, end_y),
    connectionstyle="arc3,rad=0",  # rad=0 makes it a straight line
    arrowstyle="->,head_length=10,head_width=7",  # Increased head size
    color="black",
    linewidth=2.5,  # Increased line width
    zorder=20
)
plt.gca().add_patch(arrow)

# Calculate reduction (times)
bpw_reduction = q8_0_point[0] / i2_point[0]
latency_reduction = q8_0_point[1] / i2_point[1]

# Position annotation at the down right side of the arrow
# Find a point near the arrow but shifted down and right
anno_x = end_x + 0.5
anno_y = end_y - 0.5

plt.annotate(
    f"{bpw_reduction:.1f}× BPW\n{latency_reduction:.1f}× Latency",
    # (anno_x, anno_y),
    (5, 42),
    fontsize=14,
    fontweight='bold',
    color="black",
    # bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="white", alpha=0.9),  # Removed border and reduced padding
    zorder=20
)

# Customize the plot with paper-appropriate sizes
plt.xlabel('Bits per Weight (BPW)', fontsize=14, fontweight='bold')
plt.ylabel('Latency (ms)', fontsize=14, fontweight='bold')
plt.title('GeMM (4096×4096×256) latency vs. BPW', fontsize=16, fontweight='bold', pad=10)
plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
plt.legend(loc='lower right', fontsize=12)

# Adjust tick sizes
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# Set axis limits
plt.xlim(1.0, 9.0)  # Changed from 0.5, 8.5 to 1.0, 9.0
plt.ylim(0, max_y)

ax = plt.gca()  # "get current axes"

# Set the line width for all spines
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_zorder(10)  # Higher z-order appears on top

plt.tight_layout()
plt.savefig('evaluation/figures/gemm_bits.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('evaluation/figures/gemm_bits.png', dpi=300, bbox_inches='tight')
plt.show()