import json
import os
import pandas as pd
import re
import numpy as np
from typing import Dict, List, Tuple

def load_and_process_results(device_name: str) -> pd.DataFrame:
    """
    Load and process the JSONL result files for a specific device.
    
    Args:
        device_name: Name of the device for which to load results
        
    Returns:
        DataFrame with optimal configurations for each (device, model, threads, m, k, n) combination
    """
    # Define the base directory where the files are stored
    base_dir = f"evaluation/results_tmac_{device_name}"
    
    # Dictionary to store DataFrames for each file
    dfs = {}
    
    # Regular expression to extract m, k, n, t values from the function name
    func_pattern = re.compile(r'qgemm_lut_t(\d+)_int8_m(\d+)_k(\d+)_n(\d+)_b2')
    
    # Iterate through the files in the directory
    for filename in os.listdir(base_dir):
        if not filename.endswith('.jsonl') or 'qgemm_lut' not in filename:
            continue
        
        # Extract model from the filename
        parts = filename.split('_')
        model = '_'.join(parts[:-2])
        
        # Initialize a list to store the data from each line
        data = []
        
        # Read and process each line in the file
        with open(os.path.join(base_dir, filename), 'r') as f:
            for idx, line in enumerate(f):
                try:
                    # Parse the JSON
                    record = json.loads(line)
                    
                    # Extract the function name
                    func_name = record["input"][1]
                    
                    # Extract t, m, k, n values using regex
                    match = func_pattern.match(func_name)
                    if match:
                        t, m, k, n = map(int, match.groups())
                    else:
                        continue
                    
                    # Extract latency (average of results[0] list)
                    if isinstance(record["result"][0], list):
                        latency = np.mean(record["result"][0])
                    else:
                        # For some records, result[0] might be a placeholder (e.g., 1000000000.0)
                        latency = None
                    
                    # Add the data to our list with a unique identifier
                    data.append({
                        'device': device_name,
                        'model': model,
                        'threads': t,
                        'm': m / 2, # T-MAC's M is doubled for 2-bit quant
                        'k': k,
                        'n': n,
                        'latency_s': latency,
                        'row_idx': idx  # Add a unique identifier for each row
                    })
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Error processing line in {filename}: {e}")
                    continue
        
        # Convert to DataFrame
        if data:
            dfs[f"{model}_qgemm_lut"] = pd.DataFrame(data)

    # Now, find the optimal configuration for each (device, model, threads, m, k, n) combination
    # First, ensure we only use rows with valid latency values
    valid_dfs = {k: df[df['latency_s'].notna()] for k, df in dfs.items()}
    
    # Find optimal configurations
    optimal_configs = []
    
    # Combine all valid DataFrames
    if valid_dfs:
        combined_df = pd.concat(valid_dfs.values())
        
        # Group by device, model, threads, m, k, n to find min latency for identical configs
        grouped = combined_df.groupby(['device', 'model', 'threads', 'm', 'k', 'n'])
        
        # Find the configuration with the minimum latency for each group
        for group_key, group_df in grouped:
            device, model, threads, m, k, n = group_key
            min_latency_row = group_df.loc[group_df['latency_s'].idxmin()]
            
            optimal_configs.append({
                'device': device,
                'model': model,
                'threads': threads,
                'm': int(m),
                'k': int(k),
                'n': int(n),
                'latency_s': min_latency_row['latency_s'],
                'row_idx': min_latency_row['row_idx']
            })
    
    # Convert to DataFrame and return
    result_df = pd.DataFrame(optimal_configs)
    
    # Remove the row_idx column before returning
    if 'row_idx' in result_df.columns:
        result_df = result_df.drop('row_idx', axis=1)
    
    return result_df

if __name__ == "__main__":
    # Example usage
    device_name = "aws_arm"  # Replace with actual device name
    results_df = load_and_process_results(device_name)
    print(results_df)
    
    # Save the results to a CSV file
    # results_df.to_csv(f"optimal_configs_{device_name}.csv", index=False)
    
    print("Optimal configurations saved to CSV.")