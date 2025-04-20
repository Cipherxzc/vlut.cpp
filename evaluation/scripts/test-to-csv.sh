#!/bin/bash

# Convert gemm test log to csv format

# Check if a file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file="$1"
output_file="${input_file%.*}.csv"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist"
    exit 1
fi

# Create CSV with header
echo "name,m,n,k,uspr,rps" > "$output_file"

# Use awk to process the file
awk '
BEGIN {
    processed_configs = "";
}

# Match the MUL_MAT line
/MUL_MAT\(type_a=([^,]+),.*,m=([0-9]+),n=([0-9]+),k=([0-9]+).*\):[ ]+[0-9]+ runs -[ ]+([0-9.]+) us\/run/ {
    # Extract type_a, m, n, k, us/run
    match($0, /type_a=([^,]+),/, type_match);
    type_a = type_match[1];
    
    match($0, /m=([0-9]+),/, m_match);
    m = m_match[1];
    
    match($0, /n=([0-9]+),/, n_match);
    n = n_match[1];
    
    match($0, /k=([0-9]+),/, k_match);
    k = k_match[1];
    
    match($0, /([0-9.]+) us\/run/, us_match);
    us_per_run = us_match[1];
    
    # Calculate rps (runs per second)
    rps = 1000000 / us_per_run;
    
    # Create a unique identifier for this configuration
    config = type_a "_" m "_" n "_" k;
    
    # Check if we already processed this configuration
    if (index(processed_configs, "|" config "|") == 0) {
        # Add to processed configs
        processed_configs = processed_configs "|" config "|";
        
        # Print to output
        printf "%s,%d,%d,%d,%.2f,%.2f\n", type_a, m, n, k, us_per_run, rps;
    }
}
' "$input_file" >> "$output_file"

echo "Conversion complete. Output saved to $output_file"