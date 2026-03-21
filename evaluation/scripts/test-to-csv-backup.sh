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

# Use awk to process the file - compatible with older awk versions
awk '
BEGIN {
    processed_configs = "";
}

# Match the MUL_MAT line
/MUL_MAT.*type_a=.*,m=[0-9]+,n=[0-9]+,k=[0-9]+.*\):[ ]+[0-9]+ runs -[ ]+[0-9.]+ us\/run/ {
    # Extract type_a using simpler pattern matching
    type_str = $0;
    gsub(/.*type_a=/, "", type_str);
    gsub(/,.*/, "", type_str);
    type_a = type_str;
    
    # Extract m
    m_str = $0;
    gsub(/.*m=/, "", m_str);
    gsub(/,.*/, "", m_str);
    m = m_str;
    
    # Extract n
    n_str = $0;
    gsub(/.*n=/, "", n_str);
    gsub(/,.*/, "", n_str);
    n = n_str;
    
    # Extract k
    k_str = $0;
    gsub(/.*k=/, "", k_str);
    gsub(/[,)].*/, "", k_str);
    k = k_str;
    
    # Extract us/run - specifically looking for the pattern "runs - NUMBER us/run"
    us_str = $0;
    if (match(us_str, /[0-9]+ runs - [0-9.]+/)) {
        us_str = substr(us_str, RSTART, RLENGTH);
        gsub(/[0-9]+ runs - /, "", us_str);
        us_per_run = us_str;
        
        # Calculate rps (runs per second)
        rps = 1000000 / us_per_run;
        
        # Create a unique identifier for this configuration
        config = type_a "_" m "_" n "_" k;
        
        # Check if we already processed this configuration
        if (index(processed_configs, "|" config "|") == 0) {
            # Add to processed configs
            processed_configs = processed_configs "|" config "|";
            
            # Print to output
            printf "%s,%d,%d,%d,%.2f,%f\n", type_a, m, n, k, us_per_run, rps;
        }
    }
}
' "$input_file" >> "$output_file"

echo "Conversion complete. Output saved to $output_file"