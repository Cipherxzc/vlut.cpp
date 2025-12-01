#!/bin/bash

# Search optimal LUT and TABLE_ENTRY_SIZE config for different threads, based on GeMM performance

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
RESULTS_DIR="$PROJECT_ROOT/evaluation/results_search"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Initialize scores.csv with header
echo "entry_size,threads,score" > "$RESULTS_DIR/scores.csv"

# Define the configurations
ENTRY_SIZES=(16 32 64)
THREAD_COUNTS=(1 4 8)
MODELS=("bitnet_3b" "llama3_8b" "falcon_1b")

# Loop through all configurations
for entry_size in "${ENTRY_SIZES[@]}"; do
    # Set build flags based on configuration
    BUILD_FLAGS=""
    BUILD_FLAGS="$BUILD_FLAGS -DTABLE_ENTRY_SIZE=$entry_size"
    BUILD_DIR="$PROJECT_ROOT/build-entry${entry_size}"

    cd "$PROJECT_ROOT"
    
    # Build with this configuration
    echo "Building with ENTRY_SIZE=$entry_size..."
    cmake -B "$BUILD_DIR" $BUILD_FLAGS > /dev/null 2>&1
    cmake --build $BUILD_DIR --target test-vlut-gemm --config Release -j$(nproc) > /dev/null 2>&1
    
    # If build was successful, run benchmarks with different thread counts
    if [ $? -eq 0 ]; then
        for threads in "${THREAD_COUNTS[@]}"; do
            echo "Running benchmark with ENTRY_SIZE=$entry_size, THREADS=$threads..."
            LOG_FILE="$RESULTS_DIR/s${entry_size}_t${threads}.log"

            # Loop by models sequentially to avoid OOM
            > "$LOG_FILE"
            for model in "${MODELS[@]}"; do
                "$BUILD_DIR/bin/test-vlut-gemm" search -m "$model" -t "$threads" -ns 128,512 >> "$LOG_FILE" 2>&1
            done
            echo "Results saved to $LOG_FILE"
            
            # Process the log file to CSV
            "$SCRIPT_DIR/test-to-csv.sh" "$LOG_FILE"

            # Calculate score by summing all uspr values
            CSV_FILE="${LOG_FILE%.*}.csv"
            if [ -f "$CSV_FILE" ]; then
                # Skip header and sum the uspr column (5th column)
                SCORE=$(awk -F, 'NR>1 {sum+=$5; count++} END {printf "%.2f", sum/count}' "$CSV_FILE")
                echo "$entry_size,$threads,$SCORE" >> "$RESULTS_DIR/scores.csv"
                echo "Configuration score: $SCORE (lower is better)"
            else
                echo "Warning: CSV file $CSV_FILE not found. Skipping score calculation."
            fi
        done
    else
        echo "Build failed for ENTRY_SIZE=$entry_size. Skipping benchmarks."
    fi

    # Clean up build directory
    cd "$PROJECT_ROOT"
    rm -rf "$BUILD_DIR"
    echo "Cleaned up build directory for ENTRY_SIZE=$entry_size."
    echo "----------------------------------------"
done

echo "All benchmarks completed."

# Find the best configuration for each thread count
echo "Finding optimal configurations for each thread count..."
for threads in "${THREAD_COUNTS[@]}"; do
    BEST_CONFIG=$(awk -F, -v t="$threads" '
        $2==t {
            if (min=="" || $3 < min) {
                min=$3;
                config=$1
            }
        }
        END {
            if(min!="") 
                print "Thread "t": entry_size="config", score="min
        }
    ' "$RESULTS_DIR/scores.csv")
    echo "$BEST_CONFIG"
done