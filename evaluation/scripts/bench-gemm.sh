#!/bin/bash

# Get arguments
DEVICE_NAME=$1
THREADS=$2
NS=$3
LUT2=$4
ENTRY_SIZE=$5

# Check if required arguments are provided
if [ -z "$DEVICE_NAME" ] || [ -z "$THREADS" ] || [ -z "$NS" ]; then
    echo "Usage: $0 <device_name> <threads> <ns> [lut2(on/off)] [entry_size]"
    echo "Example: $0 mydevice 4 \"128,256\" on 32"
    exit 1
fi

# Set default values if not provided
LUT2=${LUT2:-on}
ENTRY_SIZE=${ENTRY_SIZE:-32}

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
RESULTS_DIR="$PROJECT_ROOT/evaluation/results_gemm_${DEVICE_NAME}"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Set build flags based on configuration
BUILD_FLAGS=""
if [ "$LUT2" == "on" ]; then
    BUILD_FLAGS="$BUILD_FLAGS -DBITNET_LUT2=ON"
else
    BUILD_FLAGS="$BUILD_FLAGS -DBITNET_LUT2=OFF"
fi

if [ -n "$ENTRY_SIZE" ]; then
    BUILD_FLAGS="$BUILD_FLAGS -DTABLE_ENTRY_SIZE=$ENTRY_SIZE"
fi

BUILD_DIR="$PROJECT_ROOT/build-lut2${LUT2}-entry${ENTRY_SIZE}"

# Build with this configuration
echo "Building with LUT2=$LUT2, ENTRY_SIZE=$ENTRY_SIZE..."
cd "$PROJECT_ROOT"
cmake -B "$BUILD_DIR" $BUILD_FLAGS > /dev/null 2>&1
cmake --build "$BUILD_DIR" --target test-bitnet-gemm --config Release -j$(nproc) > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi

# Models to benchmark
MODELS=("llama3_8b")

# Run benchmark for each model
for model in "${MODELS[@]}"; do
    echo "Running benchmark for model $model with threads=$THREADS, ns=$NS..."
    
    # Define output file
    LOG_FILE="$RESULTS_DIR/${model}_t${THREADS}_ns${NS//,/-}_l${LUT2}_s${ENTRY_SIZE}.log"
    
    # Run the benchmark
    "$BUILD_DIR/bin/test-bitnet-gemm" perf -b CPU -t "$THREADS" -m "$model" -ns "$NS" > "$LOG_FILE" 2>&1
    
    echo "Results saved to $LOG_FILE"
    
    # Process the log file to CSV using test-to-csv.sh
    "$SCRIPT_DIR/test-to-csv.sh" "$LOG_FILE"
    
    echo "Processed results to ${LOG_FILE%.*}.csv"
done

echo "All benchmarks completed for device $DEVICE_NAME with threads=$THREADS, ns=$NS, LUT2=$LUT2, ENTRY_SIZE=$ENTRY_SIZE"