#!/bin/bash

# Default values
DEVICE_NAME="default_device"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
RESULTS_DIR="$PROJECT_ROOT/evaluation/results_tmac_${DEVICE_NAME}"

# Work in script dir
cd "$SCRIPT_DIR" || exit 1

TMAC_PATH="$PROJECT_ROOT/../T-MAC"

# Cleanup function to ensure proper restoration of files
cleanup() {
    echo "Performing cleanup..."
    
    # Restore model_utils.py if backup exists
    if [ -f "$TMAC_PATH/python/t_mac/model_utils.py.bak" ]; then
        rm -f "$TMAC_PATH/python/t_mac/model_utils.py"
        mv "$TMAC_PATH/python/t_mac/model_utils.py.bak" "$TMAC_PATH/python/t_mac/model_utils.py"
        echo "Restored original model_utils.py from backup"
    fi
    
    # Restore platform.py if backup exists
    if [ -f "$TMAC_PATH/python/t_mac/platform.py.bak" ]; then
        rm -f "$TMAC_PATH/python/t_mac/platform.py"
        mv "$TMAC_PATH/python/t_mac/platform.py.bak" "$TMAC_PATH/python/t_mac/platform.py"
        echo "Restored original platform.py from backup"
    fi
    
    echo "Cleanup completed"
}

# Set trap to call cleanup function on script exit, interrupt, or error
trap cleanup EXIT INT TERM

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE_NAME="$2"
            shift 2
            ;;
        --tmac_path)
            TMAC_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--device DEVICE_NAME] [--tmac_path TMAC_PATH]"
            exit 1
            ;;
    esac
done

# Update results directory with current device name
RESULTS_DIR="$PROJECT_ROOT/evaluation/results_gemm_tmac_${DEVICE_NAME}"

echo "Using device: $DEVICE_NAME"
echo "Using T-MAC path: $TMAC_PATH"
echo "Results will be saved to: $RESULTS_DIR"
if [ -n "$TUNE_FLAG" ]; then
    echo "Tuning enabled"
fi

# Check if T-MAC directory exists
if [ ! -d "$TMAC_PATH" ]; then
    echo "Error: T-MAC directory not found at $TMAC_PATH"
    exit 1
fi

# Backup original model_utils.py
if [ -f "$TMAC_PATH/python/t_mac/model_utils.py" ]; then
    cp "$TMAC_PATH/python/t_mac/model_utils.py" "$TMAC_PATH/python/t_mac/model_utils.py.bak"
    echo "Backed up original model_utils.py to model_utils.py.bak"
else
    echo "Error: model_utils.py not found in T-MAC directory"
    exit 1
fi

# Copy py to T-MAC/python/t_mac directory
cp tmac_model_utils.py "$TMAC_PATH/python/t_mac/model_utils.py"
echo "Copied tmac_model_utils.py to $TMAC_PATH/python/t_mac/model_utils.py"

# Backup original platform.py
if [ -f "$TMAC_PATH/python/t_mac/platform.py" ]; then
    cp "$TMAC_PATH/python/t_mac/platform.py" "$TMAC_PATH/python/t_mac/platform.py.bak"
    echo "Backed up original platform.py to platform.py.bak"
else
    echo "Error: platform.py not found in T-MAC directory"
    exit 1
fi

# Copy py to T-MAC/python/t_mac directory
cp tmac_platform.py "$TMAC_PATH/python/t_mac/platform.py"
echo "Copied tmac_platform.py to $TMAC_PATH/python/t_mac/platform.py"

# Change to T-MAC directory
cd "$TMAC_PATH" || exit 1

# Create evaluation directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Clean tune logs - use rm -f to avoid errors if files don't exist
rm -f "$TMAC_PATH/deploy/tuned/llama-3-8b-2bit_INT_N/qgemm_lut/tune.log"
rm -f "$TMAC_PATH/deploy/tuned/hf-bitnet-3b_INT_N/qgemm_lut/tune.log"
rm -f "$TMAC_PATH/deploy/tuned/llama-3-8b-2bit_INT_N/preprocessor/tune.log"
rm -f "$TMAC_PATH/deploy/tuned/hf-bitnet-3b_INT_N/preprocessor/tune.log"

# Run compiling (will override existing tuned kernel!)
echo "Sourcing T-MAC environment..."
source "$TMAC_PATH/build/t-mac-envs.sh"

echo "Running pipeline for Llama-3-8b-2bit with 1 thread..."
python tools/run_pipeline.py -o ~/models/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ -m llama-3-8b-2bit -q int_n -nt 1 -s 0

echo "Running pipeline for Llama-3-8b-2bit with 4 threads..."
python tools/run_pipeline.py -o ~/models/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ -m llama-3-8b-2bit -q int_n -nt 4 -s 0

echo "Running pipeline for Llama-3-8b-2bit with 8 threads..."
python tools/run_pipeline.py -o ~/models/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ -m llama-3-8b-2bit -q int_n -nt 8 -s 0

echo "Running pipeline for BitNet with 1 thread..."
python tools/run_pipeline.py -o ~/models/bitnet_b1_58-3B -q int_n -nt 1 -s 0

echo "Running pipeline for BitNet with 4 thread..."
python tools/run_pipeline.py -o ~/models/bitnet_b1_58-3B -q int_n -nt 4 -s 0

echo "Running pipeline for BitNet with 8 threads..."
python tools/run_pipeline.py -o ~/models/bitnet_b1_58-3B -q int_n -nt 8 -s 0

# Note: cleanup function will handle file restoration via the trap

# Copy the tune logs to the results directory
echo "Copying tune logs to results directory..."
cp -f "$TMAC_PATH/deploy/tuned/llama-3-8b-2bit_INT_N/qgemm_lut/tune.log" "${RESULTS_DIR}/llama3_8b_qgemm_lut.jsonl" 2>/dev/null || echo "Warning: Failed to copy llama3_8b_qgemm_lut.jsonl"
cp -f "$TMAC_PATH/deploy/tuned/llama-3-8b-2bit_INT_N/preprocessor/tune.log" "${RESULTS_DIR}/llama3_8b_preprocessor.jsonl" 2>/dev/null || echo "Warning: Failed to copy llama3_8b_preprocessor.jsonl"
cp -f "$TMAC_PATH/deploy/tuned/hf-bitnet-3b_INT_N/qgemm_lut/tune.log" "${RESULTS_DIR}/bitnet_3b_qgemm_lut.jsonl" 2>/dev/null || echo "Warning: Failed to copy bitnet_3b_qgemm_lut.jsonl"
cp -f "$TMAC_PATH/deploy/tuned/hf-bitnet-3b_INT_N/preprocessor/tune.log" "${RESULTS_DIR}/bitnet_3b_preprocessor.jsonl" 2>/dev/null || echo "Warning: Failed to copy bitnet_3b_preprocessor.jsonl"

echo "Done! Results have been saved to ${RESULTS_DIR}"
# (Cleanup function will be called automatically when the script exits)