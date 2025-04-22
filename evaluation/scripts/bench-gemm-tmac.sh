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
            echo "Usage: $0 [--device DEVICE_NAME] [--tmac_path TMAC_PATH] [-n N] [--tune]"
            exit 1
            ;;
    esac
done

echo "Using device: $DEVICE_NAME"
echo "Using T-MAC path: $TMAC_PATH"
if [ -n "$TUNE_FLAG" ]; then
    echo "Tuning enabled"
fi

# Check if T-MAC directory exists
if [ ! -d "$TMAC_PATH" ]; then
    echo "Error: T-MAC directory not found at $TMAC_PATH"
    exit 1
fi

# Check if tmac_model_utils.py exists in current directory
if [ ! -f "tmac_model_utils.py" ]; then
    echo "Error: tmac_model_utils.py not found in current directory"
    exit 1
fi

# Backup original model_utils.py
if [ -f "$TMAC_PATH/python/t_mac/model_utils.py" ]; then
    mv "$TMAC_PATH/python/t_mac/model_utils.py" "$TMAC_PATH/python/t_mac/model_utils.py.bak"
    echo "Backed up original model_utils.py to model_utils.py.bak"
else
    echo "Error: model_utils.py not found in T-MAC directory"
    exit 1
fi

# Copy py to T-MAC/python/t_mac directory
cp tmac_model_utils.py "$TMAC_PATH/python/t_mac/model_utils.py"
echo "tmac_model_utils.py to $TMAC_PATH/python/t_mac/model_utils.py"

# Change to T-MAC directory
cd "$TMAC_PATH" || exit 1

# Create evaluation directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Clean tune log
rm "$TMAC_PATH/deploy/tuned/llama-3-8b-2bit_INT_N/qgemm_lut/tune.log"
rm "$TMAC_PATH/deploy/tuned/hf-bitnet-3b_INT_N/qgemm_lut/tune.log"
rm "$TMAC_PATH/deploy/tuned/llama-3-8b-2bit_INT_N/preprocessor/tune.log"
rm "$TMAC_PATH/deploy/tuned/hf-bitnet-3b_INT_N/preprocessor/tune.log"

# Run compiling (will overide exsiting tuned kernel!)
source "$TMAC_PATH/build/t-mac-envs.sh"
python tools/run_pipeline.py -o ~/models/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ -m llama-3-8b-2bit -q int_n -nt 1 -s 0
python tools/run_pipeline.py -o ~/models/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ -m llama-3-8b-2bit -q int_n -nt 4 -s 0
python tools/run_pipeline.py -o ~/models/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ -m llama-3-8b-2bit -q int_n -nt 8 -s 0
python tools/run_pipeline.py -o ~/models/bitnet_b1_58-3B -q int_n -nt 1 -s 0
python tools/run_pipeline.py -o ~/models/bitnet_b1_58-3B -q int_n -nt 4 -s 0
python tools/run_pipeline.py -o ~/models/bitnet_b1_58-3B -q int_n -nt 8 -s 0

# Resume model_utils.py
if [ -f "$TMAC_PATH/python/t_mac/model_utils.py.bak" ]; then
    rm "$TMAC_PATH/python/t_mac/model_utils.py"
    mv "$TMAC_PATH/python/t_mac/model_utils.py.bak" "$TMAC_PATH/python/t_mac/model_utils.py"
    echo "Restored original model_utils.py from model_utils.py.bak"
else
    echo "Error: backup of model_utils.py not found"
    exit 1
fi

cp "$TMAC_PATH/deploy/tuned/llama-3-8b-2bit_INT_N/qgemm_lut/tune.log" "${RESULTS_DIR}/llama3_8b_qgemm_lut.csv"
cp "$TMAC_PATH/deploy/tuned/llama-3-8b-2bit_INT_N/preprocessor/tune.log" "${RESULTS_DIR}/llama3_8b_preprocessor.csv"
cp "$TMAC_PATH/deploy/tuned/hf-bitnet-3b_INT_N/qgemm_lut/tune.log" "${RESULTS_DIR}/bitnet_3b_qgemm_lut.csv"
cp "$TMAC_PATH/deploy/tuned/hf-bitnet-3b_INT_N/preprocessor/tune.log" "${RESULTS_DIR}/bitnet_3b_preprocessor.csv"

echo "Done! Results have been saved to ${RESULTS_DIR}"