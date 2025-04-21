#!/bin/bash

# Default values
DEVICE_NAME="default_device"
TUNE_FLAG=""
N_FLAG=""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
RESULTS_DIR="$PROJECT_ROOT/evaluation/results_tmac_${DEVICE_NAME}"

# Work in script dir
cd "$SCRIPT_DIR" || exit 1

TMAC_PATH="$PROJECT_ROOT/.."

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
        -n)
            N_FLAG="-n $2"
            shift 2
            ;;
        --tune)
            TUNE_FLAG="-t"
            shift
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

# Check if profile_tmac.py exists in current directory
if [ ! -f "profile_tmac.py" ]; then
    echo "Error: profile_tmac.py not found in current directory"
    exit 1
fi

# Create tools directory if it doesn't exist
mkdir -p "$TMAC_PATH/tools"

# Copy profile_tmac.py to T-MAC/tools directory
cp profile_tmac.py "$TMAC_PATH/tools/"
echo "Copied profile_tmac.py to $TMAC_PATH/tools/"

# Change to T-MAC directory
cd "$TMAC_PATH" || exit 1

# Create evaluation directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Run for bitnet_3b
echo "Running profile_tmac.py with preset bitnet_3b..."
> "out_bitnet_3b/results.csv"
python tools/profile_tmac.py --preset bitnet_3b --out_path out_bitnet_3b $TUNE_FLAG $N_FLAG

# Run for llama3_8b
echo "Running profile_tmac.py with preset llama3_8b..."
> "out_llama3_8b/results.csv"
python tools/profile_tmac.py --preset llama3_8b --out_path out_llama3_8b $TUNE_FLAG $N_FLAG

# Copy results to evaluation directory
echo "Copying results to evaluation directory..."
cp out_bitnet_3b/results.csv "${RESULTS_DIR}/bitnet_3b.csv"
cp out_llama3_8b/results.csv "${RESULTS_DIR}/llama3_8b.csv"

echo "Done! Results have been saved to:"
echo "  ${RESULTS_DIR}/bitnet_3b.csv"
echo "  ${RESULTS_DIR}/llama3_8b.csv"