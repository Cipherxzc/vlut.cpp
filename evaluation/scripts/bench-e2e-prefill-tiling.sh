#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration variables that can be easily changed
DEVICE_NAME="${DEVICE_NAME:-"mydevice"}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR/../../..}" # scripts -> evaluation -> llama.cpp-bitnet -> workspace
MODEL_DIR="${MODEL_DIR:-$HOME/models/bitnet_b1_58-3B}"
# Extract model name from model dir to separate results folder
MODEL_NAME=$(basename "$MODEL_DIR")
RESULTS_DIR="${RESULTS_DIR:-"${WORKSPACE_DIR}/llama.cpp-bitnet/evaluation/results_e2e_prefill_${DEVICE_NAME}/${MODEL_NAME}"}"
PROMPT_LENGTH="${PROMPT_LENGTH:-128,256,512}"
THREAD_COUNT="${THREAD_COUNT:-1,4,8}" # use 2 on snapdragon 8 elite
REPEAT_COUNT="${REPEAT_COUNT:-3}"

# Benchmark the bitnet inference speed of different frameworks with `bench-pp.sh`
echo "Starting benchmarks with parameters:"
echo "  Device name: $DEVICE_NAME"
echo "  Workspace directory: $WORKSPACE_DIR"
echo "  Models directory: $MODEL_DIR"
echo "  Model name: $MODEL_NAME"
echo "  Prompt length: $PROMPT_LENGTH"
echo "  Thread count: $THREAD_COUNT"
echo "  Repeat count: $REPEAT_COUNT"
echo "  Results will be saved to: $RESULTS_DIR"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Pass to bench-pp.sh
export RESULTS_DIR="$RESULTS_DIR"

# echo "Benchmarking I2_S_4 model..."
# "$SCRIPT_DIR/bench-pp.sh" -m "$MODEL_DIR/ggml-model-I2_S_4.gguf" -p "$PROMPT_LENGTH" -t "$THREAD_COUNT" -r "$REPEAT_COUNT" --csv
# echo "Benchmarking I2_S_8 model..."
# "$SCRIPT_DIR/bench-pp.sh" -m "$MODEL_DIR/ggml-model-I2_S_8.gguf" -p "$PROMPT_LENGTH" -t "$THREAD_COUNT" -r "$REPEAT_COUNT" --csv
echo "Benchmarking I1_M_2 model..."
"$SCRIPT_DIR/bench-pp.sh" -m "$MODEL_DIR/ggml-model-I1_M_2.gguf" -p "$PROMPT_LENGTH" -t "$THREAD_COUNT" -r "$REPEAT_COUNT" --csv