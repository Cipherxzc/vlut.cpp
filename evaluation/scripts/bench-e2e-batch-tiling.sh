#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration variables that can be easily changed
DEVICE_NAME="${DEVICE_NAME:-"mydevice"}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR/../../..}" # scripts -> evaluation -> llama.cpp-bitnet -> workspace
MODEL_DIR="${MODEL_DIR:-$HOME/models/bitnet_b1_58-3B}"
# Extract model name from model dir to separate results folder
MODEL_NAME=$(basename "$MODEL_DIR")
RESULTS_DIR="${RESULTS_DIR:-"${WORKSPACE_DIR}/llama.cpp-bitnet/evaluation/results_e2e_batch_${DEVICE_NAME}/${MODEL_NAME}"}"
PREFILL_LEN="${PREFILL_LEN:-16}"
TOKEN_GEN_LENS="${TOKEN_GEN_LENS:-16}"
PARALLEL_SEQS="${PARALLEL_SEQS:-64,128,256}"
THREAD_COUNT="${THREAD_COUNT:-4}" # use 2 on snapdragon 8 elite

# Benchmark the bitnet inference speed of different frameworks with `bench-bd.sh`
echo "Starting batched decoding benchmarks with parameters:"
echo "  Device name: $DEVICE_NAME"
echo "  Workspace directory: $WORKSPACE_DIR"
echo "  Models directory: $MODEL_DIR"
echo "  Model name: $MODEL_NAME"
echo "  Prefill length: $PREFILL_LEN"
echo "  Token generation lengths: $TOKEN_GEN_LENS"
echo "  Parallel sequences: $PARALLEL_SEQS"
echo "  Thread count: $THREAD_COUNT"
echo "  Results will be saved to: $RESULTS_DIR"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Pass to bench-bd.sh
export RESULTS_DIR="$RESULTS_DIR"

# Benchmark I2_S and I1_M
echo "Benchmarking I2_S_4 model..."
"$SCRIPT_DIR/bench-bd.sh" -m "$MODEL_DIR/ggml-model-I2_S_4.gguf" -p "$PREFILL_LEN" -g "$TOKEN_GEN_LENS" -n "$PARALLEL_SEQS" -t "$THREAD_COUNT" --csv
echo "Benchmarking I2_S_8 model..."
"$SCRIPT_DIR/bench-bd.sh" -m "$MODEL_DIR/ggml-model-I2_S_8.gguf" -p "$PREFILL_LEN" -g "$TOKEN_GEN_LENS" -n "$PARALLEL_SEQS" -t "$THREAD_COUNT" --csv
echo "Benchmarking I1_M_2 model..."
"$SCRIPT_DIR/bench-bd.sh" -m "$MODEL_DIR/ggml-model-I1_M_2.gguf" -p "$PREFILL_LEN" -g "$TOKEN_GEN_LENS" -n "$PARALLEL_SEQS" -t "$THREAD_COUNT" --csv

echo "All batched decoding benchmarks completed. Results stored in $RESULTS_DIR"