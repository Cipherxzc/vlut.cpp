#!/bin/bash

# Configuration variables that can be easily changed
# WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/repos}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/repos/bitnet}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/bitnet_b1_58-3B}"
RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE_DIR/llama.cpp-bitnet/evaluation/results}"
PROMPT_LENGTH="${PROMPT_LENGTH:-128,256,512}"
THREAD_COUNT="${THREAD_COUNT:-1,2,4}"
REPEAT_COUNT="${REPEAT_COUNT:-3}"
# PROMPT_LENGTH="${PROMPT_LENGTH:-128}"
# THREAD_COUNT="${THREAD_COUNT:-4}"
# REPEAT_COUNT="${REPEAT_COUNT:-1}"


# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Benchmark the bitnet inference speed of different frameworks with `bench-pp.sh`
echo "Starting benchmarks with parameters:"
echo "  Models directory: $MODEL_DIR"
echo "  Prompt length: $PROMPT_LENGTH"
echo "  Thread count: $THREAD_COUNT"
echo "  Repeat count: $REPEAT_COUNT"
echo "  Results will be saved to: $RESULTS_DIR"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Benchmark I2_S
echo "Benchmarking I2_S model..."
"$SCRIPT_DIR/bench-pp.sh" -m "$MODEL_DIR/ggml-model-I2_S.gguf" -p "$PROMPT_LENGTH" -t "$THREAD_COUNT" -r "$REPEAT_COUNT" --csv

# Benchmark llama.cpp Q2_K
echo "Benchmarking Q2_K model with llama.cpp..."
LLAMA_CPP_DIR="$WORKSPACE_DIR/llama.cpp"
LLAMA_CPP_RESULTS="$LLAMA_CPP_DIR/evaluation/results"

"$SCRIPT_DIR/bench-pp.sh" -w "$LLAMA_CPP_DIR" -m "$MODEL_DIR/ggml-model-Q2_K.gguf" -p "$PROMPT_LENGTH" -t "$THREAD_COUNT" -r "$REPEAT_COUNT" --csv

# mv results back and merge with current
if [ -d "$LLAMA_CPP_RESULTS" ] && [ "$(ls -A "$LLAMA_CPP_RESULTS")" ]; then
  echo "Moving results from $LLAMA_CPP_RESULTS to $RESULTS_DIR"
  cp -r "$LLAMA_CPP_RESULTS"/* "$RESULTS_DIR/"
fi

# Benchmark T-MAC
echo "Benchmarking T-MAC model..."
TMAC_DIR="$WORKSPACE_DIR/T-MAC"
TMAC_LLAMA_CPP_DIR="$TMAC_DIR/3rdparty/llama.cpp"
TMAC_RESULTS="$TMAC_LLAMA_CPP_DIR/evaluation/results"

"$SCRIPT_DIR/bench-pp.sh" -w "$TMAC_LLAMA_CPP_DIR" -m "$MODEL_DIR/bitnet_b1_58-3B.INT_N.gguf" -p "$PROMPT_LENGTH" -t "$THREAD_COUNT" -r "$REPEAT_COUNT" --csv

# mv results back and merge with current
if [ -d "$TMAC_RESULTS" ] && [ "$(ls -A "$TMAC_RESULTS")" ]; then
  echo "Moving results from $TMAC_RESULTS to $RESULTS_DIR"
  cp -r "$TMAC_RESULTS"/* "$RESULTS_DIR/"
fi

echo "All benchmarks completed. Results stored in $RESULTS_DIR"