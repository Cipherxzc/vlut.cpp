#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

warn() {
  echo "Warning: $*" >&2
}

FAILURES=0

# Configuration variables that can be easily changed
DEVICE_NAME="${DEVICE_NAME:-"mydevice"}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR/../../..}" # scripts -> evaluation -> vlut.cpp -> workspace
MODEL_DIR="${MODEL_DIR:-$HOME/models/bitnet_b1_58-3B}"
# Extract model name from model dir to separate results folder
MODEL_NAME=$(basename "$MODEL_DIR")
RESULTS_DIR="${RESULTS_DIR:-"${WORKSPACE_DIR}/vlut.cpp/evaluation/results_e2e_prefill_${DEVICE_NAME}/${MODEL_NAME}"}"
PROMPT_LENGTH="${PROMPT_LENGTH:-128,256,512}"
THREAD_COUNT="${THREAD_COUNT:-1,4}" # use 2 on snapdragon 8 elite
REPEAT_COUNT="${REPEAT_COUNT:-3}"


# Benchmark the inference speed of different frameworks with `bench-prefill.sh`
echo "Starting benchmarks with parameters:"
echo "  Device name: $DEVICE_NAME"
echo "  Workspace directory: $WORKSPACE_DIR"
echo "  Models directory: $MODEL_DIR"
echo "  Model name: $MODEL_NAME"
echo "  Prompt length: $PROMPT_LENGTH"
echo "  Thread count: $THREAD_COUNT"
echo "  Repeat count: $REPEAT_COUNT"
echo "  Results will be saved to: $RESULTS_DIR"

# Clean up old results
rm -rf "$RESULTS_DIR"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Pass to bench-prefill.sh
export RESULTS_DIR="$RESULTS_DIR"

run_prefill_bench() {
  local framework_name="$1"
  local model_path="$2"
  shift 2

  if [ ! -f "$model_path" ]; then
    warn "$framework_name: model file not found at $model_path. Skipping."
    return 0
  fi

  if ! "$SCRIPT_DIR/bench-prefill.sh" "$@" -m "$model_path" -p "$PROMPT_LENGTH" -t "$THREAD_COUNT" -r "$REPEAT_COUNT" --csv; then
    warn "$framework_name: benchmark failed for $(basename "$model_path")."
    FAILURES=$((FAILURES + 1))
  fi
}


# ==================== Benchmark I2_V and I1_V ====================
echo "Benchmarking I2_V_4 model..."
run_prefill_bench "vlut.cpp" "$MODEL_DIR/ggml-model-I2_V_4.gguf"
echo "Benchmarking I2_V_8 model..."
run_prefill_bench "vlut.cpp" "$MODEL_DIR/ggml-model-I2_V_8.gguf"
echo "Benchmarking I1_V_2 model..."
run_prefill_bench "vlut.cpp" "$MODEL_DIR/ggml-model-I1_V_2.gguf"



# ==================== Benchmark llama.cpp TQ2_0 and TQ1_0 ====================
echo "Benchmarking TQ2_0 and TQ1_0 model with llama.cpp..."
LLAMA_CPP_DIR="$WORKSPACE_DIR/llama.cpp"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  warn "llama.cpp directory not found at $LLAMA_CPP_DIR. Skipping llama.cpp benchmarks."
else
  run_prefill_bench "llama.cpp" "$MODEL_DIR/ggml-model-TQ2_0.gguf" -w "$LLAMA_CPP_DIR"
  run_prefill_bench "llama.cpp" "$MODEL_DIR/ggml-model-TQ1_0.gguf" -w "$LLAMA_CPP_DIR"
fi



# ==================== Benchmark T-MAC ====================
echo "Benchmarking T-MAC model..."
TMAC_DIR="$WORKSPACE_DIR/T-MAC"
TMAC_LLAMA_CPP_DIR="$TMAC_DIR/3rdparty/llama.cpp"

if [ ! -d "$TMAC_LLAMA_CPP_DIR" ]; then
  warn "T-MAC llama.cpp directory not found at $TMAC_LLAMA_CPP_DIR. Skipping T-MAC benchmarks."
else
  run_prefill_bench "T-MAC" "$MODEL_DIR/$MODEL_NAME.INT_N.gguf" -w "$TMAC_LLAMA_CPP_DIR" # model name is not ggml-model-...
fi



# ==================== Benchmark bitnet.cpp if available ====================
echo "Benchmarking bitnet.cpp model..."
BITNET_CPP_DIR="$WORKSPACE_DIR/BitNet"

if [ ! -d "$BITNET_CPP_DIR" ]; then
  warn "bitnet.cpp directory not found at $BITNET_CPP_DIR. Skipping bitnet.cpp benchmarks."
else
  # one of these would work
  run_prefill_bench "bitnet.cpp" "$MODEL_DIR/ggml-model-tl2.gguf" -w "$BITNET_CPP_DIR"
  run_prefill_bench "bitnet.cpp" "$MODEL_DIR/ggml-model-tl1.gguf" -w "$BITNET_CPP_DIR"
  run_prefill_bench "bitnet.cpp" "$MODEL_DIR/ggml-model-i2_s.gguf" -w "$BITNET_CPP_DIR"
fi

if [ $FAILURES -ne 0 ]; then
  warn "$FAILURES benchmark invocation(s) failed. Results stored in $RESULTS_DIR"
  exit 1
fi

echo "All benchmarks completed. Results stored in $RESULTS_DIR"
