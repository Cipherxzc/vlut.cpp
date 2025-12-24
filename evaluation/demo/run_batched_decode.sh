#!/usr/bin/env bash

set -euo pipefail

BIN="./build/bin/llama-batched"
MODEL_DIR="$HOME/models/Llama3-8B-1.58-100B-tokens"
# MODEL_DIR="$HOME/models/bitnet_b1_58-3B"
MODEL1="$MODEL_DIR/ggml-model-I1_V_2.gguf"	# vlut.cpp I1_V_2 (1.60 bpw ternary)
MODEL2="$MODEL_DIR/ggml-model-TQ1_0.gguf"	# llama.cpp TQ1_0 (1.69 bpw ternary)

if [[ ! -x "$BIN" ]]; then
	echo "Error: binary not found or not executable: $BIN" >&2
	echo "Please build it first, e.g.:" >&2
	echo "  cmake --build build --target llama-batched --config Release" >&2
	echo "Or update BIN in demo/run_batched_decode.sh to the correct path." >&2
	exit 1
fi

missing_models=()
[[ -f "$MODEL1" ]] || missing_models+=("$MODEL1")
[[ -f "$MODEL2" ]] || missing_models+=("$MODEL2")

if (( ${#missing_models[@]} > 0 )); then
	echo "Error: the following model file(s) are missing:" >&2
	for m in "${missing_models[@]}"; do
		echo "  $m" >&2
	done
	echo >&2
	echo "Please either:" >&2
	echo "  - Download/prepare the models at the paths above, or" >&2
	echo "  - Edit demo/run_batched_decode.sh to point to your actual model locations." >&2
	exit 1
fi

# hide system cursor
tput civis
trap 'tput cnorm' EXIT INT TERM

COMMON_ARGS="-np 32 -n 16 -t 1 --temp 0.5 --repeat-penalty 1.5"

$BIN -m $MODEL1 -p "I believe" $COMMON_ARGS
echo "" >&2
echo "[INFO] Test 1 finished. Waiting 5 seconds before starting test 2..." >&2
sleep 5
$BIN -m $MODEL2 -p "I believe" $COMMON_ARGS