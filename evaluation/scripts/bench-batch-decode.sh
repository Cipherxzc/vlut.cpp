#!/bin/bash

# Function to display help information
show_help() {
  echo "Usage: $(basename "$0") [options]"
  echo ""
  echo "A script to benchmark llama.cpp using llama-batched-bench."
  echo ""
  echo "Options:"
  echo "  -h                   Display this help message and exit"
  echo "  -w <directory>       Set the llama.cpp root directory (default: ../../ relative to script)"
  echo "  -m <path>            Path to the model to benchmark (default: ~/models/bitnet_b1_58-3B/ggml-model-I2_V.gguf)"
  echo "  -c <size>            Context size (default: auto-calculated from other parameters)"
  echo "  -p <length>          Prefill length (default: 32)"
  echo "  -g <list>            Token generation lengths in comma-separated format (default: 16,32,64)"
  echo "                       Example: -ntg 16,32,64,128"
  echo "  -n <list>            Parallel sequence numbers in comma-separated format (default: 32,64)"
  echo "                       Example: -npl 16,32,64"
  echo "  -t <list>            Thread counts (default: 4)"
  echo "                       Example: -t 1"
  echo "  --csv                Also generate a simplified CSV file from the benchmark results"
  echo "  --add=<args>         Additional arguments to pass directly to llama-batched-bench"
  echo ""
  echo "Example:"
  echo "  $(basename "$0") -m ~/models/my-model.gguf -npp 32 -ntg 16,32,64 -npl 32,64 -t 4 -r 3 --csv"
  echo ""
  echo "Note: Context size (-c) will be auto-calculated as: [npp + max(ntg)] * max(npl)"
  echo "      You can override this with the -c parameter."
  echo ""
  echo "Output will be saved to: <llama_root>/evaluation/results/<model_name>_batched_<config>_<timestamp>.txt"
  echo "CSV output (if enabled) will be saved to: <llama_root>/evaluation/results/<model_name>_batched_<config>_<timestamp>.csv"
}

# Function to parse the benchmark output and generate a CSV file
generate_csv() {
  local input_file=$1
  local output_file=$2
  
  # Extract command line
  COMMAND=$(grep "^Command:" "$input_file" | sed 's/^Command: //')
  
  # Create CSV header - based on the batched bench output format
  echo "PP,TG,B,N_KV,T_PP_s,S_PP_t/s,T_TG_s,S_TG_t/s,T_s,S_t/s" > "$output_file"
  
  # Parse all result lines and extract data
  # Skip header lines (containing headers or dashes)
  grep "^|" "$input_file" | grep -v "PP" | grep -v "\-\-\-" | while read -r line; do
    # Extract all columns
    PP=$(echo "$line" | awk -F '|' '{print $2}' | xargs)
    TG=$(echo "$line" | awk -F '|' '{print $3}' | xargs)
    B=$(echo "$line" | awk -F '|' '{print $4}' | xargs)
    N_KV=$(echo "$line" | awk -F '|' '{print $5}' | xargs)
    T_PP_S=$(echo "$line" | awk -F '|' '{print $6}' | xargs)
    S_PP_TS=$(echo "$line" | awk -F '|' '{print $7}' | xargs)
    T_TG_S=$(echo "$line" | awk -F '|' '{print $8}' | xargs)
    S_TG_TS=$(echo "$line" | awk -F '|' '{print $9}' | xargs)
    T_S=$(echo "$line" | awk -F '|' '{print $10}' | xargs)
    S_TS=$(echo "$line" | awk -F '|' '{print $11}' | xargs)
    
    # Output to CSV
    echo "$PP,$TG,$B,$N_KV,$T_PP_S,$S_PP_TS,$T_TG_S,$S_TG_TS,$T_S,$S_TS" >> "$output_file"
  done
  
  echo "CSV file generated: $output_file"
}

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
LLAMA_ROOT="$SCRIPT_DIR/../.."
MODEL_PATH="$HOME/models/bitnet_b1_58-3B/ggml-model-I2_V.gguf"
PREFILL_LEN="32"
TOKEN_GEN_LENS="16,32,64"
PARALLEL_SEQS="32,64"
THREADS="4"
ADDITIONAL_ARGS="-pps" # share prompt to reduce benchmark time
CTX_SIZE=""

# Parse command line arguments
while getopts "hw:m:c:p:g:n:t:r:-:" opt; do
  case $opt in
    h)
      show_help
      exit 0
      ;;
    w)
      LLAMA_ROOT="$OPTARG"
      ;;
    m)
      MODEL_PATH="$OPTARG"
      ;;
    c)
      CTX_SIZE="$OPTARG"
      ;;
    p)
      PREFILL_LEN="$OPTARG"
      ;;
    g)
      echo "Token generation lengths: $OPTARG"
      TOKEN_GEN_LENS="$OPTARG"
      ;;
    n)
      PARALLEL_SEQS="$OPTARG"
      ;;
    t)
      THREADS="$OPTARG"
      ;;
    -)
      case "${OPTARG}" in
        csv)
          GENERATE_CSV=true
          ;;
        add=*)
          ADDITIONAL_ARGS="${OPTARG#*=}"
          ;;
        help)
          show_help
          exit 0
          ;;
        *)
          echo "Invalid option: --${OPTARG}" >&2
          show_help
          exit 1
          ;;
      esac
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      show_help
      exit 1
      ;;
  esac
done

# Convert relative paths to absolute paths if needed
if [[ ! "$LLAMA_ROOT" = /* ]]; then
  LLAMA_ROOT="$( cd "$SCRIPT_DIR/$LLAMA_ROOT" && pwd )"
fi

if [[ ! "$MODEL_PATH" = /* ]]; then
  MODEL_PATH="$( cd "$SCRIPT_DIR/$MODEL_PATH" 2>/dev/null && pwd )/$( basename "$MODEL_PATH" )" || MODEL_PATH="$MODEL_PATH"
fi

# Set working directory to llama.cpp root
cd "$LLAMA_ROOT" || { echo "Error: Cannot change to llama.cpp root directory: $LLAMA_ROOT"; exit 1; }

# Check if llama-batched-bench exists
LLAMA_BENCH="build/bin/llama-batched-bench"
if [ ! -f "$LLAMA_BENCH" ]; then
  echo "Error: llama-batched-bench not found at $LLAMA_BENCH"
  exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model not found at $MODEL_PATH"
  exit 1
fi

# Create results directory if it doesn't exist
RESULTS_DIR="${RESULTS_DIR:-"evaluation/results"}"
mkdir -p "$RESULTS_DIR"

# Get model name from path
MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/\.gguf$//')

# Calculate context size if not provided
if [ -z "$CTX_SIZE" ]; then
  # Find max token generation length
  MAX_TG=$(echo "$TOKEN_GEN_LENS" | tr ',' '\n' | sort -n | tail -1)
  # Find max parallel sequence number
  MAX_PL=$(echo "$PARALLEL_SEQS" | tr ',' '\n' | sort -n | tail -1)
  # Calculate context size: (prefill_len + max_token_gen) * max_parallel_seq
  CTX_SIZE=$(( (PREFILL_LEN + MAX_TG) * MAX_PL ))
  echo "Auto-calculated context size: $CTX_SIZE = ($PREFILL_LEN + $MAX_TG) * $MAX_PL"
fi

# Generate timestamp for the output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Generate config string for the output filename
CONFIG_STR="npp${PREFILL_LEN}_ntg${TOKEN_GEN_LENS//,/-}_npl${PARALLEL_SEQS//,/-}_t${THREADS//,/-}"

# Define output file
OUTPUT_FILE="$RESULTS_DIR/${MODEL_NAME}_${CONFIG_STR}_${TIMESTAMP}.txt"
CSV_FILE="${OUTPUT_FILE%.txt}.csv"

echo "Starting batched benchmark with the following configuration:"
echo "Model: $MODEL_PATH"
echo "Context size: $CTX_SIZE"
echo "Prefill length: $PREFILL_LEN"
echo "Token generation lengths: $TOKEN_GEN_LENS"
echo "Parallel sequence numbers: $PARALLEL_SEQS"
echo "Threads: $THREADS"
echo "Additional arguments: $ADDITIONAL_ARGS"
echo "Output will be saved to: $OUTPUT_FILE"
if [ "$GENERATE_CSV" = true ]; then
  echo "CSV output will be saved to: $CSV_FILE"
fi

# Run benchmark
CMD="./$LLAMA_BENCH -m \"$MODEL_PATH\" -c $CTX_SIZE -npp $PREFILL_LEN -ntg $TOKEN_GEN_LENS -npl $PARALLEL_SEQS -t $THREADS -ngl 0 $ADDITIONAL_ARGS"
echo "Running command: $CMD"
echo "Command: $CMD" > "$OUTPUT_FILE"
echo "----------------------------------------" >> "$OUTPUT_FILE"
eval "$CMD" >> "$OUTPUT_FILE" 2>&1

echo "Batched benchmark completed. Results saved to $OUTPUT_FILE"

# Generate CSV if requested
if [ "$GENERATE_CSV" = true ]; then
  generate_csv "$OUTPUT_FILE" "$CSV_FILE"
fi