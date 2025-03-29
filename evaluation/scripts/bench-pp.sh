#!/bin/bash


# Function to display help information
show_help() {
  echo "Usage: $(basename "$0") [options]"
  echo ""
  echo "A script to benchmark llama.cpp using llama-bench."
  echo ""
  echo "Options:"
  echo "  -h                   Display this help message and exit"
  echo "  -w <directory>       Set the llama.cpp root directory (default: ../../ relative to script)"
  echo "  -m <path>            Path to the model to benchmark (default: ~/models/bitnet_b1_58-3B/ggml-model-I2_S.gguf)"
  echo "  -p <list>            Input lengths in comma-separated format (default: 128)"
  echo "                       Example: -p 64,128,256"
  echo "  -t <list>            Thread counts in comma-separated format (default: 1)"
  echo "                       Example: -t 1,2,4,8"
  echo "  -r <number>          Number of repeats for each benchmark (default: 5)"
  echo "  --csv                Also generate a simplified CSV file from the benchmark results"
  echo "  --add=<args>         Additional arguments to pass directly to llama-bench"
  echo ""
  echo "Example:"
  echo "  $(basename "$0") -m ~/models/my-model.gguf -p 64,128,256 -t 1,2,4 -r 3 -c --add=\"-ctx 2048\""
  echo ""
  echo "Output will be saved to: <llama_root>/evaluation/results/<model_name>_<config>_<timestamp>.txt"
  echo "CSV output (if enabled) will be saved to: <llama_root>/evaluation/results/<model_name>_<config>_<timestamp>.csv"
}

# Function to parse the benchmark output and generate a CSV file
generate_csv() {
  local input_file=$1
  local output_file=$2
  
  # Extract command line
  COMMAND=$(grep "^Command:" "$input_file" | sed 's/^Command: //')
  
  # Extract model info from the first result line
  MODEL_INFO=$(grep -A 1 "^| model" "$input_file" | tail -n 1)
  MODEL_NAME=$(echo "$MODEL_INFO" | awk -F '|' '{print $2}' | xargs)
  SIZE=$(echo "$MODEL_INFO" | awk -F '|' '{print $3}' | xargs)
  PARAMS=$(echo "$MODEL_INFO" | awk -F '|' '{print $4}' | xargs)
  BACKEND=$(echo "$MODEL_INFO" | awk -F '|' '{print $5}' | xargs)
  
  # # Create CSV header
  # echo "# $COMMAND" > "$output_file"
  # echo "# $MODEL_NAME, $SIZE, $PARAMS, $BACKEND" >> "$output_file"
  # echo "p (pp len),t (threads),t/s" >> "$output_file"
  echo "p,t,avg_ts,stdev_ts" >> "$output_file"
  
  # Parse all result lines and extract data
  # Skip header lines (containing "model" or dashes)
  grep "^| " "$input_file" | grep -v "model " | grep -v "\-\-\-" | while read -r line; do
    THREADS=$(echo "$line" | awk -F '|' '{print $6}' | xargs)
    PP_LEN=$(echo "$line" | awk -F '|' '{print $7}' | xargs | sed 's/pp//')
    TOKENS_PER_SEC=$(echo "$line" | awk -F '|' '{print $8}' | xargs)
    
    # Split the tokens per second into average and standard deviation
    AVG_TS=$(echo "$TOKENS_PER_SEC" | awk -F '±' '{print $1}' | xargs)
    STDEV_TS=$(echo "$TOKENS_PER_SEC" | awk -F '±' '{print $2}' | xargs)
    
    echo "$PP_LEN,$THREADS,$AVG_TS,$STDEV_TS" >> "$output_file"
  done
  
  echo "CSV file generated: $output_file"
}

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
LLAMA_ROOT="$SCRIPT_DIR/../.."
MODEL_PATH="$HOME/models/bitnet_b1_58-3B/ggml-model-I2_S.gguf"
PROMPT_LENS="128"
THREADS="1"
REPEATS=5
ADDITIONAL_ARGS=""

# Parse command line arguments
while getopts "hw:m:p:t:r:-:" opt; do
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
    p)
      PROMPT_LENS="$OPTARG"
      ;;
    t)
      THREADS="$OPTARG"
      ;;
    r)
      REPEATS="$OPTARG"
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

# Check if llama-bench exists
LLAMA_BENCH="build/bin/llama-bench"
if [ ! -f "$LLAMA_BENCH" ]; then
  echo "Error: llama-bench not found at $LLAMA_BENCH"
  exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model not found at $MODEL_PATH"
  exit 1
fi

# Create results directory if it doesn't exist
RESULTS_DIR="evaluation/results"
mkdir -p "$RESULTS_DIR"

# Get model name from path
MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/\.gguf$//')

# Parse prompt lengths and threads
PROMPT_ARGS=""
for p in ${PROMPT_LENS//,/ }; do
  PROMPT_ARGS="$PROMPT_ARGS -p $p"
done

THREAD_ARGS=""
for t in ${THREADS//,/ }; do
  THREAD_ARGS="$THREAD_ARGS -t $t"
done

# Generate timestamp for the output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Generate config string for the output filename
CONFIG_STR="p${PROMPT_LENS//,/-}_t${THREADS//,/-}_r${REPEATS}"

# Define output file
OUTPUT_FILE="$RESULTS_DIR/${MODEL_NAME}_${CONFIG_STR}_${TIMESTAMP}.txt"
CSV_FILE="${OUTPUT_FILE%.txt}.csv"

echo "Starting benchmark with the following configuration:"
echo "Model: $MODEL_PATH"
echo "Prompt lengths: $PROMPT_LENS"
echo "Threads: $THREADS"
echo "Repeats: $REPEATS"
echo "Additional arguments: $ADDITIONAL_ARGS"
echo "Output will be saved to: $OUTPUT_FILE"
if [ "$GENERATE_CSV" = true ]; then
  echo "CSV output will be saved to: $CSV_FILE"
fi

# Run benchmark
CMD="./$LLAMA_BENCH -m \"$MODEL_PATH\" $PROMPT_ARGS $THREAD_ARGS -ngl 0 -n 0 -r $REPEATS $ADDITIONAL_ARGS"
echo "Running command: $CMD"
echo "Command: $CMD" > "$OUTPUT_FILE"
echo "----------------------------------------" >> "$OUTPUT_FILE"
eval "$CMD" >> "$OUTPUT_FILE" 2>&1

echo "Benchmark completed. Results saved to $OUTPUT_FILE"

# Generate CSV if requested
if [ "$GENERATE_CSV" = true ]; then
  generate_csv "$OUTPUT_FILE" "$CSV_FILE"
fi