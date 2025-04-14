#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the build/bin directory
cd "$SCRIPT_DIR/../../build/bin"

# Run tests with different configurations
for t in 1 4; do
    for m in bitnet_3b llama3_8b falcon_1b trilm_1.5b; do
        for n in 128 256 512 1024; do
            ./test-bitnet-gemm perf -b CPU -t $t -m $m -ns $n
        done
    done
done

# Extract and save MUL_MAT lines to a file
mkdir -p $SCRIPT_DIR/../results_gemm
grep "^  MUL_MAT" > $SCRIPT_DIR/../results_gemm/results.txt