#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p $SCRIPT_DIR/../results_gemm

> $SCRIPT_DIR/../results_gemm/results.txt

# Change to the build/bin directory
cd "$SCRIPT_DIR/../../build/bin"

# Run tests with different configurations
for t in 1 4; do
    for m in bitnet_3b llama3_8b falcon_1b trilm_1.5b; do
        for n in 128 256 512 1024; do
            ./test-bitnet-gemm perf -b CPU -t $t -m $m -ns $n 2>&1 | grep "^  MUL_MAT" >> $SCRIPT_DIR/../results_gemm/results.txt
        done
    done
done