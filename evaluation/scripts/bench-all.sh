#!/bin/bash

# Benchmark the bitnet inference speed of different frameworks with `bench-pp.sh`

# Benchmark I2_S, I2_T
./bench-pp.sh -m ~/Projects/llama.cpp-bitnet/models/BitNet/3B-I2S.gguf -p 128,256,512 -t 1,2,4,8 -r 5 --csv
./bench-pp.sh -m ~/Projects/llama.cpp-bitnet/models/BitNet/3B-I2T.gguf -p 128,256,512 -t 1,2,4,8 -r 5 --csv

# Benchmark llama.cpp Q2_K
./bench-pp.sh -w ~/Projects/llama.cpp -m ~/Projects/llama.cpp/models/BitNet/3B-Q2K.gguf -p 128,256,512 -t 1,2,4,8 -r 5 --csv
# mv results back and merge with current
mv ~/Projects/llama.cpp/evaluation/results/* ~/Projects/llama.cpp-bitnet/evaluation/results/

# Benchmark T-MAC
./bench-pp.sh -w ~/Projects/T-MAC/3rdparty/llama.cpp -m ~/Projects/T-MAC/3rdparty/llama.cpp/models/BitNet/3B.gguf -p 128,256,512 -t 1,2,4,8 -r 5 --csv
# mv results back and merge with current
mv ~/Projects/T-MAC/3rdparty/llama.cpp/evaluation/results/* ~/Projects/llama.cpp-bitnet/evaluation/results/