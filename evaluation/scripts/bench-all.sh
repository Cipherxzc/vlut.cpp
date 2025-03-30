#!/bin/bash

# Benchmark the bitnet inference speed of different frameworks with `bench-pp.sh`

# Benchmark I2_S, I2_T
./bench-pp.sh -m ~/models/bitnet_b1_58-3B/ggml-model-I2_S.gguf -p 128,256,512 -t 1,2,4,8 -r 5 --csv
./bench-pp.sh -m ~/models/bitnet_b1_58-3B/ggml-model-I2_T.gguf -p 128,256,512 -t 1,2,4,8 -r 5 --csv

# Benchmark llama.cpp Q2_K
./bench-pp.sh -w ~/repos/bitnet/llama.cpp -m ~/models/bitnet_b1_58-3B/ggml-model-Q2_K.gguf -p 128,256,512 -t 1,2,4,8 -r 5 --csv
# mv results back and merge with current
mv ~/repos/bitnet/llama.cpp/evaluation/results/* ~/repos/bitnet/llama.cpp-bitnet/evaluation/results/

# Benchmark T-MAC
./bench-pp.sh -w ~/repos/bitnet/T-MAC/3rdparty/llama.cpp -m ~/models/bitnet_b1_58-3B/bitnet_b1_58-3B.INT_N.gguf -p 128,256,512 -t 1,2,4,8 -r 5 --csv
# mv results back and merge with current
mv ~/repos/bitnet/T-MAC/3rdparty/llama.cpp/evaluation/results* ~/repos/bitnet/llama.cpp-bitnet/evaluation/results/
