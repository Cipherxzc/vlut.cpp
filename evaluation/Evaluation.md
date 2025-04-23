# Evaluation

## Setup

#### Devices

- Intel PC (`pc_intel`): Intel Core i7-13700k, 8 P-Cores, AVX2.
- ARM Server (`aws_arm`): ARM Neoverse-V1, 8 Cores, SVE.
- AMD Laptop (`laptop_amd`): Ryzen 7 5800H, 8 Cores, AVX2.
- Smartphone (`smartphone`): Qualcomm Snapdragon 8 Elite, 2 P-Cores, NEON.

#### Models

- BitNet 3B (`bitnet_3b`)
    - https://huggingface.co/1bitLLM/bitnet_b1_58-3B.
- Llama3 8B (`llama3_8b`)
    - 1.58-bit: https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens.
    - 2-bit (T-MAC): https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ.
- Falcon 1B (`falcon_1b`)
    - https://huggingface.co/tiiuae/Falcon3-1B-Instruct-1.58bit.

#### Frameworks

- Ours (I2_S, I1_M): cmake build, fixed model.
- llama.cpp (TQ2_0, TQ1_0): cmake build, fixed model.
- T-MAC (INT_N): build in virtualenv, tvm (dependency) compiled.
    - **Note:** Need to convert model after comiling kernel.
    - **Note:** Need to tune for `n=256` in GeMM benchmark, and `n=1` (by default) in E2E.
- bitnet.cpp (i2_s/tl1/tl2): build in conda env, compiled.
    - **Note:** Need to re-compile kernels (will be overided) when changing models.

**Note:** Put all frameworks in the same workspace folder, so the scripts use relative path to find them correctly.

**Note:** Put all models folders in the same folder, **recommend** using default HF model name (e.g., `bitnet_b1_58-3B`) as folder name, and **recommend** using `llama.cpp`'s default model names: `ggml-model-{quant}.gguf` except the T-MAC models. **Rename** them to avoid potential bugs with the scripts.

**Note:** Before running evaluation, do a final check that all models are correctly quantized!
    - I find converting Llama3 8B model to bitnet (weight + scale) format consumes a lot of memory (might OOM). If you encounter this, download the quantized models directly.

## GeMM Benchmark

#### Scripts

- `bench-gemm.sh`: Ours & llama.cpp GeMM (with `test-bitnet-gemm.cpp`).
- `bench-gemm-tmac.sh`: T-MAC GeMM (T-MAC is tuned for n=256, will overide previous kernels).
    - Will copy `tmac_model_utils.py` to target directory.

#### Usage

```bash
./evaluation/scripts/bench-gemm.sh -h
Usage: ./evaluation/scripts/bench-gemm.sh <device_name> <threads> <ns> [lut2(on/off)] [entry_size]
Example: ./evaluation/scripts/bench-gemm.sh mydevice 4 "128,512,1024" on 32

./evaluation/scripts/bench-gemm-tmac.sh -h
Unknown argument: -h
Usage: ./evaluation/scripts/bench-gemm-tmac.sh [--device DEVICE_NAME] [--tmac_path TMAC_PATH]
```

#### Note
- Device name is for plotting. e.g., pc_intel, laptop_amd, etc.
- Currently only use **ns = 256** for plotting.
- See `.sh` for default values.

## E2E Prefill

#### Scripts

- `bench-e2e-prefill.sh`: evaluate all frameworks on one model, need `MODEL_DIR` as input environment variable.
    - Denpends on `bench-pp.sh`: use `llama-bench` to evaluate one framework.
    - **Note:** Ensure T-MAC's model uses the folder name as model name (by default). For example: `~/models/bitnet_b1_58-3B/bitnet_b1_58-3B.INT_N.gguf`, so the script can correctly find T-MAC's model.
    - **Note:** Ensure all other models are named `ggml-model-{quant}.gguf` (by default), so the script can correctly find them.
    - **Note:** Re-compile T-MAC and re-convert T-MAC models for n=1 before evaluation (we compiled for n=256 in GeMM benchmark, and it doesn't directly work for E2E evaluation).
    - **Note:** Re-compile bitnet.cpp **each time before evaluating a new model** since the kernels are always overided by last model's compilation.


#### Example

```bash
# Run for each model with the command below
# Run for Llama3 8B 1.58bit and 2bit separately since they are required by different frameworks
DEVICE_NAME=custom_device MODEL_DIR=~/models/Falcon3-1B-Instruct-1.58bit ./evaluation/scripts/bench-e2e-prefill.sh
```

More environment variables to customize in the script:

```sh
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration variables that can be easily changed
DEVICE_NAME="${DEVICE_NAME:-"mydevice"}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR/../../..}" # scripts -> evaluation -> llama.cpp-bitnet -> workspace
MODEL_DIR="${MODEL_DIR:-$HOME/models/bitnet_b1_58-3B}"
# Extract model name from model dir to separate results folder
MODEL_NAME=$(basename "$MODEL_DIR")
RESULTS_DIR="${RESULTS_DIR:-"${WORKSPACE_DIR}/llama.cpp-bitnet/evaluation/results_e2e_${DEVICE_NAME}/${MODEL_NAME}"}"
PROMPT_LENGTH="${PROMPT_LENGTH:-128,256,512}"
THREAD_COUNT="${THREAD_COUNT:-1,4,8}" # use 1, 2 on snapdragon 8 elite
REPEAT_COUNT="${REPEAT_COUNT:-3}"
```

## E2E Batched Decoding

TODO, use `llama-batched-bench`, similar to prefill.
