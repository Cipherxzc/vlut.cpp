# Evaluation Guide

This documentation describes how to build vlut.cpp and reproduce our main evaluation results in the paper. We provide pre-built binaries, pre-converted models, and automatic scripts to ease the evaluation.

If you are looking for a quick start guide, see [Quick Start](../README.md#quick-start).

## Preparation

### Devices

vlut.cpp supports ARM and x86 CPU, covering most modern devices.

To run vlut.cpp on your device, it is recommended that you have:

- CPU: x86_64 or ARMv8.
- RAM: 4 GB (inference only), or 16 GB (model conversion).
- Disk: 16 GB (vlut.cpp only), or 128 GB (all frameworks).
- OS: Ubuntu/WSL/Android.
- Python: >=3.10 (model conversion).

Tested devices and configurations are listed in the paper.

### Models

With flexible sub-2-bit packing, vlut.cpp supports a rich set of ternary LLMs, including [HF BitNet family](https://huggingface.co/1bitLLM), [Llama family](https://huggingface.co/HF1BitLLM), [Falcon3 family](https://huggingface.co/collections/tiiuae/falcon3), and [TriLM family](https://huggingface.co/SpectraSuite).

Belows are tested models in the paper (verified, and recommended for evaluation).

- HF BitNet 3B: [1bitLLM/bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B).
- Llama3 8B: [HF1BitLLM/Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) and [ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ).
- Falcon 1B: [tiiuae/Falcon3-1B-Instruct-1.58bit](https://huggingface.co/tiiuae/Falcon3-1B-Instruct-1.58bit).

To reproduce full comparison results, you need to download the FP16/BF16 models from Huggingface, and manually convert them for each framework.

> Try <https://hf-mirror.com> if you have proxy issues.

For evaluation of vlut.cpp only, we provide pre-converted models: TODO.

## Installation

### Pre-built binaries

TODO

### Build from source

vlut.cpp has the same building process as [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cpu-build). Please build with `cmake`:

```
cmake -B build
cmake --build build --config Release
```

Recommended options:

- For faster compilation, add the `-j <jobs>` argument, e.g., `cmake --build build --config Release -j 4`.
- For faster repeated compilation (e.g., when searching the optimal configuration), install `ccache`.

**Important notes:**

- We pre-defined several options in [vlut.cmake](../cmake/vlut.cmake), including ISA-specific optimizations and tiling configurations. Refer to [Configuration](#configuration) for details.

### (Optional) Baseline setup

To reproduce full comparison results, install and setup baseline frameworks with their official instructions:

- llama.cpp: See [build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
- bitnet.cpp: See [README.md](https://github.com/microsoft/BitNet/blob/main/README.md#installation).
- T-MAC: See [README.md](https://github.com/microsoft/T-MAC/blob/main/README.md#installation)

Notes:

- Each baseline supports only a subset of our evaluated models, as detailed in the paper.
- We recommend **installing all frameworks to the same workspace folder**, so our evaluation scripts use relative paths to find them correctly.


Your workspace should look like:

```sh
.
├── BitNet
├── T-MAC
├── llama.cpp
└── vlut.cpp
```

## Model Conversion and Quantization

### (Optional) Conversion

If you download [supported models](#Models) in Huggingface format (e.g., safetensors), you'll need to manually convert them to GGUF format before quantization. Skip this step and go to [Quantization](#quantization) if you use pre-converted models.

Setup the Python environment with `conda`, `virtualenv`, or `uv`. Install dependencies:

```
pip install -r requirements.txt
```

Then, convert models:

```sh
# Example using HF BitNet 3B
python ./convert_hf_to_gguf_vlut.py ~/models/bitnet_b1_58-3B --outfile ~/models/bitnet_b1_58-3B/bitnet_b1_58-3B.vlut.gguf
```

**Important notes:**

- Follow this naming format so our automatic scripts work correctly.

### Quantization

This step quantizes the converted GGUF models (still floating-point) to compactly packed format for inference.

After building vlut.cpp, quantize each model:

```sh
# Usage
./build/bin/llama-quantize <gguf> <type>

# Example of packing HF BitNet 3B to I1_V and I2_V
./build/bin/llama-quantize ~/models/bitnet_b1_58-3B/bitnet_b1_58-3B.vlut.gguf I1_V
./build/bin/llama-quantize ~/models/bitnet_b1_58-3B/bitnet_b1_58-3B.vlut.gguf I2_V
# Outputs: ~/models/bitnet_b1_58-3B/ggml-model-I1_V.gguf and ggml-model-I2_V.gguf
```

Besides the basic `I1_V` and `I2_V` packings, we provide more performant `I1_V_k` and `I2_V_k` packings, where `k` denotes the K-tiling configuration. Please refer to [Configuration](#configuration) for details. We recommend to run a simple performance tuning step to determine the optimal packing on your device. 

**Important notes:**

- Use the default output file name (`ggml-model-{quant}.gguf`) so our automatic scripts work correctly.
- Put all model folders in the same directory, and use their HF model names (e.g., `bitnet_b1_58-3B`) as folder names.

Your models directory should look like:

```sh
models
├── Falcon3-1B-Instruct-1.58bit
│   ├── Falcon3-1B-Instruct-1.58bit.vlut.gguf
│   ├── ggml-model-I1_V.gguf
│   ├── ...
├── Llama3-8B-1.58-100B-tokens
│   ├── Llama3-8B-1.58-100B-tokens.vlut.gguf
│   ├── ggml-model-I1_V.gguf
│   ├── ...
└── bitnet_b1_58-3B
    ├── bitnet_b1_58-3B.vlut.gguf
    ├── ggml-model-I1_V.gguf
    └── ...
```

### (Optional) Quantization for baselines

To reproduce full comparison results, follow the model quantization instructions of each baseline frameworks:

- llama.cpp: See [quantize/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).
  - Quantization types: `TQ2_0`, `TQ1_0`.
- bitnet.cpp: See [README.md](https://github.com/microsoft/BitNet/blob/main/README.md#installation).
  - Quantization types: `i2_s`, `tl1`, `tl2`.
- T-MAC: See [README.md](https://github.com/microsoft/T-MAC/blob/main/README.md#installation)
  - Quantization type: `INT_N`.

**Important notes:**

- Both T-MAC and bitnet.cpp need to re-compile kernels when changing models and quantizations. DO NOT quantize all models at once.
- DO NOT mix converted models from different frameworks. There might be compatibility issues.
- Keep their default model names.

After quantizing with all baselines, your model directory should look like:

```sh
models
├── Falcon3-1B-Instruct-1.58bit
├── Llama3-8B-1.58-100B-tokens
└── bitnet_b1_58-3B
    ├── bitnet_b1_58-3B-F16.gguf    # llama.cpp and bitnet.cpp, converted
    ├── bitnet_b1_58-3B.INT_N.gguf  # T-MAC, converted and packed
    ├── bitnet_b1_58-3B.vlut.gguf   # vlut.cpp, converted
    ├── ggml-model-I1_V.gguf        # vlut.cpp, packed
    ├── ggml-model-I1_V_2.gguf      # vlut.cpp, packed with K-tiling (optional)
    ├── ggml-model-I2_V.gguf        # vlut.cpp, packed
    ├── ggml-model-TQ1_0.gguf       # llama.cpp
    ├── ggml-model-TQ2_0.gguf       # llama.cpp
    ├── ggml-model-tl2.gguf         # bitnet.cpp
    └── ...
```

## Main Evaluation

There are 3 experiments in the main evaluation:

- GeMM benchmark (kernel-level): Benchmark the kernel-level latency with specific GeMM shapes.
- Prefilling (end-to-end):  Benchmark the end-to-end prefilling latency (i.e., TTFT).
- Parallel decoding (end-to-end): Benchmark the parallel decoding throughput (i.e., tokens/s). 

### GeMM Benchmark

#### Scripts

We provide the following shell scripts to benchmark GeMM performance (kernel-level):

- [bench-gemm.sh](scripts/bench-gemm.sh): Benchmark vlut.cpp and llama.cpp (based on [tests/test-vlut-gemm.cpp](../tests/test-vlut-gemm.cpp)).
- [bench-gemm-tmac.sh](scripts/bench-gemm-tmac.sh): Benchmark T-MAC (based on T-MAC's kernel tuning logs).
  - Make sure to setup T-MAC's compilation environment first.
  - Modify T-MAC's tuning configuration to n=256 for fair comparison.
  - This will overide previously compiled kernels of T-MAC.

#### Usage

Usage of [bench-gemm.sh](scripts/bench-gemm.sh):

```sh
# Usage
./evaluation/scripts/bench-gemm.sh <device_name> <threads> <ns> [entry_size]

# Example
./evaluation/scripts/bench-gemm.sh pc_intel 1 256 32
```

Explaination of the arguments:

| Argument      | Explaination                                                            |
| ------------- | ----------------------------------------------------------------------- |
| `device_name` | Device identifier for distinguishing test devices                       |
| `threads`     | Number of threads for testing                                           |
| `ns`          | N dimension size of the tested GeMM shape (allows comma-separated list) |
| `entry_size`  | Tile size on the N dimension (optional)                                 |

Notes:

- We use ns=256 in the paper.
- See the script for default values.

Usage of [bench-gemm-tmac.sh](scripts/bench-gemm-tmac.sh):

```sh
# Usage
./evaluation/scripts/bench-gemm-tmac.sh [--device DEVICE_NAME] [--tmac_path TMAC_PATH]
# Example
./evaluation/scripts/bench-gemm-tmac.sh --device pc_intel --tmac_path ../T-MAC
```

Explaination of the arguments:

| Argument    | Explaination                                      |
| ----------- | ------------------------------------------------- |
| `device`    | Device identifier for distinguishing test devices |
| `tmac_path` | Root directory of T-MAC                           |

### E2E Prefilling

#### Scripts

Use [bench-e2e-prefill.sh](scripts/bench-e2e-prefill.sh) to benchmark all frameworks (including vlut.cpp and baselines) on a specific model.

- Depends on [bench-prefill.sh](scripts/bench-e2e-prefill.sh), which uses `llama-bench` to evaluate each framework.

#### Usage

This script accepts environmental variables as arguments.

```sh
# Example of benchmarking with Falcon 1B
DEVICE_NAME=custom_device MODEL_DIR=~/models/Falcon3-1B-Instruct-1.58bit ./evaluation/scripts/bench-e2e-prefill.sh
```

Notes:

- Run Llama3 8B 1.58bit (for vlut.cpp, llama.cpp, and bitnet.cpp) and 2bit (for T-MAC) separately.
- Make sure the models are named correctly as in [Quantization](#optional-quantization-for-baselines).
- Make sure to re-compile T-MAC and bitnet.cpp for each model. They will overide the previous kernels.
- Make backups because the script will remove the target results directory at initialization.

More environment variables, and their default values:

```sh
# Configuration variables that can be easily changed
DEVICE_NAME="${DEVICE_NAME:-"mydevice"}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR/../../..}" # scripts -> evaluation -> vlut.cpp -> workspace
MODEL_DIR="${MODEL_DIR:-$HOME/models/bitnet_b1_58-3B}"
# Extract model name from model dir to separate results folder
MODEL_NAME=$(basename "$MODEL_DIR")
RESULTS_DIR="${RESULTS_DIR:-"${WORKSPACE_DIR}/vlut.cpp/evaluation/results_e2e_${DEVICE_NAME}/${MODEL_NAME}"}"
PROMPT_LENGTH="${PROMPT_LENGTH:-128,256,512}"
THREAD_COUNT="${THREAD_COUNT:-1,4,8}" # use 1, 2 on snapdragon 8 elite
REPEAT_COUNT="${REPEAT_COUNT:-3}"
```

## E2E Batched Decoding

#### Scripts

Use [bench-e2e-batch.sh](scripts/bench-e2e-batch.sh) to benchmark all frameworks (including vlut.cpp and baselines) on a specific model.

- Depends on [bench-batch-decode.sh](scripts/bench-batch-decode.sh), which uses `llama-batched bench` to evaluate each framework.

#### Usage

This script accepts environmental variables as arguments. The usage is similar to [bench-e2e-prefill.sh](scripts/bench-e2e-prefill.sh)

```sh
# Example of benchmarking with Falcon 1B
DEVICE_NAME=custom_device MODEL_DIR=~/models/Falcon3-1B-Instruct-1.58bit ./evaluation/scripts/bench-e2e-batch.sh
```

Notes:

- T-MAC doesn't build `llama-batched-bench` by default. You can manually build it in `T-MAC/3rdparty/llama.cpp` after each compilation, or simply add it to T-MAC's building targets (modify [this line](https://github.com/microsoft/T-MAC/blob/7042f8f73330bd083bc1e4bc5ccb3f88a4904aee/tools/run_pipeline.py#L218)).
- Make sure to read the notes for [prefilling](#usage-1). The usage is quite similar.

More environment variables, and their default values:

```sh
# Configuration variables that can be easily changed
DEVICE_NAME="${DEVICE_NAME:-"mydevice"}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$SCRIPT_DIR/../../..}" # scripts -> evaluation -> vlut.cpp -> workspace
MODEL_DIR="${MODEL_DIR:-$HOME/models/bitnet_b1_58-3B}"
# Extract model name from model dir to separate results folder
MODEL_NAME=$(basename "$MODEL_DIR")
RESULTS_DIR="${RESULTS_DIR:-"${WORKSPACE_DIR}/vlut.cpp/evaluation/results_e2e_batch_${DEVICE_NAME}/${MODEL_NAME}"}"
PREFILL_LEN="${PREFILL_LEN:-16}"
TOKEN_GEN_LENS="${TOKEN_GEN_LENS:-16}"
PARALLEL_SEQS="${PARALLEL_SEQS:-64,128,256}"
THREAD_COUNT="${THREAD_COUNT:-4}" # use 2 on snapdragon 8 elite
```

## Data Visualization

We provide Python scripts (`evaluation/scripts/plot/*.py`) to automatically plot the evaluation results. To customize:

- Modify device and type maps in [plot_utils](scripts/plot/plot_utils.py).
- Modify `combinations_to_plot`, `all_archs`, and `MULTI_THREAD_CONFIG` in [plot_gemm_combined.py](scripts/plot/plot_gemm_combined.py).
- Modify `all_archs`, `models_to_plot`, and `MULTI_THREAD_CONFIG` in [plot_e2e_prefill_combined.py](scripts/plot/plot_e2e_prefill_combined.py) and [plot_e2e_batch_combined.py](scripts/plot/plot_e2e_batch_combined.py).

Put raw results in the evaluation folder, then run corresponding plotting scripts. The evaluation folder should look like:

```sh
evaluation
├── Evaluation.md             # this file
├── figures                   # generated figures
├── reports_e2e_batch         # generated reports
├── reports_e2e_prefill
├── reports_gemm
├── results_e2e_batch_xxx     # raw results to load, marked with the device name
├── results_e2e_prefill_yyy
├── results_gemm_zzz
├── results_gemm_tmac_zzz
└── scripts                   # containing scripts
```

## Configuration

It is crucial to select the correct configuration to achieve the best performance of vlut.cpp.

## Tiling Parameters

### Introduction

As described in the paper, we conduct N-tiling and K-tiling of the LUT. They correspond to the following tunable parameters in vlut.cpp:

| Parameter | Description | Benefits |
|-----------|-------------|---------|
| `entry_size` | N-tiling size (number of LUT colomns) | SIMD addition parallelism |
| `I2_V_k` | K-tiling size for `I2` packing | Register reuse during lookup |
| `I1_V_k` | K-tiling size for `I1` packing | Register reuse during lookup |

Notes:

- N-tiling configuration is determined when [compiling vlut.cpp's binaries](#build-from-source). Adjust it by setting `TABLE_ENTRY_SIZE` when configuring the project. For example, `cmake -B build-test -DTABLE_ENTRY_SIZE=64` sets N-tiling size to 64.
- K-tiling configuration is determined when [quantizing models](#quantization) since it impacts the packed weights layout. Adjust it by setting different quantization types. For example, `I1_V_2` sets K-tiling size to 2 based on `I1_V` packing.

### Tuning guide

Based on empirical results, we recommend starting from the fault settings below:

- `entry_size = 32`
- `I2_V_4` or `I2_V_8`  
- `I1_V_2`

We also provide [search-config.sh](scripts/search-config.sh) to run automatic search. It enumerates all configurations within the search space, and profiles their GeMM performance with [test-vlut-gemm.cpp](tests/test-vlut-gemm.cpp).

Usage:

```bash
./evaluation/scripts/search-config.sh <search_mode>
```
`search_mode` determines which packing method to explore:
- Mode 1: `I1` variants
- Mode 2: `I2` variants

A full sweep typically completes in a few minutes and outputs aggregated scores into:

```
evaluation/results_search/scores<search_mode>.csv
```

After this, manually set `TABLE_ENTRY_SIZE` and quantization types to apply the changes.

### ISA-specific Options

We provide a few ISA-specific options that you might enable, if the compiler (e.g., `g++` or `clang++`) doesn't automatically apply ISA-specific SIMD optimizations. They are defined in [vlut.cmake](../cmake/vlut.cmake).

For example, on the AWS Graviton 3 which supports ARM SVE, configure the project with:

```sh
cmake -B build -DVLUT_SVE=1
```

## Limitations

TODO
