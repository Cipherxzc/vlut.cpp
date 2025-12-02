# Evaluation Guide

This documentation describes how to build vlut.cpp and reproduce our main evaluation results in the paper. We provide pre-built binaries, pre-converted models, and automatic scripts to ease the evaluation.

If you are looking for a quick start guide, see TODO.

## Preparation

### Devices

vlut.cpp supports ARM and x86 CPU, covering most modern devices.

To conduct the complete evaluation, it is recommended that you have:

- x86_64 or ARMv8 CPUs.
- \>= xx GB RAM.
- \>= yy GB Disk.
- OS: Ubuntu/WSL/Android
- (optional) Python: TODO

Notes:

- Tested devices and configurations are listed in the paper.
- Hardware and software requirements by baselines: TODO.

### Models

Thanks to our flexible sub-2-bit packing method, vlut.cpp supports a rich set of ternary LLMs, including [HF BitNet family](https://huggingface.co/1bitLLM), [Llama family](https://huggingface.co/HF1BitLLM), [Falcon3 family](https://huggingface.co/collections/tiiuae/falcon3), and [TriLM family](https://huggingface.co/SpectraSuite).

Belows are tested models in the paper (verified, and recommended for evaluation).

- HF BitNet 3B: [1bitLLM/bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B).
- Llama3 8B: [HF1BitLLM/Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) and [ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ](https://huggingface.co/ChenMnZ/Llama-3-8b-instruct-EfficientQAT-w2g128-GPTQ).
- Falcon 1B: [tiiuae/Falcon3-1B-Instruct-1.58bit](https://huggingface.co/tiiuae/Falcon3-1B-Instruct-1.58bit).

To reproduce full comparison results, you need to download the FP16/BF16 models from the above Huggingface links (or use `huggingface-cli`), and manually convert them for each framework involved in the evaluation.

> Try <https://hf-mirror.com> if you have proxy issues.

Besides, for evaluation of vlut.cpp, we provide pre-converted ternary models (gguf format) at TODO.

## Installation

### Pre-built binaries

We provide pre-built binaries of vlut.cpp in TODO:release. However, it is **NOT recommended**, because:

- The pre-built binaries are compiled with default configurations, which might be sub-optimal on your device. Building from source allows you to find the best-performing configuration.
- The pre-built binaries doesn't include Python and shell scripts. To use the scripts, you still need to clone this repo.

See building instructions below.

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

- We pre-defined several options in `cmake/vlut.cmake`, including ISA-specific optimizations and tiling configurations. Please check in TODO.

### (Optional) Baseline setup

To reproduce full comparison results, install baseline frameworks with official instructions:

- llama.cpp: See [build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
- bitnet.cpp: See [README.md](https://github.com/microsoft/BitNet/blob/main/README.md#installation).
- T-MAC: See [README.md](https://github.com/microsoft/T-MAC/blob/main/README.md#installation)

Notes:

- Each baseline supports only a subset of our evaluated models, as detailed in the paper.
- We recommend **installing all frameworks to the same workspace folder**, so our evaluation scripts use relative paths to find them correctly.

## Model Conversion and Quantization

### (Optional) Conversion

If you download [supported models](#Models) in Huggingface format, you'll need to manually convert them to GGUF format before quantization.

Setup the Python environment with `conda`, `virtualenv`, or `uv`. Install dependencies:

```
pip install -r requirements.txt
```

Then, convert models:

```
python convert_hf_to_gguf_vlut.py TODO
```

Skip this step if you use pre-converted models on TODO.

### Quantization

This step quantizes the converted GGUF models to vlut.cpp's format for inference.

After installing vlut.cpp, run for each model:

```sh
# Usage
./build/bin/llama-quantize <TODO.gguf> <type>
# Example
./build/bin/llama-quantize ~/models/bitnet_b1_58-3B/bitnet_b1_58-3B.TODO.gguf I2_V
```

Available quantization types:

TODO

To use the automatic evaluation scripts, we strongly recommend:

- Put all model folders in the same father folder, and use the default HF model name (e.g., `bitnet_b1_58-3B`) as the child folder name.
- Use llama.cpp's default model names: `ggml-model-{quant}.gguf`.
- Rename models if they are not in this format.

### (Optional) Setup baselines

To reproduce full comparison results, follow the model quantization instructions of each baseline frameworks:

- llama.cpp: See [quantize/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).
  - Quantization types: TQ2_0, TQ1_0.
- bitnet.cpp: See [README.md](https://github.com/microsoft/BitNet/blob/main/README.md#installation).
  - Quantization types: i2_s, tl1, tl2.
- T-MAC: See [README.md](https://github.com/microsoft/T-MAC/blob/main/README.md#installation)
  - Quantization type: INT_N.

Notes:

- Both T-MAC and bitnet.cpp need to re-compile kernels when changing models and quantizations. You cannot quantize all models at once and evaluate them one by one.
- DO NOT use vlut.cpp's scripts and binaries to quantize llama.cpp's models. There might be compatibility issues.
- 

<!-- - llama.cpp (TQ2_0, TQ1_0): cmake build, fixed model.
- T-MAC (INT_N): build in virtualenv, tvm (dependency) compiled.
    - **Note:** Need to convert model after comiling kernel.
    - **Note:** Need to tune for `n=256` in GeMM benchmark, and `n=1` (by default) in E2E.
- bitnet.cpp (i2_s/tl1/tl2): build in conda env, compiled.
    - **Note:** Need to re-compile kernels (will be overided) when changing models. -->


### Final check

Before evaluation, do a final check:

## Main Evaluation

There are mainly 3 types of evaluation:

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

- Denpends on [bench-prefill.sh](scripts/bench-e2e-prefill.sh), which uses `llama-bench` to evaluate each framework.

#### Usage

This script accepts environmental variables as arguments.

```sh
# Example of benchmarking with Falcon 1B
DEVICE_NAME=custom_device MODEL_DIR=~/models/Falcon3-1B-Instruct-1.58bit ./evaluation/scripts/bench-e2e-prefill.sh
```

Notes:

- Run Llama3 8B 1.58bit (for vlut.cpp, llama.cpp, and bitnet.cpp) and 2bit (for T-MAC) separately.
- Make sure T-MAC's models use their folder name as model name (by default), e.g., `bitnet_b1_58-3B/bitnet_b1_58-3B.INT_N.gguf`.
- Make sure all other models are named as `ggml-model-{quant}.gguf` (by default).
- Make sure to re-compile T-MAC and bitnet.cpp for each model. They will overide the previous kernels.
- Make backups because the script removes the target results directory at initialization.

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

- Denpends on [bench-batch-decode.sh](scripts/bench-batch-decode.sh), which uses `llama-batched bench` to evaluate each framework.

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

Put raw results in the evaluation folder, then run corresponding plotting scripts. The file strcture should look like:

```
Example
```



## Config

Selecting the right configuration is crucial for achieving optimal VLUT-based GeMM performance.

### Tunable Parameters

- **`entry_size`**  
  Defines the LUT table granularity. Larger tiles reduce indexing overhead but may increase cache pressure, while smaller tiles trade computation for better cache locality.

- **`I2_V_k` and `I1_V_K`**  
  Both I1 and I2 families offer variants such as `*_V_2`, `*_V_4`, etc., each encoding a different effective `K_tile`

### Recommended Starting Point

From extensive experiments across Intel/AMD/ARM CPUs, we find the following settings to be strong defaults:

- `entry_size = 32`  
- `I2_V_4` or `I2_V_8`  
- `I1_V_2`

### Running the Auto-Search

If you want to tune the configuration specifically for your device, we provide an automated search utility: `search-config.sh` (with `test-vlut-gemm`)

#### Usage

```bash
./evaluation/scripts/search-config.sh <search_mode>
```
`search_mode` determines which set of VLUT variants will be explored:
- **Search mode 1** — searches I1 variants
- **Search mode 2** — searches I2 variants

A full sweep typically completes in a few minutes and outputs aggregated scores into:
```
evaluation/results_search/scores<search_mode>.csv
```