# Evaluation Guide

This guide explains how to build `vlut.cpp` and reproduce the evaluation results reported in the paper.

Complete [Preparation](#preparation) first, then follow one of the three workflows:

| Workflow                                          | Goal                                               | Estimated time     |
| ------------------------------------------------- | -------------------------------------------------- | ------------------ |
| [Quick Functional Check](#quick-functional-check) | Verify the build runs correctly.                    | < 5 min            |
| [Minimal Reproduction](#minimal-reproduction)     | Reproduce vlut.cpp results on one device and model. | < 1 hr             |
| [Full Comparison](#full-comparison)               | Compare against all baselines from the paper.       | > 1 day per device |

Preparation itself takes about 30 minutes (build + model download).

---

## Preparation

### Environment

`vlut.cpp` targets modern x86 and ARM CPUs.

**Minimum requirements:**

| Resource | Inference only                           | With model conversion   |
| -------- | ---------------------------------------- | ----------------------- |
| CPU      | x86_64 with AVX2, or ARMv8 with NEON/SVE | same                    |
| RAM      | 4 GB                                     | 24 GB                   |
| Disk     | 16 GB                                    | 64 GB (full comparison) |

**Software:** Linux, WSL2, or Android (Termux). Python 3.10+.

**For stable benchmarking results:**

- Use only performance cores (P-cores) on heterogeneous CPUs.
- Minimize background processes during benchmarking.
- Set power management to high-performance mode (e.g., battery set to performance mode on laptops, or CPU governor set to `performance` on Linux).

### Models

`vlut.cpp` supports several ternary model families:

- [HF BitNet family](https://huggingface.co/1bitLLM)
- [Llama3 family](https://huggingface.co/HF1BitLLM)
- [Falcon3 family](https://huggingface.co/collections/tiiuae/falcon3)
- [TriLM family](https://huggingface.co/SpectraSuite)

The paper evaluates three models:

| Short name   | HuggingFace repository                                                                              |
| ------------ | --------------------------------------------------------------------------------------------------- |
| HF BitNet 3B | [1bitLLM/bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B)                           |
| Llama3 8B    | [HF1BitLLM/Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) |
| Falcon3 1B   | [tiiuae/Falcon3-1B-Instruct-1.58bit](https://huggingface.co/tiiuae/Falcon3-1B-Instruct-1.58bit)     |

**Pre-converted GGUF models** are hosted in this HuggingFace collection:

> <https://huggingface.co/collections/XXXXyu/vlutcpp>

| Model        | Pre-converted repository                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------------------------- |
| HF BitNet 3B | [XXXXyu/bitnet_b1_58-3B-vlut-gguf](https://huggingface.co/XXXXyu/bitnet_b1_58-3B-vlut-gguf)                         |
| Llama3 8B    | [XXXXyu/Llama3-8B-1.58-100B-tokens-vlut-gguf](https://huggingface.co/XXXXyu/Llama3-8B-1.58-100B-tokens-vlut-gguf)   |
| Falcon3 1B   | [XXXXyu/Falcon3-1B-Instruct-1.58bit-vlut-gguf](https://huggingface.co/XXXXyu/Falcon3-1B-Instruct-1.58bit-vlut-gguf) |

Store models locally using the model's short name as the directory name:

```
~/models/
├── Llama3-8B-1.58-100B-tokens/
│   ├── ggml-model-I1_V_2.gguf
│   └── ggml-model-I2_V_4.gguf
├── bitnet_b1_58-3B/
│   └── ...
└── Falcon3-1B-Instruct-1.58bit/
    └── ...
```

### Build vlut.cpp

```sh
cmake -B build
cmake --build build --config Release -j4    # adjust -j to your CPU core count
```

Optional CMake flags:

| Flag                     | Purpose                                                     |
| ------------------------ | ----------------------------------------------------------- |
| `-DVLUT_SVE=ON`          | Enable SVE intrinsics (e.g., AWS Graviton 3).                |
| `-DTABLE_ENTRY_SIZE=<N>` | Set the N-tile size as described in the paper (default: 32). |

### Set Up Python

```sh
conda create -y -n vlut-cpp python=3.10
conda activate vlut-cpp
python -m pip install huggingface_hub pandas matplotlib
```

### Download Pre-converted Models

Download the pre-converted models. For example, Llama3 8B (~2 GB per `.gguf` file):

```sh
hf download XXXXyu/Llama3-8B-1.58-100B-tokens-vlut-gguf \
  ggml-model-I1_V_2.gguf \
  ggml-model-I2_V_4.gguf \
  --local-dir ~/models/Llama3-8B-1.58-100B-tokens
```

Other models (BitNet 3B, Falcon3 1B) are smaller. Replace the repository ID and `--local-dir` accordingly.

### Convert Models (Optional)

> Skip this section if you use the pre-converted GGUF models above.

#### Step 1 — Convert HuggingFace weights to GGUF

Install conversion dependencies:

```sh
python -m pip install -r requirements.txt
```

Download the original HuggingFace weights (e.g., Llama3 8B):

```sh
hf download HF1BitLLM/Llama3-8B-1.58-100B-tokens \
  --local-dir ~/models/Llama3-8B-1.58-100B-tokens
```

Convert to the vlut.cpp intermediate GGUF format:

```sh
python ./convert_hf_to_gguf_vlut.py \
  ~/models/Llama3-8B-1.58-100B-tokens \
  --outfile ~/models/Llama3-8B-1.58-100B-tokens/Llama3-8B-1.58-100B-tokens.vlut.gguf
```

#### Step 2 — Quantize to Vec-LUT packings

Quantize the intermediate GGUF into one or more Vec-LUT packings:

```sh
./build/bin/llama-quantize \
  ~/models/Llama3-8B-1.58-100B-tokens/Llama3-8B-1.58-100B-tokens.vlut.gguf I1_V_2

./build/bin/llama-quantize \
  ~/models/Llama3-8B-1.58-100B-tokens/Llama3-8B-1.58-100B-tokens.vlut.gguf I2_V_4

./build/bin/llama-quantize \
  ~/models/Llama3-8B-1.58-100B-tokens/Llama3-8B-1.58-100B-tokens.vlut.gguf I2_V_8
```

This produces files in the same directory with default names (`ggml-model-I1_V_2.gguf`, etc.). **Do not rename them** — the evaluation scripts depend on these names.

---

## Quick Functional Check

Run a single batched-decoding pass to verify the build:

```sh
./build/bin/llama-batched \
  -m ~/models/Llama3-8B-1.58-100B-tokens/ggml-model-I1_V_2.gguf \
  -p "I believe" \
  -np 32 \
  -n 16 \
  -t 4 \
  -ngl 0 \
  --temp 0.5 \
  --repeat-penalty 1.5
```

**Expected:** The model loads without errors and the program prints generated text from 32 parallel sequences. This corresponds to `evaluation/demo/run_batched_decode.sh`.

---

## Minimal Reproduction

This section walks through a minimal evaluation of `vlut.cpp` on **one device** and **one model**. Only `vlut.cpp` and pre-converted models are needed — no baseline frameworks. The same wrapper scripts are used in the full comparison workflow; here they skip missing baselines with warnings.

To evaluate additional devices or models, repeat the steps below with different hardware and a different `MODEL_DIR`.

### Overview

The minimal workflow runs three benchmarks:

| #   | Benchmark                                                           | What it measures               | Script                 |
| --- | ------------------------------------------------------------------- | ------------------------------ | ---------------------- |
| 1   | [GeMM](#step-1--gemm-benchmark)                                     | Raw matrix-multiply throughput. | `bench-gemm.sh`        |
| 2   | [End-to-end prefilling](#step-2--end-to-end-prefilling)             | Prompt processing speed.        | `bench-e2e-prefill.sh` |
| 3   | [End-to-end batched decoding](#step-3--end-to-end-batched-decoding) | Parallel token generation.      | `bench-e2e-batch.sh`   |

### Set Variables

Choose a device identifier and point to the model directory. The device identifier is used to distinguish results from different machines when plotting them together.

```sh
export DEVICE_NAME=mydevice
export MODEL_DIR=~/models/Llama3-8B-1.58-100B-tokens
```

### Step 1 — GeMM Benchmark

Measures raw ternary GeMM throughput for the built-in model shapes.

```sh
# Single-threaded
./evaluation/scripts/bench-gemm.sh "$DEVICE_NAME" 1 256

# Multi-threaded (adjust thread count to your CPU)
./evaluation/scripts/bench-gemm.sh "$DEVICE_NAME" 4 256
```

**Arguments:** `<device_name> <threads> <sequence_length> [entry_size]`

- `device_name` — identifier used in output directory names.
- `threads` — number of threads.
- `sequence_length` — the N dimension (sequence length to benchmark).
- `entry_size` — optional, LUT entry size in bytes (default: 32).

**Output:** `evaluation/results_gemm_${DEVICE_NAME}/` containing `.log` and `.csv` files, one pair per model shape.

> **Troubleshooting:** If you encounter an `awk` error, edit `bench-gemm.sh` and replace `$SCRIPT_DIR/test-to-csv.sh` with `$SCRIPT_DIR/test-to-csv-backup.sh`.

### Step 2 — End-to-End Prefilling

Measures prompt processing throughput (tokens/sec) across prompt lengths and thread counts.

```sh
DEVICE_NAME="$DEVICE_NAME" \
MODEL_DIR="$MODEL_DIR" \
./evaluation/scripts/bench-e2e-prefill.sh
```

**Optional environment variables:**

| Variable        | Default       | Description                            |
| --------------- | ------------- | -------------------------------------- |
| `PROMPT_LENGTH` | `128,256,512` | Comma-separated prompt lengths to test. |
| `THREAD_COUNT`  | `1,4`         | Comma-separated thread counts.          |
| `REPEAT_COUNT`  | `3`           | Repetitions per configuration.          |

**Output:** `evaluation/results_e2e_prefill_${DEVICE_NAME}/<model_name>/` with `.txt` log and `.csv` files for each quantization variant found in `MODEL_DIR`.

> **Note:** This script **deletes** the target result directory before writing, so previous results for the same device and model are overwritten.

### Step 3 — End-to-End Batched Decoding

Measures parallel decoding throughput (tokens/sec) across batch sizes and thread counts.

```sh
DEVICE_NAME="$DEVICE_NAME" \
MODEL_DIR="$MODEL_DIR" \
./evaluation/scripts/bench-e2e-batch.sh
```

**Optional environment variables:**

| Variable         | Default      | Description                        |
| ---------------- | ------------ | ---------------------------------- |
| `PREFILL_LEN`    | `16`         | Prompt length before decoding.      |
| `TOKEN_GEN_LENS` | `16`         | Comma-separated generation lengths. |
| `PARALLEL_SEQS`  | `64,128,256` | Comma-separated batch sizes.        |
| `THREAD_COUNT`   | `4`          | Comma-separated thread counts.      |

**Output:** `evaluation/results_e2e_batch_${DEVICE_NAME}/<model_name>/` with `.txt` log and `.csv` files for each quantization variant found in `MODEL_DIR`.

> **Note:** This script also **deletes** the target result directory before writing.

### Verifying Results

After all three steps, you should have the following directory structure:

```
evaluation/
├── results_gemm_mydevice/
│   ├── llama3_8b_t1_ns256_s32.log
│   ├── llama3_8b_t1_ns256_s32.csv
│   ├── llama3_8b_t4_ns256_s32.log
│   └── llama3_8b_t4_ns256_s32.csv
├── results_e2e_prefill_mydevice/
│   └── Llama3-8B-1.58-100B-tokens/
│       ├── ggml-model-I1_V_2_p128-256-512_t1-4_r3_<date>_<time>.txt
│       ├── ggml-model-I1_V_2_p128-256-512_t1-4_r3_<date>_<time>.csv
│       ├── ggml-model-I2_V_4_p128-256-512_t1-4_r3_<date>_<time>.txt
│       └── ggml-model-I2_V_4_p128-256-512_t1-4_r3_<date>_<time>.csv
└── results_e2e_batch_mydevice/
    └── Llama3-8B-1.58-100B-tokens/
        ├── ggml-model-I1_V_2_npp16_ntg16_npl64-128-256_t4_<date>_<time>.txt
        ├── ggml-model-I1_V_2_npp16_ntg16_npl64-128-256_t4_<date>_<time>.csv
        ├── ggml-model-I2_V_4_npp16_ntg16_npl64-128-256_t4_<date>_<time>.txt
        └── ggml-model-I2_V_4_npp16_ntg16_npl64-128-256_t4_<date>_<time>.csv
```

### Repeating for More Devices or Models

To extend the evaluation:

1. **Another device:** Run on different hardware with a new `DEVICE_NAME` (e.g., `laptop_amd`). Results are stored in separate directories per device.
2. **Another model:** Change `MODEL_DIR` to a different model directory (e.g., `~/models/bitnet_b1_58-3B`) and re-run Steps 2–3. The GeMM benchmark (Step 1) uses built-in shapes and does not depend on model files, so it only needs to be run once per device.

### How the Wrapper Scripts Behave

When run in a vlut.cpp-only workspace, the wrapper scripts:

- Run all vlut.cpp quantization variants found in `MODEL_DIR`.
- Skip missing baseline frameworks (`llama.cpp`, `T-MAC`, `bitnet.cpp`) with warnings.
- Skip missing quantization files (e.g., `ggml-model-I2_V_8.gguf`) with warnings.
- Return nonzero exit status only if an attempted benchmark actually fails.

---

## Full Comparison

This workflow uses the **same scripts** as the minimal workflow. The difference is that the baseline repositories and model files are present, so the scripts also benchmark `llama.cpp`, `T-MAC`, and `bitnet.cpp`.

### Workspace Layout

Place all repositories as siblings:

```
workspace/
├── BitNet/          # bitnet.cpp
├── T-MAC/
├── llama.cpp/
└── vlut.cpp/
```

The wrapper scripts assume this layout by default.

### Baseline Setup

Build the baselines following their upstream instructions:

| Baseline   | Build instructions                                                     |
| ---------- | ---------------------------------------------------------------------- |
| llama.cpp  | <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>      |
| bitnet.cpp | <https://github.com/microsoft/BitNet/blob/main/README.md#installation> |
| T-MAC      | <https://github.com/microsoft/T-MAC/blob/main/README.md#installation>  |

Place the baseline model files in `MODEL_DIR` with these **exact** names:

| Framework  | Required model files                                                                   |
| ---------- | -------------------------------------------------------------------------------------- |
| llama.cpp  | `ggml-model-TQ2_0.gguf`, `ggml-model-TQ1_0.gguf`                                       |
| T-MAC      | `<model-name>.INT_N.gguf`                                                              |
| bitnet.cpp | `ggml-model-tl2.gguf`, `ggml-model-tl1.gguf`, `ggml-model-i2_s.gguf` (when applicable) |

**Important:**

- `T-MAC` and `bitnet.cpp` may need re-compilation per model or quantization variant.
- Do not mix converted model files across frameworks.

### Running Benchmarks

The commands are identical to the [Minimal Reproduction](#minimal-reproduction) steps. With baselines present, the same scripts automatically include them:

**GeMM:**

```sh
./evaluation/scripts/bench-gemm.sh "$DEVICE_NAME" 1 256
./evaluation/scripts/bench-gemm.sh "$DEVICE_NAME" 4 256
```

For T-MAC GeMM results specifically:

```sh
./evaluation/scripts/bench-gemm-tmac.sh \
  --device "$DEVICE_NAME" \
  --tmac_path ../T-MAC
```

**End-to-end prefilling:**

```sh
DEVICE_NAME="$DEVICE_NAME" \
MODEL_DIR="$MODEL_DIR" \
./evaluation/scripts/bench-e2e-prefill.sh
```

With baselines present, this also benchmarks: `llama.cpp` (`TQ2_0`, `TQ1_0`), `T-MAC` (`INT_N`), and `bitnet.cpp` (`tl2`, `tl1`, `i2_s`).

**End-to-end batched decoding:**

```sh
DEVICE_NAME="$DEVICE_NAME" \
MODEL_DIR="$MODEL_DIR" \
./evaluation/scripts/bench-e2e-batch.sh
```

> **Note:** `T-MAC` does not always build `llama-batched-bench` by default. If needed, build it manually in `T-MAC/3rdparty/llama.cpp` after each T-MAC rebuild.

---

## Plotting and Reports

The plotting scripts read the raw result directories and produce PDF figures and CSV summary reports.

```sh
python evaluation/scripts/plot/plot_gemm_combined.py --both
python evaluation/scripts/plot/plot_e2e_prefill_combined.py --both
python evaluation/scripts/plot/plot_e2e_batch_combined.py --both
```

Each script accepts `--single-thread`, `--multi-thread`, or `--both` (default).

**Output:**

| Type                  | Directory                         |
| --------------------- | --------------------------------- |
| Figures (PDF)         | `evaluation/figures/`             |
| GeMM reports (CSV)    | `evaluation/reports_gemm/`        |
| Prefill reports (CSV) | `evaluation/reports_e2e_prefill/` |
| Batch reports (CSV)   | `evaluation/reports_e2e_batch/`   |

The device identifier `mydevice` is included as a built-in dummy identifier in the plotting scripts. The paper's canonical device identifiers are: `pc_intel`, `laptop_amd`, `orangepi`, `smartphone`, `aws_arm`.

> **Note:** For the minimal reproduction (without baselines), the CSV speedup reports may be incomplete because they require baseline results for comparison. This is expected. You can still verify the raw result CSV files and the plot figures, which show vlut.cpp results on their own.

---

## Expected Results

Absolute throughput varies with hardware, compiler, and thermal conditions. The **relative trends** should match the paper:

- `vlut.cpp` generally outperforms baselines in GeMM throughput, prefilling speed, and parallel decoding throughput.
- The `I1` quantization remains competitive while using fewer bits; other sub-2-bit baselines degrade significantly relative to 2-bit.
- On AWS Graviton 3, `llama.cpp`'s `Q4_0` fallback for HF BitNet 3B can be relatively strong due to ARM-specific 4-bit kernel optimizations.

---

## Configuration Tuning (Optional)

To find the best performance on a given device, you can tune the `TABLE_ENTRY_SIZE` and quantization type.

`vlut.cpp` exposes two performance-critical tuning dimensions:

| Parameter                                        | Set at               | Description                                         |
| ------------------------------------------------ | -------------------- | --------------------------------------------------- |
| `TABLE_ENTRY_SIZE`                               | CMake configure time | N-tile size as described in the paper (default: 32). |
| Quantization type (`I1_V_2`, `I2_V_4`, `I2_V_8`) | Quantization time    | K-tiling / packed weight layout.                     |

Example — building with a different entry size:

```sh
cmake -B build-entry64 -DTABLE_ENTRY_SIZE=64
cmake --build build-entry64 --config Release -j4
```

### Automated Configuration Search

`evaluation/scripts/search-config.sh` sweeps over entry sizes (16, 32, 64) and thread counts (1, 4, 8) to find the best configuration for your hardware:

```sh
./evaluation/scripts/search-config.sh 1   # search I1 variants
./evaluation/scripts/search-config.sh 2   # search I2 variants
```

**Output:**

- `evaluation/results_search/scores1.csv` or `scores2.csv`.
- Per-thread-count optimal configuration printed to stdout.

---

## Troubleshooting

| Symptom                             | Fix                                                                                                                  |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `awk` errors in `bench-gemm.sh`     | Replace `$SCRIPT_DIR/test-to-csv.sh` with `$SCRIPT_DIR/test-to-csv-backup.sh` in the script.                         |
| Previous results overwritten        | `bench-e2e-prefill.sh` and `bench-e2e-batch.sh` delete target directories before writing; back up results if needed. |
| T-MAC `llama-batched-bench` missing | Build it manually in `T-MAC/3rdparty/llama.cpp`.                                                                     |
| Incomplete CSV speedup reports      | Expected when baselines are missing; verify raw results and figures instead.                                         |
