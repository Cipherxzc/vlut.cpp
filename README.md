# vlut.cpp

**vlut.cpp** is a lightweight extension of [llama.cpp](https://github.com/ggml-org/llama.cpp) that integrates **Vec-LUT**, an efficient lookup-table–based mixed-precision GeMM (mpGeMM) kernel for **ternary (1.58-bit) LLMs**, which is designed for **parallel multi-token inference** on resource-constrained CPUs.

This repository contains the code and artifact evaluation guide for *"Vec-LUT: Accelerate On-Device Inference of Ternary
LLMs with Vector Lookup Table"*

We conduct a comprehensive evaluation across 3 real-world ternary LLMs and 5 devices, in both single and multi-thread.

TODO: [main_experiments_picture]()

Please refer to [Evaluation.md](evaluation/Evaluation.md) for detailed instructions to evaluate FlexNN and reproduce the results.

## Demo

TODO

## Supported Models

vlut.cpp focuses on **ternary (1.58-bit) LLMs** and supports a rich set of models via flexible sub-2-bit packing. Any supported model needs to be converted to **GGUF** and then packed using vlut.cpp quantization types (e.g., `I1_V`, `I2_V`, `I1_V_k`, `I2_V_k`).

**Model families (HF):**

- **HF BitNet family**
  - Example: [`1bitLLM/bitnet_b1_58-3B`](https://huggingface.co/1bitLLM/bitnet_b1_58-3B)
- **Llama family (1.58-bit variants)**
  - Example: [`HF1BitLLM/Llama3-8B-1.58-100B-tokens`](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)
- **Falcon3 family**
  - Example: [`tiiuae/Falcon3-1B-Instruct-1.58bit`](https://huggingface.co/tiiuae/Falcon3-1B-Instruct-1.58bit)
- **TriLM family**
  - Example: [`SpectraSuite/TriLM_3.9B_Unpacked`](https://huggingface.co/SpectraSuite/TriLM_3.9B_Unpacked)

To reproduce full comparison results involving **FP16/BF16 baselines**, please refer to the [Evaluation Guide](evaluation/Evaluation.md).

## Quick Start

This section walks you through the minimum steps required to run a ternary LLM with **vlut.cpp**:

1. **Install and build** vlut.cpp.
1. **Convert** a HuggingFace model into vlut-compatible GGUF.  
2. **Quantize** the model using Vec-LUT packings (I1 / I2).  
3. **Run** inference using `llama-cli` or benchmark with `llama-bench`.

For a more detailed evaluation pipeline (GeMM, prefill, batched decoding, multi-framework comparison), see [Evaluation.md](evaluation/Evaluation.md).

### 1. Installation

vlut.cpp follows **the same build process as llama.cpp** (CPU build), see [how to build](docs/build.md#cpu-build).

Run the following commands to build vlut.cpp with 4 parallel jobs:

```bash
cmake -B build
cmake --build build --config Release -j 4
```

### 2. Convert a HuggingFace model to GGUF

Before quantization, HuggingFace-format models (safetensors) must be converted to **vlut GGUF**.

Install dependencies:

```bash
pip install -r requirements.txt
```

Convert a model (BitNet 3B for example):

```bash
python ./convert_hf_to_gguf_vlut.py ~/models/bitnet_b1_58-3B --outfile ~/models/bitnet_b1_58-3B/bitnet_b1_58-3B.vlut.gguf
```

### 3. Quantize the model with Vec-LUT packings

vlut.cpp provides lossless ternary packings **I1** and **I2**, with optional K-tiling variants (e.g., `I1_V_2`, `I2_V_4`).

Quantize the converted GGUF:

```bash
./build/bin/llama-quantize ~/models/bitnet_b1_58-3B/bitnet_b1_58-3B.vlut.gguf I1_V_2

./build/bin/llama-quantize ~/models/bitnet_b1_58-3B/bitnet_b1_58-3B.vlut.gguf I2_V_8
```

The quantized model will be saved as `ggml-model-{quant_type}.gguf`.

### 4. Run inference

Use `llama-cli` to perform a quick functional check:

```bash
./build/bin/llama-cli -m model.gguf -p "I believe the meaning of life is" -no-cnv
```

### 5. Benchmark performance

`llama-bench` lets you measure the performance of the inference for various parameters.

example:

```bash
./build/bin/llama-bench -m model.gguf -t 4 -p 128 -n 0
```

## Cite

TODO