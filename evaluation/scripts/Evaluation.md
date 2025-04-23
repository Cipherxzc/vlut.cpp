# Evaluation

Ours:
- I2_S
- I1_M (fallback to I1_S)

Baselines:
- llama.cpp (TQ1_0, TQ2_0)
- T-MAC (INT_N)
- bitnet.cpp (not sure)

Models:
- bitnet 3b
- llama3 8b
- falcon 1b


## GeMM Benchmark

Scripts:

- `bench-gemm.sh`: Ours & llama.cpp GeMM (with `test-bitnet-gemm.cpp`)
- `bench-gemm-tmac.sh`: T-MAC GeMM (T-MAC is tuned for n=256, will overide previous kernels)
    - Will copy `tmac_model_utils.py` to target directory

Usage:

```sh
./evaluation/scripts/bench-gemm.sh -h
Usage: ./evaluation/scripts/bench-gemm.sh <device_name> <threads> <ns> [lut2(on/off)] [entry_size]
Example: ./evaluation/scripts/bench-gemm.sh mydevice 4 "128,512,1024" on 32

./evaluation/scripts/bench-gemm-tmac.sh -h
Unknown argument: -h
Usage: ./evaluation/scripts/bench-gemm-tmac.sh [--device DEVICE_NAME] [--tmac_path TMAC_PATH]
```

Note:
- Device name is for plotting. e.g., pc_intel, laptop_amd, etc.
- Currently only use **ns = 256** for plotting.
- See `.sh` for default values.

## E2E Prefill

TODO, use `llama-bench`

## E2E Batched Decoding

TODO, use `llama-batched-bench`
