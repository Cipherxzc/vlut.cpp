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

- `bench-gemm.sh`: Ours & llama.cpp GeMM (with `test-vlut-gemm`).
- `bench-gemm-tmac.sh`: T-MAC GeMM (T-MAC is tuned for n=256, will overide previous kernels).
    - Will copy `tmac_model_utils.py` and `tmac_platform.py` to target directory.
        - Increased tuning timeout in `platform.py` to avoid tuning failure.
    - **Note:** Make sure to activate T-MAC's `virtualenv` and `source build/t-mac-envs.sh` first, since we need compilation.

#### Usage

```bash
./evaluation/scripts/bench-gemm.sh -h
Usage: ./evaluation/scripts/bench-gemm.sh <device_name> <threads> <ns> [entry_size]
Example: ./evaluation/scripts/bench-gemm.sh mydevice 4 256 32

./evaluation/scripts/bench-gemm-tmac.sh -h
Unknown argument: -h
Usage: ./evaluation/scripts/bench-gemm-tmac.sh [--device DEVICE_NAME] [--tmac_path TMAC_PATH]
```

#### Example

```bash
./evaluation/scripts/bench-gemm.sh pc_intel 1 256 32
./evaluation/scripts/bench-gemm-tmac.sh pc_intel
```

#### Note
- Device name is for plotting. e.g., pc_intel, laptop_amd, etc.
- Currently only use **ns = 256** for plotting.
- See `.sh` for default values.

## E2E Prefill

#### Scripts

- `bench-e2e-prefill.sh`: evaluate all frameworks on one model, need `MODEL_DIR` as input environment variable.
    - Denpends on `bench-prefill.sh`: use `llama-bench` to evaluate one framework.
    - **Note:** Ensure T-MAC's model uses the folder name as model name (by default). For example: `~/models/bitnet_b1_58-3B/bitnet_b1_58-3B.INT_N.gguf`, so the script can correctly find T-MAC's model.
    - **Note:** Ensure all other models are named `ggml-model-{quant}.gguf` (by default), so the script can correctly find them.
    - **Note:** Re-compile T-MAC and re-convert T-MAC models for n=1 before **each evaluation** (not just for model changes: we compiled for n=256 in GeMM benchmark, and it doesn't directly work for E2E evaluation).
    - **Note:** Re-compile bitnet.cpp **each time before evaluating a new model** since the kernels are always overided by last model's compilation.
    - **Note:** the script might `rm` the results directory in the beginning! Make backups.

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

Use `llama-batched-bench`, similar to prefill.

```
DEVICE_NAME=custom_device MODEL_DIR=~/models/Falcon3-1B-Instruct-1.58bit ./evaluation/scripts/bench-e2e-batch.sh
```

**Note:** T-MAC doesn't build `llama-batched-bench` by default. Manually build in `3rdparty/llama.cpp` after each compilation.

# Appendix

## CPU Specs

Use `lscpu`

AWS ARM:
```
Architecture:             aarch64
  CPU op-mode(s):         32-bit, 64-bit
  Byte Order:             Little Endian
CPU(s):                   8
  On-line CPU(s) list:    0-7
Vendor ID:                ARM
  Model name:             Neoverse-V1
    Model:                1
    Thread(s) per core:   1
    Core(s) per socket:   8
    Socket(s):            1
    Stepping:             r1p1
    BogoMIPS:             2100.00
    Flags:                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs paca pacg dcpodp svei8mm svebf16 i8mm bf16 dgh rng
Caches (sum of all):      
  L1d:                    512 KiB (8 instances)
  L1i:                    512 KiB (8 instances)
  L2:                     8 MiB (8 instances)
  L3:                     32 MiB (1 instance)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-7
Vulnerabilities:          
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Not affected
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; __user pointer sanitization
  Spectre v2:             Mitigation; CSV2, BHB
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```

Orangepi
```
Architecture:           aarch64
  CPU op-mode(s):       32-bit, 64-bit
  Byte Order:           Little Endian
CPU(s):                 8
  On-line CPU(s) list:  0-7
Vendor ID:              ARM
  Model name:           Cortex-A55
    Model:              0
    Thread(s) per core: 1
    Core(s) per socket: 4
    Socket(s):          1
    Stepping:           r2p0
    CPU max MHz:        1800.0000
    CPU min MHz:        408.0000
    BogoMIPS:           48.00
    Flags:              fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
  Model name:           Cortex-A76
    Model:              0
    Thread(s) per core: 1
    Core(s) per socket: 4
    Socket(s):          1
    Stepping:           r4p0
    CPU max MHz:        2352.0000
    CPU min MHz:        408.0000
    BogoMIPS:           48.00
    Flags:              fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
Caches (sum of all):
  L1d:                  384 KiB (8 instances)
  L1i:                  384 KiB (8 instances)
  L2:                   2.5 MiB (8 instances)
  L3:                   3 MiB (1 instance)
Vulnerabilities:
  Itlb multihit:        Not affected
  L1tf:                 Not affected
  Mds:                  Not affected
  Meltdown:             Not affected
  Mmio stale data:      Not affected
  Retbleed:             Not affected
  Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:           Mitigation; __user pointer sanitization
  Spectre v2:           Vulnerable: Unprivileged eBPF enabled
  Srbds:                Not affected
  Tsx async abort:      Not affected
```

Xiaomi 15
192KB L1 128 L1
```
Architecture:             aarch64
  CPU op-mode(s):         64-bit
  Byte Order:             Little Endian
CPU(s):                   8
  On-line CPU(s) list:    0-7
Vendor ID:                Qualcomm
  Model name:             -
    Model:                4
    Thread(s) per core:   1
    Core(s) per socket:   6
    Socket(s):            1
    Stepping:             0x4
    CPU(s) scaling MHz:   39%
    CPU max MHz:          3532.8000
    CPU min MHz:          384.0000
    BogoMIPS:             38.40
    Flags:                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp flagm2 frint i8mm bf16 rng bti ecv afp rpres
  Model name:             -
    Model:                4
    Thread(s) per core:   1
    Core(s) per socket:   2
    Socket(s):            1
    Stepping:             0x3
    CPU(s) scaling MHz:   24%
    CPU max MHz:          4320.0000
    CPU min MHz:          1017.6000
    BogoMIPS:             38.40
    Flags:                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp flagm2 frint i8mm bf16 rng bti ecv afp rpres
Vulnerabilities:
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Not affected
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; __user pointer sanitization
  Spectre v2:             Vulnerable: Unprivileged eBPF enabled
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```

laptop amd
```
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          48 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   16
  On-line CPU(s) list:    0-15
Vendor ID:                AuthenticAMD
  Model name:             AMD Ryzen 7 5800H with Radeon Graphics
    CPU family:           25
    Model:                80
    Thread(s) per core:   2
    Core(s) per socket:   8
    Socket(s):            1
    Stepping:             0
    BogoMIPS:             6387.84
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse s
                          se2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable no
                          nstop_tsc cpuid extd_apicid pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave
                          avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefet
                          ch osvw topoext perfctr_core ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms in
                          vpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves clzero xsaveerptr
                          arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vm
                          save_vmload umip vaes vpclmulqdq rdpid fsrm
Virtualization features:
  Virtualization:         AMD-V
  Hypervisor vendor:      Microsoft
  Virtualization type:    full
Caches (sum of all):
  L1d:                    256 KiB (8 instances)
  L1i:                    256 KiB (8 instances)
  L2:                     4 MiB (8 instances)
  L3:                     16 MiB (1 instance)
Vulnerabilities:
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Mitigation; safe RET
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl and seccomp
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Retpolines; IBPB conditional; IBRS_FW; STIBP always-on; RSB filling; PBRSB-eIBRS N
                          ot affected; BHI Not affected
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```

pc intel
```

Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          39 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   20
  On-line CPU(s) list:    0-19
Vendor ID:                GenuineIntel
  Model name:             12th Gen Intel(R) Core(TM) i7-12700KF
    CPU family:           6
    Model:                151
    Thread(s) per core:   2
    Core(s) per socket:   10
    Socket(s):            1
    Stepping:             2
    BogoMIPS:             7219.20
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse s
                          se2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology tsc_reliable nonst
                          op_tsc cpuid pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadl
                          ine_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd i
                          brs ibpb stibp ibrs_enhanced tpr_shadow vnmi ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 sme
                          p bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves avx_
                          vnni umip waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize flush_l
                          1d arch_capabilities
Virtualization features:
  Virtualization:         VT-x
  Hypervisor vendor:      Microsoft
  Virtualization type:    full
Caches (sum of all):
  L1d:                    480 KiB (10 instances)
  L1i:                    320 KiB (10 instances)
  L2:                     12.5 MiB (10 instances)
  L3:                     25 MiB (1 instance)
Vulnerabilities:
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Vulnerable: No microcode
  Retbleed:               Mitigation; Enhanced IBRS
  Spec rstack overflow:   Not affected
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl and seccomp
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence;
                           BHI SW loop, KVM SW loop
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```