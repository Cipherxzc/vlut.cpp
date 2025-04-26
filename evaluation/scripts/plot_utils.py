
ARCH_MAP = {
    'aws_arm': 'ARM Neoverse-V1 (SVE)',
    # 'aws2': 'Intel Xeon Platinum 8375C (AVX512)',
    'pc_intel': 'Intel Core i7-13700k (AVX2)',
    'laptop_amd': 'AMD Ryzen 7 5800H (AVX2)',
    # 'laptop1': 'Intel Core Ultra 7 258V (AVX2)',
    'smartphone': 'Qualcomm Snapdragon 8 Elite (NEON)',
}

GEMM_TYPE_MAP = {
    'i2_s': 'Ours 2-bit',
    'i1_58_m': 'Ours 1.58-bit',
    'tmac': 'T-MAC 2-bit',
    'q4_0': 'llama.cpp 4-bit',
    'tq1_0': 'llama.cpp 1.58-bit',
    'tq2_0': 'llama.cpp 2-bit',
}

# Create a mapping dictionary where keys are characters and values are their positions
GEMM_TYPE_ORDER_MAP = {
    GEMM_TYPE_MAP['i2_s']: 0,
    GEMM_TYPE_MAP['i1_58_m']: 1,
    GEMM_TYPE_MAP['tmac']: 2,
    GEMM_TYPE_MAP['q4_0']: 3,
    GEMM_TYPE_MAP['tq1_0']: 4,
    GEMM_TYPE_MAP['tq2_0']: 5,
}

E2E_TYPE_MAP = {
    'I2_S': 'Ours 2-bit',
    'I1_M': 'Ours 1.58-bit',
    'INT_N': 'T-MAC 2-bit',
    'Q4_0': 'llama.cpp 4-bit',
    'TQ2_0': 'llama.cpp 2-bit',
    'TQ1_0': 'llama.cpp 1.58-bit', 
    'i2_s': 'bitnet.cpp 2-bit',
    'tl1': 'bitnet.cpp 2-bit',
    'tl2': 'bitnet.cpp 2-bit',
}

GEMM_MODEL_MAP = {
    'bitnet_3b': 'BitNet 3B',
    'falcon_1b': 'Falcon 1B',
    'trilm_1.5b': 'TriLM 1.5B',
    'llama3_8b': 'LLaMA 3 8B',
}

E2E_MODEL_MAP = {
    'bitnet_b1_58-3B': 'BitNet 3B',
    'Falcon3-1B-Instruct-1.58bit': 'Falcon 1B',
    'Llama3-8B-1.58-100B-tokens': 'Llama3 8B',
}