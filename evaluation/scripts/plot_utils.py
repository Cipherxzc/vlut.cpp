# DEVICE_MAP = {
#     'aws_arm': 'AWS EC2 (ARM Server)',
#     'pc_intel': 'Intel PC',
#     'laptop_amd': 'Legion 5 Pro (AMD Laptop)',
#     'laptop_intel': 'ASUS Zenbook Air (Intel Laptop)',
#     'smartphone': 'Xiaomi 15 (Smartphone)',
#     'orangepi': 'Orange Pi 5 Plus (ARM Embedded)',
# }

# DEVICE_MAP = {
#     'aws_arm': 'AWS EC2\n(ARM Server)',
#     'pc_intel': 'Intel PC',
#     'laptop_amd': 'Legion 5 Pro\n(AMD Laptop)',
#     'laptop_intel': 'ASUS Zenbook Air\n(Intel Laptop)',
#     'smartphone': 'Xiaomi 15\n(Smartphone)',
#     'orangepi': 'Orange Pi 5 Plus\n(ARM Embedded)',
# }

DEVICE_MAP = {
    'aws_arm': 'AWS Gravition 3\n(ARM Neoverse-V1)',
    'pc_intel': 'Intel PC\n(Intel Core i7-13700k)',
    'laptop_amd': 'Legion 5 Pro\n(AMD Ryzen 7 5800H)',
    'laptop_intel': 'ASUS Zenbook Air\n(Intel Laptop)',
    'smartphone': 'Xiaomi 15\n(Qualcomm Snapdragon 8 Elite)',
    'orangepi': 'Orange Pi 5 Plus\n(ARM Cortex-A76)',
}


ARCH_MAP = {
    'aws_arm': 'AWS Gravition 3 (ARM Neoverse-V1)',
    'pc_intel': 'Intel Core i7-13700k',
    'laptop_amd': 'AMD Ryzen 7 5800H',
    'laptop_intel': 'Intel Core Ultra 7 258V',
    'smartphone': 'Qualcomm Snapdragon 8 Elite',
    'orangepi': 'RK3588 (ARM Cortex-A76)',
}

GEMM_TYPE_MAP = {
    'i2_s': 'Ours I2 (b2.00)',
    'i1_58_m': 'Ours I1 (b1.60)',
    'tmac': 'T-MAC INT_N (b2.00)',
    'q4_0': 'llama.cpp Q4_0 (b4.50)',
    'tq1_0': 'llama.cpp TQ1_0 (b1.69)',
    'tq2_0': 'llama.cpp TQ2_0 (b2.06)',
}

# Custom colors and patterns for each implementation type
GEMM_TYPE_STYLES = {
    GEMM_TYPE_MAP['i2_s']: {'color': '#32A178', 'hatch': ''},
    GEMM_TYPE_MAP['i1_58_m']: {'color': '#3274A1', 'hatch': '//'},
    GEMM_TYPE_MAP['tmac']: {'color': '#9067A9', 'hatch': ''},
    GEMM_TYPE_MAP['q4_0']: {'color': '#5DA8D4', 'hatch': ''},
    GEMM_TYPE_MAP['tq1_0']: {'color': '#D76B69', 'hatch': '//'},
    GEMM_TYPE_MAP['tq2_0']: {'color': '#CDB236', 'hatch': ''},
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
    'I2_S': 'Ours I2 (b2.00)',
    'I1_M': 'Ours I1 (b1.60)',
    'INT_N': 'T-MAC INT_N (b2.00)',
    'Q4_0': 'llama.cpp Q4_0 (b4.50)',
    'TQ2_0': 'llama.cpp TQ2_0 (b2.06)',
    'TQ1_0': 'llama.cpp TQ1_0 (b1.69)', 
    'i2_s': 'bitnet.cpp I2_S (b2.00)',
    'tl1': 'bitnet.cpp TL1 (b2.00)',
    'tl2': 'bitnet.cpp TL2 (b1.67)',
}

# Custom colors and patterns for each implementation type
# E2E_TYPE_STYLES = {
#     'I2_S': GEMM_TYPE_MAP['i2_s'],
#     'I1_M': GEMM_TYPE_MAP['i1_58_m'],
#     'INT_N': GEMM_TYPE_MAP['tmac'],
#     'Q4_0': GEMM_TYPE_MAP['q4_0'],
#     'TQ1_0': GEMM_TYPE_MAP['tq1_0'],
#     'TQ2_0': GEMM_TYPE_MAP['tq2_0'],
#     'i2_s': {'color': '#E18727', 'hatch': ''},
#     'tl1': {'color': '#26828E', 'hatch': ''},
#     'tl2': {'color': '#7EB875', 'hatch': ''},
# }

E2E_TYPE_STYLES = {
    E2E_TYPE_MAP['I2_S']: GEMM_TYPE_STYLES[GEMM_TYPE_MAP['i2_s']],
    E2E_TYPE_MAP['I1_M']: GEMM_TYPE_STYLES[GEMM_TYPE_MAP['i1_58_m']],
    E2E_TYPE_MAP['INT_N']: GEMM_TYPE_STYLES[GEMM_TYPE_MAP['tmac']],
    E2E_TYPE_MAP['Q4_0']: GEMM_TYPE_STYLES[GEMM_TYPE_MAP['q4_0']],
    E2E_TYPE_MAP['TQ1_0']: GEMM_TYPE_STYLES[GEMM_TYPE_MAP['tq1_0']],
    E2E_TYPE_MAP['TQ2_0']: GEMM_TYPE_STYLES[GEMM_TYPE_MAP['tq2_0']],
    E2E_TYPE_MAP['i2_s']: {'color': '#E18727', 'hatch': ''},
    E2E_TYPE_MAP['tl1']: {'color': '#26828E', 'hatch': ''},
    E2E_TYPE_MAP['tl2']: {'color': '#7EB875', 'hatch': '//'},
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
    'TriLM_3.9B_Unpacked': 'TriLM 3.9B',
}