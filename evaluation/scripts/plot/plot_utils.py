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

# DEVICE_MAP = {
#     'pc_intel': 'Intel PC\n(Intel Core i7-13700k)',
#     'laptop_amd': 'Legion 5 Pro\n(AMD Ryzen 7 5800H)',
#     'smartphone': 'Xiaomi 15\n(Qualcomm Snapdragon 8 Elite)',
#     'orangepi': 'Orange Pi 5 Plus\n(ARM Cortex-A76)',
#     'aws_arm': 'AWS Graviton 3\n(ARM Neoverse-V1)',
#     'laptop_intel': 'ASUS Zenbook Air\n(Intel Laptop)',
# }

DEVICE_MAP = {
    'pc_intel': 'Intel PC (x86)',
    'laptop_amd': 'Legion 5 Pro (x86)',
    'smartphone': 'Xiaomi 15 (ARM)',
    'orangepi': 'Orange Pi 5 Plus (ARM)',
    'aws_arm': 'AWS Graviton 3 (ARM)',
}


ARCH_MAP = {
    'pc_intel': 'Intel Core i7-13700k',
    'laptop_amd': 'AMD Ryzen 7 5800H',
    'smartphone': 'Qualcomm Snapdragon 8 Elite',
    'orangepi': 'RK3588 (ARM Cortex-A76)',
    'aws_arm': 'AWS Graviton 3 (ARM Neoverse-V1)',
    'laptop_intel': 'Intel Core Ultra 7 258V',
}

GEMM_TYPE_VARIANTS = ['i2_s','i2_s_4','i2_s_8','i1_58_m','i1_m_2']
E2E_TYPE_VARIANTS = ['I2_V','I2_V_4','I2_V_8','I1_V','I1_V_2']

GEMM_TYPE_DEVICE_MAP = {
    'aws_arm': ['i2_s','i1_m'],
    'pc_intel': ['i2_s_8','i1_m_2'],
    'laptop_amd': ['i2_s_4','i1_m_2'],
    'laptop_intel': ['i2_s_4','i1_m_2'],
    'smartphone': ['i2_s_4','i1_m_2'],
    'orangepi': ['i2_s_4','i1_m_2'],
}

E2E_TYPE_DEVICE_MAP = {
    'aws_arm': ['I2_V','I1_V'],
    'pc_intel': ['I2_V_8','I1_V_2'],
    'laptop_amd': ['I2_V_4','I1_V_2'],
    'laptop_intel': ['I2_V_4','I1_V_2'],
    'smartphone': ['I2_V_4','I1_V_2'],
    'orangepi': ['I2_V_4','I1_V_2'],
}

GEMM_TYPE_MAP = {
    'i2_s': 'Ours I2 (b2.00)',
    'i2_s_4': 'Ours I2 (b2.00)',
    'i2_s_8': 'Ours I2 (b2.00)',
    'i1_m': 'Ours I1 (b1.60)',
    'i1_m_2': 'Ours I1 (b1.60)',
    'tmac': 'T-MAC INT_N (b2.00)',
    'q4_0': 'llama.cpp Q4_0 (b4.50)',
    'tq1_0': 'llama.cpp TQ1_0 (b1.69)',
    'tq2_0': 'llama.cpp TQ2_0 (b2.06)',
}

# Custom colors and patterns for each implementation type
GEMM_TYPE_STYLES = {
    'Ours I2 (b2.00)': {'color': '#32A178', 'hatch': ''},
    'Ours I1 (b1.60)': {'color': '#3274A1', 'hatch': '//'},
    'T-MAC INT_N (b2.00)': {'color': '#9067A9', 'hatch': ''},
    'llama.cpp Q4_0 (b4.50)': {'color': '#5DA8D4', 'hatch': ''},
    'llama.cpp TQ1_0 (b1.69)': {'color': '#D76B69', 'hatch': '//'},
    'llama.cpp TQ2_0 (b2.06)': {'color': '#CDB236', 'hatch': ''},
}

E2E_TYPE_MAP = {
    'I2_V': 'Ours I2 (b2.00)',
    'I2_V_4': 'Ours I2 (b2.00)',
    'I2_V_8': 'Ours I2 (b2.00)',
    'I1_V': 'Ours I1 (b1.60)',
    'I1_V_2': 'Ours I1 (b1.60)',
    'INT_N': 'T-MAC INT_N (b2.00)',
    'Q4_0': 'llama.cpp Q4_0 (b4.50)',
    'TQ2_0': 'llama.cpp TQ2_0 (b2.06)',
    'TQ1_0': 'llama.cpp TQ1_0 (b1.69)', 
    'i2_s': 'bitnet.cpp I2_V (b2.00)',
    'tl1': 'bitnet.cpp TL1 (b2.00)',
    'tl2': 'bitnet.cpp TL2 (b1.67)',
}

E2E_TYPE_STYLES = {
    'Ours I2 (b2.00)': GEMM_TYPE_STYLES['Ours I2 (b2.00)'],
    'Ours I1 (b1.60)': GEMM_TYPE_STYLES['Ours I1 (b1.60)'],
    'T-MAC INT_N (b2.00)': GEMM_TYPE_STYLES['T-MAC INT_N (b2.00)'],
    'llama.cpp Q4_0 (b4.50)': GEMM_TYPE_STYLES['llama.cpp Q4_0 (b4.50)'],
    'llama.cpp TQ1_0 (b1.69)': GEMM_TYPE_STYLES['llama.cpp TQ1_0 (b1.69)'],
    'llama.cpp TQ2_0 (b2.06)': GEMM_TYPE_STYLES['llama.cpp TQ2_0 (b2.06)'],
    'bitnet.cpp I2_V (b2.00)': {'color': '#E18727', 'hatch': ''},
    'bitnet.cpp TL1 (b2.00)': {'color': '#26828E', 'hatch': ''},
    'bitnet.cpp TL2 (b1.67)': {'color': '#7EB875', 'hatch': '//'},
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