import json
import subprocess
import time
from pathlib import Path

# --- Configuration ---
OUTPUT_PATH = Path("generated_corpus_v2.jsonl")
LLAMA_CLI = "./build/bin/llama-cli"

# 模型路径
MODEL_PATH = str(Path("~/models/Llama3-8B-1.58-100B-tokens/ggml-model-I1_V_2.gguf").expanduser())

# 生成参数配置
N_PREDICT = 256        # 1. 增大生成长度，方便后续裁剪
TEMPERATURE = 0.2      # 2. 降低温度，增加确定性，减少幻觉
SAMPLES_PER_PROMPT = 3 # 3. 每个Topic生成多次，增加样本量

# --- Seed Prompts (带目标引导) ---
# 4. 增强引导性：
#    prompt: 起始句
#    target: 我们希望模型在后续生成中自然提到的实体（用于验证生成质量）
SEED_PROMPTS = [
    # {
    #     "topic": "Physics", 
    #     "prompt": "The theory of relativity changed our understanding of time and space, and it was famously developed by", 
    #     "target": "Einstein"
    # },
    # {
    #     "topic": "Technology", 
    #     "prompt": "In the world of smartphones, the operating system developed by Google is known as", 
    #     "target": "Android"
    # },
    # {
    #     "topic": "Biology", 
    #     "prompt": "The primary organ responsible for pumping blood throughout the human body is the", 
    #     "target": "heart"
    # },
    # {
    #     "topic": "History", 
    #     "prompt": "The Great Wall is a massive fortification built in ancient times to protect the northern borders of", 
    #     "target": "China"
    # },
    {
        "topic": "Astronomy", 
        "prompt": "The largest planet in our solar system, known for its Great Red Spot, is called", 
        "target": "Jupiter"
    },
    {
        "topic": "Literature", 
        "prompt": "Romeo and Juliet is a tragic play about two young star-crossed lovers, written by the playwright", 
        "target": "Shakespeare"
    },
    {
        "topic": "Chemistry", 
        "prompt": "Water is a chemical substance essential for all known forms of life, and its chemical formula is", 
        "target": "H2O"
    },
    {
        "topic": "Geography", 
        "prompt": "The Amazon Rainforest is the largest tropical rainforest in the world, located primarily in", 
        "target": "Brazil"
    }
]

def run_llama_generation(prompt: str, n_predict: int, temp: float):
    """
    调用 llama-cli 进行文本续写。
    """
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-c", "2048",
        "--temp", str(temp),  # 使用低温度
        "-no-cnv",
        "-t", "8",
        "-n", str(n_predict),
        "-e",
        "--no-display-prompt"
    ]

    try:
        proc = subprocess.run(  
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return proc.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running llama-cli: {e}")
        return ""

def clean_text(full_text):
    """
    简单的文本清洗：
    1. 移除换行
    2. 截取到最后一个完整的句子（通过句号判断）
    """
    text = full_text.replace("\n", " ").strip()
    
    # 如果文本很长，尝试在最后一个句号处截断，保证句子完整性
    last_period = text.rfind(".")
    if last_period != -1 and last_period > len(text) * 0.5:
        # 只有当句号出现在文本后半段时才截断，避免截得太短
        text = text[:last_period+1]
    
    return text

def main():
    total_start = time.time()
    
    print("=== Llama.cpp Guided Corpus Generation ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Config: Temp={TEMPERATURE}, N={N_PREDICT}, Samples={SAMPLES_PER_PROMPT}")
    print("------------------------------------------------------")

    # 准备写入文件
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        
        for i, item in enumerate(SEED_PROMPTS):
            topic = item["topic"]
            prompt_text = item["prompt"]
            target_entity = item["target"]
            
            print(f"\n[Topic: {topic}] Target: '{target_entity}'")
            print(f"Seed: \"{prompt_text}\"...")

            # 每个 prompt 生成多次
            for attempt in range(SAMPLES_PER_PROMPT):
                start_q = time.time()
                
                # 运行生成
                generated_part = run_llama_generation(prompt_text, N_PREDICT, TEMPERATURE)
                
                # 拼接完整文本
                full_raw_text = prompt_text + generated_part
                cleaned_text = clean_text(full_raw_text)
                
                # 检查是否包含目标实体 (不区分大小写)
                hit_target = target_entity.lower() in cleaned_text.lower()
                status_icon = "✅" if hit_target else "⚠️"

                elapsed_q = time.time() - start_q
                
                # 保存数据
                record = {
                    "topic": topic,
                    "seed": prompt_text,
                    "target_expected": target_entity,
                    "hit_target": hit_target,
                    "full_text": cleaned_text,
                    "raw_generated": generated_part
                }
                
                f_out.write(json.dumps(record) + "\n")
                # 强制刷新缓冲区，确保数据立刻写入磁盘
                f_out.flush()
                
                # 打印预览 (截取前100个字符)
                preview = cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text
                print(f"  ({attempt+1}/{SAMPLES_PER_PROMPT}) {status_icon} {preview}")

    total_end = time.time()
    total_elapsed = total_end - total_start
    print("------------------------------------------------------")
    print(f"Generation Complete. Saved to {OUTPUT_PATH}")
    print(f"Total Program Time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()