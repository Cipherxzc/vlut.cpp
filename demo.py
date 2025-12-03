import json
import subprocess
from pathlib import Path

DATA_PATH = Path("./data/longbench_en_longctx_shortans.jsonl")
LLAMA_CLI = "./build/bin/llama-cli"
MODEL_PATH = str(Path("~/models/Llama3-8B-1.58-100B-tokens/ggml-model-I2_V_8.gguf").expanduser())

MAX_EXAMPLES = 5  

prompt_template = """
You are a helpful assistant.

Context:
{context}

Question:
{question}

Answer briefly in a few words:
"""

def build_prompt(ex):
    context = ex["context"]
    question = ex["input"]
    return prompt_template.format(context=context, question=question)

def run_llama(prompt: str):
    """调用 llama-cli 并返回模型输出"""
    proc = subprocess.run(
        [
            LLAMA_CLI,
            "-m", MODEL_PATH,
            "-p", prompt,
            "-c", "8192",
            "--temp", "0.0",
            "-no-cnv",
            "-t", "8",
            "-n", "8",
            "--no-display-prompt"
        ],
        capture_output=True,
        text=True
    )
    return proc.stdout.strip()

def main():
    print("=== Llama.cpp LongBench DEMO (llama-cli) ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Data : {DATA_PATH}")
    print("--------------------------------------------")

    count = 0
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if count >= MAX_EXAMPLES:
                break
            ex = json.loads(line)

            prompt = build_prompt(ex)
            answer = run_llama(prompt)

            print(f"\n### Example {count + 1}")
            print(f"Question: {ex['input']}")
            print(f"GT Answer: {ex['answers']}")
            print(f"Model Output: {answer}")
            print("--------------------------------------------")

            count += 1

if __name__ == "__main__":
    main()