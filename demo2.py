import json
import subprocess
from pathlib import Path

DATA_PATH = Path("./data/longbench_en_longctx_shortans.jsonl")
LLAMA_CLI = "./build/bin/llama-cli"
MODEL_PATH = str(Path("~/models/Llama3-8B-1.58-100B-tokens/ggml-model-f32.gguf").expanduser())

MAX_EXAMPLES = 5  

prompt_template = """
You are an exam-taking assistant.

Please answer the following multiple-choice question.

Requirements:
1. Output ONLY the final answer letter (A/B/C/D).
2. Do NOT output any reasoning.
3. Do NOT repeat the question.
4. If unsure, choose the most likely answer.

Question:
{question}

Please respond with only one letter.
Your answer:
"""

def build_prompt(question: str):
    return prompt_template.format(question=question)

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

    
    question = """
    Emily went to the supermarket to buy ingredients for a small dinner.
    She bought tomatoes, pasta, and cheese because she wanted to cook a simple pasta dish.
    However, she realized she forgot to buy the most important item needed to boil the pasta.
    What did she forget?

    A. Water
    B. Plates
    C. Bread
    D. Chocolate
    """
    prompt = build_prompt(question)
    answer = run_llama(prompt)
    print(f"Question: {question}")
    print(f"Model Output: {answer}")



if __name__ == "__main__":
    main()