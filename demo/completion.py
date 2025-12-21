import json
import subprocess
import time
from pathlib import Path

# --- Configuration ---
DATA_PATH = Path("demo/completion.jsonl")
LLAMA_CLI = "./build/bin/llama-cli"
# Ensure this path points to your actual model file
# MODEL_PATH = str(Path("~/models/bitnet_b1_58-3B/ggml-model-I1_V.gguf").expanduser())
# MODEL_PATH = str(Path("~/models/bitnet_b1_58-3B/ggml-model-TQ1_0.gguf").expanduser())
# MODEL_PATH = str(Path("~/models/Llama3-8B-1.58-100B-tokens/ggml-model-TQ1_0.gguf").expanduser())
MODEL_PATH = str(Path("~/models/Llama3-8B-1.58-100B-tokens/ggml-model-I1_V_2.gguf").expanduser())

def run_llama_completion(prompt: str, n_predict: int):
    """
    Calls llama-cli to complete the given prompt with a specific token limit.
    """
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-c", "2048",
        # "--temp", "0.0",
        "--temp", "0.2",
        "-no-cnv",
        "-t", "4",            # Number of threads
        "-n", str(n_predict), # Dynamic token limit
        "-e",                 # Parse escapes
        "--no-display-prompt" 
    ]

    try:
        proc = subprocess.run(  
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return proc.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running llama-cli: {e}")
        return ""

def main():
    total_start = time.time()
    
    print("=== Llama.cpp Text Completion Test (Variable Length) ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Data : {DATA_PATH}")
    print("------------------------------------------------------")

    if not DATA_PATH.exists():
        print(f"Error: Data file {DATA_PATH} not found.")
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        data = json.loads(line)
        prompt_text = data.get("prompt", "")
        expected_ans = data.get("expected", "")
        n_predict = data.get("n_predict", 5) 

        print(f"\n[Test Case {i+1}]")
        print(f"Prompt: \"{prompt_text}\"")
        # print(f"Max Tokens: {n_predict}") 
        
        # Start timing for this question
        start_q = time.time()
        
        generated_output = run_llama_completion(prompt_text, n_predict)
        
        # End timing for this question
        end_q = time.time()
        elapsed_q = end_q - start_q
        
        # Combined output line with timing
        print(f"Result: \"{generated_output}\" | Expected: \"{expected_ans}\"")
        print(f"[Time: {elapsed_q:.2f}s]")

    total_end = time.time()
    total_elapsed = total_end - total_start
    print("------------------------------------------------------")
    print(f"Total Program Time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()