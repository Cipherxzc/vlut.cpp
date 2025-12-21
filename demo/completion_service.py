import json
import subprocess
import time
import requests
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
DATA_PATH = Path("demo/completion copy.jsonl")
# DATA_PATH = Path("demo/completion.jsonl")
LLAMA_SERVER_BIN = "./build/bin/llama-server"

# 48.53s 4.73s 18.69s
MODEL_PATH = str(Path("~/models/Llama3-8B-1.58-100B-tokens/ggml-model-TQ1_0.gguf").expanduser())

# Total Program Time: 21.36s 1.63s 9.74s
# MODEL_PATH = str(Path("~/models/Llama3-8B-1.58-100B-tokens/ggml-model-I1_V_2.gguf").expanduser())

# --- Execution Mode ---
# Options: "sequential" or "parallel"
# MODE = "parallel"
MODE = "sequential"

# Server settings
HOST = "127.0.0.1"
PORT = 8080
API_URL = f"http://{HOST}:{PORT}/completion"
# API_URL = f"http://{HOST}:{PORT}/v1/completions"

# Max parallel slots for the server (needed for parallel mode)
SERVER_MAX_SLOTS = 16 if MODE == "parallel" else 1

def wait_for_server(url, timeout=60):
    """Waits for the llama-server to become responsive."""
    print(f"Waiting for server at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{HOST}:{PORT}/health", timeout=1)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    print("Timeout waiting for server to start.")
    return False

def run_api_completion(index, prompt, n_predict, expected):
    """
    Sends a POST request. Returns a dict with results and timing.
    Accepts 'index' and 'expected' to keep track of which question this is.
    """
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": True
    }

    start_q = time.time()
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        # Change 2: Added .strip() to remove leading/trailing whitespace
        result_text = response.json().get("content", "").strip() 
    except requests.exceptions.RequestException as e:
        result_text = f"[Error: {e}]"
    
    end_q = time.time()
    
    return {
        "index": index,
        "prompt": prompt,
        "result": result_text,
        "expected": expected,
        "elapsed": end_q - start_q
    }

def process_sequential(dataset):
    print("\n--- Starting SEQUENTIAL Processing ---")
    correct_count = 0
    
    for i, data in enumerate(dataset):
        prompt = data.get("prompt", "")
        expected = data.get("expected", "").strip() # Ensure expected is also clean
        n_predict = data.get("n_predict", 5)

        # Change 1: Print the prompt instead of "Sending request..."
        print(f"\n[Test Case {i+1}] Prompt: \"{prompt}\"")
        
        # This blocks until the request finishes
        res = run_api_completion(i+1, prompt, n_predict, expected)
        
        print(f"Result: \"{res['result']}\" | Expected: \"{res['expected']}\"")
        print(f"[Latency: {res['elapsed']:.2f}s]")

        # Check accuracy
        if res['result'] == expected:
            correct_count += 1
            
    return correct_count

def process_parallel(dataset):
    print(f"\n--- Starting PARALLEL Processing (Max Workers: {SERVER_MAX_SLOTS}) ---")
    print("Sending all requests now...")
    
    correct_count = 0
    
    with ThreadPoolExecutor(max_workers=SERVER_MAX_SLOTS) as executor:
        # Submit all tasks
        future_to_case = {}
        for i, data in enumerate(dataset):
            prompt = data.get("prompt", "")
            expected = data.get("expected", "").strip()
            n_predict = data.get("n_predict", 5)
            
            # Submit task to thread pool
            future = executor.submit(run_api_completion, i+1, prompt, n_predict, expected)
            future_to_case[future] = i+1

        # Process results as they complete (out of order)
        for future in as_completed(future_to_case):
            case_id = future_to_case[future]
            try:
                res = future.result()
                print(f"\n[Test Case {res['index']} Finished]")
                print(f"Prompt: \"{res['prompt']}\"")
                print(f"Result: \"{res['result']}\" | Expected: \"{res['expected']}\"")
                print(f"[Latency: {res['elapsed']:.2f}s]")
                
                # Check accuracy
                # Note: res['expected'] was passed in raw, but we stripped expected locally in loop
                # It's safer to strip both here to be sure
                if res['result'] == res['expected'].strip():
                    correct_count += 1
                    
            except Exception as exc:
                print(f"\n[Test Case {case_id}] generated an exception: {exc}")
                
    return correct_count

def main():
    total_start = time.time()
    
    print("=== Llama.cpp Server-Client Completion Test ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Mode : {MODE.upper()}")
    print("------------------------------------------------------")

    if not DATA_PATH.exists():
        print(f"Error: Data file {DATA_PATH} not found.")
        return

    # Load Data
    dataset = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    # 1. Start the Server Process
    server_cmd = [
        LLAMA_SERVER_BIN,
        "-m", MODEL_PATH,
        "-c", "256",
        "--host", HOST,
        "--port", str(PORT),
        "-np", str(SERVER_MAX_SLOTS), # <--- CRITICAL: Enable parallel slots on server
        # "--n-gpu-layers", "99"
        "-t", "1"
    ]
    
    print(f"Starting llama-server (slots: {SERVER_MAX_SLOTS})...")
    server_process = subprocess.Popen(
        server_cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )

    correct_answers = 0
    
    try:
        if not wait_for_server(API_URL):
            return

        # 2. Process Data based on Mode
        if MODE.lower() == "parallel":
            correct_answers = process_parallel(dataset)
        else:
            correct_answers = process_sequential(dataset)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("\nShutting down server...")
        server_process.terminate()
        server_process.wait()
        
        total_end = time.time()
        total_items = len(dataset)
        
        # Change 3: Print Accuracy Result
        accuracy = 0.0
        if total_items > 0:
            accuracy = (correct_answers / total_items) * 100
            
        print("------------------------------------------------------")
        print(f"Total Program Time: {total_end - total_start:.2f}s")
        print(f"Accuracy: {accuracy:.2f}% ({correct_answers}/{total_items})")
        print("------------------------------------------------------")

if __name__ == "__main__":
    main()