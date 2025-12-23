import argparse
import json
import subprocess
import time
from pathlib import Path

import requests


SERVER_SLOTS = 1

def wait_for_server(url, timeout=60):
    """Waits for the llama-server to become responsive."""
    print(f"Waiting for server at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url + "/health", timeout=1)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    print("Timeout waiting for server to start.")
    return False

def run_api_completion(index, prompt, n_predict, expected, api_url):
    """
    Sends a POST request. Returns a dict with results and timing.
    Accepts 'index' and 'expected' to keep track of which question this is.
    """
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": True,
    }

    start_q = time.time()
    prompt_tokens = -1
    try:
        response = requests.post(api_url + "/completion", json=payload)
        response.raise_for_status()
        json_resp = response.json()
        # print(f"JSON Response: {json.dumps(json_resp, indent=2)}")
        # Change 2: Added .strip() to remove leading/trailing whitespace
        result_text = json_resp.get("content", "").strip()
        timings = json_resp.get("timings", {})
        prompt_tokens = timings.get("prompt_n", -1)
    except requests.exceptions.RequestException as e:
        result_text = f"[Error: {e}]"
    
    end_q = time.time()
    
    return {
        "index": index,
        "prompt": prompt,
        "result": result_text,
        "expected": expected,
        "elapsed": end_q - start_q,
        "prompt_tokens": prompt_tokens,
    }

def process_sequential(dataset, api_url):
    print("\n--- Starting SEQUENTIAL Processing ---")
    correct_count = 0
    
    for i, data in enumerate(dataset):
        prompt = data.get("prompt", "")
        expected = data.get("expected", "").strip() # Ensure expected is also clean
        n_predict = data.get("n_predict", 5)

        # Change 1: Print the prompt instead of "Sending request..."
        print(f"\n[Test Case {i+1}] Prompt: \"{prompt}\"")
        
        # This blocks until the request finishes
        res = run_api_completion(i + 1, prompt, n_predict, expected, api_url)
        
        print(f"Result: \"{res['result']}\" | Expected: \"{res['expected']}\"")
        print(f"[Latency: {res['elapsed']:.2f}s] [Prompt Tokens: {res['prompt_tokens']}]")

        # Check accuracy
        if res['result'] == expected:
            correct_count += 1
    return correct_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Llama.cpp Server-Client Completion Test",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="demo/completion.jsonl",
        help="Path to JSONL file with prompts and expected outputs.",
    )
    parser.add_argument(
        "--llama-server-bin",
        type=str,
        default="./build/bin/llama-server",
        help="Path to llama-server binary.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="~/models/Llama3-8B-1.58-100B-tokens/ggml-model-I1_V_2.gguf",
        help="Path to model GGUF file.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host where llama-server will listen.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port where llama-server will listen.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_start = time.time()
    
    data_path = Path(args.data_path)
    llama_server_bin = args.llama_server_bin
    model_path = str(Path(args.model_path).expanduser())
    host = args.host
    port = args.port
    api_base_url = f"http://{host}:{port}"

    print("=== Llama.cpp Server-Client Completion Test ===")
    print(f"Model: {model_path}")
    print("------------------------------------------------------")

    if not data_path.exists():
        print(f"Error: Data file {data_path} not found.")
        return

    # Load Data
    dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    # 1. Start the Server Process
    server_cmd = [
        llama_server_bin,
        "-m", model_path,
        "-c", "64",
        "--host", host,
        "--port", str(port),
        "-np", str(SERVER_SLOTS),
        "-t", "1"
    ]
    
    print(f"Starting llama-server (slots: {SERVER_SLOTS})...")
    server_process = subprocess.Popen(
        server_cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )

    correct_answers = 0
    
    try:
        if not wait_for_server(api_base_url):
            return
        # 2. Process Data sequentially
        correct_answers = process_sequential(dataset, api_base_url)

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