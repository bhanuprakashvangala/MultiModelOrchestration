"""Five-strategy completion runs (Table 1).

    python src/baseline/strategy_baseline.py HumanEval

This script produced results/traces/baseline_strategies.csv.gz; credentials come from the
environment. For every question id of a benchmark it sends one request per strategy
(balanced, quality, speed, cost, baseline):

- the prompt is a fixed short text per benchmark (SAMPLE_QUESTIONS), e.g. "Math problem" for GSM8K;
- the model is MODELS[hash(strategy) % 3]; Python randomizes string hashes per process,
  so the strategy-to-model assignment changes from run to run;
- success means HTTP 200 within 180 s, with max_tokens=150.

Output: results/live/baseline/<benchmark>_baseline.csv with columns
benchmark, qid, strategy, model, latency (ms), success.
"""

import concurrent.futures
import csv
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = sys.argv[1] if len(sys.argv) > 1 else "HumanEval"
BASE_URL = os.environ.get("LLM_API_BASE", "").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MAX_WORKERS = 20

MODELS = ["gemma3", "llama3-sdsc", "llama3"]  # served model names at the time
STRATEGIES = ["balanced", "quality", "speed", "cost", "baseline"]

SAMPLE_QUESTIONS = {
    "HumanEval": [{"id": f"HE_{i}", "q": "Write a Python function"} for i in range(164)],
    "MBPP": [{"id": f"MB_{i}", "q": "Python problem"} for i in range(500)],
    "TruthfulQA": [{"id": f"TQ_{i}", "q": "True or false question"} for i in range(790)],
    "ARC": [{"id": f"ARC_{i}", "q": "Science question"} for i in range(1172)],
    "GSM8K": [{"id": f"GS_{i}", "q": "Math problem"} for i in range(1319)],
    "GPQA": [{"id": f"GP_{i}", "q": "Graduate physics question"} for i in range(1725)],
    "MATH": [{"id": f"MATH_{i}", "q": "Advanced math problem"} for i in range(5000)],
    "HellaSwag": [{"id": f"HS_{i}", "q": "Complete the sentence"} for i in range(10042)],
    "MMLU-Pro": [{"id": f"MMLU_{i}", "q": "Multiple choice question"} for i in range(12032)],
}


def call_api(model, query, strategy, qid):
    start = time.time()
    try:
        r = requests.post(f"{BASE_URL}/chat/completions",
                          headers={"Authorization": f"Bearer {API_KEY}"},
                          json={"model": model, "messages": [{"role": "user", "content": query}],
                                "max_tokens": 150},
                          timeout=180)
        latency = (time.time() - start) * 1000
        if r.status_code == 200:
            return {"benchmark": BENCHMARK, "qid": qid, "strategy": strategy, "model": model,
                    "latency": latency, "success": True}
    except Exception as e:
        print(f"Error {qid}/{strategy}/{model}: {e}")
    return {"benchmark": BENCHMARK, "qid": qid, "strategy": strategy, "model": model,
            "latency": 0, "success": False}


def main():
    if not BASE_URL:
        sys.exit("Set LLM_API_BASE and LLM_API_KEY; see .env.example")
    questions = SAMPLE_QUESTIONS[BENCHMARK]
    tasks = [(MODELS[hash(s) % len(MODELS)], q["q"], s, q["id"]) for q in questions for s in STRATEGIES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = [f.result() for f in concurrent.futures.as_completed(
            [pool.submit(call_api, *t) for t in tasks])]
    out = ROOT / "results" / "live" / "baseline" / f"{BENCHMARK}_baseline.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "qid", "strategy", "model", "latency", "success"])
        w.writeheader()
        w.writerows(results)
    ok = sum(r["success"] for r in results)
    print(f"{BENCHMARK}: {ok}/{len(results)} succeeded -> {out}")


if __name__ == "__main__":
    main()
