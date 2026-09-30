"""Pick: complexity-based routing of benchmark prompts to three model tiers.

    python src/routing/smart_routing.py HumanEval --routing keyword --workers 20
    python src/routing/smart_routing.py MMLU-Pro --routing llm --workers 50 --limit 100

Each prompt is classified LOW / MEDIUM / HIGH, then sent to the model for that tier
over an OpenAI-compatible streaming API. Latency, time to first token and token counts
are recorded per prompt in results/live/<routing>/<benchmark>_<routing>.csv, with the
same columns as results/traces/routing_<routing>.csv.gz.

Configuration (environment, see .env.example):
    LLM_API_BASE   OpenAI-compatible base URL, e.g. https://host/v1
    LLM_API_KEY    API key for that endpoint
    MODEL_LOW, MODEL_MEDIUM, MODEL_HIGH, MODEL_CLASSIFIER   served model names
"""

import argparse
import csv
import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "data" / "prompts.jsonl.gz"
BENCHMARKS = ["HumanEval", "MBPP", "GSM8K", "MATH", "TruthfulQA", "ARC", "HellaSwag", "MMLU-Pro"]

# Tier -> served model name. Defaults are the names used in the recorded runs.
MODELS = {
    "llama3-small": os.environ.get("MODEL_LOW", "llama3"),
    "qwen3": os.environ.get("MODEL_MEDIUM", "qwen3"),
    "deepseek": os.environ.get("MODEL_HIGH", "deepseek-r1"),
}
CLASSIFIER_MODEL = os.environ.get("MODEL_CLASSIFIER", MODELS["llama3-small"])

COMPLEXITY_KEYWORDS = {
    "LOW": ["what is", "define", "who is", "when did", "where is",
            "true or false", "yes or no", "which of the following",
            "select", "choose", "pick", "identify"],
    "HIGH": ["explain why", "analyze", "compare and contrast", "evaluate",
             "prove", "derive", "justify", "critique", "design",
             "develop", "synthesize", "create", "formulate"],
}

FIELDS = ["qid", "question", "complexity", "routing_method", "model", "latency_ms", "ttft_ms",
          "generation_time_ms", "tokens_per_second", "prompt_tokens", "completion_tokens",
          "total_tokens", "response", "success", "error"]


def detect_complexity_keyword(query):
    """HIGH keywords are checked first, then LOW; anything else is MEDIUM."""
    q = query.lower()
    if any(k in q for k in COMPLEXITY_KEYWORDS["HIGH"]):
        return "HIGH"
    if any(k in q for k in COMPLEXITY_KEYWORDS["LOW"]):
        return "LOW"
    return "MEDIUM"


def detect_complexity_llm(client, query):
    """Ask a small model for LOW / MEDIUM / HIGH; MEDIUM on any failure."""
    prompt = ("Classify the complexity of this question as LOW, MEDIUM, or HIGH.\n\n"
              "LOW: Simple factual questions, definitions, true/false, multiple choice\n"
              "MEDIUM: Questions requiring some reasoning or multi-step thinking\n"
              "HIGH: Complex analysis, proofs, design problems, advanced reasoning\n\n"
              f"Question: {query[:500]}\n\n"
              "Respond with ONLY one word: LOW, MEDIUM, or HIGH")
    try:
        r = client.chat.completions.create(model=CLASSIFIER_MODEL,
                                           messages=[{"role": "user", "content": prompt}],
                                           max_tokens=10, temperature=0.0, timeout=10)
        label = r.choices[0].message.content.strip().upper()
        return label if label in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"
    except Exception:
        return "MEDIUM"


def route_to_model(complexity):
    return {"LOW": "llama3-small", "MEDIUM": "qwen3"}.get(complexity, "deepseek")


def call_model(client, model_key, query, timeout):
    start = time.time()
    ttft, gen_start, chunks, text, p_tok, c_tok = None, None, 0, "", 0, 0
    try:
        stream = client.chat.completions.create(model=MODELS[model_key],
                                                messages=[{"role": "user", "content": query}],
                                                max_tokens=512, temperature=0.7, stream=True,
                                                timeout=timeout)
        for chunk in stream:
            if getattr(chunk, "usage", None):
                p_tok = chunk.usage.prompt_tokens or 0
                c_tok = chunk.usage.completion_tokens or 0
            if chunk.choices:
                content = chunk.choices[0].delta.content or ""
                if content:
                    if ttft is None:
                        ttft = (time.time() - start) * 1000
                        gen_start = time.time()
                    text += content
                    chunks += 1
        total = (time.time() - start) * 1000
        gen = (time.time() - gen_start) * 1000 if gen_start else 0
        p_tok = p_tok or len(query.split())
        c_tok = c_tok or len(text.split())
        return {"latency_ms": round(total, 2), "ttft_ms": round(ttft or 0, 2),
                "generation_time_ms": round(gen, 2),
                "tokens_per_second": round(chunks / (gen / 1000), 2) if gen > 0 else 0,
                "prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok,
                "response": text[:500], "success": True, "error": None}
    except Exception as e:
        return {"latency_ms": 0, "ttft_ms": 0, "generation_time_ms": 0, "tokens_per_second": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "response": "", "success": False, "error": str(e)}


def load_prompts(benchmark):
    with gzip.open(PROMPTS, "rt", encoding="utf-8") as f:
        return [d for d in map(json.loads, f) if d["benchmark"] == benchmark]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=BENCHMARKS)
    ap.add_argument("--routing", choices=["keyword", "llm"], default="keyword")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from openai import OpenAI
    base, key = os.environ.get("LLM_API_BASE"), os.environ.get("LLM_API_KEY")
    if not base:
        raise SystemExit("Set LLM_API_BASE (and LLM_API_KEY); see .env.example")
    client = OpenAI(base_url=base, api_key=key or "none")

    prompts = load_prompts(args.benchmark)[: args.limit]

    def process(p):
        c = (detect_complexity_keyword(p["question"]) if args.routing == "keyword"
             else detect_complexity_llm(client, p["question"]))
        m = route_to_model(c)
        return {"qid": p["qid"], "question": p["question"][:200], "complexity": c,
                "routing_method": args.routing, "model": m,
                **call_model(client, m, p["question"], args.timeout)}

    out = ROOT / "results" / "live" / args.routing / f"{args.benchmark}_{args.routing}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, p): i for i, p in enumerate(prompts)}
        for n, fut in enumerate(as_completed(futs), 1):
            rows[futs[fut]] = fut.result()
            if n % 200 == 0:
                print(f"{n}/{len(prompts)}")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows[i] for i in sorted(rows))
    ok = sum(r["success"] for r in rows.values())
    print(f"{args.benchmark} [{args.routing}]: {ok}/{len(rows)} succeeded -> {out}")


if __name__ == "__main__":
    main()
