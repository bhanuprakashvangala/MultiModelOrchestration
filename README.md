# Efficient Multi-Model Orchestration for Self-Hosted LLMs

Code and experiment traces for **Efficient Multi-Model Orchestration for Self-Hosted Large Language Models**
Bhanu Prakash Vangala, Tanu Malik.
DAI Workshop at AAAI 2026. [arXiv:2512.22402](https://arxiv.org/abs/2512.22402) |
[Code](https://github.com/bhanuprakashvangala/MultiModelOrchestration)

The paper presents Pick and Spin, a framework for serving several self-hosted LLMs on Kubernetes. Pick gives each
prompt a complexity tier (LOW, MEDIUM or HIGH) and sends it to the model for that tier; Spin manages the model
deployments from one Helm umbrella chart. This repository has the routing runner and its traces for 31,019 prompts
from eight benchmarks, each routed once with keyword rules and once with an LLM prompt as the tier classifier, along
with a multi-objective model scorer, the umbrella chart, and a script that rebuilds the paper's tables and figures
from the traces.

## How the runs were made

- **Tier classifiers** (`src/routing/smart_routing.py`). Keyword rules: a prompt that contains a HIGH phrase
  ("prove", "derive", "analyze", ...) is HIGH; otherwise one that contains a LOW phrase ("what is",
  "which of the following", ...) is LOW; everything else is MEDIUM. LLM prompt: `llama3` is asked to reply LOW,
  MEDIUM or HIGH, and any other reply or a failed call gives MEDIUM.
- **Models.** LOW goes to `llama3`, MEDIUM to `qwen3` and HIGH to `deepseek-r1`, called with streaming through the
  OpenAI-compatible API of the National Research Platform's managed LLM service. The runners ran as Kubernetes Jobs,
  one per benchmark and classifier.
- **Metrics.** Success means the streamed request completed without an error or timeout; responses are not scored
  for correctness. TTFT is the time until the first streamed token.
- **Table 1** (`src/baseline/strategy_baseline.py`). For each question id, one request per strategy (balanced,
  quality, speed, cost, baseline) with a fixed short prompt per benchmark, sent to `gemma3`, `llama3-sdsc` or
  `llama3`; success is HTTP 200 within 180 s.

## Layout

```
data/prompts.jsonl.gz              the 31,019 prompts (HumanEval, MBPP, GSM8K, MATH, TruthfulQA, ARC, HellaSwag, MMLU-Pro)
results/traces/                    experiment traces (no model responses)
  routing_keyword.csv.gz           keyword rules: tier, model, latency, TTFT, tokens, success per prompt
  routing_llm.csv.gz               LLM-prompt classifier, same columns
  baseline_strategies.csv.gz       the five-strategy runs behind Table 1
scripts/reproduce.py               rebuilds Table 1 and Figs. 4-11 from results/traces/
src/routing/smart_routing.py       Pick: tier classification and routing (the runner that produced the traces)
src/routing/multi_objective.py     normalized quality / latency / cost score (Eq. 2) with the four operator profiles
src/baseline/strategy_baseline.py  the Table 1 runner
src/matrix/                        earlier model x backend matrix prototype: registry, health checks, on-demand deploy, API
deploy/helm/pick-and-spin-umbrella Helm umbrella chart: 3 models x 3 backends (vLLM, TGI, TensorRT-LLM)
deploy/jobs/                       Kubernetes Job template for the routing runs
```

## Setup

```bash
git clone https://github.com/bhanuprakashvangala/MultiModelOrchestration.git
cd MultiModelOrchestration
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce

### Tables and figures from the traces (no GPU, a few seconds)

```bash
python scripts/reproduce.py
```

This writes `results/table1_baseline.csv`, one CSV per figure (`fig4_*` to `fig11_*`), PNGs in `results/figures/`,
and `results/verification.csv` with each value next to the one printed in the paper. It also re-runs the keyword rules
on the prompts and checks that they give the tiers recorded in the trace. It exits non-zero if any value or tier
differs.

### Live runs

Point the runner at an OpenAI-compatible endpoint that serves a small, a medium and a large model:

```bash
cp .env.example .env              # set LLM_API_BASE, LLM_API_KEY and the served model names
set -a; source .env; set +a
python src/routing/smart_routing.py HumanEval --routing keyword --workers 20
python src/routing/smart_routing.py HumanEval --routing llm --workers 50
```

Add `--limit 50` for a quick check. Output goes to `results/live/<routing>/<benchmark>_<routing>.csv` in the same
format as the traces; `deploy/jobs/smart-routing-job.yaml` runs the same command as a Kubernetes Job.

### Self-hosting the model matrix

`deploy/helm/pick-and-spin-umbrella` has one subchart per model/backend pair for Llama-3, Gemma-3 and Qwen-3 with
vLLM, TGI and TensorRT-LLM (the TensorRT-LLM pairs are disabled by default). With the default values it renders six
Deployments that request 14 GPUs in total.

```bash
helm install pick-and-spin deploy/helm/pick-and-spin-umbrella -n pick-and-spin --create-namespace
```

Before installing, replace the model IDs in `values.yaml` (such as `Qwen/Qwen-3-235B`) with the Hugging Face IDs
you want to serve and pass them to the serving image; the vLLM containers need a `--model` argument, which the
subchart templates do not set. The subcharts have no autoscaler, so each Deployment starts with one replica; scale
idle pairs to zero with `kubectl scale deployment <name> --replicas=0` or with KEDA or Knative.

`src/matrix/` (dependencies in `requirements-matrix.txt`) keeps a registry of model/backend endpoints with health
checks, deploys a pair with Helm when a request needs it, shuts pairs down after 30 idle minutes, and serves a
FastAPI interface. Its registry lists domain models (BioGPT, ChemBERTa, MatSciBERT); edit `_initialize_endpoints`
in `backend_manager.py` for other models, and set `MATRIX_CHART_DIR` to your chart.

## Results

`scripts/reproduce.py` computes every value below from `results/traces/`. Each one equals the value in the paper;
`results/verification.csv` lists all 88 comparisons.

Table 1, five-strategy runs per benchmark:

| Benchmark | Runs | Success | Failures | Success (%) |
|---|---|---|---|---|
| HumanEval | 820 | 656 | 164 | 80.0 |
| GSM8K | 6,595 | 5,924 | 671 | 89.8 |
| MBPP | 2,500 | 1,736 | 764 | 69.4 |
| TruthfulQA | 3,950 | 3,167 | 783 | 80.2 |
| ARC | 5,860 | 4,704 | 1,156 | 80.3 |
| HellaSwag | 50,210 | 40,260 | 9,950 | 80.2 |
| MATH | 25,000 | 19,908 | 5,092 | 79.6 |
| MMLU-Pro | 60,160 | 42,103 | 18,057 | 70.0 |

Routing runs (31,019 prompts each):

| | Keyword rules | LLM prompt |
|---|---|---|
| LOW (Fig. 4) | 6,961 (22.4%) | 5,401 (17.4%) |
| MEDIUM | 22,594 (72.8%) | 25,264 (81.4%) |
| HIGH | 1,464 (4.7%) | 354 (1.1%) |
| Success, LOW / MEDIUM / HIGH (Fig. 5) | 100.0 / 97.2 / 99.6% | 100.0 / 95.1 / 95.8% |
| Success, overall | 98.0% | 96.0% |
| Latency, median / mean / P95 (Fig. 9) | 48.9 / 55.4 / 117.5 s | 65.4 / 65.1 / 119.8 s |
| Fig. 9 scores: success, speed, P95, mean | 8.0, 7.8, 5.6, 6.5 | 6.0, 3.6, 5.0, 3.3 |
| TTFT P50 / P95 / P99, mean over benchmarks (Fig. 11) | 45.5 / 95.4 / 106.7 s | 56.2 / 111.4 / 117.9 s |

Latency statistics are over all prompts; failed requests are logged with latency 0. TTFT statistics use the
prompts with a recorded first token. The Fig. 9 scores map success from 90-100%, median latency from 80-40 s, P95
latency from 140-100 s and mean latency from 75-45 s onto 0-10. Averaged over the eight benchmarks, the P50 TTFT is
23.5% higher with the LLM prompt.

Per benchmark (Figs. 6, 8 and 10):

| Benchmark | Median latency, keyword / LLM prompt (s) | Added latency (s / %) | Median TTFT, keyword / LLM prompt (s) | TTFT change (%) |
|---|---|---|---|---|
| HumanEval | 57.1 / 110.5 | 53.5 / 93.7 | 25.4 / 87.2 | +242.9 |
| MBPP | 103.0 / 110.4 | 7.4 / 7.1 | 85.9 / 92.5 | +7.7 |
| GSM8K | 90.6 / 108.4 | 17.9 / 19.7 | 64.7 / 101.2 | +56.5 |
| MATH | 64.3 / 76.5 | 12.3 / 19.1 | 15.3 / 29.0 | +90.1 |
| TruthfulQA | 64.5 / 95.4 | 30.9 / 48.0 | 57.5 / 31.9 | -44.5 |
| ARC | 75.4 / 80.2 | 4.7 / 6.3 | 54.5 / 35.4 | -35.2 |
| HellaSwag | 45.1 / 56.9 | 11.8 / 26.1 | 38.7 / 39.5 | +1.9 |
| MMLU-Pro | 44.1 / 55.1 | 11.0 / 24.8 | 21.6 / 32.5 | +50.1 |
| Mean over benchmarks | | 18.7 / 30.6 | | |

## Related

[Pick and Spin: Cold-Start-Aware Routing for Self-Hosted LLM Serving](https://github.com/bhanuprakashvangala/PickAndSpin)
(IEEE CLOUD 2026) extends this work to nine models with Thompson Sampling and per-model cold-start tracking.

## Citation

```bibtex
@inproceedings{vangala2026multimodel,
  title         = {Efficient Multi-Model Orchestration for Self-Hosted Large Language Models},
  author        = {Vangala, Bhanu Prakash and Malik, Tanu},
  booktitle     = {DAI Workshop at AAAI 2026},
  year          = {2026},
  eprint        = {2512.22402},
  archivePrefix = {arXiv}
}
```

## License

MIT. The benchmark prompts in `data/` come from the original benchmarks and keep their licenses.
