"""Regenerate the paper's tables and figures from the released traces.

Reads results/traces/*.csv.gz and data/prompts.jsonl.gz, writes CSV tables, figures and
a paper-vs-reproduced comparison to results/. No GPU or network access needed.

    python scripts/reproduce.py
"""

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "results" / "traces"
OUT = ROOT / "results"
FIGS = OUT / "figures"
sys.path.insert(0, str(ROOT / "src" / "routing"))
from smart_routing import detect_complexity_keyword  # noqa: E402

BENCH = ["HumanEval", "MBPP", "GSM8K", "MATH", "TruthfulQA", "ARC", "HellaSwag", "MMLU-Pro"]
TABLE1_ORDER = ["HumanEval", "GSM8K", "MBPP", "TruthfulQA", "ARC", "HellaSwag", "MATH", "MMLU-Pro"]
LEVELS = ["LOW", "MEDIUM", "HIGH"]
# Two tier classifiers: keyword rules and an LLM prompt (routing_llm trace).
METHODS = {"keyword": "Keyword", "llm": "LLM prompt"}


def read(name):
    with gzip.open(TRACES / name, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    base = read("baseline_strategies.csv.gz")
    runs = {m: read(f"routing_{m}.csv.gz") for m in METHODS}
    for rows in runs.values():
        for r in rows:
            r["success"] = r["success"] == "1"
            for k in ("latency_ms", "ttft_ms"):
                r[k] = float(r[k])

    checks = []

    def chk(claim, paper, value):
        checks.append((claim, paper, value))

    # ---------------- Table 1: five-strategy baseline runs ----------------
    t1 = defaultdict(lambda: [0, 0])
    for r in base:
        t1[r["benchmark"]][0] += 1
        t1[r["benchmark"]][1] += r["success"] == "1"
    paper_t1 = {"HumanEval": (820, 656, 164, 80.0), "GSM8K": (6595, 5924, 671, 89.8),
                "MBPP": (2500, 1736, 764, 69.4), "TruthfulQA": (3950, 3167, 783, 80.2),
                "ARC": (5860, 4704, 1156, 80.3), "HellaSwag": (50210, 40260, 9950, 80.2),
                "MATH": (25000, 19908, 5092, 79.6), "MMLU-Pro": (60160, 42103, 18057, 70.0)}
    rows = []
    for b in TABLE1_ORDER:
        n, ok = t1[b]
        rows.append([b, n, ok, n - ok, f"{100 * ok / n:.1f}"])
        pn, pok, pf, pp = paper_t1[b]
        chk(f"Table 1 {b}: runs / success / failures", f"{pn:,} / {pok:,} / {pf:,}", f"{n:,} / {ok:,} / {n - ok:,}")
        chk(f"Table 1 {b}: success (%)", f"{pp}", f"{100 * ok / n:.1f}")
    write_csv("table1_baseline.csv", ["benchmark", "runs", "success", "failures", "success_pct"], rows)

    # ---------------- Prompts and keyword classifier ----------------
    with gzip.open(ROOT / "data" / "prompts.jsonl.gz", "rt", encoding="utf-8") as f:
        prompts = {d["qid"]: d["question"] for d in map(json.loads, f)}
    chk("Prompts across 8 benchmarks", "31,019", f"{len(prompts):,}")
    kw_same = sum(detect_complexity_keyword(prompts[r["qid"]]) == r["complexity"] for r in runs["keyword"])
    kw_agree = 100 * kw_same / len(runs["keyword"])

    # ---------------- Fig. 4: complexity distribution ----------------
    paper_f4 = {"keyword": ([6961, 22594, 1464], [22.4, 72.8, 4.7]),
                "llm": ([5401, 25264, 354], [17.4, 81.4, 1.1])}
    rows = []
    for m, rs in runs.items():
        cnt = [sum(r["complexity"] == lv for r in rs) for lv in LEVELS]
        for lv, c, pc, pp in zip(LEVELS, cnt, *paper_f4[m]):
            rows.append([METHODS[m], lv, c, f"{100 * c / len(rs):.1f}"])
            chk(f"Fig. 4 {METHODS[m]} {lv}: count", f"{pc:,}", f"{c:,}")
            chk(f"Fig. 4 {METHODS[m]} {lv}: share (%)", f"{pp}", f"{100 * c / len(rs):.1f}")
    write_csv("fig4_complexity_distribution.csv", ["routing", "complexity", "queries", "pct"], rows)

    # ---------------- Fig. 5: success rate by complexity ----------------
    paper_f5 = {"keyword": [100.0, 97.2, 99.6, 98.0], "llm": [100.0, 95.1, 95.8, 96.0]}
    rows, succ = [], {}
    for m, rs in runs.items():
        vals = []
        for lv in LEVELS:
            sub = [r for r in rs if r["complexity"] == lv]
            vals.append(100 * sum(r["success"] for r in sub) / len(sub))
        vals.append(100 * sum(r["success"] for r in rs) / len(rs))
        succ[m] = vals[-1]
        for lv, v, p in zip(LEVELS + ["Overall"], vals, paper_f5[m]):
            rows.append([METHODS[m], lv, f"{v:.1f}"])
            chk(f"Fig. 5 {METHODS[m]} success, {lv} (%)", f"{p}", f"{v:.1f}")
    write_csv("fig5_success_by_complexity.csv", ["routing", "complexity", "success_pct"], rows)

    # ---------------- Fig. 6 / 8: median latency per benchmark ----------------
    def per_bench(m, key, positive_only=False):
        out = {}
        for b in BENCH:
            v = [r[key] for r in runs[m] if r["qid"].startswith(b + "_") and (r[key] > 0 or not positive_only)]
            out[b] = np.array(v) / 1000
        return out

    lat = {m: per_bench(m, "latency_ms") for m in METHODS}
    paper_f6 = {"HumanEval": (57.1, 110.5, 93.7), "MBPP": (103.0, 110.4, 7.1), "GSM8K": (90.6, 108.4, 19.7),
                "MATH": (64.3, 76.5, 19.1), "TruthfulQA": (64.5, 95.4, 48.0), "ARC": (75.4, 80.2, 6.3),
                "HellaSwag": (45.1, 56.9, 26.1), "MMLU-Pro": (44.1, 55.1, 24.8)}
    paper_f8 = {"HumanEval": 53.5, "MBPP": 7.4, "GSM8K": 17.9, "MATH": 12.3, "TruthfulQA": 30.9,
                "ARC": 4.7, "HellaSwag": 11.8, "MMLU-Pro": 11.0}
    rows, abs_over, rel_over = [], [], []
    for b in BENCH:
        k, l = np.median(lat["keyword"][b]), np.median(lat["llm"][b])
        d, pct = l - k, 100 * (l - k) / k
        abs_over.append(d)
        rel_over.append(pct)
        rows.append([b, f"{k:.1f}", f"{l:.1f}", f"{d:.1f}", f"{pct:.1f}"])
        pk, pl, pp = paper_f6[b]
        chk(f"Fig. 6 {b} median latency, keyword / LLM prompt (s)", f"{pk} / {pl}", f"{k:.1f} / {l:.1f}")
        chk(f"Fig. 6/8 {b} LLM-prompt latency overhead (%)", f"{pp}", f"{pct:.1f}")
        chk(f"Fig. 8 {b} LLM-prompt latency overhead (s)", f"{paper_f8[b]}", f"{d:.1f}")
    write_csv("fig6_fig8_median_latency.csv",
              ["benchmark", "keyword_median_s", "llm_prompt_median_s", "overhead_s", "overhead_pct"], rows)
    chk("Fig. 8 average absolute overhead (s)", "18.7", f"{np.mean(abs_over):.1f}")
    chk("Fig. 8 average relative overhead (%)", "30.6", f"{np.mean(rel_over):.1f}")

    # ---------------- Fig. 10 / 11: TTFT ----------------
    ttft = {m: per_bench(m, "ttft_ms", positive_only=True) for m in METHODS}
    paper_f10 = {"HumanEval": (25.4, 87.2), "MBPP": (85.9, 92.5), "GSM8K": (64.7, 101.2), "MATH": (15.3, 29.0),
                 "TruthfulQA": (57.5, 31.9), "ARC": (54.5, 35.4), "HellaSwag": (38.7, 39.5),
                 "MMLU-Pro": (21.6, 32.5)}
    paper_f10_pct = {"HumanEval": 242.9, "MBPP": 7.7, "GSM8K": 56.5, "MATH": 90.1, "TruthfulQA": -44.5,
                     "ARC": -35.2, "HellaSwag": 1.9, "MMLU-Pro": 50.1}
    rows = []
    for b in BENCH:
        k, l = np.median(ttft["keyword"][b]), np.median(ttft["llm"][b])
        rows.append([b, f"{k:.1f}", f"{l:.1f}", f"{100 * (l - k) / k:.1f}"])
        chk(f"Fig. 10 {b} median TTFT, keyword / LLM prompt (s)", "{} / {}".format(*paper_f10[b]), f"{k:.1f} / {l:.1f}")
        chk(f"Fig. 10 {b} TTFT change (%)", f"{paper_f10_pct[b]}", f"{100 * (l - k) / k:.1f}")
    write_csv("fig10_median_ttft.csv", ["benchmark", "keyword_s", "llm_prompt_s", "change_pct"], rows)

    pct = {m: [np.mean([np.percentile(ttft[m][b], p) for b in BENCH]) for p in (50, 95, 99)] for m in METHODS}
    write_csv("fig11_ttft_percentiles.csv", ["routing", "p50_s", "p95_s", "p99_s"],
              [[METHODS[m]] + [f"{v:.1f}" for v in pct[m]] for m in METHODS])
    for i, (name, pk, pl) in enumerate([("P50", 45.5, 56.2), ("P95", 95.4, 111.4), ("P99", 106.7, 117.9)]):
        chk(f"Fig. 11 TTFT {name}, keyword / LLM prompt (s)", f"{pk} / {pl}",
            f"{pct['keyword'][i]:.1f} / {pct['llm'][i]:.1f}")
    chk("Median TTFT increase with LLM-prompt routing (%)", "23.5",
        f"{100 * (pct['llm'][0] - pct['keyword'][0]) / pct['keyword'][0]:.1f}")

    # ---------------- Fig. 9: multi-metric radar (raw values and 0-10 scores) ----------------
    def overall(m):
        v = np.array([r["latency_ms"] for r in runs[m]]) / 1000
        return {"success": succ[m], "median": np.median(v), "p95": np.percentile(v, 95), "mean": v.mean()}

    def scale(x, lo, hi, invert=False):
        s = (hi - x) / (hi - lo) * 10 if invert else (x - lo) / (hi - lo) * 10
        return min(10, max(0, s))

    ov = {m: overall(m) for m in METHODS}
    rows = []
    for m in METHODS:
        o = ov[m]
        scores = [scale(o["success"], 90, 100), scale(o["median"], 40, 80, True),
                  scale(o["p95"], 100, 140, True), scale(o["mean"], 45, 75, True)]
        rows.append([METHODS[m], f"{o['success']:.1f}", f"{o['median']:.1f}", f"{o['p95']:.1f}", f"{o['mean']:.1f}"]
                    + [f"{s:.1f}" for s in scores])
    write_csv("fig9_multi_metric.csv", ["routing", "success_pct", "median_latency_s", "p95_latency_s",
                                        "mean_latency_s", "score_success", "score_speed", "score_p95",
                                        "score_mean"], rows)
    k, l = ov["keyword"], ov["llm"]
    chk("Fig. 9 response speed (median latency), keyword / LLM prompt (s)", "48.9 / 65.4", f"{k['median']:.1f} / {l['median']:.1f}")
    chk("Fig. 9 P95 latency, keyword / LLM prompt (s)", "117.5 / 119.8", f"{k['p95']:.1f} / {l['p95']:.1f}")
    chk("Fig. 9 mean latency, keyword / LLM prompt (s)", "55.4 / 65.1", f"{k['mean']:.1f} / {l['mean']:.1f}")
    chk("Fig. 9 scores success/speed/P95/mean, keyword", "8.0 / 7.8 / 5.6 / 6.5", " / ".join(rows[0][5:]))
    chk("Fig. 9 scores success/speed/P95/mean, LLM prompt", "6.0 / 3.6 / 5.0 / 3.3", " / ".join(rows[1][5:]))

    make_figures(runs, lat, ttft, pct)

    def norm(v):
        return v.replace(",", "").replace(" ", "")

    out = [(c, p, r, "yes" if norm(p) == norm(r) else "NO") for c, p, r in checks]
    write_csv("verification.csv", ["claim", "paper", "reproduced", "match"], out)
    width = max(len(r[0]) for r in out)
    for c, p, r, ok in out:
        print(f"{c:<{width}}  {p:>28}  {r:>28}  {ok}")
    n_ok = sum(r[3] == "yes" for r in out)
    print(f"\n{n_ok}/{len(out)} numbers match the paper.")
    print(f"Keyword classifier re-run on the prompts agrees with the recorded tiers for {kw_agree:.1f}% of prompts.")
    print(f"Wrote tables and figures to {OUT.relative_to(ROOT)}/")
    return 0 if n_ok == len(out) and kw_same == len(runs["keyword"]) else 1


def make_figures(runs, lat, ttft, pct):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    FIGS.mkdir(parents=True, exist_ok=True)
    colors = {"keyword": "#4C72B0", "llm": "#DD8452"}
    labels = {"keyword": "Keyword", "llm": "LLM prompt"}
    x = np.arange(len(BENCH))

    def bars(ax, values, fmt):
        for i, m in enumerate(METHODS):
            b = ax.bar(x + (i - 0.5) * 0.38, values[m], 0.38, label=labels[m], color=colors[m])
            ax.bar_label(b, fmt=fmt, fontsize=7)
        ax.set_xticks(x, BENCH, rotation=30, ha="right")
        ax.legend()

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(METHODS):
        cnt = [sum(r["complexity"] == lv for r in runs[m]) for lv in LEVELS]
        b = ax.bar(np.arange(3) + (i - 0.5) * 0.38, cnt, 0.38, label=labels[m], color=colors[m])
        ax.bar_label(b, fmt="{:,.0f}", fontsize=8)
    ax.set_xticks(range(3), LEVELS)
    ax.set_ylabel("Queries")
    ax.set_title("Fig. 4: complexity distribution (31,019 queries)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_complexity_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(METHODS):
        vals = [100 * np.mean([r["success"] for r in runs[m] if r["complexity"] == lv]) for lv in LEVELS]
        vals.append(100 * np.mean([r["success"] for r in runs[m]]))
        b = ax.bar(np.arange(4) + (i - 0.5) * 0.38, vals, 0.38, label=labels[m], color=colors[m])
        ax.bar_label(b, fmt="%.1f", fontsize=8)
    ax.set_xticks(range(4), LEVELS + ["Overall"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Fig. 5: success rate by complexity")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_success_rate.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars(ax, {m: [np.median(lat[m][b]) for b in BENCH] for m in METHODS}, "%.1f")
    ax.set_ylabel("Median latency (s)")
    ax.set_title("Fig. 6: median latency per benchmark")
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_median_latency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars(ax, {m: [np.median(ttft[m][b]) for b in BENCH] for m in METHODS}, "%.1f")
    ax.set_ylabel("Median TTFT (s)")
    ax.set_title("Fig. 10: median time to first token per benchmark")
    fig.tight_layout()
    fig.savefig(FIGS / "fig10_median_ttft.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(METHODS):
        b = ax.bar(np.arange(3) + (i - 0.5) * 0.38, pct[m], 0.38, label=labels[m], color=colors[m])
        ax.bar_label(b, fmt="%.1f", fontsize=8)
    ax.set_xticks(range(3), ["P50", "P95", "P99"])
    ax.set_ylabel("TTFT (s), mean over 8 benchmarks")
    ax.set_title("Fig. 11: TTFT percentiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig11_ttft_percentiles.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
