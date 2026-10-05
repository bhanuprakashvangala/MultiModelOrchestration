"""Every value printed in the paper that the reproduction compares against.

The values are Python literals with their legacy types, so f'{x}' renders them exactly as before ('80.0', '103.0',
'-44.5'), and the values the paper prints as text stay strings ('31,019', '18.7'). Keep the types: 103.0 written
as 103 would print '103', which no longer matches the reproduced '103.0'. The values are deliberately never derived
from code or data, and never loaded from a JSON, YAML, TOML or CSV file, where 103.0 can come back as 103.
"""

from collections.abc import Mapping
from typing import Final

# Table 1: runs, successes, failures and success (%) per benchmark.
TABLE1: Final[Mapping[str, tuple[int, int, int, float]]] = {
    "HumanEval": (820, 656, 164, 80.0),
    "GSM8K": (6595, 5924, 671, 89.8),
    "MBPP": (2500, 1736, 764, 69.4),
    "TruthfulQA": (3950, 3167, 783, 80.2),
    "ARC": (5860, 4704, 1156, 80.3),
    "HellaSwag": (50210, 40260, 9950, 80.2),
    "MATH": (25000, 19908, 5092, 79.6),
    "MMLU-Pro": (60160, 42103, 18057, 70.0),
}

PROMPT_COUNT: Final = "31,019"

# Fig. 4: LOW / MEDIUM / HIGH counts and shares (%) per routing method.
FIG4: Final[Mapping[str, tuple[tuple[int, int, int], tuple[float, float, float]]]] = {
    "keyword": ((6961, 22594, 1464), (22.4, 72.8, 4.7)),
    "llm": ((5401, 25264, 354), (17.4, 81.4, 1.1)),
}

# Fig. 5: success (%) for LOW, MEDIUM, HIGH and overall.
FIG5: Final[Mapping[str, tuple[float, float, float, float]]] = {
    "keyword": (100.0, 97.2, 99.6, 98.0),
    "llm": (100.0, 95.1, 95.8, 96.0),
}

# Fig. 6: median latency (s) keyword, median latency (s) LLM prompt, and the LLM-prompt overhead (%).
FIG6: Final[Mapping[str, tuple[float, float, float]]] = {
    "HumanEval": (57.1, 110.5, 93.7),
    "MBPP": (103.0, 110.4, 7.1),
    "GSM8K": (90.6, 108.4, 19.7),
    "MATH": (64.3, 76.5, 19.1),
    "TruthfulQA": (64.5, 95.4, 48.0),
    "ARC": (75.4, 80.2, 6.3),
    "HellaSwag": (45.1, 56.9, 26.1),
    "MMLU-Pro": (44.1, 55.1, 24.8),
}

# Fig. 8: LLM-prompt latency overhead (s) per benchmark, and its averages.
FIG8: Final[Mapping[str, float]] = {
    "HumanEval": 53.5,
    "MBPP": 7.4,
    "GSM8K": 17.9,
    "MATH": 12.3,
    "TruthfulQA": 30.9,
    "ARC": 4.7,
    "HellaSwag": 11.8,
    "MMLU-Pro": 11.0,
}
FIG8_MEAN_ABS_S: Final = "18.7"
FIG8_MEAN_REL_PCT: Final = "30.6"

# Fig. 10: median TTFT (s) keyword and LLM prompt, and the TTFT change (%).
FIG10: Final[Mapping[str, tuple[float, float]]] = {
    "HumanEval": (25.4, 87.2),
    "MBPP": (85.9, 92.5),
    "GSM8K": (64.7, 101.2),
    "MATH": (15.3, 29.0),
    "TruthfulQA": (57.5, 31.9),
    "ARC": (54.5, 35.4),
    "HellaSwag": (38.7, 39.5),
    "MMLU-Pro": (21.6, 32.5),
}
FIG10_CHANGE_PCT: Final[Mapping[str, float]] = {
    "HumanEval": 242.9,
    "MBPP": 7.7,
    "GSM8K": 56.5,
    "MATH": 90.1,
    "TruthfulQA": -44.5,
    "ARC": -35.2,
    "HellaSwag": 1.9,
    "MMLU-Pro": 50.1,
}

# Fig. 11: TTFT percentiles (s), mean over the 8 benchmarks: (name, keyword, LLM prompt).
FIG11: Final = (("P50", 45.5, 56.2), ("P95", 95.4, 111.4), ("P99", 106.7, 117.9))
TTFT_P50_INCREASE_PCT: Final = "23.5"

# Fig. 9: 'keyword / LLM prompt' latencies (s) and the success / speed / P95 / mean scores (0-10).
FIG9_MEDIAN: Final = "48.9 / 65.4"
FIG9_P95: Final = "117.5 / 119.8"
FIG9_MEAN: Final = "55.4 / 65.1"
FIG9_SCORES: Final[Mapping[str, str]] = {"keyword": "8.0 / 7.8 / 5.6 / 6.5", "llm": "6.0 / 3.6 / 5.0 / 3.3"}
