"""Classical metrics + aggregation: ROUGE-L, latency stats, cost."""

import statistics
from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Approximate pricing (USD per 1K tokens) — update for your models
PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
}


def rouge_l(expected: str, answer: str) -> float:
    return _scorer.score(expected, answer)["rougeL"].fmeasure


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.001, "output": 0.003})
    return (prompt_tokens / 1000) * p["input"] + (completion_tokens / 1000) * p["output"]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
