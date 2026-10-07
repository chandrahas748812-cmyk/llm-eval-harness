"""Core eval runner: execute cases, score, compare runs."""

import time
from dataclasses import dataclass, field

from .dataset import GoldenCase
from . import judges, metrics


@dataclass
class CaseResult:
    case_id: str
    answer: str
    groundedness: float = 0.0
    hallucination: float = 0.0
    relevance: float = 0.0
    rouge_l: float = 0.0
    latency_s: float = 0.0
    cost_usd: float = 0.0


@dataclass
class RunReport:
    name: str
    results: list[CaseResult] = field(default_factory=list)

    def summary(self) -> dict:
        r = self.results
        return {
            "n": len(r),
            "groundedness": metrics.summarize([x.groundedness for x in r])["mean"],
            "hallucination": metrics.summarize([x.hallucination for x in r])["mean"],
            "relevance": metrics.summarize([x.relevance for x in r])["mean"],
            "rougeL": metrics.summarize([x.rouge_l for x in r])["mean"],
            "p50_latency_s": metrics.summarize([x.latency_s for x in r])["p50"],
            "total_cost_usd": sum(x.cost_usd for x in r),
        }


class Evaluator:
    def __init__(self, generate_fn, judge_model: str = "gpt-4o-mini", cost_model: str = "gpt-4o-mini"):
        """
        generate_fn: (question, context) -> (answer, prompt_tokens, completion_tokens)
        """
        self.generate_fn = generate_fn
        self.judge_model = judge_model
        self.cost_model = cost_model

    def run(self, name: str, cases: list[GoldenCase], skip_judges: bool = False) -> RunReport:
        report = RunReport(name=name)
        for case in cases:
            t0 = time.perf_counter()
            answer, pt, ct = self.generate_fn(case.question, case.context)
            latency = time.perf_counter() - t0

            result = CaseResult(
                case_id=case.id,
                answer=answer,
                rouge_l=metrics.rouge_l(case.expected, answer),
                latency_s=latency,
                cost_usd=metrics.estimate_cost(self.cost_model, pt, ct),
            )
            if not skip_judges:
                result.groundedness = judges.groundedness(case.context, answer)
                result.hallucination = judges.hallucination(case.context, answer)
                result.relevance = judges.relevance(case.question, answer)
            report.results.append(result)
        return report


def compare(reports: list[RunReport]) -> str:
    """Render a side-by-side comparison table."""
    from tabulate import tabulate

    rows = []
    names = [r.name for r in reports]
    summaries = [r.summary() for r in reports]
    for metric in ["groundedness", "hallucination", "relevance", "rougeL", "p50_latency_s", "total_cost_usd"]:
        row = [metric] + [f"{s[metric]:.3f}" for s in summaries]
        # delta vs first run
        if len(summaries) > 1:
            base = summaries[0][metric]
            deltas = []
            for s in summaries[1:]:
                d = s[metric] - base
                arrow = "✅" if (d > 0) != (metric in ("hallucination", "p50_latency_s", "total_cost_usd")) and abs(d) > 1e-9 else ""
                deltas.append(f"{d:+.3f} {arrow}".strip())
            row.append(" / ".join(deltas) + " vs " + names[0])
        rows.append(row)
    headers = ["metric"] + names + (["delta"] if len(reports) > 1 else [])
    return tabulate(rows, headers=headers, tablefmt="github")
