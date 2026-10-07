"""LLM-as-judge scorers: groundedness, hallucination, answer relevance.

Each judge returns a float in [0, 1]. Prompts are deliberately strict —
binary-style judgments are far more consistent than nuanced rubrics.
"""

import os
from openai import OpenAI

_client = None


def _client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _score(prompt: str, model: str = "gpt-4o-mini") -> float:
    resp = _client().chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Respond with a single number between 0 and 1. Nothing else."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=10,
    )
    try:
        return max(0.0, min(1.0, float(resp.choices[0].message.content.strip())))
    except (ValueError, AttributeError):
        return 0.5


GROUNDEDNESS_PROMPT = """Rate how well the ANSWER is supported by the CONTEXT.
1.0 = every claim in the answer appears in the context.
0.0 = the answer makes claims not found in the context.

CONTEXT:
{context}

ANSWER:
{answer}"""

HALLUCINATION_PROMPT = """Does the ANSWER contain any factual claim that is NOT
supported by the CONTEXT below? 1.0 = yes, contains unsupported claims.
0.0 = no, everything is supported.

CONTEXT:
{context}

ANSWER:
{answer}"""

RELEVANCE_PROMPT = """How well does the ANSWER address the QUESTION?
1.0 = directly and completely answers it. 0.0 = irrelevant.

QUESTION:
{question}

ANSWER:
{answer}"""


def groundedness(context: str, answer: str) -> float:
    return _score(GROUNDEDNESS_PROMPT.format(context=context, answer=answer))


def hallucination(context: str, answer: str) -> float:
    return _score(HALLUCINATION_PROMPT.format(context=context, answer=answer))


def relevance(question: str, answer: str) -> float:
    return _score(RELEVANCE_PROMPT.format(question=question, answer=answer))
