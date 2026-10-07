"""Demo: evaluate two prompt versions, print the comparison."""

import os
from openai import OpenAI

from src.dataset import load_golden
from src.evaluator import Evaluator, compare

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPT_V1 = "Answer the question briefly.\n\nContext: {context}\nQuestion: {question}"
PROMPT_V2 = ("Answer using ONLY the context below. If the answer is not in the "
             "context, say so. Be concise.\n\nContext: {context}\nQuestion: {question}")


def make_generate_fn(template: str):
    def generate(question: str, context: str):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": template.format(context=context, question=question)}],
        )
        usage = resp.usage
        return resp.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens
    return generate


if __name__ == "__main__":
    cases = load_golden("data/golden.jsonl")
    print(f"Loaded {len(cases)} golden cases\n")

    ev = Evaluator(generate_fn=None)  # replaced per-run below
    reports = []
    for name, template in [("prompt-v1", PROMPT_V1), ("prompt-v2", PROMPT_V2)]:
        ev.generate_fn = make_generate_fn(template)
        print(f"Running {name}...")
        reports.append(ev.run(name, cases))

    print("\n" + compare(reports))
