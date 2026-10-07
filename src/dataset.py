"""Golden dataset loading and validation."""

import json
from pathlib import Path
from pydantic import BaseModel, field_validator


class GoldenCase(BaseModel):
    id: str
    question: str
    context: str
    expected: str

    @field_validator("question", "context", "expected")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must be non-empty")
        return v.strip()


def load_golden(path: str | Path) -> list[GoldenCase]:
    cases, seen = [], set()
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            case = GoldenCase(**json.loads(line))
            if case.id in seen:
                raise ValueError(f"duplicate case id '{case.id}' at line {lineno}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"no cases found in {path}")
    return cases
