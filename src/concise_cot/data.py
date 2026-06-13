"""Data records and prompt formatting for budget-conditioned SFT."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

BUDGETS = ("L0", "L1", "L2", "L3")


@dataclass(frozen=True)
class TeacherTrace:
    question: str
    gold: str
    long_cot: str
    answer: str
    problem_index: int | None = None
    sample_index: int | None = None
    token_count: int | None = None
    source: str | None = None
    raw_completion: str | None = None


@dataclass(frozen=True)
class BudgetedTrace:
    question: str
    gold: str
    budget: str
    reasoning: str
    answer: str
    problem_index: int | None = None
    sample_index: int | None = None
    original_token_count: int | None = None
    target_token_count: int | None = None
    compressed_token_count: int | None = None
    source: str | None = None


def make_prompt(question: str, budget: str) -> str:
    """Format the user prompt with the explicit reasoning budget token."""

    if budget not in BUDGETS:
        raise ValueError(f"unknown budget {budget!r}; expected one of {BUDGETS}")
    return f"Solve the problem. Reasoning budget: <budget={budget}>\nProblem: {question}"


def make_completion(reasoning: str, answer: str) -> str:
    """Format the assistant completion used for supervised fine-tuning."""

    return f"<think>{reasoning.strip()}</think>\n\\boxed{{{answer.strip()}}}"


def make_sft_text(row: BudgetedTrace) -> str:
    """Create a chat-style text sample for TRL SFTTrainer."""

    return (
        "<|user|>\n"
        f"{make_prompt(row.question, row.budget)}\n"
        "<|assistant|>\n"
        f"{make_completion(row.reasoning, row.answer)}"
    )


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def to_sft_rows(rows: Iterable[BudgetedTrace]) -> Iterable[dict]:
    for row in rows:
        yield {"text": make_sft_text(row), **asdict(row)}


def load_budgeted_traces(path: Path) -> list[BudgetedTrace]:
    return [BudgetedTrace(**row) for row in read_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert budgeted traces into SFT JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_budgeted_traces(args.input)
    count = write_jsonl(args.output, to_sft_rows(rows))
    print(f"wrote={count}")


if __name__ == "__main__":
    main()
