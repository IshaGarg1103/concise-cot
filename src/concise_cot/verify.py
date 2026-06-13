"""Answer extraction and correctness checks for math reasoning traces."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sympy import SympifyError, simplify, sympify

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
HASH_ANSWER_RE = re.compile(r"####\s*(.+)")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


@dataclass(frozen=True)
class VerificationResult:
    prediction: str | None
    gold: str | None
    correct: bool


def normalize_answer(answer: str | None) -> str | None:
    """Normalize common final-answer formatting without changing semantics."""

    if answer is None:
        return None
    normalized = answer.strip()
    normalized = normalized.replace(",", "")
    normalized = normalized.strip("$")
    normalized = re.sub(r"\\(?:left|right)", "", normalized)
    normalized = normalized.strip()
    return normalized or None


def extract_answer(text: str) -> str | None:
    """Extract a final answer from model output or dataset solution text."""

    boxed_matches = BOXED_RE.findall(text)
    if boxed_matches:
        return normalize_answer(boxed_matches[-1])

    hash_matches = HASH_ANSWER_RE.findall(text)
    if hash_matches:
        return normalize_answer(hash_matches[-1])

    numbers = NUMBER_RE.findall(text)
    if numbers:
        return normalize_answer(numbers[-1])

    return None


def answers_equivalent(prediction: str | None, gold: str | None) -> bool:
    """Return whether two extracted answers are equivalent."""

    prediction = normalize_answer(prediction)
    gold = normalize_answer(gold)
    if prediction is None or gold is None:
        return False
    if prediction == gold:
        return True

    try:
        return bool(simplify(sympify(prediction) - sympify(gold)) == 0)
    except (SympifyError, TypeError, ValueError):
        return False


def verify_completion(completion: str, gold_solution: str) -> VerificationResult:
    """Verify a model completion against a gold answer or full gold solution."""

    prediction = extract_answer(completion)
    gold = extract_answer(gold_solution) or normalize_answer(gold_solution)
    return VerificationResult(
        prediction=prediction,
        gold=gold,
        correct=answers_equivalent(prediction, gold),
    )


def filter_correct_jsonl(input_path: Path, output_path: Path) -> int:
    """Filter JSONL rows with ``completion`` and ``gold`` fields to correct rows."""

    kept = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as src, output_path.open("w") as dst:
        for line in src:
            row = json.loads(line)
            result = verify_completion(row["completion"], row["gold"])
            if result.correct:
                row["prediction"] = result.prediction
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter correct math completions from JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    kept = filter_correct_jsonl(args.input, args.output)
    print(f"kept={kept}")


if __name__ == "__main__":
    main()
