"""True-thinking/decorative step scoring helpers for RQ2."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

STEP_SPLIT_RE = re.compile(r"(?:\n\s*\n|(?<=\.)\s+(?=[A-Z0-9])|(?<=\))\s+(?=[A-Z0-9]))")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[+\-*/=<>%$]+")
CALC_RE = re.compile(r"\d|[+\-*/=<>%$]")


@dataclass(frozen=True)
class StepScore:
    index: int
    text: str
    load_bearing: bool
    early_answer: str | None = None


@dataclass(frozen=True)
class RQ2BudgetSummary:
    budget: str
    usable_groups: int
    mean_token_ratio: float
    mean_step_ratio: float
    removed_calc_like_step_rate: float
    removed_prose_like_step_rate: float


def split_reasoning_steps(reasoning: str) -> list[str]:
    """Split reasoning into coarse steps for ablation-style scoring."""

    return [part.strip() for part in STEP_SPLIT_RE.split(reasoning) if part.strip()]


def token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def calc_like(step: str) -> bool:
    return bool(CALC_RE.search(step))


def retained_by_compressed_step(
    l0_step: str,
    compressed_steps: list[str],
    jaccard_threshold: float,
) -> bool:
    """Approximate whether an L0 step survives in a compressed trace."""

    l0_tokens = token_set(l0_step)
    if not l0_tokens:
        return False
    for step in compressed_steps:
        compressed_tokens = token_set(step)
        if not compressed_tokens:
            continue
        if len(l0_tokens & compressed_tokens) / len(l0_tokens | compressed_tokens) >= jaccard_threshold:
            return True
    return False


def first_stable_answer_index(early_answers: list[str | None]) -> int | None:
    """Return the first index after which the same non-empty answer remains stable."""

    for index, answer in enumerate(early_answers):
        if answer is None:
            continue
        if all(later == answer for later in early_answers[index:] if later is not None):
            return index
    return None


def label_steps_by_stability(reasoning: str, early_answers: list[str | None]) -> list[StepScore]:
    """Cheap decorative-step proxy based on when the answer first stabilizes."""

    steps = split_reasoning_steps(reasoning)
    if len(steps) != len(early_answers):
        raise ValueError("early_answers must have one entry per reasoning step")

    stable_index = first_stable_answer_index(early_answers)
    return [
        StepScore(
            index=index,
            text=step,
            load_bearing=stable_index is None or index <= stable_index,
            early_answer=early_answers[index],
        )
        for index, step in enumerate(steps)
    ]


def load_budgeted_groups(path: Path) -> dict[tuple[int | None, int | None], dict[str, dict]]:
    groups: dict[tuple[int | None, int | None], dict[str, dict]] = defaultdict(dict)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            groups[(row.get("problem_index"), row.get("sample_index"))][row["budget"]] = row
    return groups


def summarize_rq2_budget(
    groups: dict[tuple[int | None, int | None], dict[str, dict]],
    budget: str,
    jaccard_threshold: float,
) -> RQ2BudgetSummary:
    usable = 0
    token_ratios = []
    step_ratios = []
    removed_calc_rates = []
    removed_prose_rates = []

    for group in groups.values():
        if "L0" not in group or budget not in group:
            continue
        usable += 1
        l0 = group["L0"]
        compressed = group[budget]
        l0_steps = split_reasoning_steps(l0["reasoning"])
        compressed_steps = split_reasoning_steps(compressed["reasoning"])
        if not l0_steps:
            continue

        calc_steps = [step for step in l0_steps if calc_like(step)]
        prose_steps = [step for step in l0_steps if not calc_like(step)]
        removed_calc = [
            step
            for step in calc_steps
            if not retained_by_compressed_step(step, compressed_steps, jaccard_threshold)
        ]
        removed_prose = [
            step
            for step in prose_steps
            if not retained_by_compressed_step(step, compressed_steps, jaccard_threshold)
        ]

        if calc_steps:
            removed_calc_rates.append(len(removed_calc) / len(calc_steps))
        if prose_steps:
            removed_prose_rates.append(len(removed_prose) / len(prose_steps))
        if l0.get("original_token_count"):
            token_ratios.append(compressed["compressed_token_count"] / l0["original_token_count"])
        step_ratios.append(len(compressed_steps) / len(l0_steps))

    return RQ2BudgetSummary(
        budget=budget,
        usable_groups=usable,
        mean_token_ratio=mean(token_ratios) if token_ratios else 0.0,
        mean_step_ratio=mean(step_ratios) if step_ratios else 0.0,
        removed_calc_like_step_rate=mean(removed_calc_rates) if removed_calc_rates else 0.0,
        removed_prose_like_step_rate=mean(removed_prose_rates) if removed_prose_rates else 0.0,
    )


def write_summary_csv(path: Path, summaries: list[RQ2BudgetSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RQ2BudgetSummary.__dataclass_fields__))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)


def write_validation_sample(
    *,
    path: Path,
    groups: dict[tuple[int | None, int | None], dict[str, dict]],
    budget: str,
    jaccard_threshold: float,
    limit: int,
) -> int:
    """Write a small sample of removed prose/calc steps for manual causal validation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w") as handle:
        for group in groups.values():
            if "L0" not in group or budget not in group:
                continue
            l0 = group["L0"]
            compressed = group[budget]
            l0_steps = split_reasoning_steps(l0["reasoning"])
            compressed_steps = split_reasoning_steps(compressed["reasoning"])
            removed_prose = [
                step
                for step in l0_steps
                if not calc_like(step)
                and not retained_by_compressed_step(step, compressed_steps, jaccard_threshold)
            ]
            removed_calc = [
                step
                for step in l0_steps
                if calc_like(step)
                and not retained_by_compressed_step(step, compressed_steps, jaccard_threshold)
            ]
            if not removed_prose or not removed_calc:
                continue
            payload = {
                "problem_index": l0.get("problem_index"),
                "sample_index": l0.get("sample_index"),
                "budget": budget,
                "question": l0["question"],
                "answer": l0["answer"],
                "removed_prose_step": removed_prose[0],
                "removed_calc_like_step": removed_calc[0],
                "compressed_reasoning": compressed["reasoning"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            if written >= limit:
                break
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Score decorative vs load-bearing CoT steps.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", default=["L1", "L2", "L3"])
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/rq2_structural_summary.csv"))
    parser.add_argument(
        "--validation-sample",
        type=Path,
        default=Path("outputs/rq2_causal_validation_sample.jsonl"),
    )
    parser.add_argument("--validation-budget", choices=["L1", "L2", "L3"], default="L2")
    parser.add_argument("--validation-limit", type=int, default=50)
    parser.add_argument("--jaccard-threshold", type=float, default=0.35)
    args = parser.parse_args()

    groups = load_budgeted_groups(args.input)
    summaries = [
        summarize_rq2_budget(groups, budget, args.jaccard_threshold) for budget in args.budgets
    ]
    write_summary_csv(args.summary_csv, summaries)
    sample_count = write_validation_sample(
        path=args.validation_sample,
        groups=groups,
        budget=args.validation_budget,
        jaccard_threshold=args.jaccard_threshold,
        limit=args.validation_limit,
    )

    print(f"groups={len(groups)}")
    for summary in summaries:
        print(
            summary.budget,
            f"usable={summary.usable_groups}",
            f"token_ratio={summary.mean_token_ratio:.3f}",
            f"step_ratio={summary.mean_step_ratio:.3f}",
            f"removed_calc={summary.removed_calc_like_step_rate:.3f}",
            f"removed_prose={summary.removed_prose_like_step_rate:.3f}",
            f"decorative_first={summary.removed_prose_like_step_rate > summary.removed_calc_like_step_rate}",
        )
    print(f"wrote_summary={args.summary_csv}")
    print(f"wrote_validation_sample={sample_count} path={args.validation_sample}")


if __name__ == "__main__":
    main()
