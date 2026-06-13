"""Audit evaluation JSONL outputs for validity and surprising budget behavior."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def print_budget_stats(rows: list[dict], budget: str, max_new_tokens: int) -> None:
    budget_rows = [row for row in rows if row["budget"] == budget]
    toks = [row["generated_tokens"] for row in budget_rows]
    correct = [bool(row["correct"]) for row in budget_rows]
    no_pred = sum(row.get("prediction") is None for row in budget_rows)
    no_box = sum("\\boxed" not in row["completion"] for row in budget_rows)
    hit_cap = sum(row["generated_tokens"] >= max_new_tokens for row in budget_rows)

    print(budget)
    print("  n:", len(budget_rows))
    print("  acc:", sum(correct) / len(correct) if correct else 0.0)
    print("  mean tokens:", mean(toks) if toks else 0.0)
    print("  median tokens:", median(toks) if toks else 0.0)
    print("  min/max tokens:", (min(toks), max(toks)) if toks else (0, 0))
    print("  hit token cap:", hit_cap)
    print("  missing prediction:", no_pred)
    print("  missing boxed:", no_box)


def audit_eval(path: Path, budgets: list[str], max_new_tokens: int, sample_limit: int) -> None:
    rows = load_rows(path)
    keys = {(row["problem_index"], row["budget"]) for row in rows}

    print("file:", path)
    print("total rows:", len(rows))
    print("unique problem/budget pairs:", len(keys))
    print("duplicates:", len(rows) - len(keys))
    print("budgets:", Counter(row["budget"] for row in rows))
    print()

    for budget in budgets:
        print_budget_stats(rows, budget, max_new_tokens)

    by_problem: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_problem[row["problem_index"]][row["budget"]] = row

    l2_right_l0_wrong = [
        problem_rows
        for problem_rows in by_problem.values()
        if problem_rows.get("L2", {}).get("correct")
        and not problem_rows.get("L0", {}).get("correct")
    ]
    l0_right_l2_wrong = [
        problem_rows
        for problem_rows in by_problem.values()
        if problem_rows.get("L0", {}).get("correct")
        and not problem_rows.get("L2", {}).get("correct")
    ]

    print()
    print("L2 correct while L0 wrong:", len(l2_right_l0_wrong))
    print("L0 correct while L2 wrong:", len(l0_right_l2_wrong))

    print()
    print("Sample L0 wrong / L2 right cases:")
    for problem_rows in l2_right_l0_wrong[:sample_limit]:
        l0 = problem_rows["L0"]
        l2 = problem_rows["L2"]
        print("=" * 80)
        print("problem_index:", l0["problem_index"])
        print("question:", l0["question"][:500])
        print("L0 tokens/correct/pred:", l0["generated_tokens"], l0["correct"], l0.get("prediction"))
        print("L2 tokens/correct/pred:", l2["generated_tokens"], l2["correct"], l2.get("prediction"))
        print("L0 completion:", l0["completion"][:1000].replace("\n", " "))
        print("L2 completion:", l2["completion"][:1000].replace("\n", " "))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit eval outputs for budget artifacts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", default=["L0", "L1", "L2", "L3"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample-limit", type=int, default=3)
    args = parser.parse_args()

    audit_eval(args.input, args.budgets, args.max_new_tokens, args.sample_limit)


if __name__ == "__main__":
    main()
