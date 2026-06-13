"""Create final report artifacts from completed eval and RQ2 summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_eval_summary(path: Path) -> dict[str, dict[str, float]]:
    return json.loads(path.read_text())


def write_pareto_csv(summary: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["budget", "accuracy", "mean_generated_tokens", "count"],
        )
        writer.writeheader()
        for budget in sorted(summary):
            row = summary[budget]
            writer.writerow(
                {
                    "budget": budget,
                    "accuracy": row["accuracy"],
                    "mean_generated_tokens": row["mean_generated_tokens"],
                    "count": row["count"],
                }
            )


def write_pareto_markdown(summary: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Budget | Accuracy | Mean generated tokens | Count |",
        "|---|---:|---:|---:|",
    ]
    for budget in sorted(summary):
        row = summary[budget]
        lines.append(
            f"| {budget} | {row['accuracy']:.4f} | "
            f"{row['mean_generated_tokens']:.1f} | {int(row['count'])} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready result artifacts.")
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--pareto-csv", type=Path, default=Path("outputs/plots/gsm8k_pareto.csv"))
    parser.add_argument(
        "--pareto-markdown",
        type=Path,
        default=Path("outputs/plots/gsm8k_pareto.md"),
    )
    args = parser.parse_args()

    summary = load_eval_summary(args.eval_summary)
    write_pareto_csv(summary, args.pareto_csv)
    write_pareto_markdown(summary, args.pareto_markdown)
    print(f"wrote={args.pareto_csv}")
    print(f"wrote={args.pareto_markdown}")


if __name__ == "__main__":
    main()
