"""Inspect and recover JSONL datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_jsonl(input_path: Path, output_path: Path | None = None) -> tuple[int, int]:
    """Validate JSONL rows, optionally writing valid rows to a clean output file."""

    valid = 0
    invalid = 0
    writer = output_path.open("w") if output_path else None
    try:
        with input_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    invalid += 1
                    print(
                        f"invalid line={line_number} char={exc.pos} msg={exc.msg} "
                        f"preview={line[:160]!r}"
                    )
                    continue
                valid += 1
                if writer:
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if writer:
            writer.close()

    return valid, invalid


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JSONL and optionally recover valid rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    valid, invalid = inspect_jsonl(args.input, args.output)
    print(f"valid={valid} invalid={invalid}")
    if args.output:
        print(f"output={args.output}")


if __name__ == "__main__":
    main()
