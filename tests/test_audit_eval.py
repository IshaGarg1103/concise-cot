import json

from concise_cot.audit_eval import load_rows


def test_load_rows_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    row = {
        "problem_index": 0,
        "budget": "L0",
        "question": "q",
        "gold": "#### 1",
        "completion": "\\boxed{1}",
        "generated_tokens": 3,
        "correct": True,
    }
    path.write_text(json.dumps(row) + "\n\n")

    assert load_rows(path) == [row]
