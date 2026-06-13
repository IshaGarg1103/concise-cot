import json
from pathlib import Path

from concise_cot.config import load_config
from concise_cot.eval import (
    EvalExample,
    build_eval_tasks,
    flatten_summary,
    load_completed_examples,
    make_eval_prompt,
    summarize_examples,
)


def test_summarize_examples_uses_precomputed_correctness() -> None:
    summary = summarize_examples(
        [
            EvalExample("q1", "#### 1", "L0", "\\boxed{1}", 10, correct=True),
            EvalExample("q2", "#### 2", "L0", "\\boxed{3}", 20, correct=False),
        ]
    )

    assert summary["L0"]["count"] == 2.0
    assert summary["L0"]["accuracy"] == 0.5
    assert summary["L0"]["mean_generated_tokens"] == 15.0


def test_make_eval_prompt_matches_sft_chat_style() -> None:
    prompt = make_eval_prompt("What is 2+2?", "L3")

    assert prompt.startswith("<|user|>\n")
    assert "Reasoning budget: <budget=L3>" in prompt
    assert prompt.endswith("<|assistant|>\n")


def test_resume_filters_completed_problem_budget_pairs(tmp_path) -> None:
    output_path = tmp_path / "eval.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "question": "q",
                "gold": "#### 1",
                "budget": "L0",
                "completion": "\\boxed{1}",
                "generated_tokens": 5,
                "problem_index": 7,
            }
        )
        + "\n"
    )

    examples, completed_keys = load_completed_examples(output_path, resume=True)
    rows = [{"problem_index": 7, "question": "q", "gold": "#### 1", "source": "test"}]
    tasks = build_eval_tasks(rows=rows, budgets=["L0", "L1"], completed_keys=completed_keys)

    assert len(examples) == 1
    assert tasks == [(rows[0], "L1")]


def test_default_config_uses_vllm_eval_backend() -> None:
    cfg = load_config(Path(__file__).parents[1] / "configs/default.yaml")

    assert cfg.eval.backend == "vllm"
    assert cfg.eval.max_new_tokens == 2048
    assert cfg.eval.vllm_chunk_size == 1024
    assert "wandb" in cfg.eval.report_to
    assert cfg.eval.wandb_project == "concise-cot"


def test_flatten_summary_uses_per_budget_metric_names() -> None:
    flattened = flatten_summary(
        {"L0": {"accuracy": 0.5, "mean_generated_tokens": 100.0, "count": 2.0}}
    )

    assert flattened == {
        "eval/L0/accuracy": 0.5,
        "eval/L0/mean_generated_tokens": 100.0,
        "eval/L0/count": 2.0,
    }
