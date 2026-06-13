"""Evaluation metrics for budget-conditioned reasoning."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from concise_cot.config import load_config
from concise_cot.data import make_prompt
from concise_cot.verify import verify_completion


@dataclass(frozen=True)
class EvalExample:
    question: str
    gold: str
    budget: str
    completion: str
    generated_tokens: int
    prediction: str | None = None
    correct: bool | None = None
    source: str | None = None
    problem_index: int | None = None


def summarize_examples(examples: Iterable[EvalExample]) -> dict[str, dict[str, float]]:
    """Aggregate accuracy and token count per budget."""

    grouped: dict[str, list[EvalExample]] = defaultdict(list)
    for example in examples:
        grouped[example.budget].append(example)

    summary = {}
    for budget, rows in grouped.items():
        correctness = [
            row.correct if row.correct is not None else verify_completion(row.completion, row.gold).correct
            for row in rows
        ]
        summary[budget] = {
            "count": float(len(rows)),
            "accuracy": mean(correctness) if correctness else 0.0,
            "mean_generated_tokens": mean(row.generated_tokens for row in rows) if rows else 0.0,
        }
    return summary


def flatten_summary(summary: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten per-budget eval metrics into W&B-friendly scalar names."""

    return {
        f"eval/{budget}/{metric}": value
        for budget, metrics in summary.items()
        for metric, value in metrics.items()
    }


def log_summary_to_wandb(
    *,
    summary: dict[str, dict[str, float]],
    project: str,
    run_name: str | None,
    run_id: str | None,
    resume: str,
    backend: str,
    source: str,
    adapter_path: Path | None,
    output_path: Path | None,
    summary_output_path: Path | None,
) -> None:
    """Log evaluation scalars and Pareto data to Weights & Biases."""

    import wandb

    run = wandb.init(
        project=project,
        name=run_name,
        id=run_id,
        resume=resume if run_id else None,
        config={
            "eval_backend": backend,
            "eval_source": source,
            "adapter": str(adapter_path) if adapter_path else None,
            "output": str(output_path) if output_path else None,
            "summary_output": str(summary_output_path) if summary_output_path else None,
        },
    )
    table = wandb.Table(columns=["budget", "accuracy", "mean_generated_tokens", "count"])
    for budget in sorted(summary):
        metrics = summary[budget]
        table.add_data(
            budget,
            metrics["accuracy"],
            metrics["mean_generated_tokens"],
            metrics["count"],
        )

    wandb.log(
        {
            **flatten_summary(summary),
            "eval/pareto_table": table,
            "eval/pareto_accuracy_vs_tokens": wandb.plot.line(
                table,
                "mean_generated_tokens",
                "accuracy",
                title="Accuracy vs Generated Tokens",
            ),
        }
    )
    run.finish()


def load_gsm8k_test(limit: int | None = None) -> list[dict]:
    """Load GSM8K test rows."""

    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [
        {
            "problem_index": index,
            "question": str(row["question"]),
            "gold": str(row["answer"]),
            "source": "gsm8k/test",
        }
        for index, row in enumerate(dataset)
    ]


def make_eval_prompt(question: str, budget: str) -> str:
    """Create the same chat-style prompt format used for SFT."""

    return (
        "<|user|>\n"
        f"{make_prompt(question, budget)}\n"
        "<|assistant|>\n"
    )


def load_completed_examples(output_path: Path, resume: bool) -> tuple[list[EvalExample], set[tuple[int | None, str]]]:
    """Load existing eval rows for resumable generation."""

    if not resume or not output_path.exists():
        return [], set()
    examples = load_eval_outputs(output_path)
    completed_keys = {(example.problem_index, example.budget) for example in examples}
    print(f"resuming from {len(completed_keys)} existing generations", flush=True)
    return examples, completed_keys


def write_eval_example(handle, example: EvalExample) -> None:
    """Write one evaluation example as JSONL."""

    handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def build_eval_tasks(
    *,
    rows: list[dict],
    budgets: list[str],
    completed_keys: set[tuple[int | None, str]],
) -> list[tuple[dict, str]]:
    """Create the remaining problem/budget generation tasks."""

    tasks = [
        (row, budget)
        for row in rows
        for budget in budgets
        if (row["problem_index"], budget) not in completed_keys
    ]
    print(
        f"loaded {len(rows)} problems x {len(budgets)} budgets; remaining={len(tasks)} generations",
        flush=True,
    )
    return tasks


def generate_adapter_outputs(
    *,
    config_path: Path,
    adapter_path: Path,
    output_path: Path,
    budgets: list[str],
    limit: int | None,
    max_new_tokens: int,
    batch_size: int,
    progress_every: int,
    resume: bool,
) -> list[EvalExample]:
    """Generate evaluation outputs from a LoRA adapter."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.models.student,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    rows = load_gsm8k_test(limit)
    examples, completed_keys = load_completed_examples(output_path, resume)
    tasks = build_eval_tasks(rows=rows, budgets=budgets, completed_keys=completed_keys)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(mode) as handle:
        next_progress = min(progress_every, len(tasks))
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            prompts = [make_eval_prompt(row["question"], budget) for row, budget in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            input_length = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for (row, budget), output in zip(batch, outputs, strict=True):
                generated_ids = output[input_length:]
                completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
                verification = verify_completion(completion, row["gold"])
                example = EvalExample(
                    question=row["question"],
                    gold=row["gold"],
                    budget=budget,
                    completion=completion,
                    generated_tokens=len(generated_ids),
                    prediction=verification.prediction,
                    correct=verification.correct,
                    source=row["source"],
                    problem_index=row["problem_index"],
                )
                write_eval_example(handle, example)
                examples.append(example)
            handle.flush()
            completed = min(start + len(batch), len(tasks))
            if completed >= next_progress or completed == len(tasks):
                print(f"processed={completed}/{len(tasks)}", flush=True)
                while next_progress <= completed:
                    next_progress += progress_every
    return examples


def generate_adapter_outputs_vllm(
    *,
    config_path: Path,
    adapter_path: Path,
    output_path: Path,
    budgets: list[str],
    limit: int | None,
    max_new_tokens: int,
    chunk_size: int,
    resume: bool,
) -> list[EvalExample]:
    """Generate evaluation outputs from a LoRA adapter with vLLM."""

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    cfg = load_config(config_path)
    rows = load_gsm8k_test(limit)
    examples, completed_keys = load_completed_examples(output_path, resume)
    tasks = build_eval_tasks(rows=rows, budgets=budgets, completed_keys=completed_keys)

    llm = LLM(
        model=cfg.models.student,
        enable_lora=True,
        max_lora_rank=cfg.training.lora_rank,
        tensor_parallel_size=cfg.generation.tensor_parallel_size,
        gpu_memory_utilization=cfg.generation.gpu_memory_utilization,
        dtype=cfg.generation.dtype,
        trust_remote_code=cfg.generation.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    lora_request = LoRARequest("concise_cot_adapter", 1, str(adapter_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(mode) as handle:
        for start in range(0, len(tasks), chunk_size):
            chunk = tasks[start : start + chunk_size]
            prompts = [make_eval_prompt(row["question"], budget) for row, budget in chunk]
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

            for (row, budget), request_output in zip(chunk, outputs, strict=True):
                completion_output = request_output.outputs[0]
                completion = completion_output.text.strip()
                verification = verify_completion(completion, row["gold"])
                example = EvalExample(
                    question=row["question"],
                    gold=row["gold"],
                    budget=budget,
                    completion=completion,
                    generated_tokens=len(completion_output.token_ids),
                    prediction=verification.prediction,
                    correct=verification.correct,
                    source=row["source"],
                    problem_index=row["problem_index"],
                )
                write_eval_example(handle, example)
                examples.append(example)
            handle.flush()
            completed = min(start + len(chunk), len(tasks))
            print(f"processed_remaining={completed}/{len(tasks)}", flush=True)
    return examples


def load_eval_outputs(path: Path) -> list[EvalExample]:
    examples = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(EvalExample(**row))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or summarize evaluation outputs.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--source", choices=["gsm8k"], default="gsm8k")
    parser.add_argument("--budgets", nargs="+", default=["L0", "L1", "L2", "L3"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument("--backend", choices=["vllm", "hf"], default=None)
    parser.add_argument("--vllm-chunk-size", type=int, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-to", nargs="+", default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-resume", type=str, default="allow")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = args.backend or cfg.eval.backend
    max_new_tokens = args.max_new_tokens or cfg.eval.max_new_tokens
    batch_size = args.batch_size or cfg.eval.batch_size
    progress_every = args.progress_every or cfg.eval.progress_every
    vllm_chunk_size = args.vllm_chunk_size or cfg.eval.vllm_chunk_size
    report_to = tuple(args.report_to) if args.report_to is not None else tuple(cfg.eval.report_to)
    wandb_project = args.wandb_project or cfg.eval.wandb_project

    if args.adapter:
        if args.output is None:
            raise ValueError("--output is required when generating adapter outputs")
        if args.source != "gsm8k":
            raise ValueError(f"unsupported source: {args.source}")
        if backend == "vllm":
            examples = generate_adapter_outputs_vllm(
                config_path=args.config,
                adapter_path=args.adapter,
                output_path=args.output,
                budgets=args.budgets,
                limit=args.limit,
                max_new_tokens=max_new_tokens,
                chunk_size=vllm_chunk_size,
                resume=args.resume,
            )
        else:
            examples = generate_adapter_outputs(
                config_path=args.config,
                adapter_path=args.adapter,
                output_path=args.output,
                budgets=args.budgets,
                limit=args.limit,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                progress_every=progress_every,
                resume=args.resume,
            )
    else:
        if args.input is None:
            raise ValueError("--input is required when summarizing existing outputs")
        examples = load_eval_outputs(args.input)

    summary = summarize_examples(examples)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(payload + "\n")
    else:
        print(payload)
    if "wandb" in report_to:
        log_summary_to_wandb(
            summary=summary,
            project=wandb_project,
            run_name=args.wandb_run_name,
            run_id=args.wandb_run_id,
            resume=args.wandb_resume,
            backend=backend,
            source=args.source,
            adapter_path=args.adapter,
            output_path=args.output or args.input,
            summary_output_path=args.summary_output,
        )


if __name__ == "__main__":
    main()
