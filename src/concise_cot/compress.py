"""Build budgeted traces from verified long teacher traces."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from concise_cot.config import load_config
from concise_cot.data import BudgetedTrace, TeacherTrace, read_jsonl, write_jsonl
from concise_cot.gen_teacher import extract_reasoning
from concise_cot.verify import verify_completion


def target_token_count(long_token_count: int, budget_ratio: float) -> int:
    """Compute the requested compressed token count for a budget ratio."""

    if not 0 < budget_ratio <= 1:
        raise ValueError("budget_ratio must be in (0, 1]")
    return max(1, round(long_token_count * budget_ratio))


def target_token_range(target_tokens: int) -> tuple[int, int]:
    """Allow a small range around the target to improve length adherence."""

    return max(1, round(target_tokens * 0.9)), max(1, round(target_tokens * 1.1))


def budget_instruction(budget: str) -> str:
    """Budget-specific rewrite guidance."""

    instructions = {
        "L1": (
            "Light-to-moderate compression: preserve the original solution structure, most "
            "explanatory detail, and all intermediate calculations. Remove only obvious "
            "repetition, chatty text, and redundant self-checks. Do not summarize into a "
            "short solution; the output should still read like a full worked explanation."
        ),
        "L2": (
            "Medium compression: keep the essential derivation and important equations, but "
            "use compact phrasing. One short sentence per step is enough. Avoid full worked "
            "explanations and remove all redundant prose."
        ),
        "L3": (
            "Very strong compression: include only the decisive computation and the minimum "
            "context needed to make it understandable. Equations and terse fragments are "
            "preferred. Avoid full sentences when an equation is sufficient."
        ),
    }
    return instructions.get(budget, "Preserve the original reasoning.")


def make_rewrite_prompt(trace: TeacherTrace, budget: str, target_tokens: int) -> str:
    """Prompt text for rewrite-to-budget compression."""

    min_tokens, max_tokens = target_token_range(target_tokens)
    lower_bound_note = ""
    if budget == "L1":
        lower_bound_note = (
            f"Important for L1: the reasoning must not be shorter than about {min_tokens} tokens. "
            "If your first draft is too short, keep more of the original derivation and explanatory "
            "steps until it fits the requested range.\n\n"
        )
    elif budget in {"L2", "L3"}:
        lower_bound_note = (
            f"Important for {budget}: prioritize staying at or below {max_tokens} reasoning tokens. "
            "Do not pad the answer to reach the lower end of the range.\n\n"
        )
    return (
        "Rewrite the reasoning below to match the requested reasoning budget while preserving "
        "the answer-determining logic. Aim for the target range, not the shortest possible "
        "answer. Remove filler, repeated checks, chatty text, and decorative self-reflection "
        "before removing useful derivation steps.\n\n"
        f"{lower_bound_note}"
        f"Budget guidance: {budget_instruction(budget)}\n\n"
        "Return exactly this format:\n"
        "<think>\n"
        "compressed reasoning here\n"
        "</think>\n"
        f"\\boxed{{{trace.answer}}}\n\n"
        f"Budget: {budget}\n"
        f"Target reasoning tokens: {target_tokens}\n"
        f"Acceptable reasoning-token range: {min_tokens}-{max_tokens}\n"
        f"Problem:\n{trace.question}\n\n"
        f"Original reasoning:\n{trace.long_cot}\n\n"
        f"Final answer: {trace.answer}"
    )


def keep_verified_rewrite(
    trace: TeacherTrace,
    budget: str,
    rewritten_reasoning: str,
    answer: str,
) -> BudgetedTrace | None:
    """Return a budgeted trace only if the rewrite still verifies."""

    completion = f"<think>{rewritten_reasoning}</think>\n\\boxed{{{answer}}}"
    if not verify_completion(completion, trace.gold).correct:
        return None
    return BudgetedTrace(
        question=trace.question,
        gold=trace.gold,
        budget=budget,
        reasoning=rewritten_reasoning,
        answer=answer,
        problem_index=trace.problem_index,
        sample_index=trace.sample_index,
        original_token_count=trace.token_count,
        target_token_count=None,
        compressed_token_count=None,
        source=trace.source,
    )


def pass_through_l0(input_path: Path, output_path: Path) -> int:
    """Create L0 rows directly from verified long traces."""

    rows = []
    for raw in read_jsonl(input_path):
        trace = TeacherTrace(**raw)
        rows.append(
            asdict(
                BudgetedTrace(
                    question=trace.question,
                    gold=trace.gold,
                    budget="L0",
                    reasoning=trace.long_cot,
                    answer=trace.answer,
                    problem_index=trace.problem_index,
                    sample_index=trace.sample_index,
                    original_token_count=trace.token_count,
                    target_token_count=trace.token_count,
                    compressed_token_count=trace.token_count,
                    source=trace.source,
                )
            )
        )
    return write_jsonl(output_path, rows)


def load_traces(input_path: Path, limit: int | None = None) -> list[TeacherTrace]:
    """Load verified teacher traces."""

    traces = []
    for raw in read_jsonl(input_path):
        traces.append(TeacherTrace(**raw))
        if limit is not None and len(traces) >= limit:
            break
    return traces


def render_rewrite_prompts(
    model_name: str,
    traces: list[TeacherTrace],
    budget: str,
    budget_ratio: float,
) -> tuple[list[str], list[int]]:
    """Render rewrite prompts and target token counts."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompts = []
    targets = []
    for trace in traces:
        original_tokens = trace.token_count or max(1, len(trace.long_cot.split()))
        target_tokens = target_token_count(original_tokens, budget_ratio)
        targets.append(target_tokens)
        messages = [{"role": "user", "content": make_rewrite_prompt(trace, budget, target_tokens)}]
        try:
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
    return prompts, targets


def build_budgeted_rows(
    *,
    model_name: str,
    traces: list[TeacherTrace],
    budgets: dict[str, float],
    output_path: Path,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    trust_remote_code: bool,
) -> int:
    """Build L0 rows and vLLM-compressed L1/L2/L3 rows."""

    from vllm import LLM, SamplingParams

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    with output_path.open("w") as handle:
        if "L0" in budgets:
            for trace in traces:
                row = BudgetedTrace(
                    question=trace.question,
                    gold=trace.gold,
                    budget="L0",
                    reasoning=trace.long_cot,
                    answer=trace.answer,
                    problem_index=trace.problem_index,
                    sample_index=trace.sample_index,
                    original_token_count=trace.token_count,
                    target_token_count=trace.token_count,
                    compressed_token_count=trace.token_count,
                    source=trace.source,
                )
                handle.write(json_dumps(asdict(row)))
                kept += 1

        for budget, ratio in budgets.items():
            if budget == "L0":
                continue

            prompts, targets = render_rewrite_prompts(model_name, traces, budget, ratio)
            max_target = max(target_token_range(target)[1] for target in targets) if targets else 1
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max(64, min(1024, max_target + 48)),
                stop=["<|im_end|>", "<|endoftext|>"],
            )

            for trace, target, request_output in zip(
                traces,
                targets,
                llm.generate(prompts, sampling_params),
                strict=True,
            ):
                completion = request_output.outputs[0]
                raw_completion = completion.text.strip()
                reasoning = extract_reasoning(raw_completion)
                verified = verify_completion(raw_completion, trace.gold)
                if not verified.correct:
                    continue

                row = BudgetedTrace(
                    question=trace.question,
                    gold=trace.gold,
                    budget=budget,
                    reasoning=reasoning,
                    answer=verified.prediction or trace.answer,
                    problem_index=trace.problem_index,
                    sample_index=trace.sample_index,
                    original_token_count=trace.token_count,
                    target_token_count=target,
                    compressed_token_count=len(completion.token_ids),
                    source=trace.source,
                )
                handle.write(json_dumps(asdict(row)))
                kept += 1

    return kept


def json_dumps(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create budgeted traces from teacher traces.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budgets", nargs="+", default=["L0", "L1", "L2", "L3"])
    parser.add_argument(
        "--l0-only",
        action="store_true",
        help="Emit only full-length L0 rows; useful before rewrite generation exists.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or Path(cfg.data.traces_path)
    output_path = args.output or Path(cfg.data.budgeted_path)

    if args.l0_only:
        count = pass_through_l0(input_path, output_path)
        print(f"wrote={count}")
        return

    selected_budgets = {budget: cfg.budgets[budget] for budget in args.budgets}
    traces = load_traces(input_path, args.limit)
    count = build_budgeted_rows(
        model_name=cfg.models.teacher,
        traces=traces,
        budgets=selected_budgets,
        output_path=output_path,
        tensor_parallel_size=cfg.generation.tensor_parallel_size,
        gpu_memory_utilization=cfg.generation.gpu_memory_utilization,
        dtype=cfg.generation.dtype,
        trust_remote_code=cfg.generation.trust_remote_code,
    )
    print(f"wrote={count} output={output_path}")


if __name__ == "__main__":
    main()
