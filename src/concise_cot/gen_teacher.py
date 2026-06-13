"""Teacher trace generation with rejection sampling."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from concise_cot.config import load_config
from concise_cot.verify import extract_answer, verify_completion

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{")


def make_teacher_prompt(question: str) -> str:
    """Prompt the teacher for a verifiable math solution."""

    return (
        "Solve the math problem. Show only the reasoning needed to determine the answer. "
        "End with exactly one final answer in \\boxed{}. Do not ask follow-up questions, "
        "do not include code, and stop immediately after the boxed answer.\n\n"
        f"Problem: {question}"
    )


def render_teacher_prompts(model_name: str, questions: list[dict[str, str]]) -> list[str]:
    """Render teacher prompts with the model chat template when available."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompts = []
    for row in questions:
        user_prompt = make_teacher_prompt(row["question"])
        messages = [{"role": "user", "content": user_prompt}]
        try:
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
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
    return prompts


def load_gsm8k_questions(limit: int | None) -> list[dict[str, str]]:
    """Load GSM8K train examples for teacher trace generation."""

    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [
        {
            "problem_index": index,
            "question": str(row["question"]),
            "gold": str(row["answer"]),
            "source": "gsm8k/train",
        }
        for index, row in enumerate(dataset)
    ]


def extract_reasoning(completion: str) -> str:
    """Extract reasoning content from a teacher completion."""

    think_matches = THINK_RE.findall(completion)
    if think_matches:
        return think_matches[-1].strip()

    boxed_match = BOXED_RE.search(completion)
    if boxed_match:
        return completion[: boxed_match.start()].strip()

    return completion.strip()


def generate_verified_traces(
    *,
    model_name: str,
    questions: list[dict[str, str]],
    output_path: Path,
    samples_per_problem: int,
    temperature: float,
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    trust_remote_code: bool,
) -> int:
    """Sample teacher completions with vLLM and keep only answer-correct traces."""

    from vllm import LLM, SamplingParams

    prompts = render_teacher_prompts(model_name, questions)
    sampling_params = SamplingParams(
        n=samples_per_problem,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_path.open("w") as handle:
        for row, request_output in zip(questions, llm.generate(prompts, sampling_params), strict=True):
            for sample_index, sample in enumerate(request_output.outputs):
                verification = verify_completion(sample.text, row["gold"])
                if not verification.correct:
                    continue
                raw_completion = sample.text.strip()
                payload = {
                    "problem_index": row["problem_index"],
                    "sample_index": sample_index,
                    "question": row["question"],
                    "gold": row["gold"],
                    "long_cot": extract_reasoning(raw_completion),
                    "answer": verification.prediction or extract_answer(row["gold"]),
                    "token_count": len(sample.token_ids),
                    "source": row["source"],
                    "raw_completion": raw_completion,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate verified teacher CoT traces.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", choices=["gsm8k"], default="gsm8k")
    parser.add_argument("--samples-per-problem", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = args.output or Path(cfg.data.traces_path)

    if args.source != "gsm8k":
        raise ValueError(f"unsupported source: {args.source}")

    questions = load_gsm8k_questions(args.limit)
    kept = generate_verified_traces(
        model_name=cfg.models.teacher,
        questions=questions,
        output_path=output,
        samples_per_problem=args.samples_per_problem or cfg.generation.samples_per_problem,
        temperature=cfg.generation.temperature,
        max_tokens=args.max_tokens or cfg.generation.max_tokens,
        tensor_parallel_size=cfg.generation.tensor_parallel_size,
        gpu_memory_utilization=cfg.generation.gpu_memory_utilization,
        dtype=cfg.generation.dtype,
        trust_remote_code=cfg.generation.trust_remote_code,
    )
    print(f"wrote={kept} output={output}")


if __name__ == "__main__":
    main()
