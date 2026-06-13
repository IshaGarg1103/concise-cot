"""Generate samples from a LoRA adapter for quick sanity checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from concise_cot.config import load_config
from concise_cot.data import make_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a trained concise-cot LoRA adapter.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--budgets", nargs="+", default=["L0", "L3"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.models.student,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    model.eval()

    for budget in args.budgets:
        prompt = (
            "<|user|>\n"
            f"{make_prompt(args.question, budget)}\n"
            "<|assistant|>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False)
        print("=" * 80)
        print(f"budget={budget}")
        print(decoded.strip())


if __name__ == "__main__":
    main()
