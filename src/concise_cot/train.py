"""bf16 LoRA SFT entry point for the budget-conditioned student."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

from concise_cot.config import load_config

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_sft_config(config_cls, **kwargs):
    """Build SFTConfig while tolerating TRL field-name changes."""

    signature = inspect.signature(config_cls.__init__)
    supported = set(signature.parameters)
    if "max_seq_length" in kwargs and "max_seq_length" not in supported:
        if "max_length" in supported:
            kwargs["max_length"] = kwargs.pop("max_seq_length")
        else:
            kwargs.pop("max_seq_length")
    if "completion_only_loss" in kwargs and "completion_only_loss" not in supported:
        kwargs.pop("completion_only_loss")
    if "dataset_text_field" in kwargs and "dataset_text_field" not in supported:
        kwargs.pop("dataset_text_field")
    return config_cls(**{key: value for key, value in kwargs.items() if key in supported})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the concise CoT student with bf16 LoRA.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--report-to", nargs="+", default=["wandb"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-resume", type=str, default="allow")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_jsonl = args.train_jsonl or Path(cfg.data.budgeted_path)

    # Keep heavyweight ML imports inside the command path so tests can import the package quickly.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model = AutoModelForCausalLM.from_pretrained(
        cfg.models.student,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("json", data_files=str(train_jsonl), split="train")
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))

    output_dir = args.output_dir or cfg.training.output_dir
    run_name = args.run_name or Path(output_dir).name
    if "wandb" in args.report_to:
        os.environ.setdefault("WANDB_PROJECT", "concise-cot")
        if args.wandb_run_id:
            os.environ["WANDB_RUN_ID"] = args.wandb_run_id
            os.environ["WANDB_RESUME"] = args.wandb_resume

    peft_config = LoraConfig(
        r=cfg.training.lora_rank,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    sft_config = build_sft_config(
        SFTConfig,
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs or cfg.training.num_train_epochs,
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=(
            args.per_device_train_batch_size or cfg.training.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps or cfg.training.gradient_accumulation_steps
        ),
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_seq_length=cfg.training.max_seq_length,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        completion_only_loss=True,
        dataset_text_field="text",
        report_to=args.report_to,
        run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=peft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
