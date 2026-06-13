"""Configuration loading for the Concise-CoT pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 13
    output_dir: str = "outputs"


@dataclass(frozen=True)
class ModelConfig:
    teacher: str = "Qwen/Qwen3-32B"
    student: str = "Qwen/Qwen3-4B"


@dataclass(frozen=True)
class DataConfig:
    train_sources: tuple[str, ...] = ("gsm8k", "math")
    eval_sources: tuple[str, ...] = ("gsm8k_test", "math_500")
    traces_path: str = "data/traces/teacher_traces.jsonl"
    budgeted_path: str = "data/budgeted/budgeted_traces.jsonl"


@dataclass(frozen=True)
class GenerationConfig:
    samples_per_problem: int = 4
    temperature: float = 0.7
    max_tokens: int = 4096
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "ckpts/concise-cot"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05


@dataclass(frozen=True)
class EvalConfig:
    backend: str = "vllm"
    max_new_tokens: int = 2048
    batch_size: int = 8
    progress_every: int = 32
    vllm_chunk_size: int = 1024
    report_to: tuple[str, ...] = ("wandb",)
    wandb_project: str = "concise-cot"


@dataclass(frozen=True)
class PipelineConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    budgets: dict[str, float] = field(
        default_factory=lambda: {"L0": 1.0, "L1": 0.6, "L2": 0.35, "L3": 0.15}
    )
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _tupleify(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def load_config(path: str | Path = "configs/default.yaml") -> PipelineConfig:
    """Load the YAML pipeline config into typed dataclasses."""

    raw = yaml.safe_load(Path(path).read_text()) or {}
    data_raw = raw.get("data", {})

    return PipelineConfig(
        project=ProjectConfig(**raw.get("project", {})),
        models=ModelConfig(**raw.get("models", {})),
        data=DataConfig(
            train_sources=_tupleify(data_raw.get("train_sources", DataConfig.train_sources)),
            eval_sources=_tupleify(data_raw.get("eval_sources", DataConfig.eval_sources)),
            traces_path=data_raw.get("traces_path", DataConfig.traces_path),
            budgeted_path=data_raw.get("budgeted_path", DataConfig.budgeted_path),
        ),
        budgets={str(key): float(value) for key, value in raw.get("budgets", {}).items()}
        or PipelineConfig().budgets,
        generation=GenerationConfig(**raw.get("generation", {})),
        training=TrainingConfig(**raw.get("training", {})),
        eval=EvalConfig(**raw.get("eval", {})),
    )
