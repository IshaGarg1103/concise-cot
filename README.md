# Concise-CoT

Length-controlled reasoning distillation from `Qwen/Qwen3-32B` into a compact `Qwen/Qwen3-4B` student.

This project trains one student model to follow explicit reasoning budgets:

- `L0`: full reasoning
- `L1`: light compression
- `L2`: medium compression
- `L3`: strong compression

The goal is to measure the accuracy-vs-token Pareto curve and test whether compression removes decorative reasoning before load-bearing calculation.

## Training Curve

bf16 LoRA SFT was tracked in W&B under `qwen3-4b-budget-sft-full-1000`.

![Qwen3-4B budget SFT training curves](asset/qwen3-4b-budget-sft-full-1000.png)

## Results

Final fixed GSM8K test evaluation used vLLM LoRA inference with `checkpoint-1000`.

| Budget | Accuracy | Mean Generated Tokens | Count |
|---|---:|---:|---:|
| `L0` | 0.9356 | 496.7 | 1319 |
| `L1` | 0.8506 | 350.6 | 1319 |
| `L2` | 0.8878 | 99.0 | 1319 |
| `L3` | 0.8484 | 79.7 | 1319 |

Main insight: `L0` is the accuracy ceiling, while `L2` is the best compressed Pareto point: **88.8% accuracy at about 99 generated tokens**.

## RQ2: What Gets Removed?

Structural analysis on the budgeted traces supports decorative-first compression:

| Budget | Mean Token Ratio | Mean Step Ratio | Removed Calc-Like Steps | Removed Prose-Like Steps |
|---|---:|---:|---:|---:|
| `L1` | 0.653 | 0.589 | 0.538 | 0.825 |
| `L2` | 0.179 | 0.192 | 0.697 | 0.970 |
| `L3` | 0.137 | 0.102 | 0.892 | 0.995 |

Prose-like reasoning is removed more aggressively than calculation-like reasoning at every budget. This is structural evidence, not full causal proof.

## Pipeline Summary

1. Generate verified teacher traces from GSM8K train with `Qwen/Qwen3-32B` and vLLM.
2. Rewrite each correct trace into `L1/L2/L3` budgeted versions and re-verify correctness.
3. Convert budgeted traces into chat-style SFT rows.
4. Train `Qwen/Qwen3-4B` with bf16 LoRA SFT.
5. Evaluate the adapter on GSM8K test with vLLM LoRA inference.
6. Build Pareto artifacts and RQ2 structural-removal artifacts.

## Repository

```text
configs/default.yaml              # model, generation, training, and eval config
src/concise_cot/
  config.py                       # typed YAML config loading
  data.py                         # data records and SFT formatting
  verify.py                       # answer extraction and equivalence checks
  gen_teacher.py                  # vLLM teacher trace generation
  compress.py                     # rewrite-to-budget compression
  train.py                        # bf16 LoRA SFT entry point
  eval.py                         # vLLM/HF adapter eval and W&B logging
  tts_score.py                    # RQ2 structural removal analysis
  make_report_artifacts.py        # Pareto CSV/Markdown artifacts
  inspect_jsonl.py                # JSONL inspection helper
  sample_adapter.py               # quick adapter sampling helper
report/report.md                  # final write-up
learning.md                       # detailed experiment log
asset/                            # README/report images
tests/                            # unit tests for pipeline utilities
```

Generated datasets, checkpoints, W&B logs, and copied VM artifacts are ignored by git.

## Setup

Install ROCm PyTorch and ROCm vLLM explicitly for the target MI300X environment. Then install this project without replacing the ROCm stack:

```bash
python -m pip install --no-deps -e ".[dev]"
```

For local utility tests without the GPU stack:

```bash
PYTHONPATH=src python -m pytest tests -q
```

## Experiment Tracking

- Project: `ishagarg-research/concise-cot`
- Training run: `qwen3-4b-budget-sft-full-1000`
- Fixed eval run: `gsm8k-checkpoint1000-vllm-fixed-eval`

The training run contains bf16 LoRA SFT loss and token-accuracy curves. The fixed eval run contains the budget-wise accuracy/token metrics and Pareto plot.

## References and Inspiration

- Efficient reasoning survey: [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/html/2503.16419v1) and the [Awesome Efficient Reasoning LLMs](https://github.com/Eclipsess/Awesome-Efficient-Reasoning-LLMs) tracker.
- Faithfulness and decorative reasoning: [Can Aha Moments Be Fake? Identifying True and Decorative Thinking Steps in Chain-of-Thought](https://arxiv.org/abs/2510.24941) and the [True vs Decorative Thinking project page](https://andotalao24.github.io/Identify_true_decorative_thinking/).
- R1-style self-reflection caution: [There May Not be Aha Moment in R1-Zero-like Training](https://sail.sea.com/blog/articles/62) and [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/html/2503.20783).
- Reflection analysis: [First Try Matters: Revisiting the Role of Reflection in Reasoning Models](https://arxiv.org/html/2510.08308v1).
- Controllable CoT compression: [TokenSkip: Controllable Chain-of-Thought Compression in LLMs](https://arxiv.org/html/2502.12067v1), [C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness](https://arxiv.org/html/2412.11664), and [CoT-Valve: Length-Compressible Chain-of-Thought Tuning](https://arxiv.org/html/2502.09601).
- Concise reasoning and token budgets: [Self-Training Elicits Concise Reasoning in Large Language Models](https://arxiv.org/abs/2502.20122), [Token-Budget-Aware LLM Reasoning / TALE](https://aclanthology.org/2025.findings-acl.1274/), and [Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/abs/2502.18600v2).
- Teacher lineage: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/html/2501.12948v1), [DeepSeek-R1 in Nature](https://www.nature.com/articles/s41586-025-09422-z), [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B), and [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B).

## Limitations

- GSM8K is the completed benchmark; MATH-500 remains a useful harder follow-up.
- RQ2 is structural evidence, not full causal proof.
- Evaluation caps matter: under-capping `L0` can make full reasoning look artificially weak.
