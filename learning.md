# Concise-CoT Learning Log

This file tracks decisions, setup lessons, experiment observations, and results as the project progresses.

## Final Experiment Summary

### Project Goal

Distill length-controlled reasoning from a strong teacher into a compact student:

- Teacher: `Qwen/Qwen3-32B`
- Student: `Qwen/Qwen3-4B`
- Method: budget-conditioned SFT with `L0`, `L1`, `L2`, `L3`
- Main question: can a small model control reasoning length, and does compression remove decorative reasoning before load-bearing reasoning?

### Final Pipeline

The aligned pipeline is:

1. Generate verified teacher CoTs with vLLM.
2. Rewrite teacher CoTs into budgeted traces (`L0/L1/L2/L3`) with vLLM and re-verification.
3. Train `Qwen3-4B` student with bf16 LoRA SFT.
4. Evaluate the trained adapter with vLLM LoRA inference.
5. Analyze the accuracy-vs-token Pareto curve.
6. Analyze RQ2 decorative-first removal on the budgeted dataset.

Important correction: final adapter evaluation must use vLLM. The earlier Hugging Face generation path was too slow and is only a fallback/debugging path.

### Final GSM8K Result

Final fixed evaluation:

- checkpoint: `ckpts/concise-cot-qwen3-4b-full/checkpoint-1000`
- backend: vLLM LoRA inference
- dataset: GSM8K test
- examples: 1,319 problems x 4 budgets = 5,276 generations
- W&B run: `gsm8k-checkpoint1000-vllm-fixed-eval`

| Budget | Accuracy | Mean generated tokens | Count |
|---|---:|---:|---:|
| `L0` | 0.9356 | 496.7 | 1319 |
| `L1` | 0.8506 | 350.6 | 1319 |
| `L2` | 0.8878 | 99.0 | 1319 |
| `L3` | 0.8484 | 79.7 | 1319 |

Key insight:

- `L0` is the full-reasoning accuracy ceiling.
- `L2` is the best compressed Pareto point: it keeps 88.8% accuracy at about 99 generated tokens.
- `L3` compresses further but loses accuracy relative to `L2`.
- The project result is coherent: full reasoning wins when tokens are unconstrained; medium compression gives a strong accuracy/token tradeoff.

### Evaluation Fix

The first vLLM full eval used `max_new_tokens=512` for all budgets. This unfairly capped `L0`; audit samples showed `L0` hitting 512 tokens and stopping mid-reasoning. We fixed this without rerunning the full experiment:

- reran only `L0` with `max_new_tokens=2048`;
- kept existing `L1/L2/L3` rows;
- combined them into `outputs/gsm8k_checkpoint1000_eval_fixed_vllm.jsonl`;
- logged the corrected run to W&B.

This was the right minimal fix because the trained model, dataset, checkpoint, backend, and test set stayed unchanged.

### RQ2: Decorative-First Compression

RQ2 dataset:

- path: `data/budgeted/gsm8k_qwen3_32b_budgeted_full.jsonl`
- rows: 106,601
- groups: 26,685
- budgets: `L0=26685`, `L1=26599`, `L2=26676`, `L3=26641`

RQ2 structural removal results:

| Budget | Usable groups | Mean token ratio | Mean step ratio | Removed calc-like steps | Removed prose-like steps | Decorative-first |
|---|---:|---:|---:|---:|---:|---|
| `L1` | 26599 | 0.653 | 0.589 | 0.538 | 0.825 | true |
| `L2` | 26676 | 0.179 | 0.192 | 0.697 | 0.970 | true |
| `L3` | 26641 | 0.137 | 0.102 | 0.892 | 0.995 | true |

Key insight:

- Compression removes prose-like/decorative steps more aggressively than calculation-like/load-bearing steps at every budget.
- This structurally supports the project thesis.
- At `L3`, compression is so aggressive that many calculation-like steps are also removed, explaining the accuracy drop relative to `L2`.
- This is structural evidence, not full causal proof. A small manual/causal validation sample was created for follow-up.

### Final Artifacts

Created artifacts:

- `outputs/gsm8k_checkpoint1000_eval_fixed_vllm.jsonl`
- `outputs/gsm8k_checkpoint1000_eval_fixed_vllm.summary.json`
- `outputs/rq2_structural_summary.csv`
- `outputs/rq2_causal_validation_sample.jsonl`
- `outputs/plots/gsm8k_pareto.csv`
- `outputs/plots/gsm8k_pareto.md`
- `report/report.md`

The report-ready conclusion is:

> The 4B student learned budget-conditioned reasoning. `L0` provides the best accuracy, while `L2` gives the best compressed Pareto point. Structural RQ2 analysis suggests that compression removes decorative/prose-like reasoning before calculation-like reasoning, especially at moderate compression.

### Remaining Limitations

- RQ2 is currently supported by structural analysis, not full causal scoring.
- GSM8K is the only completed benchmark; MATH-500 remains a harder optional follow-up.
- Evaluation caps matter: under-capping `L0` can make full reasoning look artificially weak.

## 2026-06-13 — Hot Aisle MI300X Setup

### Goal

Set up a Hot Aisle AMD MI300X VM for the Concise-CoT experiment:

- Teacher: `Qwen/Qwen3-32B`
- Student: `Qwen/Qwen3-4B`
- Inference stack: vLLM on ROCm
- Training stack: bf16 LoRA on ROCm PyTorch

### Environment

- VM: `enc1-gpuvm004`
- IP: `23.183.40.69`
- GPU: `AMD Instinct MI300X VF`
- Python: `3.12.3`
- PyTorch observed working stack: ROCm-enabled torch with `torch.cuda.is_available() == True`
- vLLM observed working stack: ROCm-compatible vLLM import with GPU visible

### Key Learnings

- On ROCm PyTorch, AMD GPUs are still exposed through the normal `torch.cuda` API.
- A correct AMD setup should report an AMD device from:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

- Plain `pip install vllm` can install CUDA/NVIDIA wheels and replace ROCm torch. This broke the environment once by installing a `+cu130` torch build.
- For this project, `torch` and `vllm` should not be default package dependencies in `pyproject.toml`. Install them explicitly for the target GPU stack.
- vLLM for ROCm should be installed from the ROCm wheel path, preferably with `uv`:

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

- The project package should be installed with `--no-deps` after ROCm torch/vLLM are already correct:

```bash
python -m pip install --no-deps -e ".[dev]"
```

### Current Verification

The VM successfully loaded project config and passed local tests:

```text
teacher: Qwen/Qwen3-32B
student: Qwen/Qwen3-4B
device: AMD Instinct MI300X VF
7 passed
```

### Next Step

Run a tiny vLLM teacher-generation smoke test on GSM8K with `--limit 1`, verify the generated JSONL, then scale cautiously to `--limit 5` and `--limit 20`.

## 2026-06-13 — First Qwen3-32B vLLM Smoke Test

### Result

The first `--limit 1` GSM8K teacher-generation run successfully loaded `Qwen/Qwen3-32B` with vLLM and produced verified correct samples.

Observed output:

```text
rows: 4
answer: 72
shortest sample: 168 tokens
longest samples: 4096 tokens
```

### Interpretation

This is a successful hardware and pipeline smoke test:

- vLLM can load the 32B teacher on the MI300X.
- The verifier accepted all sampled answers for the first GSM8K problem.
- The JSONL schema is usable for downstream compression and SFT data building.

However, it is not yet good enough as production trace generation. Two samples hit the max-token limit and continued with repeated self-checking / chatty filler after already reaching the answer. That is exactly the decorative reasoning behavior the project studies, but for dataset construction we still need traces that terminate cleanly.

### Code Adjustment

Updated `gen_teacher.py` to:

- render prompts with the model tokenizer chat template;
- explicitly tell the teacher to stop after one boxed answer;
- add vLLM stop strings for chat end tokens;
- allow smoke-run overrides for `--max-tokens` and `--samples-per-problem`.

### Next Step

Rerun the smoke test with the updated prompt path and a smaller token cap, then inspect whether samples stop cleanly.

## 2026-06-13 — Clean Teacher Trace Format

### Result

The updated prompt path produced short, correct Qwen3-32B samples around a few hundred tokens instead of repeatedly running to the token cap.

### Schema Decision

Teacher JSONL should keep:

- `long_cot`: extracted reasoning only, without the final boxed answer wrapper;
- `answer`: verified final answer;
- `raw_completion`: full model output for audit/debugging.

This keeps downstream compression aligned with the README, where the compression target is the reasoning trace and the answer is tracked separately.

## 2026-06-13 — Main Teacher Batch Plan

### Decision

Stop running additional smoke tests. The hardware, vLLM stack, verifier, and prompt path are sufficiently validated.

### Next Run

Generate the first main GSM8K teacher-trace batch:

- source: `gsm8k/train`
- limit: `200`
- samples per problem: `4`
- max tokens per sample: `1024`
- teacher: `Qwen/Qwen3-32B`
- output: `data/traces/gsm8k_qwen3_32b_200.jsonl`

### Success Criteria

- vLLM completes without ROCm/runtime errors.
- Output JSONL contains verified rows only.
- Token distribution stays well below the `1024` cap for most samples.
- Rows include `long_cot`, `answer`, and `raw_completion`.

If the 200-problem batch is healthy, scale to a larger GSM8K run before adding MATH.

## 2026-06-13 — GSM8K 200-Problem Teacher Batch Result

### Result

The first main teacher batch completed successfully.

```text
requested problems: 200
samples per problem: 4
requested generations: 800
verified rows kept: 719
keep rate: 89.9%
mean tokens: 560.0
median tokens: 515
min tokens: 188
max tokens: 1024
```

### Interpretation

The batch is healthy enough to scale:

- vLLM completed without ROCm/runtime failures.
- The verifier kept a high fraction of generations.
- Median trace length is well below the `1024` cap.
- Some traces still hit the cap, so token-cap rate should be tracked during larger runs.

### Schema Adjustment Before Scaling

Added `problem_index` and `sample_index` to new teacher trace rows. This makes coverage and per-problem yield auditable without relying on question text.

### Next Step

Run a larger GSM8K teacher batch with the updated schema. Use `1000` problems first, then decide whether to continue toward the full GSM8K train split or add MATH traces.

## Teacher Trace JSONL Schema

Each line in `data/traces/*.jsonl` is one verified teacher generation. Incorrect generations are filtered out and are not written.

Fields:

- `problem_index`: Index of the original problem in the dataset split. For `gsm8k/train`, `0` is the first training problem, `1` is the second, and so on.
- `sample_index`: Which sampled generation this row came from for that problem. With `samples_per_problem=4`, valid values are usually `0`, `1`, `2`, and `3`.
- `question`: The math problem text shown to the teacher model.
- `gold`: The dataset's official answer or solution. For GSM8K, this usually includes the worked solution and final answer after `####`.
- `long_cot`: The extracted teacher reasoning trace. This is the main field used later for compression and distillation.
- `answer`: The final answer extracted from the teacher output after verification. The row is kept only if this matches `gold`.
- `token_count`: Number of generated tokens in this teacher sample. This is used to measure reasoning length and build token statistics.
- `source`: Dataset split that produced the row, such as `gsm8k/train`.
- `raw_completion`: The full original teacher output before cleanup. This is kept for debugging and auditability.

Important distinction:

- `long_cot` is the cleaned reasoning trace we will compress and train on.
- `raw_completion` is the full evidence of what Qwen3-32B produced.

## 2026-06-13 — GSM8K 1000-Problem Teacher Batch Result

### Result

The larger GSM8K teacher batch completed successfully.

```text
requested problems: 1000
samples per problem: 4
requested generations: 4000
verified rows kept: 3537
keep rate: 88.4%
output: data/traces/gsm8k_qwen3_32b_1000.jsonl
```

### Next Check

Compute coverage and token statistics before scaling further:

- number of covered problems;
- zero-yield problems;
- mean/median/min/max token counts;
- number and rate of traces hitting the `1024` token cap.

### Coverage And Token Stats

```text
rows: 3537
covered problems: 947 / 1000
zero-yield problems: 53
mean traces per problem: 3.54
mean tokens: 538.3
median tokens: 497
min tokens: 157
max tokens: 1024
capped rows: 116
capped rate: 3.3%
```

### Interpretation

The 1000-problem GSM8K batch is healthy:

- coverage is high;
- average yield is close to the requested `4` samples per problem;
- token lengths are centered around `500` tokens;
- only a small fraction hit the `1024` cap.

This is good enough to scale to the full GSM8K train split before moving to compression.

## 2026-06-13 — Full GSM8K Teacher Batch Started

### Status

Started full GSM8K teacher-trace generation:

```text
problems: 7473
samples per problem: 4
requested generations: 29892
teacher: Qwen/Qwen3-32B
output: data/traces/gsm8k_qwen3_32b_full.jsonl
```

### Runtime Health Check

The vLLM logs indicate a healthy ROCm run:

- Qwen3-32B loaded from the local Hugging Face cache;
- vLLM selected ROCm attention backend;
- GPU KV cache was allocated;
- graph capture and warmup completed;
- generation progressed beyond 4000 processed samples.

The log still uses names like `cuda` and `CUDA graphs` because ROCm PyTorch exposes AMD GPUs through the PyTorch CUDA API. This is expected and does not mean NVIDIA/CUDA wheels are being used.

### Near-Completion Status

The full GSM8K teacher batch reached near completion:

```text
processed generations: 28648 / 29892
progress: 96%
elapsed time: 54m17s
estimated remaining time: 1m39s
throughput: 12.49 generations/s
```

The run remained healthy through the end of generation, with stable vLLM processing and no visible ROCm runtime errors. Final kept-row and token statistics should be recorded after the command returns.

### Completion Result

The full GSM8K teacher batch completed successfully.

```text
requested problems: 7473
samples per problem: 4
requested generations: 29892
verified rows kept: 26685
keep rate: 89.3%
elapsed generation time: 55m46s
output: data/traces/gsm8k_qwen3_32b_full.jsonl
```

Final coverage and token-distribution stats still need to be computed from the output JSONL.

### Final Coverage And Token Stats

```text
rows: 26685
covered problems: 7132 / 7473
zero-yield problems: 341
mean traces per problem: 3.57
mean tokens: 545.1
median tokens: 512
min tokens: 120
max tokens: 1024
capped rows: 644
capped rate: 2.4%
```

### Interpretation

The full GSM8K teacher dataset is healthy and ready for the compression stage:

- keep rate stayed consistent with the smaller runs;
- problem coverage is high at `95.4%`;
- median reasoning length is around `512` tokens;
- only `2.4%` of kept traces hit the token cap.

The 341 zero-yield problems can be revisited later with more samples or a different prompt, but they should not block the main compression/SFT pipeline.

### Experiment Status

The experiment is on track. The first major data milestone from the README is complete:

```text
Environment: done
Teacher traces: done for GSM8K train
Budgeted compression dataset: next
Student SFT: not started
Evaluation/Pareto curve: not started
Decorative-step analysis: not started
```

### Next Decision

Proceed to budgeted dataset construction before generating more teacher traces. The current GSM8K trace set is large enough to validate the compression and SFT pipeline.

Recommended next step:

1. Build `L0` rows directly from `long_cot`.
2. Implement rewrite-to-budget compression for `L1`, `L2`, and `L3`.
3. Re-verify every compressed trace before it enters training.
4. Train a first Qwen3-4B LoRA run on a controlled subset before scaling full SFT.

Why not generate more data immediately:

- GSM8K coverage is already high enough for the first training pass.
- Compression and re-verification are now the main methodological risk.
- The project thesis depends on comparing budget levels, not just collecting more long traces.

## 2026-06-13 — Compression Stage Implementation

### Decision

Move from teacher-trace collection to budgeted dataset construction.

The compression command now supports:

- `L0` pass-through rows from full teacher traces;
- vLLM rewrite-to-budget compression for `L1`, `L2`, and `L3`;
- answer re-verification before keeping compressed rows;
- preservation of `problem_index`, `sample_index`, source, original token count, target token count, and compressed token count.

### Budget Meanings

```text
L0: 100% of the teacher trace
L1: ~60% of the teacher trace
L2: ~35% of the teacher trace
L3: ~15% of the teacher trace
```

### Compression Invariant

Every compressed row must still verify against the gold answer. If a rewrite changes or loses the answer, it is dropped.

### Next Run

Build the full `L0` budgeted dataset first. Then run rewrite compression for `L1-L3` on a controlled subset before scaling the full compression job, because compression quality and re-verification rate are now the main risks.

## 2026-06-13 — Budgeted Compression 2000-Trace Tranche

### Result

The first controlled compression tranche completed successfully.

```text
input teacher traces: 2000
budgets requested: L0, L1, L2, L3
maximum possible rows: 8000
rows written: 7998
output: data/budgeted/gsm8k_qwen3_32b_budgeted_2000.jsonl
```

### Interpretation

This is a very strong compression-pipeline result:

- `L0` pass-through worked.
- vLLM rewrite compression completed for `L1`, `L2`, and `L3`.
- Re-verification dropped at most two rows from the full `8000` possible budgeted examples.
- The next check is not correctness yield, which looks healthy, but length adherence: whether `L1`, `L2`, and `L3` actually got close to their target token budgets.

### Next Check

Compute rows per budget, mean compression ratio, and target adherence before scaling compression to the full GSM8K trace set.

### Budget Adherence Check

```text
total rows: 7998

L0:
  rows: 2000
  mean tokens: 557.4
  median tokens: 521
  mean target: 557.4
  mean ratio: 1.000
  median ratio: 1.000

L1:
  rows: 2000
  mean tokens: 97.7
  median tokens: 91
  mean target: 334.4
  mean ratio: 0.186
  median ratio: 0.181

L2:
  rows: 2000
  mean tokens: 91.0
  median tokens: 84.5
  mean target: 195.1
  mean ratio: 0.172
  median ratio: 0.164

L3:
  rows: 1998
  mean tokens: 87.0
  median tokens: 81
  mean target: 83.6
  mean ratio: 0.165
  median ratio: 0.160
```

### Interpretation

Correctness/re-verification is excellent, but budget adherence is not yet acceptable.

The compression model over-compressed `L1` and `L2`; all compressed budgets collapsed near the `L3` length range. That would weaken the central experiment because the student would not learn a clean length-control knob. The full compression job should not be scaled until `L1`, `L2`, and `L3` are better separated.

### Required Fix

Change the rewrite prompt from a simple "maximum token budget" instruction to a target-range instruction:

- `L1`: moderate compression, preserve most reasoning detail;
- `L2`: compact reasoning, preserve essential steps;
- `L3`: minimal reasoning, only answer-determining computations;
- ask the rewriter to target a range around the budget, not just stay below a maximum.

### Prompt Iteration Result

A second 500-trace compression run with target ranges improved separation:

```text
L0 mean ratio: 1.000
L1 mean ratio: 0.399
L2 mean ratio: 0.206
L3 mean ratio: 0.174
rows kept: 2000 / 2000
```

This is better than the collapsed first run, but `L1` and `L2` are still too short relative to the intended `0.60` and `0.35` targets. Correctness/re-verification is excellent; length adherence remains the active issue.

### Second Prompt Adjustment

Strengthen only `L1` and `L2`:

- `L1`: light-to-moderate compression; preserve original solution structure and most explanatory detail;
- `L2`: medium compression; keep every answer-determining step and brief explanations;
- add an explicit warning not to go below the lower end of the target token range.

Keep `L3` largely unchanged because it is already near the target.

### Third Prompt Iteration Result

The next 500-trace compression run produced:

```text
rows kept: 1991 / 2000

L0 mean ratio: 1.000
L1 mean ratio: 0.667
L2 mean ratio: 0.488
L3 mean ratio: 0.264
```

This fixed `L1`: it is now close to the intended moderate-compression budget. However, the stronger warning against short outputs overcorrected `L2` and `L3`; both are now too long relative to their intended `0.35` and `0.15` targets.

### Next Prompt Adjustment

Keep the current `L1` behavior. Make `L2` and `L3` stricter:

- `L2`: target compact reasoning around `0.35`; allow concise equations plus brief explanations.
- `L3`: target very short reasoning around `0.15`; do not require full explanatory prose.
- apply the lower-bound warning only to `L1`, not to all budgets.

### Final Compression Policy Before Scaling

We will stop running calibration tranches and make the next compression run the full run.

Final prompt policy:

- `L1`: keep the successful light-to-moderate compression behavior and retain the lower-bound warning.
- `L2`: make phrasing compact, with one short sentence per step and no full worked explanation.
- `L3`: prefer equations and terse fragments; avoid full sentences when equations are enough.
- apply lower-bound pressure only to `L1`;
- for `L2` and `L3`, prioritize staying at or below the upper end of the target range;
- reduce generation slack from target upper bound +128 tokens to +48 tokens.

This accepts practical budget tolerance rather than spending more GPU time trying to make the rewriter hit exact ratios.

## 2026-06-13 — Full Budgeted Compression Result

### Result

The full GSM8K budgeted compression run completed successfully.

```text
teacher traces: 26685
requested budgeted rows: 106740
actual budgeted rows: 106601
dropped rows: 139
output: data/budgeted/gsm8k_qwen3_32b_budgeted_full.jsonl
```

Rows by budget:

```text
L0 rows: 26685
L1 rows: 26599
L2 rows: 26676
L3 rows: 26641
```

Length ratios:

```text
L0 mean ratio: 1.000
L1 mean ratio: 0.653
L2 mean ratio: 0.179
L3 mean ratio: 0.137
```

Token lengths:

```text
L0 mean tokens: 545.1, median: 512
L1 mean tokens: 346.6, median: 333
L2 mean tokens: 91.1, median: 86
L3 mean tokens: 71.2, median: 66
```

### Interpretation

Compression is complete and usable for SFT.

The final budget ladder is:

```text
L0: full reasoning
L1: medium reasoning
L2: short reasoning
L3: very short reasoning
```

`L2` is more aggressive than the originally intended `0.35` target, but it remains clearly separated from both `L1` and `L3`. This is acceptable for the first student training run. If later evaluation shows `L2` and `L3` collapse in behavior or accuracy, regenerate only `L2` with a revised prompt.

### Next Step

Convert the budgeted JSONL into SFT training rows, then train the Qwen3-4B student with budget-conditioned prompts.

## 2026-06-13 — SFT Dataset And Student Setup

### SFT Dataset

Converted the full budgeted JSONL into SFT-format JSONL.

```text
input: data/budgeted/gsm8k_qwen3_32b_budgeted_full.jsonl
output: data/budgeted/gsm8k_qwen3_32b_sft_full.jsonl
rows: 106601
```

Each SFT row contains a `text` field with:

```text
<|user|>
Solve the problem. Reasoning budget: <budget=Lx>
Problem: ...
<|assistant|>
<think>...</think>
\boxed{...}
```

### Student Model

Cached the student model on the Hot Aisle VM:

```text
student: Qwen/Qwen3-4B
```

### Experiment Tracking

Installed and logged into Weights & Biases on the VM. Use W&B for SFT training and evaluation metrics.

### Training Plan

Run a small LoRA SFT subset first to validate:

- Qwen3-4B loads correctly on ROCm;
- LoRA adapters attach to Qwen modules;
- SFT JSONL is read correctly;
- loss logs to W&B;
- checkpoints save.

If the subset run is healthy, proceed to the full SFT run.

## 2026-06-13 — Qwen3-4B SFT Subset Run

### Run

Started a small LoRA SFT validation run:

```text
model: Qwen/Qwen3-4B
train rows used: 4000
max steps: 50
batch size: 2
gradient accumulation: 8
run name: qwen3-4b-budget-sft-subset-50
tracking: Weights & Biases
```

### W&B Observations

The W&B curves look healthy:

- training loss decreases sharply over the 50 steps;
- mean token accuracy increases;
- learning rate follows the expected schedule;
- gradient norm remains finite;
- no NaN/instability is visible.

This validates the core SFT path: Qwen3-4B loads on ROCm, the SFT JSONL is readable, W&B logging works, and the model can optimize on the budget-conditioned data.

### Next Check

After the run finishes, verify:

- checkpoint exists under `ckpts/concise-cot-qwen3-4b-subset`;
- adapter files are saved;
- a tiny generation sanity check produces the expected `<think>...</think>` plus `\boxed{...}` format for at least two budgets.

If those checks pass, proceed to the full SFT run.

### Completion Result

The 50-step subset run completed successfully.

```text
final train loss: 0.465
mean token accuracy: 0.8706
train runtime: 166.9s
checkpoint: ckpts/concise-cot-qwen3-4b-subset/checkpoint-50
```

W&B run:

```text
qwen3-4b-budget-sft-subset-50
```

The subset run validates that the Qwen3-4B LoRA SFT path works on the Hot Aisle ROCm VM. Next step is adapter generation sanity checking before full SFT.

### Adapter Generation Sanity Check

Tested `checkpoint-50` on the Natalia GSM8K example with `L0` and `L3`.

Observed:

- both outputs used the expected `<think>...</think>` and `\boxed{72}` format;
- both outputs produced the correct answer;
- `L3` was not shorter than `L0` yet and included extra self-checking.

Interpretation:

The 50-step subset checkpoint validates the training and generation plumbing, but it is too early to expect strong budget control. The full/longer SFT run is needed before judging length adherence.

Proceed to full SFT, then evaluate budget adherence systematically on held-out prompts.

## 2026-06-13 — Full-Data 1000-Step SFT Run

### Status

Started the first full-data SFT run:

```text
model: Qwen/Qwen3-4B
dataset: data/budgeted/gsm8k_qwen3_32b_sft_full.jsonl
train rows: 106601
max steps: 1000
effective batch size: 16
run name: qwen3-4b-budget-sft-full-1000
```

At about `457/1000` steps, the run is healthy:

```text
loss: ~0.29-0.35
mean token accuracy: ~0.89-0.91
grad norm: finite
training speed: ~2.7s/step
```

### W&B Continuation Plan

If the 1000-step checkpoint is still improving and budget behavior needs more training, continue from `checkpoint-1000` to `max_steps=3000`.

To keep W&B as one continuous graph, resume the same W&B run by setting:

```text
WANDB_RUN_ID=<existing run id>
WANDB_RESUME=allow
```

Training code now supports:

```text
--resume-from-checkpoint
--wandb-run-id
--wandb-resume
```

This lets a `1000 -> 3000` continuation appear on the same W&B run instead of creating a disconnected chart.

### 1000-Step Run Result

The 1000-step full-data run completed successfully.

W&B observations:

- loss dropped quickly and then flattened at a low value;
- mean token accuracy rose and stabilized around roughly `0.91`;
- gradient norm remained stable after the initial steps;
- learning-rate schedule behaved as expected;
- no visible instability or NaNs.

Interpretation:

The model has learned the supervised format and is optimizing well on the full budgeted SFT dataset. Before extending to 3000 steps, run generation checks from `checkpoint-1000` for `L0/L1/L2/L3`. If budget adherence is clearly improving, either evaluate this checkpoint or continue training based on the sample quality.

### Checkpoint-1000 Generation Sanity Check

Tested `checkpoint-1000` on the Natalia GSM8K example for all budgets.

Observed:

- all budgets produced the correct final answer `\boxed{72}`;
- all budgets used the expected `<think>...</think>` plus boxed-answer format;
- output length decreased clearly by budget:
  - `L0`: full explanatory reasoning;
  - `L1`: shorter worked explanation;
  - `L2`: compact equation-style reasoning;
  - `L3`: very terse reasoning.

Interpretation:

Budget conditioning has emerged by `checkpoint-1000`. This checkpoint is ready for systematic GSM8K test evaluation before deciding whether to continue training to 3000 steps.

## 2026-06-13 — Evaluation Step Prepared

### Goal

Evaluate `checkpoint-1000` on GSM8K test across budgets:

```text
L0, L1, L2, L3
```

Metrics:

- accuracy;
- mean generated tokens;
- raw outputs for inspection.

### Implementation

Added adapter-based evaluation support to `eval.py`. It can now:

- load Qwen3-4B plus a LoRA adapter;
- generate GSM8K test completions per budget;
- write JSONL outputs;
- summarize accuracy and generated-token counts.

Start with a small held-out eval subset before running the full GSM8K test set.

### Eval Runner Fix

The first full eval attempt was interrupted because generation appeared stuck. The issue was the eval runner, not the checkpoint:

- it generated one prompt at a time;
- it did not print progress;
- it could overwrite the output JSONL with the final summary when `--output` was used.

Updated eval to:

- batch prompts with left padding;
- print progress every N generated examples;
- keep raw JSONL output and summary output separate.

This makes the main GSM8K eval observable and avoids corrupting the output file.

### Eval Backend Correction

The eval pipeline must use vLLM for adapter inference on the MI300X. The Hugging Face `generate` path is now only a fallback backend.

Why this matters:

- the full GSM8K eval is 1,319 test problems x 4 budgets = 5,276 generations;
- vanilla Transformers generation was too slow for this workload;
- vLLM is the intended inference engine for this project and should handle the adapter eval with much better throughput.

Code changes:

- `eval.backend` now defaults to `vllm` in `configs/default.yaml`;
- `eval.py` supports `--backend vllm` with vLLM LoRA adapter loading;
- vLLM uses the configured student model and the configured training LoRA rank;
- raw JSONL output and summary output remain separate;
- resumable eval remains available, but the clean main eval should use a fresh vLLM output file rather than mixing partial HF generations with vLLM generations.

Main eval command pattern:

```bash
python -m concise_cot.eval \
  --config configs/default.yaml \
  --adapter ckpts/concise-cot-qwen3-4b-full/checkpoint-1000 \
  --source gsm8k \
  --budgets L0 L1 L2 L3 \
  --output outputs/gsm8k_checkpoint1000_eval_full_vllm.jsonl \
  --summary-output outputs/gsm8k_checkpoint1000_eval_full_vllm.summary.json
```

### Main GSM8K Eval Result: checkpoint-1000

The full GSM8K test eval completed with vLLM in minutes after the Hugging Face eval path proved too slow.

Run:

- student: `Qwen/Qwen3-4B` + `ckpts/concise-cot-qwen3-4b-full/checkpoint-1000`
- backend: vLLM LoRA inference
- dataset: GSM8K test
- budgets: `L0`, `L1`, `L2`, `L3`
- total generations: 1,319 problems x 4 budgets = 5,276

Results:

- `L0`: accuracy 0.7551, mean generated tokens 389.0
- `L1`: accuracy 0.8506, mean generated tokens 350.6
- `L2`: accuracy 0.8878, mean generated tokens 99.0
- `L3`: accuracy 0.8484, mean generated tokens 79.7

Audit:

- The first `L0` eval used `max_new_tokens=512`.
- Audit samples showed many `L0` generations hitting the cap and stopping mid-reasoning.
- That made `L0` artificially weak and not a fair full-budget baseline.

Fixed result:

- Re-ran only `L0` with `max_new_tokens=2048`.
- Combined fixed `L0` with the existing vLLM `L1/L2/L3` rows.
- Logged the corrected run to W&B as `gsm8k-checkpoint1000-vllm-fixed-eval`.

Corrected GSM8K metrics:

- `L0`: accuracy 0.9356, mean generated tokens 496.7
- `L1`: accuracy 0.8506, mean generated tokens 350.6
- `L2`: accuracy 0.8878, mean generated tokens 99.0
- `L3`: accuracy 0.8484, mean generated tokens 79.7

Corrected interpretation:

- Budget conditioning worked: token usage dropped sharply from `L0/L1` to `L2/L3`.
- `L0` is now the accuracy ceiling, as expected for full reasoning.
- `L2` is the best compressed Pareto point: ~88.8% accuracy at ~99 tokens.
- `L3` gives the strongest compression but loses some accuracy relative to `L2`.
- The result supports the project direction: strong compression preserves much of GSM8K accuracy, but full reasoning still wins when token cost is unconstrained.

### RQ2 Structural Removal Analysis

Ran the first RQ2 analysis on `data/budgeted/gsm8k_qwen3_32b_budgeted_full.jsonl`.

Goal: test whether budget compression removes prose-like/decorative reasoning more aggressively than calculation-like/load-bearing reasoning.

Results:

- `L1`: mean token ratio 0.653, mean step ratio 0.589
  - removed calc-like step rate: 0.538
  - removed prose-like step rate: 0.825
  - prose removed more than calc: true
- `L2`: mean token ratio 0.179, mean step ratio 0.192
  - removed calc-like step rate: 0.697
  - removed prose-like step rate: 0.970
  - prose removed more than calc: true
- `L3`: mean token ratio 0.137, mean step ratio 0.102
  - removed calc-like step rate: 0.892
  - removed prose-like step rate: 0.995
  - prose removed more than calc: true

Interpretation:

- The structural RQ2 signal supports decorative-first compression.
- Compression removes prose-like steps much more aggressively than calculation-like steps at all budgets.
- The signal weakens at `L3` because very aggressive compression also removes many calculation-like steps.
- This is structural evidence, not final causal evidence. A small causal validation sample is the next rigorous step if time permits.

Artifacts:

- `outputs/rq2_structural_summary.csv`: report-ready RQ2 summary table.
- `outputs/rq2_causal_validation_sample.jsonl`: 50 examples for manual/small causal validation, each containing one removed prose-like step and one removed calculation-like step.
- `outputs/plots/gsm8k_pareto.csv`: report-ready GSM8K accuracy-vs-token table.
- `outputs/plots/gsm8k_pareto.md`: Markdown version of the GSM8K Pareto table.

The artifact generation is analysis-only. It does not rerun the model or change the checkpoint.

## Experiment Tracking Plan

Use Weights & Biases for training and evaluation, not for every teacher-generation row.

Recommended W&B usage:

- log SFT training loss, learning rate, epoch, and checkpoint metadata;
- log evaluation metrics per budget: accuracy, mean generated tokens, length adherence;
- log Pareto-curve tables and plots;
- optionally log aggregate teacher-trace statistics after each dataset build.

Do not stream raw teacher traces to W&B by default. The JSONL files should remain local artifacts because they can be large and contain full model outputs.
