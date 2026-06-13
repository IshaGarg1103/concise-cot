# Concise-CoT: Length-Controlled Reasoning Distillation

## Summary

This project distills budget-controlled reasoning from `Qwen/Qwen3-32B` into a `Qwen/Qwen3-4B` student. The student is trained to respond to explicit reasoning budgets (`L0`, `L1`, `L2`, `L3`) and trade answer accuracy against generated reasoning length.

The final GSM8K evaluation supports the main project direction: strong compression preserves much of the full-reasoning accuracy, and the budget token controls output length.

## Method

- Teacher: `Qwen/Qwen3-32B`.
- Student: `Qwen/Qwen3-4B` with bf16 LoRA SFT.
- Data: GSM8K teacher traces, then rewrite-to-budget compression into `L0/L1/L2/L3`.
- Inference backend: vLLM for teacher sampling, compression, and final adapter evaluation.
- Evaluation: GSM8K test, exact answer verification after extracting `\boxed{...}` / GSM8K-style answers.

## GSM8K Pareto Result

Final fixed evaluation used `checkpoint-1000` and corrected an initial `L0` truncation artifact by rerunning only `L0` with a larger generation cap.

| Budget | Accuracy | Mean generated tokens | Count |
|---|---:|---:|---:|
| L0 | 0.9356 | 496.7 | 1319 |
| L1 | 0.8506 | 350.6 | 1319 |
| L2 | 0.8878 | 99.0 | 1319 |
| L3 | 0.8484 | 79.7 | 1319 |

Interpretation:

- `L0` is the accuracy ceiling, as expected for full reasoning.
- `L2` is the strongest compressed Pareto point: 88.8% accuracy at about 99 generated tokens.
- `L3` compresses further but loses some accuracy relative to `L2`.
- The corrected result is coherent: full reasoning wins when token cost is unconstrained, while medium compression gives a strong accuracy/token tradeoff.

## RQ2: Decorative-First Compression

RQ2 asks whether compression preferentially removes decorative reasoning before load-bearing calculation. A structural removal analysis compared each compressed trace against its `L0` source and categorized removed steps as prose-like or calculation-like.

| Budget | Mean token ratio | Mean step ratio | Removed calc-like steps | Removed prose-like steps |
|---|---:|---:|---:|---:|
| L1 | 0.653 | 0.589 | 0.538 | 0.825 |
| L2 | 0.179 | 0.192 | 0.697 | 0.970 |
| L3 | 0.137 | 0.102 | 0.892 | 0.995 |

Interpretation:

- Prose-like steps are removed more aggressively than calculation-like steps at every budget.
- This supports the decorative-first compression hypothesis structurally.
- At `L3`, compression is so aggressive that many calculation-like steps are also removed, which explains the accuracy drop relative to `L2`.

## Limitations

- The RQ2 result is structural evidence, not full causal proof. A small causal validation sample should be inspected or scored separately before making a stronger causal claim.
- GSM8K is easier than MATH-500; a harder benchmark would test whether the same Pareto curve holds under more difficult reasoning.
- Length control should be evaluated with the corrected generation caps; under-capping `L0` can artificially depress full-budget accuracy.

## Conclusion

The experiment demonstrates a working concise-CoT pipeline: a 4B student learns budget-conditioned reasoning, vLLM enables practical full-set evaluation, and compression yields a meaningful accuracy/token Pareto curve. The best compressed point in this run is `L2`, which keeps most of the `L0` accuracy while using about one-fifth of the generated tokens.
