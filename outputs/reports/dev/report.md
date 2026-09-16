# Measured experiment analysis



## Absolute heldout quality and cost



| Policy | Success | Calls | Tokens | Wall s | p95 s | Controller token share | Controller latency share | Configured cost USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_agent | 0.6667 | 8.0000 | 286.3333 | 0.0005 | 0.0006 | 0.0000 | 0.1085 | 0.0000 |
| Tiny-MoE-Init | 0.5000 | 6.0000 | 2356.0000 | 0.0064 | 0.0066 | 0.5671 | 0.8764 | 0.0000 |
| conductor_preference | 0.7222 | 2.5556 | 762.2778 | 0.0032 | 0.0038 | 0.8002 | 0.8813 | 0.0000 |
| conductor_sft | 0.8333 | 1.8333 | 589.1667 | 0.0024 | 0.0056 | 0.8567 | 0.8972 | 0.0000 |
| random_top_k | 0.8889 | 6.0000 | 668.5556 | 0.0008 | 0.0009 | 0.0000 | 0.0802 | 0.0000 |
| rule_based | 1.0000 | 1.3333 | 53.6667 | 0.0002 | 0.0003 | 0.0000 | 0.1510 | 0.0000 |

Tokens use backend accounting; tiny controller counts are lexical proxies. Monetary cost uses configured rates; zero rates do not establish free inference. Controller cost share is unavailable at zero total cost.

## Paired differences (candidate minus baseline)



conductor_sft versus all_agent: 18 paired tasks; success-rate change +0.167; mean token change +302.8. These measurements describe this task set and configured backends only.

conductor_preference versus all_agent: 18 paired tasks; success-rate change +0.056; mean token change +475.9. These measurements describe this task set and configured backends only.

conductor_sft versus rule_based: 18 paired tasks; success-rate change -0.167; mean token change +535.5. These measurements describe this task set and configured backends only.

conductor_preference versus rule_based: 18 paired tasks; success-rate change -0.278; mean token change +708.6. These measurements describe this task set and configured backends only.

conductor_sft versus random_top_k: 18 paired tasks; success-rate change -0.056; mean token change -79.4. These measurements describe this task set and configured backends only.

conductor_preference versus random_top_k: 18 paired tasks; success-rate change -0.167; mean token change +93.7. These measurements describe this task set and configured backends only.

conductor_sft versus static_supervisor: unanswered (no paired trajectories).

conductor_preference versus static_supervisor: unanswered (no paired trajectories).

conductor_sft versus base_moe: 18 paired tasks; success-rate change +0.333; mean token change -1766.8. These measurements describe this task set and configured backends only.

conductor_preference versus base_moe: 18 paired tasks; success-rate change +0.222; mean token change -1593.7. These measurements describe this task set and configured backends only.

conductor_preference versus conductor_sft: 18 paired tasks; success-rate change -0.111; mean token change +173.1. These measurements describe this task set and configured backends only.

Internal expert comparisons use identical heldout initial states; task-load correlations remain descriptive and do not establish semantic specialization.

Routing ablation k2_interval1 versus k1_interval1: 5 paired actual heldout runner tasks; success-rate change +0.000, mean call change +0.00, mean wall-time change -0.000140 seconds.

Routing ablation k2_interval2 versus k1_interval2: 5 paired actual heldout runner tasks; success-rate change +0.200, mean call change +0.80, mean wall-time change +0.000092 seconds.

Routing ablation k1_interval2 versus k1_interval1: 5 paired actual heldout runner tasks; success-rate change +0.000, mean call change +0.40, mean wall-time change -0.000516 seconds.

Routing ablation k2_interval2 versus k2_interval1: 5 paired actual heldout runner tasks; success-rate change +0.200, mean call change +1.20, mean wall-time change -0.000283 seconds.

## Task categories



| Policy | Category | Split | Generalization | Success | Calls |
| --- | --- | --- | --- | --- | --- |
| all_agent | code | eval | seen_family | 1.0000 | 8.0000 |
| all_agent | composed_math_string | eval | unseen_composition | 0.0000 | 8.0000 |
| all_agent | composed_retrieval_math | eval | unseen_composition | 0.0000 | 8.0000 |
| all_agent | math | eval | seen_family | 1.0000 | 8.0000 |
| all_agent | retrieval | eval | seen_family | 1.0000 | 8.0000 |
| base_moe | code | eval | seen_family | 1.0000 | 6.0000 |
| base_moe | composed_math_string | eval | unseen_composition | 0.0000 | 6.0000 |
| base_moe | composed_retrieval_math | eval | unseen_composition | 0.0000 | 6.0000 |
| base_moe | math | eval | seen_family | 1.0000 | 6.0000 |
| base_moe | retrieval | eval | seen_family | 0.0000 | 6.0000 |
| conductor_preference | code | eval | seen_family | 1.0000 | 3.0000 |
| conductor_preference | composed_math_string | eval | unseen_composition | 0.0000 | 3.0000 |
| conductor_preference | composed_retrieval_math | eval | unseen_composition | 0.3333 | 2.3333 |
| conductor_preference | math | eval | seen_family | 1.0000 | 3.0000 |
| conductor_preference | retrieval | eval | seen_family | 1.0000 | 1.0000 |
| conductor_sft | code | eval | seen_family | 1.0000 | 1.0000 |
| conductor_sft | composed_math_string | eval | unseen_composition | 0.0000 | 1.0000 |
| conductor_sft | composed_retrieval_math | eval | unseen_composition | 1.0000 | 6.0000 |
| conductor_sft | math | eval | seen_family | 1.0000 | 1.0000 |
| conductor_sft | retrieval | eval | seen_family | 1.0000 | 1.0000 |
| random_top_k | code | eval | seen_family | 1.0000 | 6.0000 |
| random_top_k | composed_math_string | eval | unseen_composition | 0.6667 | 6.0000 |
| random_top_k | composed_retrieval_math | eval | unseen_composition | 0.6667 | 6.0000 |
| random_top_k | math | eval | seen_family | 1.0000 | 6.0000 |
| random_top_k | retrieval | eval | seen_family | 1.0000 | 6.0000 |
| rule_based | code | eval | seen_family | 1.0000 | 1.0000 |
| rule_based | composed_math_string | eval | unseen_composition | 1.0000 | 2.0000 |
| rule_based | composed_retrieval_math | eval | unseen_composition | 1.0000 | 2.0000 |
| rule_based | math | eval | seen_family | 1.0000 | 1.0000 |
| rule_based | retrieval | eval | seen_family | 1.0000 | 1.0000 |

## Seen families and unseen compositions



| Policy | Group | Split | Tasks | Success | Calls |
| --- | --- | --- | --- | --- | --- |
| all_agent | seen_family | eval | 12 | 1.0000 | 8.0000 |
| all_agent | unseen_composition | eval | 6 | 0.0000 | 8.0000 |
| base_moe | seen_family | eval | 12 | 0.7500 | 6.0000 |
| base_moe | unseen_composition | eval | 6 | 0.0000 | 6.0000 |
| conductor_preference | seen_family | eval | 12 | 1.0000 | 2.5000 |
| conductor_preference | unseen_composition | eval | 6 | 0.1667 | 2.6667 |
| conductor_sft | seen_family | eval | 12 | 1.0000 | 1.0000 |
| conductor_sft | unseen_composition | eval | 6 | 0.5000 | 3.5000 |
| random_top_k | seen_family | eval | 12 | 1.0000 | 6.0000 |
| random_top_k | unseen_composition | eval | 6 | 0.6667 | 6.0000 |
| rule_based | seen_family | eval | 12 | 1.0000 | 1.0000 |
| rule_based | unseen_composition | eval | 6 | 1.0000 | 2.0000 |

Unseen compositions are separately heldout task families; results do not establish broader language-model generalization.

## Internal experts on fixed initial states



| Policy | Layer | Utilized fraction | Normalized entropy | Load CV |
| --- | --- | --- | --- | --- |
| base_moe | 0 | 0.7500 | 0.9998 | 0.8089 |
| conductor_sft | 0 | 0.5000 | 0.8546 | 1.0000 |
| conductor_preference | 0 | 1.0000 | 0.1693 | 0.8425 |

Initial-state probe hashes match across stages: True. Rollout statistics reflect different visited states and are not a controlled before/after comparison.

## Scope and missing evidence



Development deterministic agents do not establish real language-model coordination quality.

Random-initialized tiny models are not pretrained MoE baselines.

No GPU cluster performance claim follows from CPU/local measurements.

Expert task-load divergence and routing ablations: analysis.json. Top-k settings: {"all_agent": [2], "rule_based": [2], "random_top_k": [2], "base_moe": [2], "conductor_sft": [2], "conductor_preference": [2]}.

## Interpretation of this development run

SFT reached 83.3% success on 18 heldout synthetic tasks. Preference training reached 72.2%, with 2.56 mean calls versus 1.83 for SFT. This experiment does not support the claim that the current preference objective/data reduce unnecessary activation relative to SFT. Preferences cover initial states only; collecting counterfactual late-state and stopping decisions is a concrete next experiment, not an established explanation of this result.

The deterministic rule baseline reached 100.0%. Fewer specialist calls did not yield lower total latency than All-Agent in this cheap fixture: SFT averaged 0.002424s versus 0.000495s for All-Agent. Controller processing dominates these tiny specialist workloads. Token totals combine explicitly documented backend proxies; realistic language-model economics require tokenizer-accounted frozen LLM specialists and a configured monetary cost model.

## Matched inference study and reproduction

Actual float32, bfloat16 and warm feature-cache comparisons are archived in [the optimization report](artifacts/optimizations/report.md), with request timings, hardware, compute dtype, quality observations and machine-readable ratios. Results concern this tiny CPU controller and repeated states. They do not establish pretrained transformer KV-cache, GPU kernel, or general changing-state speedups.

All raw run artifacts are in [artifacts/](artifacts/README.md), and source/hardware/runtime references are in [provenance.json](provenance.json). Run individual SFT and preference commands on the versioned `data/dev/` files to preserve the recorded training inputs. The complete `run_dev.sh` command generates a new dataset/experiment; measured trajectory latencies and timing-sensitive preference labels can differ even with the same task-generation seed.

The full pretrained OLMoE/HPC experiment and a frozen prompting supervisor have not been run in this development study. Only the HF/PEFT backend integration was verified on locally created random tiny OLMoE weights. These results justify pipeline-validation claims, not pretrained MoE post-training effectiveness claims.
