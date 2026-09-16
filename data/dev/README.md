# Development data

This is an inexpensive **synthetic mechanics dataset**, not a representative agent benchmark or evidence about a pretrained MoE. Frozen deterministic specialists calculate answers from public problem text and previously delivered outputs. No grading labels or private task metadata are serialized into controller or specialist inputs.

`tasks.jsonl` contains explicit task-ID splits. The configured development generation uses 36 training tasks and 18 held-out tasks. IDs hash the complete public text. Public operands and string inputs differ across splits; neither IDs nor prompts overlap. Training families are arithmetic (`math`), reference lookup (`retrieval`), and bounded string operations (`code`). Held-out examples include those families and unseen dependency families (`composed_retrieval_math`, `composed_math_string`). The latter require information to pass between specialists. These are template-family shifts, not claims of broad language-task generalization.

Public templates:

- Compute an addition expression.
- Look up a named numeric value in a supplied JSON reference.
- Reverse a supplied string or count its vowels.
- Retrieve a value and multiply it by a supplied factor (held-out family).
- Calculate a sum and reverse its digits (held-out family).

`trajectories.jsonl` records actual executions under all-agent, rule-based, and random top-k policies, including successes and failures. Every state is an immutable snapshot. Final answers come from specialist outputs; a separate exact grader accesses the private expected answer afterward. Deterministic-agent token counts are **whitespace estimates of serialized input plus output**, explicitly identified in output metadata. Wall-clock latencies use actual `perf_counter` measurements. Free deterministic local computation has zero configured estimated dollar cost; it is not a measured commercial inference price.

`communication_graph` records controller-to-agent requests and agent-to-controller responses, plus logical directed edges where a previous agent's output is forwarded in a later state. Forwarding edges are dependencies within a serialized request, not additional physical RPCs. Own-agent history does not count as inter-agent communication. New trajectories separately report `controller_agent_messages` and `logical_inter_agent_edges` in metadata. Request byte sizes measure serialized public state payloads (they exclude backend-specific system prompts, API envelopes, and protocol overhead); logical forwarding bytes must not be added to request bytes to estimate physical network traffic. Parallel agents see the same pre-execution state. Sequential agents receive earlier outputs from the same round.

`sft.jsonl` includes successful **training-split** executions only. Decisions exceeding the sparse coordination k limit are excluded and counted in generation metrics; they are never silently projected into sparse targets. Terminal decisions are included. Configurable budgets clip activations, while usage retains measured/estimated actual tokens and reports indivisible-invocation overruns.

`preferences.jsonl` compares alternative **first decisions at exactly the same canonical initial state**. Every counterfactual then uses the identical rule-based continuation policy, with routing reuse disabled. Candidates include each individual specialist, termination, and a seeded, bounded sample of feasible parallel subsets and ordered sequential selections. Candidate selection does not inspect task types or grading labels. Only successful chosen executions with strictly higher configured success/cost reward are retained; chosen quality cannot be below rejected quality. Both alternatives include measured execution metrics, final answers, and subsequent decisions. All preference records belong to the training split. Short deterministic latency differences are noisy and should not support inference claims; preference weights/scales are design choices.

The generation configuration and actual counts/runtime/hardware/Git provenance are stored at `outputs/generation/dev/run.json` and `outputs/generation/dev/metrics.json`. Reproduce with:

```bash
python -m conductor.generate --config configs/generation.yaml
```

The task inventory is deterministic for its seed/configuration. Measured latency fields vary between runs. Larger language-model experiments must regenerate trajectories using configured frozen model specialists and a genuine pretrained MoE controller.
