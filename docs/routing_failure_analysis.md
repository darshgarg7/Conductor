# Routing failures: original pilot and completed repair

Conductor has completed real pretrained MoE SFT and DPO. It has not demonstrated
that learned coordination beats strong baselines. The repair recovers familiar
single-agent tasks but fails every unseen composition. This distinction is the
starting point for the next experiment.

## What was measured

| Policy | Original six-task pilot | Repair's 48-task inventory |
| --- | --- | --- |
| Original SFT | 2/6 | 16/48 |
| Original DPO | 2/6 | 16/48 |
| Frozen-backbone fitted head | Not measured | 24/48 |
| Repaired LoRA SFT | Not measured | 32/48 |
| Repaired DPO | Not measured | 32/48 |
| Rule-based router | 6/6 | 48/48 |
| Random top-k | 4/6 | 39/48 |

The old and repaired checkpoints are evaluated on the **same 48 tasks** in the
48-task column. Comparing 2/6 with 32/48 alone would mix a model change with a
different inventory. [Original measurements](../results/granite-pilot/report.md)
and [repair measurements](../results/routing-repair/report.md) retain raw traces,
partitions, hardware, source commits and checksums. Both studies use CPU FP32
and eight frozen deterministic specialists, rather than downstream LLMs.

In the original pilot, both learned controllers always call coder and then
stop. In the repair, both solve **32/32 seen-template tasks and 0/16 unseen
compositions**, averaging one agent call per task. Their improved familiar-task
performance is not learned dependency execution. DPO's validation pair-ranking
accuracy rises from 92.6% in the original study to 96.7% in the repair; it yields
no additional online success or reduction in agent calls over repaired SFT.
These are different validation corpora, so the ranking percentages are not a
controlled improvement comparison.

## Diagnosed defects and remaining limits

| Observation | Evidence and interpretation | Next experiment |
| --- | --- | --- |
| Critical progress was truncated | All 48 later original training states lose the four critical progress keys under legacy token allocation. The tokenizer-only audit establishes missing inputs, not a unique cause of collapse. | Preserve and audit progress plus relevant evidence in initial, intermediate and failure states. |
| Successful rollouts gave contradictory targets | All 12 original initial states have conflicting actions among ultimately successful trajectories. A successful final answer does not make every preceding call useful. | Compare state-matched actions and measured continuations; retain failures and ambiguous labels. |
| Intermediate supervision was incomplete | Original DPO has no step-one states. Repair SFT covers steps zero/one/two with 36/108/36 labels. | Include handoffs, incomplete outputs, tool errors, recovery, revisits and justified stopping. |
| Coordination actions are absent | Repair labels are 127 stop, 27 coder, 14 retriever and 12 math; every continuation selects one agent. | Require demonstrated multi-agent dependencies and audit chosen count, order and mode. Factorization alone cannot create missing labels. |
| Validation is unbalanced | The nine validation tasks are three lookup, three reverse-string and three vowel-count tasks. No math task or math-agent target is present. | Group-disjoint, family-stratified train/development partitions; disclose every coverage cell before fitting. |
| Cheap routing already solves the grammar | Rules solve 48/48; random routing solves 39/48. A sequential dense control solves 40/48. | Use public-state rules, linear and MLP routers, reliable prompted supervision and dependency-capable dense controls. |
| Offline ranking is insufficient | DPO's 96.7% pair accuracy coexists with 0/16 composition success and unchanged calls. | Collect SFT's on-policy mistakes on training/development data and require online quality plus efficiency gates. |

The repair changes serialization, context length, labels, data volume and head
initialization together. It does not identify the causal contribution of each
change. Pooling/normalization probes are narrower representation ablations.
Single-seed results and related synthetic templates cannot establish general
customer-task accuracy, preference-training benefit or the need for a MoE.

## What happens next

The [coordination v2 protocol](coordination_v2_protocol.md) commits three seeds,
strong cheap controls, quality/call acceptance rules and a fresh structural-OOD
test. The [factorized action design](factorized_routing.md) defines normalized
complete-decision probabilities and prerequisite label audits. These are plans;
the factorized controller and v2 experiments have not been implemented or run.

The inspected six- and 48-task inventories are development evidence now. They
cannot become fresh tests after further tuning. An unfavorable locked result
must remain published. CPU timings, estimated tool tokens and fewer calls do
not establish dollar savings or NVIDIA performance. A successful research run
also does not establish deployment or customer-support acceptance.
