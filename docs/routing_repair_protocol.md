# Routing repair: a bounded CPU experiment

The first pilot completed SFT and DPO, but both policies called the coder and
then stopped on every held-out task. Lower loss and preference ranking accuracy
did not translate into useful coordination. This experiment tests a repair of
the input and supervision pipeline; it does not erase the original result.

## Problems observed before selecting the repair

- The legacy 128-token allocation loses step, budget, called agents, and routing
  history from all 48 later SFT states. Eighteen of 72 task/type headers exceed
  their fixed allocation, even though some complete initial states would fit.
- All twelve initial training states have conflicting successful-trajectory
  targets. An ultimately successful rollout can start with an unnecessary call.
- DPO covers initial and step-two states, but no step-one states. Its pairwise
  metric can improve while an unexamined third action wins the full catalog.
- Six held-out tasks provide little evidence about template generalization.

The tokenizer audit diagnoses lost information. It does not establish that
truncation, pooling, or supervision alone caused collapse.

## Fixed protocol

The committed generation configuration locks seed 314159, 36 training tasks,
48 held-out tasks, three trajectory policies, and common execution budgets.
The inventory and its SHA-256 are written before representation selection.
Training retains the original arithmetic, lookup, and two string-task families.
Held-out evaluation adds two composition categories absent from training.
Distinct IDs are sampling units; these tasks still share synthetic templates.

1. Generate trajectories and exact-state counterfactual preferences. Include
   intermediate states rather than taking only the first and last state.
2. Curate one successful, highest measured reward action per state. Retain up
   to two measured negatives, including a correctness failure when available.
   The continuation remains rule based; this teacher is not a global optimum.
3. Extract frozen pretrained representations using `priority_v1` inputs.
   Use a 256-token cap; the tokenizer audit preserves all original pilot tasks
   and progress fields at that length. The legacy checkpoints retain 128.
   Compare last-token and masked-mean pooling, each with or without stateless
   layer normalization. Fit linear heads for 200 epochs at learning rate 0.01.
   Select representation and epoch by global action accuracy, then cross
   entropy, on a task-disjoint 25% internal validation partition. Held-out
   outcomes never select this checkpoint.
4. Initialize genuine coordinator LoRA SFT from the selected head and zero
   LoRA updates. Train two epochs at learning rate 0.0001. Then run one epoch
   of categorical DPO at 0.00003 against that exact frozen SFT reference.
5. Open held-out evaluation once for the old checkpoints, frozen-head probe,
   new SFT/DPO, random initial head, rules, random sparse routing, and the
   original one-round parallel All-Agent control. Also run a locked one-round
   sequential dense control with the same total budgets. Report every outcome.

The main endpoints are exact task success, agent calls, and category success.
Report paired task differences, global action distributions, stopping behavior,
and a permutation null for task-type/action association. A concentrated action
distribution alone is not a failure when tasks need the same action.

The base controller keeps the legacy representation so its historical identity
stays intact; the old/new checkpoint comparison bundles multiple repairs.
`inference.preserve_checkpoint_context: true` keeps each loaded checkpoint's
recorded input cap while the untrained base retains its configured 128 tokens.
The four head probes isolate pooling/normalization within the new input format.
They do not isolate serialization, curation, or data-volume effects. Offline
head accuracy is distinct from online execution quality.

## Limits and acceptance

This is one seed on a CPU host with frozen deterministic tools. It cannot prove
GPU throughput, broad language-task generalization, monetary savings, customer
readiness, or that MoE is preferable to a smaller router. The fixed rule policy
is an important control because it already solves this task grammar.

Archive raw data, configuration, partitions, training traces, checkpoint hashes,
hardware, source commit, evaluation traces, and unfavorable results. Any
successful repair warrants a larger task-family and seed study before a broad
resume or deployment claim. Existing serving acceptance remains a separate gate.

```bash
bash scripts/run_routing_repair.sh
```

The experiment needs the pinned Granite weights and enough host RAM for its
float32 backbone. Use fresh output directories; the scripts refuse to overwrite
committed training checkpoints.
