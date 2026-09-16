# Design decisions

## Predict a routing action, not an agent name

The controller pools a pretrained MoE representation and predicts a categorical
distribution over complete actions. Each action specifies agents, execution
mode and stopping. A mask enforces the external agent budget before sampling or
argmax. Sequential actions preserve order because later agents see earlier
outputs; parallel subsets do not carry an ordering.

This gives SFT and DPO a small, explicit probability space. It avoids generating
and then repairing agent-name strings. The pretrained backbone still needs a
new action head: the base comparison is therefore a pretrained backbone with a
random head, not an instruction-tuned coordinator. Action enumeration grows
quickly with the number of agents and k; this implementation deliberately caps
the catalog at eight agents and k ≤ 3.

## Keep the two routing problems separate

External top-k limits specialist calls. Internal MoE top-k selects neural
experts. Lower external k does not reduce the number of internal experts per
token. Granite uses its pretrained internal routing configuration. LoRA changes
attention and expert-router projections; expert feedforward weights stay frozen.
Expert/task correlations are exploratory measurements, not evidence that an
expert acquired a causal specialization.

## Prefer exact-state counterfactuals to whole-trajectory comparisons

Two successful trajectories can reach different states. Comparing their next
actions as if they shared an input introduces a confound. Preference generation
replays candidate actions from one serialized state, with identical budgets and
one fixed continuation. Initial and late states include stopping decisions.
The pilot's latency reward weight is zero because cheap fixture timing is noisy.
Actual latency is still recorded. Candidate trials are durably journaled so a
restart preserves the first measurement and preference labels.

DPO uses categorical chosen/rejected action log probabilities against the exact
SFT checkpoint. Reference probabilities are cached once; training does not keep
a second large backbone resident. The cache identity includes the SFT weights,
records, order and k.

## Commit optimizer windows rather than partial gradients

Checkpoints contain weights, optimizer, scheduler, scaler, per-rank RNG and the
execution cursor. Publication happens after a complete accumulation window.
The last short window divides by its actual example count. DDP ranks with an
empty final shard execute a zero-weight sample so collective counts match.

CPU resume is checked against uninterrupted training. CUDA arithmetic and
numerics require separate device validation; compatible serialization alone
does not imply bitwise equivalence across different hardware.

## Give one worker ownership of the model

Each service process has one exclusive model worker. The queue and in-flight
requests share a bounded admission limit. Compatible k values form batches;
per-request token/cost telemetry is copied before the next model call.
Timeouts cancel delivery, not an already-running accelerator kernel. Readiness
reports a stalled worker so a process supervisor can restart it.

Tokenization and canonical-state caches are bounded. The classification path
does not reuse a generative KV cache: most execution-state fields change between
rounds, and a correct reuse scheme would need explicit prefix invariance.

## Measure workload boundaries explicitly

Native offline batches, queued individual requests and queued dynamic batches
answer different latency questions. Reports name the boundary. HF stage probes
separate serialization, tokenization, transfer, forward and decision work;
CUDA events measure device stages, while host clocks measure synchronized
elapsed time. Profiler-instrumented samples are kept outside ordinary timings.

Input tokens per second is not generated tokens per second. Development-tool
tokens are whitespace estimates; zero price rates mean no monetary model was
configured. Neither can establish real LLM billing savings. Small held-out
corpora and task-paired uncertainty intervals are disclosed with every result.
