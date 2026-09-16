# Research protocol

The hypothesis is that post-training a pretrained sparse MoE coordination
model can preserve task quality while reducing specialist activations and
communication. The smallest local experiment establishes correctness of the
training and measurement pipeline. The compact Granite pilot performs genuine
pretrained-model post-training on a CPU. Larger independent corpora and frozen
LLM specialists are needed to test the broader hypothesis; NVIDIA performance
requires a real device run.

## Data and objectives

Generate both successful and unsuccessful execution trajectories using
multiple policies. SFT uses successful training trajectories. Preferences
compare counterfactual first actions from identical serialized states under
the same continuation policy. This controls continuation-policy confounding;
configurable late-state trials include stopping actions. Candidate coverage
remains sampled and does not establish globally optimal coordination.
Cost preference weights and normalization scales are configurable. DPO uses
the exact saved SFT policy as its frozen reference.

Successful trajectory imitation may over-represent dense or redundant actions.
Report SFT action counts and filter counts rather than assuming that successful
execution implies optimal coordination. Sweep reward weights to test whether
efficiency incentives trade away quality. Do not use held-out scores to choose
training weights or checkpoints; add a separate validation split before a
large hyperparameter study.

## Evaluation controls

Hold specialists, their model revisions, prompts, tool corpus, task order,
budgets, hardware and precision constant across routing policies. Repeat with
multiple training and evaluation seeds. Pair outcome differences by task ID.
Report both aggregate and unseen-category results. The included development
split has different examples and unseen compositions, but shares task
templates and vocabulary; it is a weak generalization test, not an external
benchmark. Add established externally graded task suites before claiming
broad transfer.

Record controller and specialist tokens separately, actual elapsed latency,
estimated cost assumptions, graph edges, invalid actions, activation counts,
termination and budget reasons. Whitespace counts from deterministic agents
are estimates, not tokenizer measurements. A zero cost rate means no monetary
cost model has been supplied, not that hardware inference is free.

## Inference studies

Benchmark actual controller forwards with warmups, synchronized accelerator
timers and repeated requests. Compare batching against single-request forwards,
including queue delay for dynamic batching. Sweep contexts, concurrency, agent
top-k and routing frequency. Report the exact represented context length;
hashed feature contexts are not transformer sequence-length scaling results.
Measure controller share of system latency only from end-to-end executions.
Unsupported KV/prefix caching is reported explicitly. Serialization caching
includes canonical-key construction cost and may be slower.

Expert utilization is observational. Task-conditioned routing correlations
alone do not prove causal specialization. Token-level HF expert statistics
must exclude padding. A separate intervention or ablation experiment is
needed to substantiate specialization. Repeat expert comparisons on an
identical probe set at initialization, SFT and preference checkpoints.

## Evidence standard

Every published number must originate from a saved run and retain its config,
code commit, dirty status, seed, hardware, checkpoint and runtime. Synthetic
or tiny random-initialization results may justify an infrastructure claim.
Claims about pretrained MoE post-training require completed training on pinned
pretrained weights with linked SFT/DPO artifacts. Claims about inference
optimization require matched measured results under the named workload and
hardware. CPU post-training is valid evidence of model training; it cannot
establish NVIDIA latency, throughput or cost improvements. Missing evidence stays an unanswered question.
