# Factorized routing and label coverage: proposed next experiment

This is a design proposal, not an implemented controller or a new experimental
result. The current learned controllers use a normalized categorical catalog of
complete routing decisions. That implementation remains the compatibility path
and an experimental control. Internal MoE expert selection is distinct from the
external specialist-agent selection described here.

The proposed change replaces the growing complete-action head with conditional
heads for **stop → count → mode → agent choices**. It still predicts one complete
structured decision. Its training objective uses the probability of that whole
decision, rather than treating independently scored agents as a joint policy.

## A normalized policy over valid decisions

Let `s` be the public execution state, `A(s)` the currently admissible specialist
agents, and `C = |A(s)|`. Use a fixed, versioned global agent order, initially
`AGENT_NAMES`; write its order/hash into checkpoint metadata. Prior calls do not
remove an agent from `A(s)`. Retrieval, criticism, verification, or tool execution
may legitimately be needed again after new evidence arrives.

For a validated state with a nonnegative call budget, the maximum new calls are
`M = min(k, floor(remaining_agent_calls), C)`. Exhausted token budget also makes
continuation inadmissible. A separate public admission rule can reduce `A(s)`
when an agent cannot execute under an enforced budget, but must not consult a
private answer or grader. Positive token budget alone does not guarantee that
all future output tokens fit; existing hard-budget admission remains necessary.

1. Predict a masked categorical `stop ∈ {true, false}`. If `M = 0` or continuation
   is otherwise inadmissible, only stop is legal. That forced event has
   probability one and log probability zero.
2. Given `stop=false`, predict `count ∈ {1, …, M}` with a categorical softmax.
3. Given count greater than one, predict `mode ∈ {parallel, sequential}` with a
   categorical softmax. A single-agent decision has one canonical mode,
   `parallel`, with probability one; no redundant mode loss is trained.
   Historical single-agent targets marked `sequential` map to that same action.
4. Given state, count, mode, and earlier choices, predict each next agent using
   a softmax over its valid choices. These conditionals can share the learned
   coordination representation and consume embeddings of earlier decisions.

A stopped action has no count, mode, or agents. Its structured response sets
`terminate=true` and `selected_agents=[]`; the irrelevant execution-mode field
can keep its canonical schema default. A continuing action has exactly its
predicted count of distinct agents and `terminate=false`.

### Parallel subsets

Represent a parallel subset only once, in increasing global agent order. If
`c` agents were requested, the choice at position `j` may select an index `i`
only when it is greater than the preceding selected index and there remain at
least `c-j` admissible indices greater than `i`. This **remaining-feasibility
mask** prevents an early choice from leaving too few agents to finish the subset.
Masks are computed over `A(s)`, including any admission exclusions, rather than
assuming all global agent indices are available.

For example, with ordered agents `[planner, retriever, coder, verifier]` and
count two, `verifier` cannot be the first choice. Choosing `coder` first forces
`verifier` second. Choosing `retriever` first permits either `coder` or
`verifier` second. Each valid two-agent subset has one path; no permutation
probabilities need to be summed or divided by a factorial.

### Sequential orders

For sequential execution, mask agents already selected **within this decision**,
then normalize over the remaining agents. Do not impose increasing order:
`[math, coder]` and `[coder, math]` are different actions with different execution
dependencies and probabilities. Since `c ≤ C`, sampling without replacement
always leaves enough choices. Both sequential and parallel routing allow agents
called in an earlier coordination round to be selected again.

### Joint probability

For a stopped decision `a`, `log π(a|s) = log p(stop=true|s)`.
For a continuing decision:

```text
log π(a|s) = log p(stop=false | s)
          + log p(count=c | s, stop=false)
          + log p(mode=m | s, stop=false, count=c)     # zero when c=1
          + Σ_j log p(agent_j | s, c, m, agent_<j, valid_mask_j)
```

Every conditional is normalized on its nonempty valid support. Together with
the unique parallel path, this yields a normalized distribution over all legal
complete decisions. Teacher-forced probability computation must apply the same
masks as inference at every stage. Reject an illegal target explicitly and
report its exclusion; do not create an infinite loss by silently training a
masked target. Confidence, if exported, should have a documented definition,
such as the complete decision probability, rather than an average of stage
confidences.

Canonicalize parallel target order and irrelevant single/stop fields before
likelihood computation. Chosen/rejected examples that become the same complete
action after canonicalization are not a preference and must be rejected.

## Consistent SFT and preference optimization

SFT minimizes the negative joint log probability above. Stop, count, mode, and
selection losses are its terms, rather than independently reweighted objectives
that would cease to equal the stated likelihood. Auxiliary objectives, if later
introduced, must be reported separately. Forced choices contribute zero loss.

DPO compares chosen and rejected **complete decisions**, both valid under the
same public state and mask rules:

```text
Δ = [log πθ(chosen|s) - log πθ(rejected|s)]
  - [log πref(chosen|s) - log πref(rejected|s)]
loss = -log sigmoid(β * Δ)
```

The reference is the exact saved SFT controller, frozen in evaluation mode. Its
factorization version, agent order, state representation, head configuration,
and mask semantics must match the actor. A reference-probability cache must
identify that checkpoint's digest, source/partition hashes, public state,
complete chosen/rejected actions, top-k, and mask version. Actor and reference
must not normalize over different action supports. Training still updates only
the coordination model and its heads/adapters; specialists remain frozen.

An independent sigmoid/BCE score for each agent does not supply this joint
distribution. Top-k selection changes the sample space, omits a learned stop
and count distribution, and does not encode sequential ordering. Products or
sums of selected sigmoid scores generally do not normalize over the legal
top-k decisions. They cannot be substituted as pseudo log probabilities in DPO
without deriving and implementing a consistent normalized policy.

The complete categorical catalog remains the control for likelihood, masking,
and rollout comparisons. Factorized checkpoints need a distinct format/head
identifier; they must not be loaded as catalog checkpoints or silently reinterpret
existing saved weights. Both policies should decode into the existing
`RoutingDecision` interface.

## Label coverage must precede training

Factorization alone cannot teach coordination behaviors missing from the data.
The completed routing-repair curation has **180 states from 36 training tasks**:
127 stop labels and 53 single-agent labels (27 coder, 14 retriever, 12 math).
It has no chosen multi-agent, parallel multi-agent, or ordered multi-agent label.
Steps zero/one/two contain 36/108/36 states respectively. Thus its count and
multi-agent mode targets do not cover the proposed head's behavior.

Its task-disjoint internal validation partition has nine tasks: six code and
three retrieval, **no math task**. The training partition contains all nine math
tasks. Representation selection on that partition cannot validate math routing
or category-balanced generalization. These observations concern the repair's
training/internal-validation corpus, not a newly selected held-out result.

The following audit is a prerequisite for a new corpus and protocol. Empty or
rare cells must be disclosed before a training run; there is no implied coverage
target achieved today.

| Audit dimension | Required breakdown | Why it matters |
| --- | --- | --- |
| Task type and family | Tasks and states per family in train, internal validation, and locked held-out inventory | Prevent a family from disappearing from selection validation; group paraphrases/related instances together. |
| Dependency structure | Independent evidence gathering, ordered handoff, iterative refinement, no specialist needed | A task category alone does not specify execution dependencies. |
| Coordination step | Initial, intermediate, late/budget-limited states | Policies must learn continuation and stopping after actual execution. |
| Prefix quality | Correct usable answer, partial evidence, incorrect answer, failed tool/agent call, conflicting evidence, unresolved ambiguity | Include recovery and continuation after plausible wrong outputs. |
| Agent count | Stop, one, two, three where top-k permits | Audit quality/cost-optimal chosen counts, not just candidate availability. |
| Execution mode | Single canonical mode; parallel and sequential for counts greater than one | Train both simultaneous independent work and ordered dependencies. |
| Handoff/order | Predecessor → consumer, required public artifact, successful and broken handoffs | Distinguish `[math, coder]` from its reversal and test artifact use. |
| Stop decision | Appropriate termination, premature stop, unnecessary continuation | Separate task completion from the presence of an output string. |
| Cross-round recalls | Agent revisits after new evidence; useful versus redundant repetition | A within-round duplicate mask must not forbid legitimate iterative work. |

Public `answer_present` is a feature, **not a completion certificate**. A
specialist can emit a partial, incorrect, or unsupported answer. Private grading
may label offline trajectories and preferences, but its scores or expected
answers must never enter the execution state or admission masks. The corpus
needs successful continuations from failed/partial prefixes and chosen stop
decisions supported by a completed task, rather than deriving stop directly
from public answer presence.

Measured labels should compare identical states under a fixed, identified
continuation, grader, specialist configuration, and reward specification. For
multi-agent demonstrations, include tasks where a cheaper single-agent route
fails or has worse quality, parallelism is appropriate for independent work,
and order changes success. Report contradictory observations rather than
quietly replacing them with a desired action. A rule continuation is an offline
teacher with limits, not evidence of a globally optimal policy.

## Proposed acceptance gates and tests

These gates describe future implementation and experiments:

- Enumerate a small agent universe and verify complete decision probabilities
  sum to one for each top-k/budget, including canonical parallel paths and every
  sequential order. Verify probability one for forced stop/single-mode events.
- Test parallel remaining-feasibility masks, unavailable-agent holes, within-round
  duplicate rejection, ordered sequential targets, and valid cross-round recalls.
  Assert every sampled/greedy decision passes the existing structured validator.
- Compare teacher-forced joint log probability with explicit path products.
  Check gradients for each learned conditional and zero contributions from
  forced terms; reject illegal SFT/preference labels with recorded reasons.
- Test DPO at actor/reference equality, reference immutability, complete
  chosen/rejected likelihoods, and cache invalidation for changed state/action,
  checkpoint, agent order, masks, representation, or top-k.
- Publish the coverage table and grouped task-disjoint partitions before fitting.
  Require each represented training family in internal validation; lock a new
  larger held-out inventory before selection and keep private outcomes closed.
- Compare catalog and factorized SFT/DPO with identical specialist interfaces,
  state inputs, budgets, seeds, and task inventories. Retain rule, random sparse,
  dense, and cheap learned non-MoE controls. Report global action quality,
  count/mode/stop/handoff behavior and end-to-end success; pairwise preference
  accuracy alone is insufficient.
- Declare quality/cost and generalization endpoints in advance, archive all
  outcomes, and repeat seeds before a broad claim. Benchmark head/controller
  latency separately; a smaller head is not an inference improvement until
  measured. This proposal supplies no GPU, deployment, or customer-quality
  evidence.
