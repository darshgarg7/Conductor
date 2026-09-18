# Coordination v2: proposed CPU experiment

This is a plan for the next experiment, not a completed result or an executable
training configuration. The [declarative protocol](../configs/research/coordination_v2_protocol.yaml)
freezes three training seeds, **42, 137, and 2027**, and the acceptance rules below.
Do not pass that YAML file to `conductor.train` or `conductor.evaluate`. This commit records design only;
training has not started.

The completed routing repair is useful development evidence. Its original
48 held-out tasks have already been inspected and now belong to the development
exposure ledger. They cannot become an unseen final test by renaming their IDs.
The next study asks whether a controller can execute genuine dependencies,
ignore distractors, recover from tool errors, and coordinate independent work
without relying on a task-category tag that identifies the route.

## Eight ordered steps

The eight items match the research improvement plan. Practical prerequisites
matter: the seed list, data-split rules, model-selection rule and acceptance
thresholds in item 7 are frozen by this protocol **before any new training**,
not introduced after items 5 and 6. A change needs a new committed revision
before training; none may follow inspection of the locked final outcomes.

1. **Publish both prior analyses and record exposure.** Keep the original
   collapsed pilot and completed routing repair, including failures and overhead.
   Preserve the [routing failure analysis](routing_failure_analysis.md) alongside
   the [repair protocol](routing_repair_protocol.md).
   Record source commits, model revisions, specialist identities, development
   inventories and protocol hashes. The repair's 48 inspected tasks are now
   development regressions. Commit this protocol before creating and sealing
   the fresh final suite. Model, prompt, baseline and release choices use
   development data; final outcomes cannot change them.

2. **Build genuine workflows and distinct frozen specialists.** Families cover
   retrieval/calculation, retrieval/coding/verification, research/verification,
   parallel evidence gathering, distractor handling and recovery from tool
   failure. Require prerequisites, intermediate artifacts and public evidence,
   rather than merely renaming an agent that already solves the complete task.
   Retrieval returns records; research synthesizes or reconciles them; math
   calculates; coding produces transformations or artifacts; the tool executor
   invokes an allowlisted operation; criticism identifies defects; verification
   independently checks public evidence or assertions; planning proposes
   dependencies. A planner cannot supply a graded answer, and a verifier cannot
   read the answer key. Test these boundaries and freeze agent implementations,
   prompts, model revisions, tools and corpora before comparisons.

   Grade source-grounded answers, executable artifact validity and verification
   evidence requested by the task, not preset route names or mandatory call
   counts. Permit any valid alternative workflow. Demonstrate prerequisites
   through measured ablations: omit a prerequisite, reverse the order, supply a
   wrong consumer artifact, or offer an unsupported final-text shortcut. A
   controlled synthetic capability boundary can establish behavior in that
   fixture; disclose its artificiality and do not infer broad multi-agent
   necessity from it.

3. **Collect initial, intermediate, handoff, failure and stopping supervision.**
   Start with 72 training and 36 development tasks, 12 and six per family.
   Split underlying problems and source-document groups together: paraphrases,
   operand variants and intermediate states cannot cross the split. Stratify
   development by family and two-/three-stage dependencies, with at least 12
   tasks in each dependency group and every training family represented.
   Collect successful and unsuccessful measured trajectories from multiple
   policies, including broken handoffs, distractors, wrong or partial answers,
   tool errors, legitimate agent recalls and completed workflows. A public
   `answer_present` flag alone is not an appropriate-stop certificate.

   Use constant public `task_type=coordination_workflow`. Family, graph, required
   agent list, reference answers, rewards and private grader metadata remain
   offline. Controllers and specialists see only user intent, public tool
   evidence and execution progress. Audit initial, intermediate, failure and
   completion serializations, including truncation and admission masks; neither
   can consult the private grader. Curated winners must be successful measured
   counterfactuals from identical states under a shared identified continuation,
   specialist configuration and grader. A rule continuation is a bounded teacher,
   not proof of global optimality. Report contradictory labels and exclusions.

4. **Design normalized factorization and audit label coverage before fitting.**
   The [factorized-routing proposal](factorized_routing.md) specifies a complete
   joint policy: stop, then count, conditional mode, and agents without
   replacement. Every conditional is normalized on legal support. Stop has no
   agents; a single agent uses canonical parallel mode with probability one.
   Parallel subsets follow versioned agent order with a remaining-feasibility
   mask; sequential selections retain order. Within-round duplicate masks must
   allow useful cross-round recalls. SFT and DPO use complete joint log
   probabilities, never independent sigmoid scores as pseudo likelihoods.

   Keep the existing normalized categorical catalog as a control. Actor and
   frozen reference must share support, masks, state format and factorization.
   Checkpoints and reference caches identify agent order/hash, mask/head version,
   top-k and budget support, state/action hashes, source partitions and exact SFT
   reference digest. Do not silently load catalog weights into the new format.
   Before training publish a task-group/family/state coverage table for chosen
   counts zero/one/two/three, parallel and sequential multi-agent actions,
   successful/broken or blocked handoffs, failure prefixes/recovery and stopping
   decisions. Private route IDs never enter model input. Candidate
   availability is not chosen-label coverage. Empty or rare cells prevent claims
   about untrained behavior. This document proposes that implementation and audit;
   it does not say they exist today.

5. **Establish strong cheap controllers, then gate cached heads and LoRA.**
   Compare public-state rules, a sparse linear router, a small MLP, iterative
   sequential dense execution, parallel All-Agent, Random Top-K, a functioning
   frozen prompted supervisor, a pretrained MoE/random head and a frozen MoE/
   fitted head. Rules cannot read family tags. Dense schedules must permit
   dependencies and verification within the same total budgets. Routed policies
   use `k=3`; All-Agent has an explicit eight-agent per-step exception while
   retaining the same 12-call total budget. Its scheduling difference is part
   of the independent policy variable, not a claim of identical per-step sparsity.

   The prompted supervisor needs at least 99% structurally valid development
   decisions; actual correctness and errors remain separate. Freeze its prompt
   and parser, and account for any explicit fallback. An invalid-JSON control
   cannot support superiority claims. Choose the primary cheap baseline from
   rules, linear routing and the small MLP by highest mean family-balanced
   development success across the fixed seeds, then fewer whole-inventory calls,
   then lower total CPU latency, then lexical identifier. It must meet 95%
   known-workflow development correctness. Lock its identity before final lock;
   report all controls, stronger secondary results and unavailable comparisons.

   Start learned controllers with cached frozen Granite features and small heads.
   Missing pinned weights fail closed, without random substitution. Reuse features
   only for identical inputs and backbone weights. Limit representation variants
   to four and heads to 200 epochs, with choices fixed on development data.
   Every seed must solve at least `ceil(0.95 × N)` tasks in each two- and
   three-stage development group with correct answers and required public
   verification evidence before LoRA. Report full-catalog scores, stopping and
   family breakdowns as well as rollouts. Only passing heads receive two SFT
   epochs with rank-four coordinator adapters. Verify real saved attention/router
   updates and frozen specialist identities, then repeat the same development
   gate before DPO. Failed seeds stop their dependent stage and remain reported.

6. **Use on-policy mistakes and measured rescues for preference training.**
   Collect current SFT rollouts on both training and development tasks; refresh
   state coverage around actual premature stops, bad counts/modes, wrong-agent
   calls, handoffs and failures. On training-task states, compare that actual
   mistake with measured successful rescue or counterfactual decisions using the
   same public state and shared continuation. Include cost-only pairs only at
   comparable correctness. Reject chosen/rejected routes that are canonically
   identical, including irrelevant confidence and single-agent mode differences.
   Demonstrate coverage of two-/three-agent and parallel/sequential choices where
   tasks genuinely need them. Do not invent positive labels to fill an audit cell.

   Fit one DPO epoch per eligible seed against that seed's exact frozen SFT
   checkpoint. Development rollout pairs are validation/diagnostic data only,
   never DPO fitting data. Final-test rollouts never generate preferences.
   Preserve source-state, actor, continuation, reward and reference identities.
   Report new on-policy coverage and exclusions; pair-ranking accuracy does not
   substitute for full-policy prediction or actual workflow success.

7. **Repeat the three fixed seeds under already frozen acceptance criteria.**
   Seeds 42, 137 and 2027 share data, fixed model/representation choices and
   execution conditions. Publish each seed, including failed and blocked stages,
   and mean/range/standard deviation; do not choose the best seed or quietly
   replace one that fails. Freeze candidate-stage and release rules on development
   before final lock. Seed 42 supplies the released artifact if its stage passes;
   an unfavorable DPO result need not be deployed. The numerical gates below are
   proposed tolerances frozen by this design commit, not measured performance.
   A full study can finish and correctly reject the candidate.

8. **Lock fresh unseen structures, open once, and publish every outcome.**
   After the protocol commit and all development choices, an independent
   generation/curation process seals 120 fresh tasks, 20 per family. Ten per
   family use dependency structures or failure patterns absent from training
   and development. Exclude exposed underlying problems/source groups; seal
   inputs, private answers/contract graders, strata and content hashes before
   candidate-dependent inspection. Prefer an independent reviewer to hold the
   suite; document weaker independence if one researcher must curate it.

   Only frozen candidates and baselines access final inputs. Use the same eight
   specialists, six rounds, 12 total calls and 16,384 labeled token-budget units,
   with the declared All-Agent per-step exception. Record all attempts, retries,
   fallbacks, abstentions and controller overhead. Report seen structures and
   structural OOD separately. Final failure ends this confirmatory cycle; repairs
   need a revised committed protocol and a new unopened final suite. The failed
   inventory becomes development evidence.

## Predeclared fixed-inventory acceptance

These are proposed tolerances, not achieved accuracy or statistical guarantees.
Quality means a correct answer **and** a grounded workflow contract, including
required verification. Valid routing JSON alone never counts as task success.
All three seeds must have completed, auditable SFT and DPO stages for the full
three-seed study to pass; failures and blocked phases remain in the report.

For every seed, an accepted learned candidate must finish within **five percentage
points** of the locked primary cheap baseline on the full final inventory, and
achieve at least **90% quality in every workflow family**. With equal family sizes,
report both ordinary task proportions and equally weighted family means. Publish
seen-structure and structural-OOD results separately; do not hide an OOD failure
inside overall accuracy. Passing these finite-suite tolerances does not justify
calling a lower-quality candidate “quality preserving.”

For every matched seed, accepted DPO must also finish within five percentage
points of its SFT quality, meet the 90% family floor, and use at least **10% fewer
actual agent activation attempts over the whole final inventory** than SFT.
This comparison includes all tasks, not only successful completions. As an
additional anti-abstention guard, the same 10% reduction must hold when each
failed or abstained task is charged the full 12-call budget. Preserve these
failure-charged acceptance values separately from actual measured calls; the
charge represents a decision rule and is not work that was physically executed.
If the baseline has zero calls, a relative reduction is undefined and the gate
cannot pass.

Report actual calls on jointly correct tasks as a separate conditional cohort,
with its size and excluded failures. This cohort cannot support a population or
causal savings claim. Report controller tokens, downstream model tokens, tool
estimates, feature extraction, state serialization, retries/fallbacks and total
CPU latency even when they move against call reduction. A 10% call reduction
with worse total latency is a tradeoff, not an overall inference improvement.
Do not convert proxy tool tokens or configured zero prices into monetary savings.

## Statistical and deployment limits

Show each seed's raw paired task outcomes, baseline-only and candidate-only
success counts, family/structure counts, quality differences, call differences,
mean, range and standard deviation across the three seeds. A deterministic
fixture gate is distinct from uncertainty about future customer tasks.
Use family-cluster resampling only as an exploratory description, with 1,999
predeclared draws; retain whole families and all seed results together. Six
handcrafted workflow families supply little population information. More task
variants and repeated CPU timings do not create more independent families, and
three seeds do not establish population guarantees about training randomness.
Unadjusted multiple comparisons remain exploratory. These data cannot certify
broad statistical noninferiority or superiority; a future independently sampled
family study needs its own probability model, power analysis and protocol.

A superiority claim requires adequate quality evidence against the predeclared
strong cheap baseline plus a measured advantage in the stated efficiency
endpoint. Passing an observed five-point tolerance alone supplies no such
statistical certificate. A perfect grammar rule need not be beaten for the
study to be useful: matching its observed quality with a larger controller may
show the learned mechanism works while showing the simpler option is preferable.

Measure CPU latency with three repetitions on fixed hardware and software,
including the controller and specialist execution; record warmup/cache conditions
and separate one-time feature preparation from online work. This plan schedules
no GPU or cluster validation. CUDA paths, SLURM scripts and container files do
not establish NVIDIA throughput, memory capacity, dollar savings or production
readiness. Customer support acceptance and serving reliability remain separate
from this workflow study. The study may be complete while the candidate fails
acceptance; publish that outcome without rewriting its gates.
