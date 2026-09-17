# Support demonstration walkthrough

The demonstration connects a trained routing service to fixed support
specialists. It tests the entire state → route → tools → updated-state loop and
separately exercises the HTTP boundary. No ticket causes a real system command.

## Run it

Install the development and serving dependencies using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hf,serve,profiling]'
bash scripts/run_support_demo.sh
```

If the default tiny checkpoint is absent, the script generates development
trajectories and runs tiny SFT first. It reuses an existing artifact and does not
download pretrained weights. An explicit `--checkpoint` skips that preparation.
The demo rejects a missing or random checkpoint and requires a fresh result
directory; it does not overwrite an earlier run.

For an existing pretrained Granite preference checkpoint:

```bash
python -m conductor.demo \
  --config configs/demos/support.yaml \
  --serving-config configs/serving/granite_pilot.yaml \
  --checkpoint outputs/research/granite-pilot/preference \
  --output outputs/demos/support-granite
```

Granite needs the pinned base weights and more memory than the tiny path. The
pretrained [research pilot](../README.md#reproduce-pretrained-post-training)
creates that checkpoint. The demo supplies a temporary API key to its isolated
localhost child service. No key is written to the results.

Add `--require-model-acceptance` when diagnostic or configured load failures
should block a validation job. The default exit status reports service-boundary
acceptance; a completed comparison can still reject the model candidate. The
strict flag requires the service, diagnostic, and illustrative load gates to
pass together. Inspect `metrics.json` rather than treating exit zero as a model
quality result.

## Archived checkpoints and fresh preparation

The published CPU report uses the existing local tiny SFT and Granite preference
artifacts, identified by their checkpoint hashes. A fresh bootstrap trains a new
tiny artifact; its routing outcomes need not match the archived checkpoint.
The [Linux CI run](https://github.com/darshgarg7/Conductor/actions/runs/35275229939)
passed service checks and completed 48 valid HTTP load requests. Its new tiny
checkpoint solved 3/12 fixtures, versus 0/12 for the archived local tiny artifact.
The workload hashes match and the checkpoint hashes differ. Both fail diagnostic
acceptance. Compare checkpoint, data, and environment identities before treating
two executions as reproductions of the same trained model.

## What to inspect

1. **One ticket.** Read its observed error, the agents selected, the evidence
   returned, and the final diagnostic response. The expected response belongs to
   the grader and is absent from every execution state.
2. **A dependency.** Retrieval must finish before synthesis can consume its
   evidence. Compare sequential and parallel decisions in the recorded model
   trajectories rather than assuming parallelism is always useful.
3. **A failed answer.** A learned controller can return a valid routing decision
   and still fail the diagnostic contract. Show the failure instead of filtering
   it out of the demo.
4. **Service acceptance.** Inspect unauthenticated access, invalid/private state
   fields, the exhausted-budget guard, request identity, readiness, queue drain,
   and shutdown checks.
5. **Concurrent requests.** Compare raw client timing with queue and batch
   timings. The latency target is a configured example, not a customer SLA.

## A short technical presentation

Use the [customer scenario](customer_case_study.md) to explain the requirements,
then walk through one real recorded trajectory. Explain why the controller owns
routing while fixed tools own evidence and the human owns system changes.

Show the diagnostic comparison before the latency chart. If rules are more
reliable for this corpus, recommend rules for this workflow and a shadow trial
for the learned model. End with the next test that would change that decision:
an independently collected workload, reliable frozen LLM specialists, or actual
NVIDIA-host validation. Do not call implementation paths completed deployments.

The [recorded results](../results/support-demo/report.md) provide a reviewable
example without requiring a model download. The
[operations guide](support_operations.md) explains how to interpret failures.
