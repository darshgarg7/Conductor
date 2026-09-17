# Operating the support routing component

This guide covers Conductor's routing service, not administration of a customer's
GPU cluster. The support tools use synthetic observations and never run the
commands mentioned in a diagnostic response.

## Before starting

- Check the checkpoint's trained stage and action catalog. A LoRA artifact still
  requires the pinned base model; a merged export has different packaging.
- Run the device preflight on the actual serving host. CPU checks do not validate
  CUDA, the driver, BF16 support, or container GPU access.
- Assign one model worker per service process. Bound queue plus in-flight work,
  state size, timeouts, and downstream call budgets for the intended workload.
- Keep the API key outside source/configuration files. An external endpoint needs
  a trusted TLS ingress and a ticket-data retention/redaction policy.

The automated demonstration manages an isolated loopback service and ephemeral
key. It does not provision ingress, rotate production secrets, or establish an
external deployment's capacity.

## Failure triage

| Symptom | Inspect first | Response |
| --- | --- | --- |
| Child exits before readiness | Local `server.log`, checkpoint stage, model revision, available memory | Fix the startup cause. Never substitute a random controller to make readiness pass. |
| HTTP 401 | Key configured at startup and supplied header | Correct the credential path without logging the secret. |
| HTTP 422 | Public state schema, budget, checkpoint k | Fix the request. Private grader fields are intentionally rejected. |
| HTTP 429 | Admission limit and pending work | Reduce offered work or back off. Do not add unbounded retries. |
| HTTP 503 | Readiness, worker availability, model failure | Stop sending work until the component is ready; a stalled worker may need process replacement. |
| HTTP 504 | Queue time, batch time, request deadline | Determine whether waiting or execution dominated. A cancelled response does not preempt an executing kernel. |
| Correct HTTP response, wrong diagnosis | Routing trajectory, retrieval evidence, synthesis ordering, stop decision | Fail the quality gate and retain the baseline. Transport success does not make the answer correct. |
| High p95 at higher concurrency | Raw client samples, server queue/batch timings, failed requests | Reproduce under the same offered load before changing batching or capacity. |

Error and deadline semantics have unit coverage. The archived demo records the
checks it actually executes; it does not inject every failure above into a real
trained GPU service. Readiness is a necessary admission condition, not proof of
diagnostic quality or a promised latency target.

## Interpret telemetry correctly

Per-request client elapsed time includes the local HTTP boundary and queue wait.
Batch/model timing is shared by requests in a batch and must not be summed as
independent device work. State/tool token counts are estimates; HF controller
token counts use its tokenizer. With no billing rates configured, monetary cost
is unknown.

The service's recent latency window is useful for diagnosis but is not a durable
SLO histogram. Use preserved raw samples for the short demo's p50/p95 values.
Keep diagnostic task counts separate from repeated load-request counts.

## Escalation and rollback

A support answer should request discriminating evidence or escalate when the
runbook does not support a diagnosis. NVIDIA's
[Xid guidance](https://docs.nvidia.com/deploy/xid-errors/introduction.html)
explains why a code alone may have several causes. The customer operator decides
whether to collect more information or perform a system change.

If the learned controller fails the demonstration's diagnostic acceptance,
retain the rule policy for the fixed demo workload and capture the learned
decisions for analysis. This is a policy recommendation, not an implemented
production traffic-switching mechanism. Retraining, a new checkpoint, or revised
tools requires rerunning both diagnostic and service acceptance under the same
recorded configuration.
