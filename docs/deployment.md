# Routing service deployment

Conductor exposes its coordination model through a bounded HTTP service. This
provides a deployable research component; production capacity, NVIDIA performance
and cluster operation still require validation on the target hardware. Local CPU
checks cannot certify CUDA kernels or justify GPU speedup claims.

## Device and artifact preflight

Create a Python environment and install the required optional dependencies:

```bash
pip install -e '.[hf,serve,profiling]'
python -m conductor.doctor --device cpu --dtype float32 --probe \
  --output outputs/device-cpu.json
```

On a CUDA host, run an explicit check before loading a large coordinator:

```bash
python -m conductor.doctor --device cuda:0 --dtype bfloat16 --require-cuda \
  --probe --output outputs/device-cuda.json
```

An incompatible or unavailable device fails with a nonzero exit code. Native
BF16 support is checked rather than inferred from a GPU name. This small
arithmetic probe establishes basic device compatibility; benchmark the actual
model and workload separately.

Use an explicit trained checkpoint and its action catalog. For an HF coordinator,
the base model revision must remain pinned and accessible to the serving process.
A LoRA checkpoint contains coordinator adapters and the routing head; it still
needs the frozen base weights. Export a merged, self-contained artifact when the
base dependency should be bundled:

```bash
python -m conductor.controller.export \
  --checkpoint outputs/checkpoints/YOUR_TRAINED_HF_COORDINATOR \
  --output outputs/exports/YOUR_COORDINATOR \
  --device cpu --dtype float32
```

Exports retain their source identity and adapter evidence. Benchmark merged and
unmerged inference under identical workloads before claiming a merge improves
performance. Never treat randomly initialized smoke checkpoints as pretrained
post-training evidence.

## NVIDIA container recipe

`containers/Dockerfile.nvidia` uses the NVIDIA PyTorch image selected by its
`BASE_IMAGE` argument and retains the image's optimized PyTorch build instead
of replacing it through PyPI. Dependency installation respects the base image's
constraint file. The supplied recipe has not been built locally: the Docker
daemon was unavailable, and no NVIDIA device was present. Pin a verified image
digest and validate the resulting image on the target host before a release.

```bash
docker build -f containers/Dockerfile.nvidia -t conductor:nvidia .
docker run --rm --gpus all -p 127.0.0.1:8000:8000 \
  -e CONDUCTOR_ROUTING_API_KEY \
  --mount type=bind,src=/ABSOLUTE_CHECKPOINT_ROOT,dst=/opt/conductor/outputs/checkpoints,readonly \
  conductor:nvidia
```

The checkpoint mount must contain the configured trained preference artifact.
Stage its pinned base weights and mount a cache writable by UID 10001, or supply
a self-contained merged export and matching configuration. Driver/GPU support
comes from the host, and container startup warmup is not a capacity benchmark.

## Start and call the service

Local development serves the existing trained tiny SFT checkpoint:

```bash
python -m conductor.serve --config configs/serving/development.yaml \
  --host 127.0.0.1 --port 8000
```

Development mode has no API key and is restricted to a loopback bind. For an
NVIDIA deployment, supply the API key through the configured environment variable
and use a trained preference checkpoint:

```bash
export CONDUCTOR_ROUTING_API_KEY='YOUR_RANDOM_SECRET'
python -m conductor.serve --config configs/serving/nvidia.yaml \
  --checkpoint outputs/checkpoints/YOUR_PREFERENCE_COORDINATOR \
  --host 127.0.0.1 --port 8000
```

Terminate TLS and apply external rate limits at a trusted ingress before exposing
the endpoint beyond the host. Keep secrets outside YAML and source control. Run
one serving process per assigned GPU; adding Uvicorn workers would independently
load additional model copies. For container GPU access, configure the host driver
and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
A container image cannot supply a missing host GPU.

```bash
curl --fail http://127.0.0.1:8000/v1/route \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $CONDUCTOR_ROUTING_API_KEY" \
  --data '{"state":{"user_task":"Compute 12 * 7.","task_type":"math","remaining_budget":{"agent_calls":2,"tokens":1024}},"k":2}'
```

The state schema includes task/type, conversation history, prior specialist
outputs, prior calls, tool results, budget, step and routing history. Grader
answers and private task metadata are not API fields. Unknown fields and invalid
budgets are rejected. Agent `k` is bounded by the checkpoint catalog and remaining
activation budget; it is separate from the base MoE's internal expert top-k.
An exhausted budget returns a termination decision with `decision_source` set to
`budget_guard` and zero model work. It is an orchestration guard, not a trained
model prediction or a promise that remaining token budget covers model dispatch.

## Admission, timing and failure behavior

A single exclusive model worker groups compatible requests by agent `k`, using
configured batch size and collection delay. Pending work is bounded; overload
returns HTTP 429 with `Retry-After`. Unavailable workers return 503 and request
deadlines return 504. Timings distinguish queue wait, model batch work and total
request time. A request UUID supports client correlation without placing task
text in service counters.

The service bounds body size, body-read duration, history lengths and execution
steps. Authentication is required by default; startup fails if the configured
key is absent or the checkpoint stage mismatches. These controls do not replace
host isolation, dependency scanning, ingress protections or workload-specific
resource validation.

- `GET /health/live` reports process liveness.
- `GET /health/ready` returns 200 only while the warmed worker accepts work; a
  stalled worker returns 503.
- `GET /metrics` requires the API key in authenticated mode and exports pending
  work, counters and a bounded recent latency window. The latency window is not
  an all-time histogram or a stable service-level percentile.

A response deadline cannot interrupt an already executing CUDA kernel or Python
thread safely. Shutdown bounds waiting and resolves queued requests; use an
external process supervisor to replace a stuck worker. Measure cancellation,
queue saturation and recovery on target hardware before promising a service
latency objective. Do not retry cancelled in-flight work unboundedly.

## Validation before a GPU release

Preserve the device doctor record, exact container/dependency versions and source
commit. Run the CUDA-marked correctness checks, checkpoint reload/export checks,
strict held-out evaluation and actual inference benchmarks on the chosen device.
Measure allocated and reserved peak memory with explicit synchronization, native
forward timing, serialization/tokenization overhead and queue-inclusive p50/p95
under expected concurrency. Include failure/deadline rates and raw JSON/CSV.
Record unsupported optimizations as unavailable; no cache, batching or precision
mode earns a performance claim solely because it is configurable.
