# SLURM and Minnesota MSI

Conductor can launch experiments locally or through SLURM. Scripts in
`scripts/slurm/` cover trajectory generation, SFT, preference optimization,
evaluation, inference benchmarks and configuration sweeps. Their resource
defaults are starting templates, not measured capacity promises. No NVIDIA or
MSI allocation was available for local validation.

MSI requires matching resource requests to the chosen partition; GPU type and
eligibility depend on the allocation. Check the current
[MSI submission guide](https://userdocs.msi.umn.edu/compute/slurm_job_submission.html)
and [shared partitions](https://userdocs.msi.umn.edu/compute/shared_partitions.html)
before submission. The `--gres=gpu:TYPE:COUNT` syntax requests a GPU type offered
by that partition. A published V100 example does not imply BF16 support or
compatibility with a modern NVIDIA container.

## Set up once and stage immutable inputs

Install the environment on a setup node rather than in every job. The scripts
accept `CONDUCTOR_ROOT` and `CONDUCTOR_PYTHON`; set these to the checked-out project
and environment interpreter. Stage a pinned base model and datasets in shared
storage appropriate to the allocation. For HF runs, use a fixed revision and
`local_files_only` after download to avoid network surprises on compute nodes.
Do not embed account credentials in configuration.

```bash
export CONDUCTOR_ROOT=/YOUR_PROJECT/Conductor
export CONDUCTOR_PYTHON=/YOUR_ENV/bin/python
sbatch --account=YOUR_ACCOUNT -p msismall \
  scripts/slurm/generate.slurm configs/generation.yaml
sbatch --account=YOUR_ACCOUNT -p YOUR_GPU_PARTITION --gres=gpu:YOUR_GPU_TYPE:1 \
  scripts/slurm/sft.slurm configs/training/olmoe_sft.yaml
sbatch --account=YOUR_ACCOUNT -p YOUR_GPU_PARTITION --gres=gpu:YOUR_GPU_TYPE:1 \
  scripts/slurm/preference.slurm configs/training/olmoe_preference.yaml
```

Run `python -m conductor.doctor --device cuda:0 --dtype bfloat16 --require-cuda
--probe --output outputs/device-cuda.json` **inside the GPU allocation**. Login
node CUDA availability does not characterize compute-node availability. The
probe is compatibility evidence only. Run a short actual model experiment to
measure peak memory before increasing sequence length, batches or DPO scale:
MoE total parameters, activations and adapter/head state consume memory even
when each token activates a sparse subset of experts.

## Independent arrays and recovery

`scripts/make_sweep.py` materializes independent YAML configurations with distinct
output/checkpoint directories. An array index selects one configuration:

```bash
python scripts/make_sweep.py --config configs/training/tiny_sft.yaml \
  --output configs/sweeps/dev
sbatch --array=0-8%3 scripts/slurm/sweep.slurm configs/sweeps/dev/manifest.txt
```

For trajectory generation, set `jobshards` and `shard_index` in each independent
configuration. Stable task-ID hashing assigns tasks to shards; per-task/policy
seeds preserve isolation from scheduler ordering. Shards write separate
directories. Supply an external task inventory through `task_source.path` when
moving beyond development templates; the exact-answer grader still requires
private `expected_answer` values. Public task ID/text collisions and train/held-out
leakage are rejected before execution.

New generation journals are checksum sealed, single-writer locked and fsynced.
Resume requires the same source/configuration fingerprint; a changed policy or
reward belongs in a new run directory. Restart can recover a torn final journal
line; interior corruption fails closed. Run `python -m conductor.audit --dataset
YOUR_SHARD_ROOT --strict` to read-check the resulting inventory. Do not concatenate
shards into training data without checking disjoint tasks and journal IDs.

Training writes immutable resume directories and a pointer to the committed
checkpoint. A trusted local resume includes optimizer/scheduler/scaler state,
RNG state and the training cursor; it is different from initializing a new run
from inference weights:

```bash
python -m conductor.train --config YOUR_SFT.yaml \
  --resume outputs/checkpoints/YOUR_RUN/resume/YOUR_SAVED_DIRECTORY
```

Resume state uses trusted PyTorch serialization; do not load an unknown
`training_state.pt`. Keep the same data, configuration identity and world size.
The exact-resume guarantees are exercised by local tests and need a target
CUDA/distributed run before being extended to a GPU production claim.

For supported single-node multi-GPU jobs, use one `torchrun` process per assigned
GPU with a shared experiment directory, for example:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m conductor.train --config YOUR_TWO_GPU_TRAINING.yaml
```

Allocate the matching GPU count through SLURM and use the cluster's supported
launcher/network setup. Rank zero publishes shared artifacts; all ranks must
complete checkpoint collectives. Consult the
[PyTorch torchrun documentation](https://docs.pytorch.org/docs/stable/elastic/run.html)
for launch semantics. Multi-node network tuning and elastic resizing are not
validated by this project.

## Preserve evidence and inspect failures

Every run records configuration, source commit/dirty state, random seed,
hardware, runtime and metrics; training adds checkpoint, data and reference
fingerprints. Retain raw inference JSON/CSV alongside summaries and compare
identical workloads with repeated trials. CPU development timing is not a GPU
performance forecast.

Use `squeue` for pending/running jobs and `sacct` for completed-job accounting,
including exit state, elapsed time and memory. OOM, deadline and failed grader
outcomes belong in experiment reports. See [deployment](deployment.md) for
serving admission and shutdown limits, and [resume evidence](resume_evidence.md)
for truthful claim wording.
