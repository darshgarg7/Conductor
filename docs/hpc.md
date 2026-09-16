# SLURM and Minnesota MSI

The scripts run locally without SLURM, or submit through `sbatch`. Create the
environment on a login or setup node first; do not install packages repeatedly
inside array jobs. Pin a downloaded model revision and stage model weights in
a shared cache or project scratch directory appropriate to your allocation.
The scripts select configuration files, not hardcoded cluster credentials.

MSI documents `msismall` for CPU jobs and `msigpu` with explicit GPU resource
requests; partition eligibility and hardware differ by allocation. See the
[MSI submission guide](https://userdocs.msi.umn.edu/compute/slurm_job_submission.html)
and [shared partitions](https://userdocs.msi.umn.edu/compute/shared_partitions.html).
Pass account/partition/GPU choices at submission, for example:

```bash
sbatch --account=YOUR_ACCOUNT -p msismall scripts/slurm/generate.slurm configs/generation.yaml
sbatch --account=YOUR_ACCOUNT -p msigpu --gres=gpu:a100:1 scripts/slurm/sft.slurm configs/training/olmoe_sft.yaml
sbatch --account=YOUR_ACCOUNT -p msigpu --gres=gpu:a100:1 scripts/slurm/preference.slurm configs/training/olmoe_preference.yaml
```

Resource defaults are templates, not validated OLMoE capacity promises. OLMoE
has substantially more total parameters than active parameters; optimizer,
activation and reference-model memory must be measured. Test a short job and
record peak memory before scaling context, batch size or DPO.

`scripts/make_sweep.py` materializes independent YAML files with unique output
and checkpoint directories. Submit the resulting manifest with a job array;
an array index selects exactly one configuration:

```bash
python scripts/make_sweep.py --config configs/training/tiny_sft.yaml --output configs/sweeps/dev
sbatch --array=0-8%3 scripts/slurm/sweep.slurm configs/sweeps/dev/manifest.txt
```

Generation arrays should use shard-specific seeds and dataset/output paths.
Merge shards only after validating disjoint task IDs/text, then rebuild SFT
and preferences from training splits. Dataset generation is offline and
independent of specialist parameter training. The minimal generator writes
one shard per invocation; large datasets should use streamed shards rather
than one shared file or concurrent writes to the same output directory.
