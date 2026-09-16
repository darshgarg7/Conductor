# Archived measured development run

These are raw machine-readable artifacts from code commit `8c11775`, with a clean working tree. Checkpoint tensors remain local under `outputs/checkpoints/`; regenerate them using `bash scripts/run_dev.sh`. The exact development inputs are versioned under `data/dev/`.

The optimization study separately measures actual bfloat16 computation and warm repeated-state feature memoization. Its measured comparisons and per-category quality observations are in `optimizations/comparison.json`. No GPU or pretrained-model outcome is established by these CPU/random-initialization artifacts.

To preserve the versioned training inputs, use the individual SFT/DPO commands rather than regenerating trajectories first. The end-to-end script produces a new measured experiment whose timing-dependent preference labels may differ.
