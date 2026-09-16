# Contributing

Use Python 3.12 and install `pip install -e '.[dev,hf,serve,profiling]'` in a
virtual environment. Run `ruff check .` and `pytest -q` before opening a pull
request. The CPU tests include small local MoE fixtures, checkpoint recovery and
two-worker Gloo training; they do not download the pilot's pretrained weights.

Keep model logic separate from specialist execution. Changes to training must
preserve task-level splits, coordinator-only optimization and the exact frozen
DPO reference. Changes to serving must preserve bounded admission and exclusive
ownership of the model worker.

For performance changes, include the configuration, device, precision, source
commit, raw request timings and a matched baseline. Keep profiler samples out of
throughput timing. An unavailable device or unsupported optimization should be
reported explicitly. A CPU test passing does not validate a CUDA path.

Working data and checkpoints belong under ignored `data/dev/`, `data/generated/`
and `outputs/`. Add a results archive only after the experiment finishes; retain
the original per-phase provenance and record failed attempts where relevant.
Never commit API keys, model weights or optimizer/RNG binaries.
