# autoresearch (Apple Silicon)

This is an experiment to have the LLM do its own research, ported to Apple Silicon with dual-backend support (PyTorch MPS + MLX).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — MPS backend training script. Modify this if using MPS.
   - `train_mlx.py` — MLX backend training script. Modify this if using MLX.
   - `backends/` — hardware detection, optimizers. Do not modify.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Select backend**: The system auto-detects the best backend. Override with `AUTORESEARCH_BACKEND=mps` or `AUTORESEARCH_BACKEND=mlx`.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Platform: Apple Silicon

This port runs on Apple Silicon Macs (M1/M2/M3/M4 and Pro/Max/Ultra variants). Key differences from the original CUDA version:

- **No FlashAttention-3**: Uses PyTorch SDPA (MPS) or native MLX attention instead.
- **No torch.compile**: MPS backend runs in eager mode. MLX uses `mx.compile` for kernel fusion.
- **Unified memory**: Apple Silicon uses unified CPU/GPU memory — no explicit data transfers needed.
- **MFU metric**: Approximate only. Based on estimated Apple Silicon bf16 FLOPS (varies by chip).
- **Hyperparameters**: Auto-tuned for detected hardware. Max/Ultra chips get larger batch sizes and model depth.

### Backend selection

```bash
# Auto-detect (default: prefers MLX)
uv run train.py

# Force MPS backend
AUTORESEARCH_BACKEND=mps uv run train.py

# Force MLX backend (or run directly)
AUTORESEARCH_BACKEND=mlx uv run train.py
uv run train_mlx.py
```

### Installing dependencies

```bash
# MPS only
uv pip install -e '.[mps]'

# MLX only
uv pip install -e '.[mlx]'

# Both backends
uv pip install -e '.[all]'
```

## Experimentation

Each experiment runs on a single Apple Silicon GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` (MPS backend) or `train_mlx.py` (MLX backend) — these are the files you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify `backends/`. The optimizer and hardware detection code is shared infrastructure.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Apple Silicon has unified memory (64-192GB on Max/Ultra). Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     8192.0
mfu_percent:      45.20
total_tokens_M:   125.0
num_steps:        250
num_params_M:     50.3
depth:            8
backend:          mlx
chip:             Apple M4 Max
```

Note that the script is configured to always stop after 5 minutes. Performance will vary by chip — Max/Ultra will process more tokens/sec than base M-series. You can extract the key metric:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune the training script (`train.py` or `train_mlx.py`) with an experimental idea
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace
7. Record the results in the tsv (do not commit results.tsv)
8. If val_bpb improved (lower), keep the commit
9. If val_bpb is equal or worse, git reset back

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup overhead). If a run exceeds 10 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. The loop runs until the human interrupts you, period.
