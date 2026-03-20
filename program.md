# autoresearch (CUDA / NVIDIA GPU)

This is an experiment to have the LLM do its own research, optimized for NVIDIA CUDA GPUs (H100, H200, L40S, RTX series). Forked from autoresearch-arm (Apple Silicon).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train_cuda.py` — CUDA backend training script. Modify this for experiments.
   - `backends/` — hardware detection, optimizers. Do not modify.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Select backend**: Set `AUTORESEARCH_BACKEND=cuda` or let auto-detect find CUDA.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Platform: NVIDIA CUDA

This fork runs on NVIDIA GPUs. Key optimizations over the Apple Silicon version:

- **FlashAttention-2**: Automatic via PyTorch SDPA on CUDA — `is_causal=True` dispatches to flash attention.
- **torch.compile**: Model forward pass compiled with `mode="reduce-overhead"` for CUDA graph capture and kernel fusion.
- **bf16 autocast**: Native bfloat16 on H100/H200 via `torch.amp.autocast`.
- **Muon optimizer**: torch.compile'd step functions with on-device scalar tensors (no CPU→GPU transfers).
- **Dedicated VRAM**: No unified memory — batch sizes tuned for GPU VRAM (80GB on H100).
- **MFU metric**: Based on published bf16 dense TFLOPS for each GPU (H100: 756 TFLOPS).

### Backend selection

```bash
# Auto-detect (default: prefers CUDA)
uv run train.py

# Force CUDA backend
AUTORESEARCH_BACKEND=cuda uv run train.py

# Run CUDA script directly
uv run train_cuda.py
```

### Installing dependencies

```bash
# CUDA backend
uv pip install -e '.[cuda]'

# With agent/TUI support
uv pip install -e '.[all]'
```

## Experimentation

Each experiment runs on a single NVIDIA GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train_cuda.py` — this is the file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify `backends/`. The optimizer and hardware detection code is shared infrastructure.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a hard constraint. H100 has 80GB dedicated VRAM. Monitor with `nvidia-smi`. OOM will crash the run.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     32768.0
mfu_percent:      45.20
total_tokens_M:   500.0
num_steps:        1000
num_params_M:     50.3
depth:            12
backend:          cuda
chip:             NVIDIA H100 80GB HBM3
```

Note that the script is configured to always stop after 5 minutes. Performance will vary by GPU — H100 will process significantly more tokens/sec than RTX consumer cards. You can extract the key metric:

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
2. Tune the training script (`train_cuda.py`) with an experimental idea
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace
7. Record the results in the tsv (do not commit results.tsv)
8. If val_bpb improved (lower), keep the commit
9. If val_bpb is equal or worse, git reset back

**Timeout**: Each experiment should take ~5 minutes total (+ startup for torch.compile warmup). If a run exceeds 10 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. The loop runs until the human interrupts you, period.
