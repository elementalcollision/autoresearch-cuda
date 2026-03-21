# autoresearch-CUDA

NVIDIA GPU optimized fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) with full `torch.compile`, FlashAttention-2, and Muon optimizer support on CUDA. Designed for rapid iteration on DigitalOcean GPU droplets.

> **Latest results**: See the [experiment wiki](https://github.com/elementalcollision/autoresearch-cuda/wiki) for full details per GPU and dataset.
>
> | GPU | Dataset | Best val_bpb | tok/sec | MFU | Experiments |
> |-----|---------|-------------|---------|-----|-------------|
> | **RTX 4000 Ada** (20 GB) | [Climbmix](https://github.com/elementalcollision/autoresearch-cuda/wiki) | **1.230** | 67,709 | 26.7% | 3 (in progress) |
>
> **Cross-platform comparison**: The RTX 4000 Ada ($0.76/hr) achieves 67K tok/sec vs 58K on the M5 Max — with `torch.compile` and FlashAttention-2 providing the performance advantage despite being a lower-tier GPU. See the [Apple Silicon fork](https://github.com/elementalcollision/autoresearch/wiki) for Metal/MLX baselines.

## What is this?

[Autoresearch](https://github.com/karpathy/autoresearch) is Karpathy's framework for autonomous AI-driven LLM training experiments. An AI agent modifies the training code, runs a 5-minute experiment, checks if results improved, keeps or discards, and repeats.

This fork targets **NVIDIA CUDA GPUs** — from consumer RTX cards to datacenter H100s. It leverages `torch.compile` for kernel fusion, FlashAttention-2 via PyTorch SDPA, and bf16 tensor cores for maximum throughput. It also includes a complete DigitalOcean GPU integration for provisioning, deploying, monitoring, and collecting results from cloud GPU instances.

### Key features

- **`torch.compile`**: Full model compilation with `mode="reduce-overhead"` for CUDA graph capture — the single biggest performance advantage over eager mode
- **FlashAttention-2**: Automatic dispatch via `F.scaled_dot_product_attention` on CUDA
- **Full Muon optimizer**: Newton-Schulz orthogonalization with `@torch.compile` for kernel fusion, bf16 tensor cores, CUDA-optimized scalar placement
- **Hardware auto-detection**: Identifies GPU tier (consumer/professional/datacenter), VRAM, SM count, and peak FLOPS. Scales hyperparameters accordingly.
- **DigitalOcean GPU integration**: One-command provisioning, deployment, monitoring, and teardown of cloud GPU instances
- **Headless orchestrator**: AI-driven experiment loop via Claude Sonnet API, runs unattended on remote GPU droplets

## Quick start

### Local (NVIDIA GPU)

**Requirements**: NVIDIA GPU with CUDA support, Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Clone and install
git clone https://github.com/elementalcollision/autoresearch-cuda.git
cd autoresearch-cuda
uv sync --extra cuda

# 2. Download data and train tokenizer (~2 min)
uv run prepare.py

# 3. Run a training experiment (~5 min)
uv run train_cuda.py
```

### DigitalOcean GPU droplet

```bash
# 1. Provision a GPU droplet (RTX 4000 Ada — $0.76/hr)
./gpu_provision.sh gpu-rtx4000-ada-1x nyc2

# 2. Bootstrap the droplet (install deps, clone repo, download data)
./setup_cuda.sh <droplet-ip>

# 3. Push latest code
./cuda_push.sh <droplet-ip>

# 4. Launch AI experiment loop (runs unattended)
./cuda_experiment.sh <droplet-ip> climbmix 100 mar20

# 5. Monitor progress
./cuda_monitor.sh <droplet-ip> climbmix --watch

# 6. Collect results when done
./cuda_collect.sh <droplet-ip> climbmix

# 7. Destroy droplet
./gpu_destroy.sh <droplet-name>
```

## Backend selection

The system auto-detects CUDA when available. Override with an environment variable:

```bash
# Auto-detect (prefers CUDA)
uv run train.py

# Force CUDA
AUTORESEARCH_BACKEND=cuda uv run train.py

# Run CUDA training directly
uv run train_cuda.py
```

Check detected hardware and suggested config:

```bash
uv run -c "from backends import print_hardware_summary; print_hardware_summary()"
```

## TUI Dashboard & Agent Mode

A real-time terminal dashboard with autonomous LLM-driven experiment optimization.

```bash
uv sync --extra all

uv run dashboard.py                                # Single training run with live metrics
uv run dashboard.py --agent --tag my-run           # Autonomous experiment loop
uv run dashboard.py --agent --tag my-run --max 50  # Limit to 50 experiments
uv run dashboard.py --watch                        # Watch mode (monitor results.tsv)
```

Agent mode requires an Anthropic API key — run `uv run dashboard.py --setup-key` for one-time setup.

### Headless mode (remote GPU)

For unattended operation on cloud GPU instances without a terminal UI:

```bash
uv run -m tui.headless --max 100 --tag mar20
```

This is what `cuda_experiment.sh` launches on the droplet via `nohup`.

## Hardware recommendations

### Auto-detected defaults

| GPU tier | VRAM | Model depth | Device batch | Total batch |
|----------|------|-------------|-------------|-------------|
| Consumer (RTX 4060-4090) | 8-24 GB | 8 | 16 | 32K tokens |
| Professional (RTX 4000/6000 Ada, L40S) | 20-48 GB | 10 | 32 | 64K tokens |
| Datacenter (H100, H200) | 80+ GB | 12 | 64 | 128K tokens |

### DigitalOcean GPU options

| Droplet type | GPU | VRAM | Cost | Use case |
|-------------|-----|------|------|----------|
| `gpu-rtx4000-ada-1x` | RTX 4000 Ada | 20 GB | $0.76/hr | Development, single-dataset runs |
| `gpu-h100x1-base` | H100 SXM | 80 GB | $3.39/hr | Full-suite multi-dataset sweeps |

## Multi-dataset experiments

Run experiments across different training datasets:

```bash
uv run convert_dataset.py fineweb-edu     # Download + convert a dataset
uv run run_suite.py                       # Full multi-dataset sweep
uv run compare_datasets.py                # Cross-dataset analysis + charts
```

**Available datasets**: climbmix (default), fineweb-edu, fineweb-edu-high, cosmopedia-v2, slimpajama, fineweb, github-code-python.

For remote multi-dataset sweeps:

```bash
./cuda_suite.sh <droplet-ip> 100 120      # All datasets, 100 experiments each, 120min timeout
```

## Project structure

```
train_cuda.py           CUDA training script (agent modifies this)
train.py                Backend dispatch (auto-detects CUDA/MPS/MLX)
prepare.py              Data prep, tokenizer, dataloader, evaluation (do not modify)
dashboard.py            TUI dashboard entry point
program.md              Agent instructions for autonomous experiments
run_suite.py            Multi-dataset experiment orchestrator
compare_datasets.py     Cross-dataset analysis and visualization
compare_backends.py     Cross-platform CUDA vs Apple Silicon comparison
convert_dataset.py      Download and convert alternative datasets
monitor.py              CLI experiment results monitor
backends/
  __init__.py           Hardware detection, GPU tier, FLOPS lookup, hyperparameter suggestions
  muon_cuda.py          Muon+AdamW optimizer for CUDA (torch.compile, bf16 tensor cores)
  muon_mps.py           Muon+AdamW optimizer for PyTorch MPS
  muon_mlx.py           Muon+AdamW optimizer for MLX
tui/
  app.py                Textual Application, layout, subprocess management
  headless.py           Headless orchestrator for remote/unattended operation
  widgets.py            TrainingPanel, HardwarePanel, ExperimentsTable, ActivityLog
  orchestrator.py       Autonomous experiment loop (LLM → modify → train → evaluate)
  llm_backend.py        Claude Sonnet API integration for experiment generation
  credentials.py        API key resolution: env var → macOS Keychain → Claude Code creds
  git_manager.py        Git operations: branch, commit, revert
  results.py            results.tsv read/write/history formatting
  parser.py             Regex parser for training stdout
  hardware.py           GPU hardware detection (NVIDIA + Apple Silicon fallback)
  experiments.py        results.tsv loader for TUI table display
  styles.tcss           CSS layout for panel styling
results/
  <dataset>/results.tsv Per-dataset experiment results
```

**What the agent edits**: `train_cuda.py`. Everything is fair game: architecture, optimizer settings, hyperparameters, batch size, model depth.

**What is fixed**: `prepare.py` (evaluation, data loading, constants), `backends/` (optimizer, hardware detection).

## Output format

After a 5-minute run, the script prints:

```
---
val_bpb:          1.238870
training_seconds: 300.2
total_seconds:    367.8
peak_vram_mb:     15872.0
mfu_percent:      26.70
total_tokens_M:   20.3
num_steps:        620
num_params_M:     67.1
depth:            10
backend:          cuda
chip:             NVIDIA RTX 4000 Ada Generation
```

The key metric is **val_bpb** (validation bits per byte) — lower is better.

## Technical notes

### CUDA backend
- `torch.compile(mode="reduce-overhead")` for CUDA graph capture and kernel fusion
- FlashAttention-2 via `F.scaled_dot_product_attention` (auto-dispatched on CUDA)
- bf16 autocast via `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`
- Muon optimizer compiled with `@torch.compile` for fused Newton-Schulz iterations
- `torch.cuda.empty_cache()` for explicit VRAM management

### Muon optimizer
The Muon optimizer combines Newton-Schulz orthogonalization (Polar Express) with Nesterov momentum, NorMuon variance reduction, and cautious weight decay. Applied to 2D matrix parameters in transformer blocks, while embeddings and scalars use standard AdamW. The CUDA version uses `@torch.compile` for kernel fusion and bf16 tensor cores for the Newton-Schulz iterations.

### DigitalOcean GPU integration
Scripts in the companion [DigitalOceanGPU](https://github.com/elementalcollision/autoresearch-cuda) repo handle the full lifecycle:

| Script | Purpose |
|--------|---------|
| `gpu_provision.sh` | Create GPU droplet via DO API |
| `setup_cuda.sh` | Bootstrap: install uv, clone repo, install deps, download data |
| `cuda_push.sh` | Rapid code sync via rsync (~2 seconds) |
| `cuda_experiment.sh` | Launch headless AI experiment loop via nohup |
| `cuda_monitor.sh` | Remote monitoring (--once, --watch, --log, --sync) |
| `cuda_collect.sh` | Pull results, configs, environment info |
| `gpu_destroy.sh` | Tear down droplet |

## Differences from the Apple Silicon fork

| Feature | Apple Silicon fork | This fork (CUDA) |
|---------|-------------------|-------------------|
| Attention | PyTorch SDPA (MPS) / native (MLX) | FlashAttention-2 via SDPA |
| Compilation | Eager mode (MPS) / `mx.compile` (MLX) | `torch.compile` with CUDA graphs |
| Memory model | Unified CPU/GPU memory | Discrete GPU VRAM |
| MFU metric | Approximate (estimated per-chip) | Precise (known GPU FLOPS) |
| Deployment | Local Mac | Local or DigitalOcean GPU cloud |
| Precision | bf16 manual casting (MPS) / native (MLX) | bf16 via autocast + tensor cores |
| Target hardware | M1-M5 (8-192 GB unified) | RTX consumer to H100 datacenter |

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy/autoresearch) — original autoresearch framework
- [elementalcollision/autoresearch](https://github.com/elementalcollision/autoresearch) — Apple Silicon fork with MLX/MPS backends
- [Jordan Keller](https://kellerjordan.github.io/posts/muon/) — Muon optimizer

## License

MIT
