"""
Hardware auto-detection, chip tier classification, and hyperparameter suggestions.
Supports CUDA (NVIDIA GPUs), PyTorch MPS, and MLX backends.
"""

import os
import sys
import subprocess
import re


def detect_backend():
    """
    Auto-detect best available backend.
    Priority: CUDA > MLX > MPS.
    Override with AUTORESEARCH_BACKEND env var: 'cuda', 'mlx', 'mps', or 'auto'.
    """
    override = os.environ.get("AUTORESEARCH_BACKEND", "auto").lower()
    if override not in ("auto", "cuda", "mlx", "mps"):
        raise ValueError(f"AUTORESEARCH_BACKEND must be 'auto', 'cuda', 'mlx', or 'mps', got '{override}'")

    if override == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError("PyTorch is installed but CUDA is not available")
        except ImportError:
            raise RuntimeError("AUTORESEARCH_BACKEND=cuda but torch is not installed. "
                               "Install with: uv pip install 'autoresearch-cuda[cuda]'")

    if override == "mlx":
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                return "mlx"
            raise RuntimeError("MLX is installed but Metal is not available")
        except ImportError:
            raise RuntimeError("AUTORESEARCH_BACKEND=mlx but mlx is not installed")

    if override == "mps":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("PyTorch is installed but MPS is not available")
        except ImportError:
            raise RuntimeError("AUTORESEARCH_BACKEND=mps but torch is not installed")

    # Auto-detect: prefer CUDA > MLX > MPS
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    if sys.platform == "darwin":
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                return "mlx"
        except ImportError:
            pass

        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

    raise RuntimeError(
        "No compatible backend found. Install at least one:\n"
        "  CUDA: uv pip install 'autoresearch-cuda[cuda]'\n"
        "  MLX:  uv pip install mlx  (macOS only)\n"
        "  MPS:  uv pip install torch  (macOS only)"
    )


def get_hardware_info():
    """
    Returns hardware info dict with keys:
      memory_gb, chip_name, chip_tier, gpu_cores (estimated)
    Works for both NVIDIA GPUs and Apple Silicon.
    """
    info = {
        "memory_gb": 0,
        "chip_name": "unknown",
        "chip_tier": "unknown",
        "gpu_cores": 0,
    }

    # Try NVIDIA GPU first
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["chip_name"] = props.name
            info["memory_gb"] = props.total_memory / (1024 ** 3)
            info["gpu_cores"] = props.multi_processor_count

            # Classify tier based on GPU name
            name_lower = props.name.lower()
            if any(x in name_lower for x in ["h100", "h200", "a100", "h800", "b200", "b300"]):
                info["chip_tier"] = "datacenter"
            elif any(x in name_lower for x in ["l40", "a40", "rtx 6000", "a6000", "rtx pro"]):
                info["chip_tier"] = "professional"
            elif any(x in name_lower for x in ["rtx 4000", "rtx 3000", "rtx 5000"]):
                info["chip_tier"] = "professional"
            elif "rtx" in name_lower:
                info["chip_tier"] = "consumer"
            else:
                info["chip_tier"] = "unknown"
            return info
    except ImportError:
        pass

    # Fall back to Apple Silicon detection
    if sys.platform == "darwin":
        try:
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True).strip())
            info["memory_gb"] = mem_bytes / (1024 ** 3)
        except (subprocess.CalledProcessError, ValueError):
            pass

        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["chip_name"] = brand
        except subprocess.CalledProcessError:
            pass

        chip = info["chip_name"].lower()
        if "ultra" in chip:
            info["chip_tier"] = "ultra"
            info["gpu_cores"] = 80
        elif "max" in chip:
            info["chip_tier"] = "max"
            info["gpu_cores"] = 40
        elif "pro" in chip:
            info["chip_tier"] = "pro"
            info["gpu_cores"] = 18
        elif "m1" in chip or "m2" in chip or "m3" in chip or "m4" in chip:
            info["chip_tier"] = "base"
            info["gpu_cores"] = 10
        else:
            info["chip_tier"] = "unknown"

        m = re.search(r"(m[1-5])\s*(pro|max|ultra)?", chip)
        if m:
            gen = m.group(1)
            tier = m.group(2) or "base"
            gpu_core_map = {
                ("m1", "base"): 8, ("m1", "pro"): 16, ("m1", "max"): 32, ("m1", "ultra"): 64,
                ("m2", "base"): 10, ("m2", "pro"): 19, ("m2", "max"): 38, ("m2", "ultra"): 76,
                ("m3", "base"): 10, ("m3", "pro"): 18, ("m3", "max"): 40, ("m3", "ultra"): 76,
                ("m4", "base"): 10, ("m4", "pro"): 20, ("m4", "max"): 40, ("m4", "ultra"): 80,
            }
            info["gpu_cores"] = gpu_core_map.get((gen, tier), info["gpu_cores"])

    return info


def get_peak_flops(hw_info=None):
    """
    Estimate peak bf16 TFLOPS for MFU calculation.
    Returns FLOPS (not TFLOPS) for direct use in MFU computation.
    """
    if hw_info is None:
        hw_info = get_hardware_info()

    chip = hw_info["chip_name"].lower()

    # NVIDIA GPU FLOPS lookup (bf16 dense, not sparse)
    nvidia_flops = {
        # Blackwell
        "b200": 2250e12,
        "b300": 2250e12,
        "rtx pro 6000": 420e12,     # Blackwell professional
        "rtx 5090": 419e12,         # Blackwell consumer
        # Hopper
        "h100": 756e12,
        "h200": 756e12,
        "h800": 756e12,
        # Ampere
        "a100": 312e12,
        "l40s": 362e12,
        "l40": 181e12,
        "rtx 6000 ada": 363e12,
        "rtx 4000 ada": 105e12,
        "rtx 4090": 330e12,
        "rtx 4080": 194e12,
        "rtx 3090": 142e12,
    }

    for key, flops in nvidia_flops.items():
        if key in chip:
            # Scale FLOPS for MIG/vGPU partitions by VRAM ratio
            # (PyTorch reports full SM count even on MIG slices)
            full_vram = {"a100": 80, "h100": 80, "h200": 141, "a40": 48}
            actual_vram = hw_info.get("memory_gb", 0)
            for gpu_key, full_gb in full_vram.items():
                if gpu_key in chip and actual_vram > 0 and actual_vram < full_gb * 0.9:
                    return flops * (actual_vram / full_gb)
            return flops

    # Fall back to Apple Silicon estimate
    gpu_cores = hw_info["gpu_cores"]
    m = re.search(r"(m[1-5])", chip)
    gen = m.group(1) if m else "m4"

    flops_per_core = {
        "m1": 0.5e12,
        "m2": 0.55e12,
        "m3": 0.65e12,
        "m4": 0.7e12,
    }.get(gen, 0.65e12)

    return gpu_cores * flops_per_core


def suggest_hyperparameters(hw_info=None):
    """
    Suggest hyperparameters based on hardware tier.
    Returns dict with: depth, device_batch_size, total_batch_size, eval_tokens_multiplier
    """
    if hw_info is None:
        hw_info = get_hardware_info()

    mem_gb = hw_info["memory_gb"]
    tier = hw_info["chip_tier"]

    # NVIDIA GPU tiers — VRAM-aware to handle MIG slices correctly
    # (e.g., A100 MIG 20C has only 20 GB VRAM but "datacenter" tier by name)
    if tier in ("datacenter", "professional", "consumer"):
        if mem_gb >= 60:
            # Full datacenter GPUs: H100 (80 GB), A100 (80 GB), H200 (141 GB)
            return {
                "depth": 12,
                "device_batch_size": 64,
                "total_batch_size": 2**17,  # 128K tokens
                "eval_tokens_multiplier": 10,
            }
        elif mem_gb >= 30:
            # Mid-range: L40S (48 GB), RTX 6000 Ada (48 GB), A100 MIG 40 GB
            return {
                "depth": 10,
                "device_batch_size": 32,
                "total_batch_size": 2**16,  # 64K tokens
                "eval_tokens_multiplier": 10,
            }
        elif mem_gb >= 16:
            # Constrained: RTX 4000 Ada (20 GB), A100 MIG 20C, A40 MIG
            return {
                "depth": 10,
                "device_batch_size": 32,
                "total_batch_size": 2**16,  # 64K tokens
                "eval_tokens_multiplier": 10,
            }
        else:
            # Small VRAM: A100 MIG 10 GB, A16, consumer GPUs < 16 GB
            return {
                "depth": 8,
                "device_batch_size": 16,
                "total_batch_size": 2**15,  # 32K tokens
                "eval_tokens_multiplier": 10,
            }

    # Apple Silicon tiers
    if tier == "ultra" or mem_gb >= 128:
        return {
            "depth": 10,
            "device_batch_size": 32,
            "total_batch_size": 2**16,
            "eval_tokens_multiplier": 10,
        }
    elif tier == "max" or mem_gb >= 48:
        return {
            "depth": 8,
            "device_batch_size": 16,
            "total_batch_size": 2**15,
            "eval_tokens_multiplier": 10,
        }
    elif tier == "pro" or mem_gb >= 18:
        return {
            "depth": 6,
            "device_batch_size": 8,
            "total_batch_size": 2**13,
            "eval_tokens_multiplier": 5,
        }
    else:
        return {
            "depth": 4,
            "device_batch_size": 4,
            "total_batch_size": 2**12,
            "eval_tokens_multiplier": 3,
        }


def sync_device(device_type):
    """Synchronize device for accurate timing."""
    if device_type == "cuda":
        import torch
        torch.cuda.synchronize()
    elif device_type == "mps":
        import torch
        torch.mps.synchronize()


def get_peak_memory_mb(device_type):
    """Get peak memory usage in MB."""
    if device_type == "cuda":
        import torch
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device_type == "mps":
        import torch
        try:
            return torch.mps.driver_allocated_memory() / 1024 / 1024
        except AttributeError:
            return 0.0
    elif device_type == "mlx":
        import mlx.core as mx
        try:
            return mx.get_peak_memory() / 1024 / 1024
        except AttributeError:
            return 0.0
    return 0.0


def print_hardware_summary():
    """Print a summary of detected hardware and suggested config."""
    hw = get_hardware_info()
    hp = suggest_hyperparameters(hw)
    peak_flops = get_peak_flops(hw)

    print(f"Hardware: {hw['chip_name']}")
    print(f"  Memory: {hw['memory_gb']:.0f} GB")
    print(f"  GPU cores/SMs: {hw['gpu_cores']}")
    print(f"  Tier: {hw['chip_tier']}")
    print(f"  Peak bf16 FLOPS: {peak_flops:.2e}")
    print(f"Suggested config:")
    print(f"  Depth: {hp['depth']}")
    print(f"  Device batch size: {hp['device_batch_size']}")
    print(f"  Total batch size: {hp['total_batch_size']:,}")


if __name__ == "__main__":
    print_hardware_summary()
    print(f"\nDetected backend: {detect_backend()}")
