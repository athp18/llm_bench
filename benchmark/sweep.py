"""
Sweep runner: runs a matrix of benchmark configs and saves all results.
Useful for comparing single-GPU vs FSDP, different dtypes, grad checkpointing on/off, etc.
"""
import json
import os
from copy import deepcopy
from typing import List, Dict, Any

from .config import BenchmarkConfig, FSDPConfig, GradCheckpointConfig, TrainingConfig
from .trainer import run_benchmark
from .metrics import BenchmarkResult


def make_sweep_configs(base: BenchmarkConfig, overrides: List[Dict[str, Any]]) -> List[BenchmarkConfig]:
    """
    Given a base config and a list of override dicts, produce one config per override.

    Override keys use dot-notation for nested fields, e.g.:
        {"fsdp.enabled": True, "grad_checkpoint.enabled": False, "run_id": "fsdp_no_gc"}
    """
    configs = []
    for ovr in overrides:
        cfg = deepcopy(base)
        for key, val in ovr.items():
            parts = key.split(".")
            obj = cfg
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)
        configs.append(cfg)
    return configs


def run_sweep(configs: List[BenchmarkConfig]) -> List[BenchmarkResult]:
    """Run all configs sequentially and return results."""
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n{'#'*60}")
        print(f"  Sweep run {i+1}/{len(configs)}: {cfg.run_id}")
        print(f"{'#'*60}")
        result = run_benchmark(cfg)
        results.append(result)

    # Save sweep summary
    summary = [
        {
            "run_id": r.run_id,
            "num_gpus": r.num_gpus,
            "fsdp": r.fsdp_enabled,
            "sharding": r.sharding_strategy,
            "grad_ckpt": r.grad_checkpoint_enabled,
            "dtype": r.dtype,
            "mean_throughput_tps": r.mean_throughput_tps,
            "peak_gpu_memory_gb": r.peak_gpu_memory_gb,
            "mean_step_time_ms": r.mean_step_time_s * 1000,
            "total_time_s": r.total_time_s,
        }
        for r in results
    ]

    out_dir = configs[0].output_dir if configs else "./results"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[sweep] Summary saved to {summary_path}")
    return results


# ---------------------------------------------------------------------------
# Preset sweeps
# ---------------------------------------------------------------------------

def single_vs_fsdp_sweep(base: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Compare single-GPU (no FSDP) vs FSDP FULL_SHARD."""
    if base is None:
        base = BenchmarkConfig()
    overrides = [
        {"run_id": "single_gpu_fp32",   "fsdp.enabled": False, "training.dtype": "fp32"},
        {"run_id": "single_gpu_bf16",   "fsdp.enabled": False, "training.dtype": "bf16"},
        {"run_id": "fsdp_full_shard",   "fsdp.enabled": True,  "fsdp.sharding_strategy": "FULL_SHARD"},
        {"run_id": "fsdp_shard_grad",   "fsdp.enabled": True,  "fsdp.sharding_strategy": "SHARD_GRAD_OP"},
    ]
    return run_sweep(make_sweep_configs(base, overrides))


def grad_checkpoint_sweep(base: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Compare throughput and memory with grad checkpointing on vs off."""
    if base is None:
        base = BenchmarkConfig()
    overrides = [
        {"run_id": "no_gc",   "grad_checkpoint.enabled": False},
        {"run_id": "with_gc", "grad_checkpoint.enabled": True},
    ]
    return run_sweep(make_sweep_configs(base, overrides))


def full_ablation_sweep(base: BenchmarkConfig = None) -> List[BenchmarkResult]:
    """Full ablation: dtype x grad_ckpt x sharding."""
    if base is None:
        base = BenchmarkConfig()
    overrides = [
        {"run_id": "bf16_no_gc_full_shard",   "training.dtype": "bf16", "grad_checkpoint.enabled": False, "fsdp.sharding_strategy": "FULL_SHARD"},
        {"run_id": "bf16_gc_full_shard",       "training.dtype": "bf16", "grad_checkpoint.enabled": True,  "fsdp.sharding_strategy": "FULL_SHARD"},
        {"run_id": "bf16_no_gc_shard_grad",    "training.dtype": "bf16", "grad_checkpoint.enabled": False, "fsdp.sharding_strategy": "SHARD_GRAD_OP"},
        {"run_id": "bf16_gc_shard_grad",       "training.dtype": "bf16", "grad_checkpoint.enabled": True,  "fsdp.sharding_strategy": "SHARD_GRAD_OP"},
        {"run_id": "fp32_no_gc_full_shard",    "training.dtype": "fp32", "grad_checkpoint.enabled": False, "fsdp.sharding_strategy": "FULL_SHARD"},
        {"run_id": "fp32_gc_full_shard",       "training.dtype": "fp32", "grad_checkpoint.enabled": True,  "fsdp.sharding_strategy": "FULL_SHARD"},
    ]
    return run_sweep(make_sweep_configs(base, overrides))
