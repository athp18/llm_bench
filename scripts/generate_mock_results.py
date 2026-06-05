"""
Generates realistic-looking mock benchmark results for dashboard development.
Run this first if you don't have a GPU to actually run the benchmark.
"""
import json
import os
import random
import math

random.seed(42)

RUNS = [
    # run_id, num_gpus, fsdp, sharding, grad_ckpt, dtype, base_tps, base_mem
    ("single_gpu_fp32",    1, False, "none",          False, "fp32",  2_800,  14.2),
    ("single_gpu_bf16",    1, False, "none",          False, "bf16",  5_400,  8.1),
    ("fsdp_full_shard",    4, True,  "FULL_SHARD",    False, "bf16",  18_200, 7.3),
    ("fsdp_shard_grad",    4, True,  "SHARD_GRAD_OP", False, "bf16",  14_900, 10.8),
    ("bf16_no_gc_full_shard", 4, True, "FULL_SHARD",  False, "bf16",  18_200, 7.3),
    ("bf16_gc_full_shard",    4, True, "FULL_SHARD",  True,  "bf16",  12_400, 4.9),
    ("bf16_no_gc_shard_grad", 4, True, "SHARD_GRAD_OP", False, "bf16", 14_900, 10.8),
    ("bf16_gc_shard_grad",    4, True, "SHARD_GRAD_OP", True, "bf16", 10_100, 6.2),
    ("fp32_no_gc_full_shard", 4, True, "FULL_SHARD",  False, "fp32",  8_600, 14.6),
    ("fp32_gc_full_shard",    4, True, "FULL_SHARD",  True,  "fp32",  5_900, 9.1),
]

def make_steps(base_tps, base_mem, num_steps=50):
    steps = []
    for s in range(num_steps):
        warmup_factor = min(1.0, (s + 1) / 5)
        tps = base_tps * warmup_factor * random.uniform(0.97, 1.03)
        mem = base_mem * min(1.0, 0.7 + 0.3 * (s / num_steps)) * random.uniform(0.98, 1.02)
        step_time = (4 * 512 * 4) / tps  # batch*seq*grad_accum / tps
        loss = 3.5 * math.exp(-0.02 * s) + random.uniform(-0.05, 0.05)
        steps.append({
            "step": s,
            "loss": round(loss, 4),
            "step_time_s": round(step_time, 4),
            "tokens_per_second": round(tps, 1),
            "gpu_memory_allocated_gb": round(mem, 3),
            "gpu_memory_reserved_gb": round(mem * 1.15, 3),
            "gpu_utilization_pct": round(random.uniform(88, 98), 1),
        })
    return steps

os.makedirs("results", exist_ok=True)

summary = []
for (run_id, num_gpus, fsdp, sharding, gc, dtype, base_tps, base_mem) in RUNS:
    steps = make_steps(base_tps, base_mem)
    tps_vals = [s["tokens_per_second"] for s in steps]
    mem_vals = [s["gpu_memory_allocated_gb"] for s in steps]
    times = [s["step_time_s"] for s in steps]
    result = {
        "run_id": run_id,
        "num_gpus": num_gpus,
        "fsdp_enabled": fsdp,
        "sharding_strategy": sharding,
        "grad_checkpoint_enabled": gc,
        "dtype": dtype,
        "batch_size": 4,
        "seq_len": 512,
        "steps": steps,
        "mean_throughput_tps": round(sum(tps_vals)/len(tps_vals), 1),
        "peak_gpu_memory_gb": round(max(mem_vals), 3),
        "mean_step_time_s": round(sum(times)/len(times), 4),
        "total_time_s": round(sum(times), 2),
        "tokens_trained": int(sum(s["tokens_per_second"]*s["step_time_s"] for s in steps)),
    }
    with open(f"results/{run_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    summary.append({
        "run_id": run_id,
        "num_gpus": num_gpus,
        "fsdp": fsdp,
        "sharding": sharding,
        "grad_ckpt": gc,
        "dtype": dtype,
        "mean_throughput_tps": result["mean_throughput_tps"],
        "peak_gpu_memory_gb": result["peak_gpu_memory_gb"],
        "mean_step_time_ms": round(result["mean_step_time_s"]*1000, 1),
        "total_time_s": result["total_time_s"],
    })

with open("results/sweep_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Generated {len(RUNS)} mock runs in ./results/")
print("Summary:")
for r in summary:
    print(f"  {r['run_id']:35s} {r['mean_throughput_tps']:>8,.0f} tok/s  {r['peak_gpu_memory_gb']:.1f} GB")
