# Distributed LLM Fine-Tuning Benchmark

Benchmarks training throughput, GPU memory usage, and step time across:
- Single-GPU vs multi-GPU (FSDP)
- Sharding strategies: `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`
- Dtypes: `fp32`, `fp16`, `bf16`
- Gradient checkpointing on/off

## Structure

```
llm_bench/
├── benchmark/
│   ├── __init__.py
│   ├── config.py        # BenchmarkConfig dataclasses
│   ├── metrics.py       # StepMetrics, BenchmarkResult, MetricsCollector
│   ├── trainer.py       # Training loop (single-GPU + FSDP)
│   └── sweep.py         # Sweep runner + preset sweeps
├── results/             # JSON outputs per run + sweep_summary.json
├── dashboard.html       # Self-contained results dashboard
├── inject_results.sh    # Embeds results/ data into dashboard.html
├── run_benchmark.py     # CLI entry point
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# Single run
python run_benchmark.py --run_id my_run --dtype bf16 --steps 50

# Single run with FSDP (requires multi-GPU)
torchrun --nproc_per_node=4 run_benchmark.py --run_id fsdp_run --fsdp --sharding FULL_SHARD

# Preset sweeps
python run_benchmark.py --sweep single_vs_fsdp
python run_benchmark.py --sweep grad_checkpoint
python run_benchmark.py --sweep full_ablation

# With torchrun for multi-GPU sweeps
torchrun --nproc_per_node=4 run_benchmark.py --sweep full_ablation
```

## Dashboard

After running a benchmark, inject the results and open the file:

```bash
./inject_results.sh
open dashboard.html   # or just double-click it
```

`inject_results.sh` reads every JSON file in `results/` and embeds the data directly into `dashboard.html`, which is then a self-contained static file with no server needed.

Features:
- **Throughput & memory** bar chart with filter chips (BF16, FP32, FSDP, Single GPU, GC on/off)
- **Efficiency frontier** scatter plot — throughput vs peak memory per configuration
- **Grad checkpoint impact** — side-by-side throughput and memory comparison for matched pairs
- **Training dynamics** — per-step curves for throughput, loss, and GPU memory (all runs, individually toggleable)
- **All runs table** — sorted by throughput, with inline bar charts

## Programmatic API

```python
from benchmark import BenchmarkConfig, FSDPConfig, GradCheckpointConfig, run_benchmark

cfg = BenchmarkConfig(
    run_id="custom_run",
    fsdp=FSDPConfig(enabled=True, sharding_strategy="FULL_SHARD"),
    grad_checkpoint=GradCheckpointConfig(enabled=True),
)
result = run_benchmark(cfg)
print(f"Mean throughput: {result.mean_throughput_tps:,.0f} tok/s")
print(f"Peak GPU memory: {result.peak_gpu_memory_gb:.2f} GB")
```

## Custom sweep

```python
from benchmark import BenchmarkConfig, run_sweep, make_sweep_configs

base = BenchmarkConfig(model=ModelConfig(model_name="meta-llama/Llama-2-7b-hf"))
overrides = [
    {"run_id": "7b_bf16_fsdp",    "training.dtype": "bf16", "fsdp.enabled": True},
    {"run_id": "7b_bf16_fsdp_gc", "training.dtype": "bf16", "fsdp.enabled": True, "grad_checkpoint.enabled": True},
]
results = run_sweep(make_sweep_configs(base, overrides))
```

## Notes

- Uses **synthetic token batches** by default — no dataset download needed for benchmarking.
  Swap in your real DataLoader in `trainer.py:make_synthetic_batch` for training fidelity.
- FSDP requires `torchrun` to initialize the process group. Single-GPU mode works with
  plain `python run_benchmark.py`.
- `pynvml` is optional; GPU utilization % is skipped if not installed.
- Results are saved as JSON to `./results/`. Run `./inject_results.sh` to update the dashboard.
