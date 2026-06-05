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
├── scripts/
│   └── generate_mock_results.py   # Mock data for dashboard dev
├── results/             # JSON outputs per run + sweep_summary.json
├── dashboard.html       # Standalone results dashboard
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

## Generate mock data (no GPU needed)

```bash
cd llm_bench
python scripts/generate_mock_results.py
# then open dashboard.html in a browser
```

## Dashboard

Open `dashboard.html` directly in a browser. It loads results from the embedded mock
data by default. To use your real results, replace the `RAW` array in the script with
the contents of `results/sweep_summary.json`.

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
- Results are saved as JSON to `./results/` and can be fed directly into the dashboard.
