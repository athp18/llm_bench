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

## Results

> Ran on a Vast.ai 4× A100 instance. TinyLlama-1.1B, batch size 4, sequence length 512, 50 steps per configuration.

**BF16 is not optional on A100s.** The single biggest lever by far — BF16 is 7.1× faster than FP32 (67k vs 9.4k tok/s) and cuts memory in half. A100s have native BF16 tensor core support so this is basically free performance. There's no practical reason to run FP32 fine-tuning on this hardware.

**FSDP is worth it, and the communication overhead is smaller than you'd expect.** The best FSDP run (BF16, SHARD_GRAD_OP, no gradient checkpointing) comes in at 60,960 tok/s — only 9% slower than running on a single GPU — while dropping per-GPU memory from 9.00 GB to 3.63 GB. Once your model starts pushing the memory limits of a single card, that's a very cheap tax to pay.

**FULL_SHARD vs SHARD_GRAD_OP is mostly a memory decision.** SHARD_GRAD_OP keeps full parameter replicas on each GPU and only shards gradients and optimizer states, so it's a touch faster (60,705 vs 59,733 tok/s, about 1.6%). But it uses 2.3 GB more memory per GPU (4.77 vs 2.49 GB). The throughput gap is small enough that FULL_SHARD is the safer default — that headroom matters once you scale up model size or batch size.

**Gradient checkpointing didn't save memory here, and that's not a bug.** GC cost around 21–22% throughput across every configuration, which is consistent and expected. The surprising part: it actually *increased* peak allocated memory slightly (+1.06 GB for BF16, +2.11 GB for FP32). At batch size 4, TinyLlama-1.1B isn't close to its memory ceiling, so the recomputed activations during the backward pass briefly spike the peak before they're freed — higher than the steady-state footprint without GC. This will flip at larger batch sizes or with bigger models where activation memory starts to dominate.

| Configuration | Throughput | Peak mem/GPU |
|---|---|---|
| Single GPU · BF16 | 67,086 tok/s | 9.00 GB |
| FSDP · SHARD_GRAD_OP · BF16 · no GC | 60,960 tok/s | 3.63 GB |
| FSDP · FULL_SHARD · BF16 · no GC | 59,733 tok/s | 2.49 GB |
| FSDP · FULL_SHARD · BF16 · GC | 46,343 tok/s | 3.55 GB |
| Single GPU · FP32 | 9,396 tok/s | 17.97 GB |

For a model this size the sweet spot is **FSDP FULL_SHARD + BF16 + no gradient checkpointing** — under 2.5 GB per GPU with only an 11% throughput hit, leaving plenty of headroom to push batch size or swap in a larger model.

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
