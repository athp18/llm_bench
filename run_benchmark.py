#!/usr/bin/env python3
"""
CLI entry point for the distributed LLM fine-tuning benchmark.

Usage examples:

  # Single custom run
  python run_benchmark.py --run_id my_run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --batch_size 4 --steps 50 --dtype bf16 --fsdp --sharding FULL_SHARD

  # Preset sweep: single-GPU vs FSDP
  python run_benchmark.py --sweep single_vs_fsdp

  # Preset sweep: grad checkpoint ablation
  python run_benchmark.py --sweep grad_checkpoint

  # Full ablation
  python run_benchmark.py --sweep full_ablation

  # With torchrun for multi-GPU:
  torchrun --nproc_per_node=4 run_benchmark.py --sweep full_ablation
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from benchmark import (
    BenchmarkConfig, ModelConfig, TrainingConfig, FSDPConfig, GradCheckpointConfig,
    run_benchmark,
    single_vs_fsdp_sweep, grad_checkpoint_sweep, full_ablation_sweep,
)


def parse_args():
    p = argparse.ArgumentParser(description="Distributed LLM Fine-Tuning Benchmark")

    # Sweep shortcuts
    p.add_argument("--sweep", choices=["single_vs_fsdp", "grad_checkpoint", "full_ablation"],
                   help="Run a preset sweep instead of a single config.")

    # Single-run options
    p.add_argument("--run_id",    default="run_001")
    p.add_argument("--model",     default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--batch_size",type=int, default=4)
    p.add_argument("--seq_len",   type=int, default=512)
    p.add_argument("--steps",     type=int, default=50)
    p.add_argument("--warmup",    type=int, default=5)
    p.add_argument("--grad_accum",type=int, default=4)
    p.add_argument("--dtype",     choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--fsdp",      action="store_true", default=False)
    p.add_argument("--sharding",  default="FULL_SHARD",
                   choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"])
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--log_interval", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()

    if args.sweep == "single_vs_fsdp":
        single_vs_fsdp_sweep()
    elif args.sweep == "grad_checkpoint":
        grad_checkpoint_sweep()
    elif args.sweep == "full_ablation":
        full_ablation_sweep()
    else:
        cfg = BenchmarkConfig(
            run_id=args.run_id,
            model=ModelConfig(
                model_name=args.model,
                max_seq_len=args.seq_len,
            ),
            training=TrainingConfig(
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                num_steps=args.steps,
                warmup_steps=args.warmup,
                dtype=args.dtype,
            ),
            fsdp=FSDPConfig(
                enabled=args.fsdp,
                sharding_strategy=args.sharding,
                cpu_offload=args.cpu_offload,
            ),
            grad_checkpoint=GradCheckpointConfig(enabled=args.grad_checkpoint),
            output_dir=args.output_dir,
            log_interval=args.log_interval,
        )
        run_benchmark(cfg)


if __name__ == "__main__":
    main()
