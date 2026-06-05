"""
Training loop: single-GPU or FSDP multi-GPU.
Wraps a HuggingFace causal LM and runs a fixed number of steps for benchmarking.
"""
import os
import time
import torch
import torch.distributed as dist
from torch.optim import AdamW
from contextlib import nullcontext
from typing import Optional

from .config import BenchmarkConfig
from .metrics import BenchmarkResult, MetricsCollector


# ---------------------------------------------------------------------------
# FSDP imports (graceful fallback if not available)
# ---------------------------------------------------------------------------
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp import StateDictType
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


def _sharding_strategy(name: str):
    mapping = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    return mapping.get(name, ShardingStrategy.FULL_SHARD)


def _dtype(name: str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def setup_distributed():
    """Initialize process group when launched via torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        return rank, dist.get_world_size()
    return 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_model(cfg: BenchmarkConfig, rank: int = 0):
    """Load model from HuggingFace hub and optionally wrap with FSDP."""
    from transformers import AutoModelForCausalLM, AutoConfig

    is_main = rank == 0

    if is_main:
        print(f"[trainer] Loading model: {cfg.model.model_name}")

    # Load on CPU first to avoid OOM on rank 0 when sharding
    hf_config = AutoConfig.from_pretrained(cfg.model.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        config=hf_config,
        torch_dtype=_dtype(cfg.training.dtype),
    )

    # Gradient checkpointing
    if cfg.grad_checkpoint.enabled:
        model.gradient_checkpointing_enable()
        if is_main:
            print("[trainer] Gradient checkpointing enabled")

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if cfg.fsdp.enabled and FSDP_AVAILABLE and num_gpus > 1:
        # Identify the transformer layer class for auto-wrap
        layer_cls = _get_transformer_layer_cls(model)
        mp_policy = None
        if cfg.fsdp.mixed_precision:
            dt = _dtype(cfg.training.dtype)
            mp_policy = MixedPrecision(param_dtype=dt, reduce_dtype=dt, buffer_dtype=dt)

        model = FSDP(
            model,
            sharding_strategy=_sharding_strategy(cfg.fsdp.sharding_strategy),
            mixed_precision=mp_policy,
            cpu_offload=CPUOffload(offload_params=cfg.fsdp.cpu_offload),
            auto_wrap_policy=transformer_auto_wrap_policy(
                transformer_layer_cls={layer_cls}
            ) if layer_cls else None,
            device_id=torch.cuda.current_device(),
        )
        if is_main:
            print(f"[trainer] FSDP enabled | strategy={cfg.fsdp.sharding_strategy} | gpus={num_gpus}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        if is_main:
            print(f"[trainer] Single-device training on {device}")

    return model


def _get_transformer_layer_cls(model):
    """Try to auto-detect the decoder layer class."""
    candidates = [
        "LlamaDecoderLayer", "GPT2Block", "OPTDecoderLayer",
        "BloomBlock", "FalconDecoderLayer", "MistralDecoderLayer",
    ]
    for name, mod in model.named_modules():
        cls_name = type(mod).__name__
        if cls_name in candidates:
            return type(mod)
    return None


def make_synthetic_batch(cfg: BenchmarkConfig, device: str):
    """
    Returns a synthetic batch of random token IDs.
    Avoids the need for a real dataset for benchmarking purposes.
    """
    B = cfg.training.batch_size
    T = cfg.model.max_seq_len
    ids = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
    return {"input_ids": ids, "labels": ids}


def run_benchmark(cfg: BenchmarkConfig) -> BenchmarkResult:
    """Entry point: run the benchmark and return a BenchmarkResult."""

    rank, world_size = setup_distributed()
    is_main = rank == 0

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    result = BenchmarkResult(
        run_id=cfg.run_id,
        num_gpus=max(num_gpus, 1),
        fsdp_enabled=cfg.fsdp.enabled and num_gpus > 1,
        sharding_strategy=cfg.fsdp.sharding_strategy if cfg.fsdp.enabled else "none",
        grad_checkpoint_enabled=cfg.grad_checkpoint.enabled,
        dtype=cfg.training.dtype,
        batch_size=cfg.training.batch_size,
        seq_len=cfg.model.max_seq_len,
    )

    tokens_per_step = (
        cfg.training.batch_size
        * cfg.model.max_seq_len
        * cfg.training.gradient_accumulation_steps
        * world_size
    )

    model = build_model(cfg, rank=rank)
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate)
    collector = MetricsCollector(result, tokens_per_step)

    if is_main:
        print(f"\n{'='*60}")
        print(f"  Run ID        : {cfg.run_id}")
        print(f"  Model         : {cfg.model.model_name}")
        print(f"  GPUs          : {world_size}")
        print(f"  FSDP          : {result.fsdp_enabled}")
        print(f"  GradCkpt      : {cfg.grad_checkpoint.enabled}")
        print(f"  Dtype         : {cfg.training.dtype}")
        print(f"  Steps         : {cfg.training.num_steps}")
        print(f"  Batch x SeqLen: {cfg.training.batch_size} x {cfg.model.max_seq_len}")
        print(f"{'='*60}\n")

    # Clear CUDA cache before benchmarking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model.train()
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=_dtype(cfg.training.dtype))
        if cfg.training.dtype in ("fp16", "bf16") and not cfg.fsdp.enabled
        else nullcontext()
    )

    for step in range(cfg.training.num_steps):
        batch = make_synthetic_batch(cfg, device)
        optimizer.zero_grad()

        collector.step_start()

        # Gradient accumulation
        for micro in range(cfg.training.gradient_accumulation_steps):
            with autocast_ctx:
                outputs = model(**batch)
                loss = outputs.loss / cfg.training.gradient_accumulation_steps
            loss.backward()

        optimizer.step()

        m = collector.step_end(step, loss.item() * cfg.training.gradient_accumulation_steps)

        if is_main and (step % cfg.log_interval == 0 or step == cfg.training.num_steps - 1):
            print(
                f"  step {step:4d}/{cfg.training.num_steps} | "
                f"loss={m.loss:.4f} | "
                f"{m.tokens_per_second:,.0f} tok/s | "
                f"mem={m.gpu_memory_allocated_gb:.2f}GB"
            )

    # -----------------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------------
    result.compute_summary()

    if is_main:
        print(f"\n{'='*60}")
        print(f"  Mean throughput : {result.mean_throughput_tps:,.0f} tok/s")
        print(f"  Peak GPU memory : {result.peak_gpu_memory_gb:.2f} GB")
        print(f"  Mean step time  : {result.mean_step_time_s*1000:.1f} ms")
        print(f"  Total time      : {result.total_time_s:.1f} s")
        print(f"{'='*60}\n")
        result.save(cfg.output_dir)

    cleanup_distributed()
    return result
