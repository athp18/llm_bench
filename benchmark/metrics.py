"""
Metrics collection: throughput, GPU memory, step timing.
"""
import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import torch


@dataclass
class StepMetrics:
    step: int
    loss: float
    step_time_s: float
    tokens_per_second: float
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
    gpu_utilization_pct: Optional[float] = None   # populated if pynvml available


@dataclass
class BenchmarkResult:
    run_id: str
    num_gpus: int
    fsdp_enabled: bool
    sharding_strategy: str
    grad_checkpoint_enabled: bool
    dtype: str
    batch_size: int
    seq_len: int
    steps: List[StepMetrics] = field(default_factory=list)

    # Summary stats (filled in at end)
    mean_throughput_tps: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    mean_step_time_s: float = 0.0
    total_time_s: float = 0.0
    tokens_trained: int = 0

    def compute_summary(self):
        if not self.steps:
            return
        tps = [s.tokens_per_second for s in self.steps]
        times = [s.step_time_s for s in self.steps]
        mems = [s.gpu_memory_allocated_gb for s in self.steps]
        self.mean_throughput_tps = sum(tps) / len(tps)
        self.mean_step_time_s = sum(times) / len(times)
        self.peak_gpu_memory_gb = max(mems)
        self.total_time_s = sum(times)
        self.tokens_trained = int(sum(s.tokens_per_second * s.step_time_s for s in self.steps))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[metrics] Saved results to {path}")
        return path


class MetricsCollector:
    """Collects per-step metrics during training."""

    def __init__(self, result: BenchmarkResult, tokens_per_step: int):
        self.result = result
        self.tokens_per_step = tokens_per_step
        self._step_start: float = 0.0

    def step_start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def step_end(self, step: int, loss: float) -> StepMetrics:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._step_start
        tps = self.tokens_per_step / elapsed if elapsed > 0 else 0.0

        mem_alloc = mem_res = 0.0
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_res = torch.cuda.memory_reserved() / 1e9

        gpu_util = _get_gpu_util()

        m = StepMetrics(
            step=step,
            loss=loss,
            step_time_s=elapsed,
            tokens_per_second=tps,
            gpu_memory_allocated_gb=mem_alloc,
            gpu_memory_reserved_gb=mem_res,
            gpu_utilization_pct=gpu_util,
        )
        self.result.steps.append(m)
        return m


def _get_gpu_util() -> Optional[float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return None
