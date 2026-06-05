"""
Benchmark configuration dataclasses.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json


@dataclass
class ModelConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # swap for any HF model
    max_seq_len: int = 512
    vocab_size: int = 32000


@dataclass
class TrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_steps: int = 50          # short run for benchmarking; increase for real training
    warmup_steps: int = 5
    learning_rate: float = 2e-5
    dtype: str = "bf16"          # "fp32" | "fp16" | "bf16"


@dataclass
class FSDPConfig:
    enabled: bool = True
    sharding_strategy: str = "FULL_SHARD"   # FULL_SHARD | SHARD_GRAD_OP | NO_SHARD
    cpu_offload: bool = False
    mixed_precision: bool = True


@dataclass
class GradCheckpointConfig:
    enabled: bool = False
    offload_to_cpu: bool = False


@dataclass
class BenchmarkConfig:
    run_id: str = "run_001"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    grad_checkpoint: GradCheckpointConfig = field(default_factory=GradCheckpointConfig)
    profile_memory: bool = True
    log_interval: int = 10       # log every N steps
    output_dir: str = "./results"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkConfig":
        with open(path) as f:
            d = json.load(f)
        cfg = cls()
        cfg.model = ModelConfig(**d.get("model", {}))
        cfg.training = TrainingConfig(**d.get("training", {}))
        cfg.fsdp = FSDPConfig(**d.get("fsdp", {}))
        cfg.grad_checkpoint = GradCheckpointConfig(**d.get("grad_checkpoint", {}))
        for k in ("run_id", "profile_memory", "log_interval", "output_dir"):
            if k in d:
                setattr(cfg, k, d[k])
        return cfg
