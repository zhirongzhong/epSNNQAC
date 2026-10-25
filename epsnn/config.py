"""Configuration shared by training and checkpoint-based evaluation."""

from dataclasses import asdict, dataclass, fields
import math


@dataclass
class Config:
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 100
    conv_features: int = 128
    num_res_blocks: int = 3
    ep_hidden: int = 256
    num_ep_blocks: int = 3
    v_ref_min: float = -5.0
    v_ref_max: float = 5.0
    gain: float = 0.1
    offset: float = 0.0
    bits: tuple = (None, 8, 6, 4, 2)
    rounds: int = 1
    seed: int = 42
    num_workers: int = 0
    cpu_threads: int = 1

    def validate(self):
        for name in (
            "batch_size", "epochs", "conv_features", "ep_hidden", "rounds", "cpu_threads"
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("num_res_blocks", "num_ep_blocks", "num_workers", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed + self.rounds > 2**32 - 1:
            raise ValueError("seed + rounds must be below 2**32")
        for name in ("learning_rate", "v_ref_min", "v_ref_max", "gain", "offset"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0 or self.gain == 0 or self.v_ref_min >= self.v_ref_max:
            raise ValueError("Require learning_rate > 0, gain != 0, v_ref_min < v_ref_max")
        if not isinstance(self.bits, (list, tuple)) or not self.bits:
            raise ValueError("bits must be a nonempty list")
        if any(b is not None and (type(b) is not int or not 1 <= b <= 24) for b in self.bits):
            raise ValueError("Each bit depth must be null or an integer from 1 to 24")
        if len(set(self.bits)) != len(self.bits):
            raise ValueError("Duplicate bit depths are not allowed")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values).validate()
