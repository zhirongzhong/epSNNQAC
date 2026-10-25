"""split loading, ADC quantization, and synthetic demo data."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


def validate_arrays(features, labels, source="arrays"):
    x, y = np.asarray(features), np.asarray(labels)
    if x.ndim != 3 or min(x.shape) < 1:
        raise ValueError(f"{source}: features must have nonempty shape (N, T, C)")
    if x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise ValueError(f"{source}: features must contain finite real numbers")
    if y.ndim not in (1, 2) or (y.ndim == 2 and min(y.shape) != 1):
        raise ValueError(f"{source}: labels must have shape (N,), (N, 1), or (1, N)")
    y = y.reshape(-1)
    if y.size != x.shape[0]:
        raise ValueError(f"{source}: features and labels have different sample counts")
    if y.dtype.kind not in "fiu" or not np.isfinite(y).all():
        raise ValueError(f"{source}: labels must contain finite integers")
    if np.any(y < 0) or np.any(y != np.floor(y)) or np.any(y >= 2**63):
        raise ValueError(f"{source}: labels must be nonnegative int64 integers")
    with np.errstate(over="ignore"):
        x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{source}: features exceed the float32 range")
    return x, y.astype(np.int64)


def load_split(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    try:
        data = loadmat(path)
    except NotImplementedError as exc:
        raise ValueError("MATLAB v7.3 is unsupported; save as v7 or earlier") from exc
    if "features" not in data or "labels" not in data:
        raise ValueError(f"{path}: expected MATLAB keys 'features' and 'labels'")
    return validate_arrays(data["features"], data["labels"], str(path))


def validate_splits(train, val, test):
    classes = np.unique(train[1])
    if not np.array_equal(classes, np.arange(len(classes))) or len(classes) < 2:
        raise ValueError("Training labels must cover contiguous classes 0, ..., K-1 (K >= 2)")
    channels = train[0].shape[2]
    for name, (x, y) in (("validation", val), ("test", test)):
        if x.shape[2] != channels:
            raise ValueError(f"{name}: channel count differs from the training split")
        if y.max() >= len(classes):
            raise ValueError(f"{name}: label is outside the training class range")
    return channels, len(classes)


def quantize(x, bits, config):
    if bits is None:
        return x.copy()
    voltage = config.gain * x + config.offset
    step = (config.v_ref_max - config.v_ref_min) / (2**bits - 1)
    quantized = np.clip(
        np.round((voltage - config.v_ref_min) / step) * step + config.v_ref_min,
        config.v_ref_min,
        config.v_ref_max,
    )
    return ((quantized - config.offset) / config.gain).astype(np.float32)


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, bits, config):
        self.x = x
        self.y = y
        self.bits = bits
        self.config = config

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        return torch.from_numpy(x), torch.from_numpy(quantize(x, self.bits, self.config)), self.y[index]


def synthetic_split(samples, steps, channels, classes, seed):
    """Generate labeled waveforms without reading any external dataset."""
    if min(samples, steps, channels) < 1 or classes < 2 or samples < classes:
        raise ValueError("Require positive dimensions, classes >= 2, and samples >= classes")
    rng = np.random.default_rng(seed)
    labels = np.arange(samples, dtype=np.int64) % classes
    rng.shuffle(labels)
    t = np.linspace(0, 1, steps, endpoint=False, dtype=np.float32)
    frequency = (labels + 1)[:, None, None]
    phase = np.arange(channels, dtype=np.float32)[None, None, :] * 0.3
    signal = 2 * np.sin(2 * np.pi * frequency * t[None, :, None] + phase)
    signal += labels[:, None, None] * 1.5
    signal += rng.normal(0, 0.15, (samples, steps, channels))
    return signal.astype(np.float32), labels


def synthetic_splits(spec):
    return tuple(
        synthetic_split(
            spec[name], spec["steps"], spec["channels"], spec["classes"], spec["seed"] + index
        )
        for index, name in enumerate(("train_samples", "val_samples", "test_samples"))
    )


DEFAULT_SYNTHETIC = {
    "train_samples": 96,
    "val_samples": 48,
    "test_samples": 48,
    "steps": 32,
    "channels": 3,
    "classes": 3,
    "seed": 2026,
}
