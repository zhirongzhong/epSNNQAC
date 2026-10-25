"""Training and statistics evaluation."""

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import TimeSeriesDataset


def seed_everything(seed, cpu_threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(split, bits, config, shuffle=False, seed=42):
    return DataLoader(
        TimeSeriesDataset(*split, bits, config),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=seed_worker,
    )


def prepare_input(x, r, buss, device, training=False):
    if buss is None:
        return x.to(device)
    shape = r.shape
    quantized = r.numpy().reshape(-1, shape[-1])
    if training:
        buss.update(x.numpy().reshape(-1, shape[-1]), quantized)
    reconstructed, _ = buss.infer(quantized)
    return torch.from_numpy(reconstructed.reshape(shape)).to(device)


def metrics_from_confusion(confusion):
    counts = confusion.to(torch.float64)
    tp = counts.diag()
    precision = tp / counts.sum(0).clamp_min(1)
    recall = tp / counts.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": (tp.sum() / counts.sum()).item(),
        "precision_macro": precision.mean().item(),
        "recall_macro": recall.mean().item(),
        "f1_macro": f1.mean().item(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
        "samples": int(counts.sum().item()),
    }


def run_epoch(model, loader, buss, device, num_classes, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.set_grad_enabled(training):
        for x, r, labels in loader:
            inputs = prepare_input(x, r, buss, device, training)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; check input scaling and learning rate")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch = labels.shape[0]
            total_loss += loss.item() * batch
            total += batch
            indices = labels.detach().cpu() * num_classes + logits.argmax(1).detach().cpu()
            confusion += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    if total == 0:
        raise ValueError("Cannot train or evaluate an empty split")
    return {"loss": total_loss / total, **metrics_from_confusion(confusion)}
