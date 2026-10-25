"""Portable checkpoints and JSON results."""

import inspect
import json
from pathlib import Path

import torch

from ..config import Config
from ..models import EPSNNQAC
from .bussgang import BussgangLMMSE


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(path, model, buss, config, channels, num_classes, bits, epoch, val_metrics, seed, synthetic_spec):
    state = {
        "format_version": 1,
        "config": config.to_dict(),
        "channels": channels,
        "num_classes": num_classes,
        "bits": bits,
        "epoch": epoch,
        "seed": seed,
        "validation": val_metrics,
        "synthetic_spec": synthetic_spec,
        "feat": {key: value.detach().cpu() for key, value in model.feat.state_dict().items()},
        "ep": {key: value.detach().cpu() for key, value in model.ep.state_dict().items()},
        "buss": None if buss is None else buss.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path, device):
    load_options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = True
    state = torch.load(path, **load_options)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint; use a checkpoint produced by this demo")
    config = Config.from_dict(state["config"])
    model = EPSNNQAC(state["channels"], state["num_classes"], config).to(device)
    model.feat.load_state_dict(state["feat"])
    model.ep.load_state_dict(state["ep"])
    buss = None
    if state["bits"] is not None:
        if state["buss"] is None:
            raise ValueError("Quantized checkpoint is missing Bussgang statistics")
        buss = BussgangLMMSE(state["channels"])
        buss.load_state_dict(state["buss"])
    elif state["buss"] is not None:
        raise ValueError("Unquantized checkpoint unexpectedly contains Bussgang statistics")
    return model, buss, config, state
