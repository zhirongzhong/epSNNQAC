"""Command-line training and checkpoint evaluation for the epSNN demo."""

import argparse
import json
from pathlib import Path

import torch

from .config import Config
from .models import EPSNNQAC
from .utils.bussgang import BussgangLMMSE
from .utils.checkpoint import load_checkpoint, save_checkpoint, write_json
from .utils.data import DEFAULT_SYNTHETIC, load_split, synthetic_splits, validate_splits
from .utils.training import make_loader, run_epoch, seed_everything


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu, cuda, cuda:N, or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def parse_bits(value):
    if value.lower() in ("none", "null", "unquantized"):
        return None
    try:
        bits = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use none or an integer from 1 to 24") from exc
    if not 1 <= bits <= 24:
        raise argparse.ArgumentTypeError("Bit depth must be from 1 to 24")
    return bits


def train(args):
    values = {} if args.config is None else json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("epochs", "batch_size", "rounds", "seed", "bits"):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    config = Config.from_dict(values)
    device = resolve_device(args.device)
    synthetic_spec = DEFAULT_SYNTHETIC.copy() if args.synthetic else None
    if synthetic_spec is not None:
        splits = synthetic_splits(synthetic_spec)
    else:
        splits = tuple(load_split(args.data_dir / f"{name}_LRE.mat") for name in ("train", "val", "test"))
    channels, num_classes = validate_splits(*splits)
    output = args.output_dir.resolve()
    planned_paths = [
        output / f"round_{index}" / f"bits_{'none' if bits is None else bits}" / "best.pt"
        for index in range(config.rounds) for bits in config.bits
    ]
    if not args.overwrite and any(path.exists() for path in planned_paths):
        raise FileExistsError("An output checkpoint already exists; select another --output-dir or use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    results = []
    for round_index in range(config.rounds):
        seed = config.seed + round_index
        for bits in config.bits:
            seed_everything(seed, config.cpu_threads)
            run_dir = output / f"round_{round_index}" / f"bits_{'none' if bits is None else bits}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "best.pt"
            model = EPSNNQAC(channels, num_classes, config).to(device)
            buss = None if bits is None else BussgangLMMSE(channels)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            train_loader = make_loader(splits[0], bits, config, shuffle=True, seed=seed)
            val_loader = make_loader(splits[1], bits, config)
            best_accuracy, best_loss = -1.0, float("inf")
            history = []
            for epoch in range(1, config.epochs + 1):
                training = run_epoch(model, train_loader, buss, device, num_classes, optimizer)
                validation = run_epoch(model, val_loader, buss, device, num_classes)
                accuracy, loss = validation["accuracy"], validation["loss"]
                if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
                    best_accuracy, best_loss = accuracy, loss
                    save_checkpoint(
                        checkpoint, model, buss, config, channels, num_classes,
                        bits, epoch, validation, seed, synthetic_spec,
                    )
                history.append({"epoch": epoch, "train": training, "validation": validation})
                print(
                    f"round={round_index} bits={bits} epoch={epoch}/{config.epochs} "
                    f"train_loss={training['loss']:.4f} val_loss={loss:.4f} "
                    f"val_acc={accuracy:.4f} best_val_acc={best_accuracy:.4f}",
                    flush=True,
                )
            model, buss, _, state = load_checkpoint(checkpoint, device)
            test_loader = make_loader(splits[2], bits, config)
            metrics = run_epoch(model, test_loader, buss, device, num_classes)
            record = {
                "round": round_index,
                "seed": seed,
                "bits": bits,
                "best_epoch": state["epoch"],
                "checkpoint": checkpoint.relative_to(output).as_posix(),
                "validation": state["validation"],
                "test": metrics,
            }
            write_json(run_dir / "history.json", history)
            write_json(run_dir / "metrics.json", record)
            results.append(record)
            write_json(output / "results.json", {
                "data_source": "synthetic" if args.synthetic else "private_mat",
                "runs": results,
            })
            print(f"test_acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f}", flush=True)
    print(f"Results: {output / 'results.json'}", flush=True)


def test(args):
    device = resolve_device(args.device)
    model, buss, config, state = load_checkpoint(args.checkpoint, device)
    seed_everything(state["seed"], config.cpu_threads)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        config.validate()
    if args.synthetic:
        if state["synthetic_spec"] is None:
            raise ValueError("This checkpoint was not trained with synthetic data; provide --data-file")
        split = synthetic_splits(state["synthetic_spec"])[2]
    else:
        split = load_split(args.data_file)
    if split[0].shape[2] != state["channels"] or split[1].max() >= state["num_classes"]:
        raise ValueError("Test channels or labels do not match the checkpoint")
    metrics = run_epoch(
        model, make_loader(split, state["bits"], config), buss, device, state["num_classes"]
    )
    result = {"bits": state["bits"], "best_epoch": state["epoch"], "test": metrics}
    write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    print(f"Results: {args.output.resolve()}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="epSNNQAC time-series classification demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train", help="Train, select on validation data, and test")
    source = training.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path, help="Directory containing train/val/test_LRE.mat")
    source.add_argument("--synthetic", action="store_true", help="Use generated in-memory waveforms")
    training.add_argument("--config", type=Path)
    training.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    training.add_argument("--device", default="auto")
    training.add_argument("--epochs", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--rounds", type=int)
    training.add_argument("--seed", type=int)
    training.add_argument("--bits", type=parse_bits, nargs="+")
    training.add_argument("--overwrite", action="store_true", help="Allow replacing checkpoints in the output directory")
    training.set_defaults(func=train)
    evaluation = subparsers.add_parser("test", help="Evaluate a saved checkpoint with frozen statistics")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    source = evaluation.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-file", type=Path, help="MATLAB test split")
    source.add_argument("--synthetic", action="store_true")
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--batch-size", type=int)
    evaluation.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    evaluation.set_defaults(func=test)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
