#!/usr/bin/env python3
"""Run the synthetic epSNN demo directly or pass command-line arguments."""

from pathlib import Path
import sys
from tempfile import mkdtemp

from epsnn.cli import main as run_cli


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "configs" / "demo.json"
DEVICE = "auto"


def main():
    if len(sys.argv) > 1:
        run_cli()
        return

    output_root = PROJECT_DIR / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(mkdtemp(prefix="demo_", dir=output_root))
    run_cli([
        "train",
        "--synthetic",
        "--config", str(CONFIG_PATH),
        "--device", DEVICE,
        "--output-dir", str(output_dir),
    ])


if __name__ == "__main__":
    main()
