# epSNNQAC: Expectation-Propagating Spiking Neural Networks with Quantization-Aware Compensation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10.2](https://img.shields.io/badge/PyTorch-1.10.2-red.svg)](https://pytorch.org/)

Official implementation of Expectation-Propagating Spiking Neural Networks with Quantization-Aware Compensation Under Extreme Data Quantization, will publish soon in *Reliability Engineering and System Safety*.

> **Zhirong Zhong, Hongfei Ren, Zhongyi Zhang, Zhi Zhai, Meng Ma, Jinxin Liu**
>
> *Reliability Engineering and System Safety*, 2026.

## Overview

epSNN combines expectation-propagating spiking neural networks with quantization-aware compensation for fault diagnosis under quantized sensing.

## Installation

### Prerequisites

The package versions are pinned in `requirements.txt`.

### Install dependencies

Run from the project root in PowerShell:

```powershell
-m pip install -r requirements.txt
```

## Quick Start

Command-line usage is supported:

```powershell
main.py train --synthetic --config configs/demo.json --device cpu --output-dir outputs/demo
main.py test --synthetic --checkpoint outputs/demo/round_0/bits_4/best.pt --device cpu --output outputs/test_bits4.json
```

`configs/demo.json` runs a small synthetic-data example with unquantized and 4-bit inputs.  Configuration fields are defined by `Config` in `epsnn/config.py`; command-line options such as `--epochs`, `--bits`, and `--rounds` override the JSON settings.

## Training

Preparing the data, then run: 

```powershell
main.py train --data-dir /path/to/private/LRE --config configs/lre.json --output-dir outputs/lre
main.py test --data-file /path/to/private/LRE/test_LRE.mat --checkpoint outputs/lre/round_0/bits_4/best.pt --output outputs/lre_test_bits4.json
```

Training reloads the best validation checkpoint before testing; Bussgang statistics are fitted on training data and frozen during validation and testing. Each run saves `best.pt`, `history.json`, and `metrics.json` under `round_<index>/bits_<depth>/`, with aggregate results in `results.json`. Reported metrics include accuracy, macro precision/recall/F1, and the confusion matrix.

## Citation

If you use this work in your research, please cite:

```bibtex

```

## License

This project is released under the [MIT License](LICENSE).
