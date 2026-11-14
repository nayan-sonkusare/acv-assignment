# Reproducible Semantic Segmentation: Baseline U-Net vs Attention U-Net

## Overview

This project compares two semantic-segmentation models on the **Pascal VOC 2012** dataset:

- **Baseline model — `BaselineUNet`:** a ResNet-18 encoder with a decoder that uses standard skip connections.
- **Modified model — `AttentionUNet`:** the same ResNet-18 encoder and output head, but attention gates filter the encoder skip features before they are fused with decoder features.

The goal is to evaluate whether attention-based skip fusion improves segmentation quality, measured primarily with mean Intersection over Union (**mIoU**), while also tracking model parameter count and inference latency.

## Repository structure

```text
acv assignment/
├── configs/
│   └── config.yaml             # Dataset, training and evaluation settings
├── models/
│   ├── baseline.py             # Baseline ResNet-18 U-Net
│   └── modified.py             # Attention-gated U-Net
├── src/
│   ├── dataset.py              # Pascal VOC dataset and transforms
│   ├── metrics.py              # Confusion matrix and per-class IoU/mIoU
│   └── utils.py                # Random seeds and environment logging
├── logs/
│   ├── baseline_training_log.csv
│   ├── modified_training_log.csv
│   └── evidence/               # Saved evaluation tables, plots and summary
├── train.py                    # Train either model
├── evaluate.py                 # Compare saved baseline and modified checkpoints
├── requirements.txt
└── README.md
```

Training creates a `checkpoints/` directory for the best model weights. It is not included in the repository listing shown here, so evaluation requires that the checkpoint files have been generated or supplied separately.

## Requirements

- Python 3.10 or newer recommended
- PyTorch and Torchvision
- The packages listed in `requirements.txt`
- A CUDA-capable GPU is recommended; the code falls back to CPU if CUDA is unavailable, but training may be slower.

Install the dependencies in a virtual environment:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m pip install numpy Pillow
```

Install a PyTorch build compatible with your operating system and CUDA setup using the official PyTorch installation selector if needed.

## Dataset

The data loader uses **Pascal VOC 2012** with 21 classes and ignore label `255`. The configured image size is `256 × 256`.

The training script calls the dataset loader with `download=False`. Therefore, make sure the dataset is already available under the configured data directory before training. The default is:

```text
data/
```

If the dataset is stored elsewhere, update `data.data_dir` in `configs/config.yaml`. The loader uses Torchvision's `VOCSegmentation` dataset layout.

## Configuration

Edit `configs/config.yaml` to change the experiment settings. Current defaults are:

| Setting | Default |
|---|---:|
| Dataset | Pascal VOC 2012 |
| Number of classes | 21 |
| Ignore label | 255 |
| Image size | 256 |
| Batch size | 16 |
| Data-loader workers | 4 |
| Random seed | 42 |
| Maximum epochs | 40 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Early-stopping patience | 8 epochs |

The training device is configured as `cuda`, with automatic CPU fallback when CUDA is unavailable.

## Train the models

Run commands from the repository root.

Train the baseline:

```bash
python train.py --model baseline
```

Train the attention-based model:

```bash
python train.py --model modified
```

Each run records environment information and training/validation metrics in `logs/`, and saves the best checkpoint according to validation mIoU:

```text
checkpoints/baseline_best.pth
checkpoints/modified_best.pth
```

The code sets random seeds and requests deterministic cuDNN behavior to improve reproducibility. Exact results can still vary with hardware, library versions and other runtime conditions.

## Evaluate and compare

After **both** checkpoint files exist and the dataset is available, run:

```bash
python evaluate.py
```

The evaluation script compares the models on the validation split and writes evidence files under `logs/evidence/`, including:

- `evaluation_summary.json` — mIoU, parameter counts, parameter overhead and latency measurements
- `per_class_iou.csv` — baseline and modified IoU for each class
- `training_curves.png` — training/validation curves
- `visual_improvements.png` — examples with the largest per-image mIoU gains
- `failure_cases.png` — examples with the lowest modified-model per-image mIoU

Latency measurements depend on the hardware and runtime environment. Compare models on the same machine and under the same settings.

## Metrics

- **Intersection over Union (IoU):** overlap between predicted and ground-truth pixels for a class.
- **Mean IoU (mIoU):** average IoU over classes represented in the evaluation confusion matrix.
- **Parameter count:** number of learned model parameters.
- **Inference latency:** measured runtime for selected batch sizes.

The segmentation loss is cross-entropy with label `255` ignored.

## Important notes

- The current training and evaluation scripts use `download=False`; they do not automatically download the dataset.
- `evaluate.py` expects both `checkpoints/baseline_best.pth` and `checkpoints/modified_best.pth`. Train both models first or place compatible checkpoint files in `checkpoints/`.
- Existing logs and evidence are experiment artifacts; they should not be treated as proof that a fresh run will reproduce the same values unless the corresponding environment, data and checkpoints are also available.
- The project compares the baseline and attention variants; whether attention improves performance should be determined from the evaluation outputs rather than assumed.

## Acknowledgements

The project uses PyTorch/Torchvision and the Pascal VOC 2012 semantic-segmentation dataset.
