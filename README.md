# BACL TorchVision port for the 170-class mollusk dataset

This repository is a runnable modern-PyTorch port of
[Tianhao-Qi/BACL](https://github.com/Tianhao-Qi/BACL). It trains Faster R-CNN
directly from a YOLO detection dataset and keeps the central BACL ideas:

- decoupled representation and classifier training;
- Foreground Classification Balance Loss (FCBL);
- cumulative foreground confusion statistics;
- per-class feature means/variances;
- feature hallucination biased toward poorly classified classes.

It does **not** vendor the original historical MMDetection fork and is not a
bit-exact reproduction. See [ATTRIBUTION.md](ATTRIBUTION.md).

## Expected dataset layout

```text
dataset-root/
├── dataset.yaml
├── classes.txt
├── train.txt
├── val.txt
├── test.txt
├── images/
└── labels/
```

Labels must use standard normalized YOLO detection rows:
`class_id center_x center_y width height`. The loader converts class IDs from
YOLO's `0..K-1` convention to TorchVision's `1..K` convention internally.

## Server installation

Python 3.10 or 3.11 is recommended. Install a CUDA-enabled PyTorch build that
matches the server driver first, then install this project:

```bash
git clone https://github.com/ziran001/BACL-custom.git
cd BACL-custom
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -e .
```

Validate the transferred dataset before using GPU time:

```bash
python -m tools.validate_dataset --data /data/mollusks/dataset.yaml \
  --output dataset_report.json
```

The resolver ignores a stale Windows `path:` in `dataset.yaml` when the YAML's
own directory contains `images/` and the split files.

## Training

Stage 1 learns the representation with one-vs-rest BCE:

```bash
python -m tools.train \
  --data /data/mollusks/dataset.yaml \
  --stage representation \
  --epochs 12 \
  --batch-size 2 \
  --workers 8 \
  --output runs/representation
```

Stage 2 loads stage 1, freezes the backbone/FPN/box feature MLP, trains the RPN
and box predictors, and enables FCBL + FHM:

```bash
python -m tools.train \
  --data /data/mollusks/dataset.yaml \
  --stage classifier \
  --checkpoint runs/representation/last.pth \
  --epochs 12 \
  --batch-size 2 \
  --workers 8 \
  --output runs/classifier
```

The default learning rate follows the original linear-scaling convention:
`0.02 × total_batch_size / 16`. Use `--lr` to override it. If memory is tight,
reduce `--batch-size`, `--max-size`, or `--statistics-boxes-per-gt`.

For multi-GPU stage training:

```bash
torchrun --standalone --nproc_per_node=4 -m tools.train \
  --data /data/mollusks/dataset.yaml \
  --stage representation \
  --epochs 12 \
  --batch-size 2 \
  --output runs/representation
```

Training checkpoints are deliberately ignored by Git. Distributed training
saves checkpoints on rank 0; run evaluation afterward with one GPU.

## Evaluation

```bash
python -m tools.evaluate \
  --data /data/mollusks/dataset.yaml \
  --checkpoint runs/classifier/last.pth \
  --split test \
  --output runs/classifier/test_map50.json
```

The included evaluator reports 101-point interpolated `mAP@0.50` and per-class
AP50. It is dependency-light and is not the COCO `mAP@0.50:0.95` metric.

## Fast diagnostics

Run one train/backward/inference step without downloading pretrained weights:

```bash
python -m tools.smoke_test --data /data/mollusks/dataset.yaml
```

For a short end-to-end training-path check:

```bash
python -m tools.train \
  --data /data/mollusks/dataset.yaml \
  --stage representation \
  --epochs 1 \
  --max-train-batches 2 \
  --max-val-batches 2 \
  --no-pretrained \
  --min-sizes 256 \
  --max-size 384 \
  --output runs/debug
```
