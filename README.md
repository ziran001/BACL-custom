# BACL TorchVision port for custom LVIS / YOLO detection datasets

This repository is a runnable modern-PyTorch port of
[Tianhao-Qi/BACL](https://github.com/Tianhao-Qi/BACL). It trains Faster R-CNN
directly from LVIS JSON or a YOLO detection dataset and keeps the central BACL ideas:

- decoupled representation and classifier training;
- Foreground Classification Balance Loss (FCBL);
- cumulative foreground confusion statistics;
- per-class feature means/variances;
- feature hallucination biased toward poorly classified classes.

It does **not** vendor the original historical MMDetection fork and is not a
bit-exact reproduction. See [ATTRIBUTION.md](ATTRIBUTION.md).

## Expected dataset layout

### LVIS (AutoDL 默认配置)

所有数据命令的默认根目录为 `/root/autodl-tmp/datasets`。已上传的标注可直接读取：

```text
/root/autodl-tmp/datasets/
├── annotations/
│   ├── lvis_v1_train.json
│   ├── lvis_v1_val.json
│   └── lvis_v1_test.json
└── images/...
```

图片位置由 JSON `images[].file_name` 决定：例如 `images/train/a.jpg` 对应
`/root/autodl-tmp/datasets/images/train/a.jpg`。官方 LVIS 没有 `file_name` 时，
使用 `coco_url` 最后两段（如 `train2017/000000123456.jpg`）定位本地图片，不下载图片。

无需 `dataset.yaml`、`classes.txt`、分割 TXT 或 YOLO 标签。类别数和名称从
`categories` 自动读取；按数值排序的 category ID 映射为 `1..K`，背景为 `0`。
训练、验证和测试必须有相同的完整类别 ID／名称表，顺序可不同，允许某个类别没有目标。

现有 [configs/mollusks_lvis.yaml](configs/mollusks_lvis.yaml) 已配置你的服务器路径。
非默认布局可用 `--data configs/mollusks_lvis.yaml`，修改其中的 `image_root`：
若 JSON 是 `train/a.jpg`、实际图片在 `images/train/a.jpg`，设为 `images`。
也可将配置存为数据目录的 `dataset_lvis.yaml`，传入根目录时会优先读取它。
标准 LVIS JSON 优先于旧 YOLO YAML；多个同分割 JSON 同时存在时要求用 YAML 明确选择。

标准 LVIS 每张图片应有 `neg_category_ids` 和 `not_exhaustive_category_ids`。
若自定义导出缺少这些字段，**仅当数据确实完整标注所有类别**时，在 YAML 设置
`exhaustive: true`。读取器会补齐缺失字段，不修改原始 JSON，也不覆盖已有字段。
不要对部分标注数据使用此选项。

本项目仅做边界框检测，读取像素单位 `bbox: [x, y, width, height]`；
`segmentation: []` 可用，不训练实例掩码。当前端口不支持 `iscrowd=1` 或 `ignore=1`
训练框，读取时会报错。负类别／非完整标注字段用于评估；训练仍沿用该端口的
BCE/FCBL 和 RPN 样本策略。

### YOLO (仍然支持)

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
To explicitly select YOLO when both formats exist, pass `--data-format yolo`.

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

服务器已有仓库时，在仓库目录执行 `git pull --ff-only origin main`，然后
`python -m pip install -e .`。

Validate the transferred dataset before using GPU time:

```bash
python -m tools.validate_dataset --data /root/autodl-tmp/datasets \
  --output dataset_report.json
```

The resolver ignores a stale Windows `path:` in `dataset.yaml` when the YAML's
own directory contains `images/` and the split files.
Validation fully decodes images and checks boxes, JSON dimensions, category
consistency and duplicate image paths within/across splits. A missing optional
LVIS test split is reported as skipped; explicitly configured missing files fail.

## Training

LVIS training enables the original base LVIS config's class-balanced repeating
with threshold `1e-3`. Each image is repeated
`ceil(max(1, sqrt(threshold / class_image_frequency)))` times, taking the maximum
over its categories. Empty images are retained once. Use `--repeat-threshold 0`
to disable it. YOLO, validation and test datasets are not repeated.

Stage 1 learns the representation with one-vs-rest BCE:

```bash
python -m tools.train \
  --data /root/autodl-tmp/datasets \
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
  --data /root/autodl-tmp/datasets \
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
The first 500 iterations use the original linear warmup schedule, beginning at
0.001 times the target learning rate. Use `--warmup-iters 0` to disable it.

For multi-GPU stage training:

```bash
torchrun --standalone --nproc_per_node=4 -m tools.train \
  --data /root/autodl-tmp/datasets \
  --stage representation \
  --epochs 12 \
  --batch-size 2 \
  --output runs/representation
```

Training checkpoints are deliberately ignored by Git. Distributed training
saves checkpoints on rank 0; run evaluation afterward with one GPU.
Checkpoints and `dataset_config.json` record class names and the LVIS category
mapping. Loading checks consistency. Legacy YOLO checkpoints can initialize LVIS
training if their class names and order match exactly.

## Evaluation

```bash
python -m tools.evaluate \
  --data /root/autodl-tmp/datasets \
  --checkpoint runs/classifier/last.pth \
  --split test \
  --output runs/classifier/test_map50.json
```

The included evaluator reports 101-point interpolated `mAP@0.50` and per-class
AP50. It is dependency-light and is not the COCO `mAP@0.50:0.95` metric.
For LVIS it ignores unverified categories and unmatched detections in
non-exhaustively annotated categories using the LVIS image metadata. It is still
**not** the official LVIS `AP@0.50:0.95` or APr/APc/APf; do not compare it directly
with the paper's tables. Metrics use the 0–1 scale. If only a validation split
exists, specify `--split val`; a missing test set is never silently replaced.

## Fast diagnostics

Run one train/backward/inference step without downloading pretrained weights:

```bash
python -m tools.smoke_test --data /root/autodl-tmp/datasets
```

For a short end-to-end training-path check:

```bash
python -m tools.train \
  --data /root/autodl-tmp/datasets \
  --stage representation \
  --epochs 1 \
  --max-train-batches 2 \
  --max-val-batches 2 \
  --no-pretrained \
  --min-sizes 256 \
  --max-size 384 \
  --output runs/debug
```

Run regression tests and an optional synthetic-data integration check:

```bash
python -m unittest discover -s tests -v
python -m tests.run_lvis_integration
```

The integration check uses temporary LVIS data and no pretrained downloads. It
runs validation, train/backward/inference, both training stages, checkpoint
handoff and test evaluation on CUDA when available (otherwise CPU). It does not
replace full validation of the real server dataset.
