# BACL-custom：原始 BACL + 自定义 LVIS bbox 数据

默认后端直接调用 [Tianhao-Qi/BACL 原源码](https://github.com/Tianhao-Qi/BACL/tree/65536623361587286d99c70133b886792a84463d)，
固定提交 `65536623361587286d99c70133b886792a84463d`，包含逐文件 SHA 校验。
训练模型、BCE/FCBL、FHM、BoxGn 和官方 LVIS 评估沿用原实现。

你的 JSON 只有框，没有实例掩码，因此按已确认的 bbox 适配关闭第一阶段 CopyPaste，
同时关闭 mask 加载和从 mask 重算 bbox。其余原始实验配置保留。
这是“原源码 + 明确的数据适配”，不是在自定义数据上宣称复现原论文数值。

完整对照、依赖安装和验证边界见 [原源码对照说明](docs/UPSTREAM_ALIGNMENT.md)。

## 数据

默认目录，无需重新转换 JSON：

```text
/root/autodl-tmp/datasets/
├── annotations/
│   ├── lvis_v1_train.json
│   ├── lvis_v1_val.json
│   └── lvis_v1_test.json
└── images/...
```

图片按原 JSON 的 `file_name` 相对于数据根目录解析，例如 `images/example.jpg`。
也支持 [configs/mollusks_lvis.yaml](configs/mollusks_lvis.yaml)。
从 train 的 categories 读取类别及 ID 映射，不修改上传标注、不生成矩形伪掩码。
当前数据为 170 类，train/val/test 分别 13673/1721/1721 张图。
训练只要求 train/val；独立 test 评估额外要求 test 标注。

## 安装与检查

先按 [Linux 独立环境说明](docs/UPSTREAM_ALIGNMENT.md#linuxautodl-独立环境) 安装历史
PyTorch 1.7 / CUDA 11.0 / mmcv-full 1.2.7 / mmlvis 10.5.3 环境。
不要在原有现代 TorchVision 环境中混装或直接降级。
服务器 GPU 必须支持该环境；新 GPU 的兼容性需要实际检查。

```bash
python -m tools.validate_dataset --data /root/autodl-tmp/datasets \
  --output outputs/dataset_validation.json
python -m tools.smoke_test --data /root/autodl-tmp/datasets
```

validate 默认完整解码图片；快速检查可加 `--skip-decode`。
smoke_test 检查源码、配置、数据及 CUDA NMS/RoIAlign 内核，不等于模型端到端训练通过。

## 两阶段训练

必须从原后端的第一阶段重新训练；旧 TorchVision 的 `last.pth`/`best.pth` 不兼容。

```bash
# 可先追加 --dry-run，仅导出配置，不训练、不检查 checkpoint/CUDA。
python -m tools.train --stage representation \
  --data /root/autodl-tmp/datasets \
  --output runs/official_representation --batch-size 2 --workers 2

python -m tools.train --stage classifier \
  --data /root/autodl-tmp/datasets \
  --checkpoint runs/official_representation/epoch_12.pth \
  --output runs/official_classifier --batch-size 2 --workers 2
```

两阶段默认各 12 epochs；保留原 warmup 500、step=[8,11]、FP32 和无梯度裁剪。
第一阶段 weight decay=0.00005，第二阶段=0.0001。
默认学习率按 `0.02 × 全局batch / 16` 缩放；单卡 batch=2 时为 0.0025。
每轮验证计算官方 bbox AP，最佳 checkpoint 按 `bbox_AP` 选取。
类别、JSON SHA256、原代码版本、环境和实际配置写入输出目录。

单卡也会初始化分布式进程组，因为原 FCBL 使用 all_reduce。多卡使用旧 PyTorch 自带启动器：

```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env \
  -m tools.train --stage representation \
  --data /root/autodl-tmp/datasets \
  --output runs/official_representation_4gpu --batch-size 2
```

同阶段恢复训练使用 `--resume-from runs/official_classifier/epoch_6.pth`；
不要同时传 `--checkpoint`。不要把新实验写进已有 checkpoint 的目录。

## 官方 LVIS 评估

```bash
python -m tools.evaluate --stage classifier \
  --data /root/autodl-tmp/datasets \
  --checkpoint runs/official_classifier/epoch_12.pth \
  --split test --output outputs/lvis_test.json
```

主指标 `metrics.bbox_AP` 是 IoU=0.50:0.05:0.95 的平均 AP；
`bbox_AP50` 和 `bbox_AP75` 是独立的单阈值指标，另有 APr/APc/APf/APs/APm/APl。
JSON 和原终端表保留 0..1 标度、三位小数；换算百分数时乘以 100。
当前 categories 没有 rare 类，所以 APr=-1 表示“不适用”，并非模型 AP 为 0。
未连接服务器进行完整模型训练时，不会把配置/CPU 测试描述为训练成功。

## 任意图片批量检测

默认递归检测 `/root/autodl-tmp/test` 中的常见图片，使用第二阶段
`best_bbox_AP.pth`，并将画框图片和 JSON 结果写入数据盘：

```bash
python -m tools.detect \
  --input /root/autodl-tmp/test \
  --output /root/autodl-tmp/test_results \
  --data /root/autodl-tmp/datasets \
  --checkpoint /root/BACL-custom/runs/official_classifier/best_bbox_AP.pth \
  --score-thr 0.3 \
  --line-width 2 \
  --font-size 10
```

输出图片保持输入的相对子目录和文件名，不改写原图。
`detections.json` 同时记录原 LVIS `category_id`、类别名、置信度、
`bbox_xyxy` 和 `bbox_xywh`。`--input` 也可直接指定单张图片。
`--line-width` 和 `--font-size` 只控制画框图片的外观，不改变模型预测结果。

## 兼容旧移植版

旧 TorchVision 代码保留在 `bacl/` 和 `tools/*_torchvision.py`，
但仅通过 `--backend torchvision` 显式选择，评估仍为 AP50，不属于原源码后端。
旧版独立环境需要 Python>=3.10，安装 `python -m pip install -e ".[torchvision]"`。
用法见 [旧版说明](docs/TORCHVISION_LEGACY.md)，不要混用两套权重。

CPU 原源码/指标审计：

```bash
# 单独 Python 3.8-3.11 环境；不是 native CUDA 训练环境
python -m pip install -r requirements-audit.txt
python -m unittest tests.test_official_backend -v
```

旧版回归测试为 `tests.test_lvis_data` / `tests.run_lvis_integration`，只验证 TorchVision 后端。
许可证及原作者信息见 [ATTRIBUTION.md](ATTRIBUTION.md) 和 [原始许可证](third_party/BACL/LICENSE)。
