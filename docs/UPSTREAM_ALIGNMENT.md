# 原始 BACL 对照与 bbox 数据适配

## 对照基准

原仓库：[Tianhao-Qi/BACL](https://github.com/Tianhao-Qi/BACL/tree/65536623361587286d99c70133b886792a84463d)。
固定提交：`65536623361587286d99c70133b886792a84463d`。

默认 `official` 后端直接运行 `third_party/BACL/tools/train.py` 和 `tools/test.py`。
导入的 `mmdet` 必须来自这个目录。模型、损失、FHM、BoxGn 和训练流程不重新实现。
导入的 284 个原文件逐字节保留，每次启动按 `UPSTREAM_LOCK.json` 校验 Git blob SHA。
范围是完整 mmdet 运行时、三个 R50 Faster R-CNN LVIS v1 配置、训练/测试入口及依赖和许可证；
没有声称复制了上游所有示例、图片、数据或其他骨干网络的配置。

实际实验配置为：

- `faster_rcnn_r50_fpn_mstrain_1x_lvis_v1.py`
- `bacl_representation_faster_rcnn_r50_fpn_1x_lvis_v1.py`
- `bacl_classifier_faster_rcnn_r50_fpn_mstrain_1x_lvis_v1.py`

不能把未被这些实验配置引用的通用 `_base_/datasets/lvis_v1_detection.py` 当作训练依据。
此前将该通用配置的 ClassBalancedDataset/0.001 重复采样作为默认值不准确，现已纠正。

## 关键差异与处理

| 项目 | 之前的 TorchVision 移植版 | 默认原源码后端 |
| --- | --- | --- |
| 检测器 | TorchVision Faster R-CNN | 原 MMDetection Faster R-CNN R50-FPN |
| 预训练 | 整个 COCO 检测器权重 | `torchvision://resnet50`，仅 ImageNet 骨干 |
| 回归损失 | TorchVision smooth-L1 路径 | 原 RPN/ROI 的 L1Loss |
| 分类标签 | 背景 0、前景 1..K | 原前景 0..K-1、背景 K，输出 K+1 |
| FHM | 近似统计、采样和框扰动 | 原 FhmRoIHead/FhmShared2FCBBoxHead/BoxGn |
| 训练冻结 | 移植版选择逻辑 | 原 `selectp=1`：fc_cls/fc_reg/rpn（有 mask head 时也训练） |
| 重复采样 | 之前默认阈值 0.001 | 两阶段实验没有 ClassBalancedDataset |
| weight decay | 两阶段默认 0.00005 | 第一阶段 0.00005；第二阶段 0.0001 |
| 精度/裁剪 | 默认 AMP、梯度裁剪 10 | 原 FP32、`grad_clip=None` |
| 学习率日程 | 随 epochs 比例改变 milestones | 原 12 epochs，warmup 500/0.001，step=[8,11] |
| 评估 | 自定义 AP50 | 原 `LVISV1Dataset.evaluate` → mmlvis LVISEval |
| 最佳 checkpoint | AP50 | `bbox_AP`，即 AP@0.50:0.95 |

FHM 的 per-rank 均值/无偏方差、EMA、先抽类别再跳过未观测类别、16 个扰动框和
FCBL 的混淆统计/reweight 均调用原文件；不添加方差下限、不改变抽样分布。
测试保持 score threshold=0.0001、NMS=0.5、每图最多 300 个检测。

原学习率 .02 对应全局 batch=16；按照原 README 的线性缩放要求，适配入口计算
`.02 * batch_size_per_gpu * WORLD_SIZE / 16`。单卡 batch=2 为 .0025。
不依赖上游未实际接入训练 API 的 auto_scale_lr 开关。显式 `--lr` 或 `--epochs`
会记录在运行信息中；改 epochs 不会自动修改原 step=[8,11]。

## 用户批准的必要差异

数据是自定义 170 类 LVIS 格式，不是官方 1203 类 LVIS 数据集。
读取原 JSON 的 ID/名称/frequency，按 ID 排序建立映射；不重写 JSON、不重新转换、不补造掩码。
`file_name` 优先于 `coco_url`，根目录默认为 `/root/autodl-tmp/datasets`。
类别数量在 bbox head 和 loss 同时更新。评估时预测标签映射回原 JSON category_id。

已确认标注均为 bbox、`segmentation=[]`。用户同意 bbox 适配：

1. 第一阶段 `LoadAnnotations.with_mask=False`；
2. 第一阶段 `RandomCrop.recompute_bbox=False`（原来从真实 mask 重算 bbox）；
3. 仅移除 CopyPaste，保留原 Resize 1280/ratio_range、RandomCrop、Filter、Flip、Pad、Normalize 等。

第二阶段的数据增强保持原配置。未来有真实实例掩码时可显式 `--with-masks` 恢复完整原增强。
矩形伪掩码不是实例掩码，不作为原 CopyPaste 的等价替代。

其他适配仅为运行接口：配置路径/输出目录、单卡也初始化 DDP（原 FCBL 调用 all_reduce）、
数据与权重检查、来源记录、评估结果另存 JSON。JSON 报告包装不改评分算法。
原评估返回值和 mmlvis 终端表保留三位小数，分数范围 0..1；换算百分数时乘以 100。
`bbox_mAP_copypaste` 是上游指标汇总字符串的名字，不表示正在使用 CopyPaste 增强。

本地确认的 JSON 内容（用户说明与服务器上传版本相同；未通过 SSH 独立验证服务器）：

| split | images | annotations | 有实例掩码的 annotations |
| --- | ---: | ---: | ---: |
| train | 13673 | 17536 | 0 |
| val | 1721 | 2277 | 0 |
| test | 1721 | 2170 | 0 |

JSON 的 frequency 包含 common=130、frequent=40、rare=0；无 rare 类时官方 APr=-1，
不能将其解读为 0 AP，也不能据此声称复现原论文的 LVIS rare 类结果。

## Linux/AutoDL 独立环境

不要在已有 TorchVision 训练环境中直接降级。原作者测试的是 Python 3.8、
PyTorch 1.7.0、CUDA 11.0、MMCV 1.2.7。原 README 写的 torchvision 0.4.0
与 PyTorch 1.7 不匹配；下面使用对应系列 0.8.1。

```bash
conda create -n bacl-original python=3.8 -y
conda activate bacl-original
python -m pip install "pip<25" "setuptools<70" wheel
python -m pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 \
  -f https://download.pytorch.org/whl/torch_stable.html
# 先安装 Cython/numpy，供 mmpycocotools 的历史构建流程使用。
python -m pip install numpy==1.23.5 Cython==0.29.36
python -m pip install -r requirements-official.txt
python -m pip install mmcv-full==1.2.7 --no-deps \
  -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
python -m pip install -e . --no-deps
python -m tools.smoke_test --data /root/autodl-tmp/datasets
```

依赖选择依据：[原 BACL 环境说明](https://github.com/Tianhao-Qi/BACL/tree/65536623361587286d99c70133b886792a84463d#requirements)、
[PyTorch 旧版安装](https://docs.pytorch.org/get-started/previous-versions/)、
[TorchVision 0.8.1](https://github.com/pytorch/vision/releases/tag/v0.8.1)、
[OpenMMLab CUDA 11.0 / Torch 1.7.0 wheel](https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html)。

`mmpycocotools` 与 `pycocotools`、`mmlvis` 与 `lvis`、`mmcv-full` 与纯 Python `mmcv`
分别占用相同模块名，不能混装。NumPy 1.24+ 删除了原 mmlvis 使用的 `np.float`，故锁定 1.23.5。
纯 Python mmcv 只能跑配置审计，不能训练。若 mmpycocotools 编译失败，需要服务器的 gcc/g++
开发工具；如果 GPU 不支持这套旧 CUDA/PyTorch，需另外评估兼容迁移，不能宣称直接等价。

当前没有服务器 GPU/驱动信息，因此不能保证任意 AutoDL 新 GPU 都支持此历史环境。
默认 smoke_test 检查实际 CUDA NMS/RoIAlign 前后向内核，但不是完整模型训练测试。

## 权重和验证边界

旧 TorchVision `last.pth`/`best.pth` 的 `model` 字段不是原 MMDetection `state_dict`；
不能直接续训或直接用新后端评估。请用新的输出目录从第一阶段重训。
阶段二使用原后端阶段一 `epoch_12.pth`，保留原流程，不自动改为 stage1 最佳权重。
默认每轮进行官方 bbox 评估，最佳权重按 bbox_AP 选取。恢复同阶段训练用 `--resume-from`。
仅加载自己信任的 checkpoint（历史 PyTorch checkpoint 使用 pickle）。

若旧版适配入口在启动第二阶段、恢复训练或评估时报
`AttributeError: type object 'Config' has no attribute 'fromstring'`，
这是适配层误用了 MMCV 1.2.7 没有的 API。更新本仓库后重试即可，
不需要升级 MMCV，也不需要因此重训第一阶段。修复使用临时配置文件和
`Config.fromfile(..., import_custom_modules=False)` 读取 checkpoint 的配置元数据；
保留类别顺序和阶段校验，不改写权重文件、数据标注或原 BACL 模型。

CPU 审计在单独环境进行：

```bash
# Python 3.8-3.11；不要装进上面的 native 训练环境
python -m pip install -r requirements-audit.txt
python -m unittest tests.test_official_backend -v
```

审计包括原文件 SHA、完整模型配置差异、bbox-only pipeline、类别/路径/联邦标注检查、
配置 round-trip、含配置元数据的 checkpoint 兼容性与错误拦截、默认 CLI dry-run，
以及真实 mmlvis 和原 evaluate 方法的多 IoU 测试。
其中原 evaluate 方法测试只替代结果存储层，不加载 native 检测器，不能当作训练通过。
`--dry-run`/`--config-only` 明确不测试 checkpoint、模型前后向或服务器 CUDA。
旧版测试 `tests.run_lvis_integration` 只验证 TorchVision 后端，与 native BACL 验证分开。
