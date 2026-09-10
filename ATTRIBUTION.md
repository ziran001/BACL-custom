# Attribution

This project includes the original implementation and a separately retained
TorchVision approximation of:

> Tianhao Qi, Hongtao Xie, Pandeng Li, Jiannan Ge, Yongdong Zhang,
> "Balanced Classification: A Unified Framework for Long-Tailed Object Detection,"
> IEEE Transactions on Multimedia, 2023.

Original implementation: https://github.com/Tianhao-Qi/BACL (Apache-2.0).

The default backend vendors 284 unchanged files from upstream commit
`65536623361587286d99c70133b886792a84463d` in `third_party/BACL/`, including
the original MMDetection runtime, BACL losses/heads, R50 LVIS v1 configs and
entrypoints. Original copyright notices and the Apache-2.0 license are retained.
`third_party/UPSTREAM_LOCK.json` records the upstream Git blob SHA of each file.

Custom dataset/runtime adapters are in `bacl_official/`; the bbox-only dataset
adaptation disables mask-dependent CopyPaste and is documented explicitly.
This does not claim identical paper results on a different dataset.

The previous modern-PyTorch implementation remains under `bacl/` and is selected
only by `--backend torchvision`. It is not an exact reproduction; its AP50
evaluator and checkpoints are separate from the original backend.
