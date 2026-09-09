# Attribution

This project is a modern, lightweight reimplementation of the training ideas in:

> Tianhao Qi, Hongtao Xie, Pandeng Li, Jiannan Ge, Yongdong Zhang,
> "Balanced Classification: A Unified Framework for Long-Tailed Object Detection,"
> IEEE Transactions on Multimedia, 2023.

Original implementation: https://github.com/Tianhao-Qi/BACL (Apache-2.0).

The port uses TorchVision Faster R-CNN instead of vendoring the historical
MMDetection source tree. Class conventions and device/distributed handling were
adapted for current TorchVision releases. It is intended to preserve FCBL,
feature-distribution tracking, feature hallucination, and decoupled training,
but it is not a bit-for-bit reproduction of the paper repository.
