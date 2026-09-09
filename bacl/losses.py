from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _distributed_sum_(tensor: torch.Tensor) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)


class OneVsRestBCELoss(nn.Module):
    """BCE classifier used by the representation-learning stage of BACL."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / max(
            logits.shape[0], 1
        )


class ForegroundClassificationBalanceLoss(nn.Module):
    """FCBL adapted to TorchVision's background-first class convention.

    Foreground labels are 1..K and background is 0. The cumulative foreground
    confusion matrix supplies pairwise class margins and hard-competitor weights.
    """

    def __init__(self, num_foreground_classes: int, alpha: float = 0.85, prob_threshold: float = 0.7):
        super().__init__()
        if not 0.0 < prob_threshold < 1.0:
            raise ValueError("prob_threshold must be between 0 and 1")
        self.num_foreground_classes = num_foreground_classes
        self.alpha = alpha
        self.prob_threshold = prob_threshold
        self.register_buffer(
            "confusion_numerator", torch.zeros(num_foreground_classes, num_foreground_classes)
        )
        self.register_buffer("ground_truth_count", torch.zeros(num_foreground_classes))

    def confusion_matrix(self) -> torch.Tensor:
        return self.confusion_numerator / self.ground_truth_count[:, None].clamp(min=1.0)

    def correct_class_ratio(self) -> torch.Tensor:
        return torch.diagonal(self.confusion_matrix())

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        foreground = probabilities[:, 1:] * (1.0 - probabilities[:, :1])
        return torch.cat((probabilities[:, :1], foreground), dim=1)

    @torch.no_grad()
    def _update_confusion(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        positive = labels > 0
        numerator = logits.new_zeros(
            (self.num_foreground_classes, self.num_foreground_classes), dtype=torch.float32
        )
        counts = logits.new_zeros(self.num_foreground_classes, dtype=torch.float32)
        if positive.any():
            gt = labels[positive] - 1
            distribution = F.softmax(logits[positive, 1:].float(), dim=1)
            one_hot = F.one_hot(gt, num_classes=self.num_foreground_classes).to(distribution.dtype)
            numerator = one_hot.transpose(0, 1) @ distribution
            counts = torch.bincount(gt, minlength=self.num_foreground_classes).to(distribution.dtype)
        _distributed_sum_(numerator)
        _distributed_sum_(counts)
        self.confusion_numerator.add_(numerator)
        self.ground_truth_count.add_(counts)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        reweight: bool = True,
    ) -> torch.Tensor:
        number_of_samples, number_of_channels = logits.shape
        expected_channels = self.num_foreground_classes + 1
        if number_of_channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} logits, got {number_of_channels}")
        target = F.one_hot(labels, num_classes=number_of_channels).to(dtype=logits.dtype)
        margin = torch.zeros_like(logits)
        positive = labels > 0

        if positive.any():
            gt = labels[positive] - 1
            confusion = self.confusion_matrix().detach()
            numerator = confusion[gt, :].clamp(min=1e-3)
            denominator = confusion[:, gt].transpose(0, 1).clamp(min=1e-3)
            margin[positive, 1:] = (numerator / denominator).log() * self.alpha

        weights = torch.ones_like(logits)
        if reweight and positive.any():
            probabilities = self.probabilities(logits[positive].detach())
            gt_probability = probabilities.gather(1, labels[positive, None])
            foreground_weights = (
                (probabilities[:, 1:] >= gt_probability)
                | (probabilities[:, 1:] >= self.prob_threshold)
            ).to(logits.dtype)
            positive_weights = torch.cat(
                (torch.ones((foreground_weights.shape[0], 1), device=logits.device), foreground_weights),
                dim=1,
            )
            weights[positive] = positive_weights

        loss = F.binary_cross_entropy_with_logits(logits + margin, target, reduction="none")
        loss = (weights * loss).sum() / max(number_of_samples, 1)
        self._update_confusion(logits.detach(), labels)
        return loss
