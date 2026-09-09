from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops

from .losses import (
    ForegroundClassificationBalanceLoss,
    OneVsRestBCELoss,
    _distributed_sum_,
)


def _box_regression_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: torch.Tensor,
    regression_targets: torch.Tensor,
) -> torch.Tensor:
    positive = torch.where(labels > 0)[0]
    reshaped = box_regression.reshape(class_logits.shape[0], class_logits.shape[1], 4)
    if not positive.numel():
        return reshaped.sum() * 0.0
    loss = F.smooth_l1_loss(
        reshaped[positive, labels[positive]],
        regression_targets[positive],
        beta=1.0 / 9.0,
        reduction="sum",
    )
    return loss / max(labels.numel(), 1)


class BACLRoIHeads(RoIHeads):
    """TorchVision RoI head with BACL's FCBL and feature hallucination."""

    def __init__(
        self,
        *args,
        num_foreground_classes: int,
        stage: str = "representation",
        alpha: float = 0.85,
        probability_threshold: float = 0.7,
        feature_decay: float = 0.1,
        sampled_classes: int = 8,
        sampled_features_per_class: int = 12,
        statistics_boxes_per_gt: int = 16,
        max_statistics_gt_per_image: int = 100,
        fhm_start_epoch: int = 0,
        reweight_start_epoch: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if stage not in {"representation", "classifier"}:
            raise ValueError(f"Unknown BACL stage: {stage}")
        self.num_foreground_classes = num_foreground_classes
        self.stage = stage
        self.feature_decay = feature_decay
        self.sampled_classes = sampled_classes
        self.sampled_features_per_class = sampled_features_per_class
        self.statistics_boxes_per_gt = statistics_boxes_per_gt
        self.max_statistics_gt_per_image = max_statistics_gt_per_image
        self.fhm_start_epoch = fhm_start_epoch
        self.reweight_start_epoch = reweight_start_epoch
        self.current_epoch = 0
        self.representation_loss = OneVsRestBCELoss()
        self.fcbl = ForegroundClassificationBalanceLoss(
            num_foreground_classes,
            alpha=alpha,
            prob_threshold=probability_threshold,
        )

        feature_dim = int(self.box_predictor.cls_score.in_features)
        self.register_buffer("feature_mean", torch.zeros(num_foreground_classes, feature_dim))
        self.register_buffer("feature_variance", torch.zeros(num_foreground_classes, feature_dim))
        self.register_buffer("feature_seen", torch.zeros(num_foreground_classes))

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @property
    def fhm_enabled(self) -> bool:
        return self.stage == "classifier" and self.current_epoch >= self.fhm_start_epoch

    @torch.no_grad()
    def _update_feature_distribution(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        values = embeddings.detach().float()
        feature_sum = values.new_zeros(self.feature_mean.shape)
        square_sum = values.new_zeros(self.feature_variance.shape)
        counts = values.new_zeros(self.feature_seen.shape)
        if embeddings.numel():
            foreground_labels = labels - 1
            feature_sum.index_add_(0, foreground_labels, values)
            square_sum.index_add_(0, foreground_labels, values.square())
            counts.index_add_(
                0, foreground_labels, torch.ones_like(foreground_labels, dtype=values.dtype)
            )
        _distributed_sum_(feature_sum)
        _distributed_sum_(square_sum)
        _distributed_sum_(counts)

        present = counts > 0
        if not present.any():
            return
        batch_mean = feature_sum[present] / counts[present, None]
        batch_variance = square_sum[present] / counts[present, None] - batch_mean.square()
        batch_variance = batch_variance.clamp(min=0.0)
        already_seen = self.feature_seen[present] > 0
        old_mean = self.feature_mean[present]
        old_variance = self.feature_variance[present]
        decay = self.feature_decay
        self.feature_mean[present] = torch.where(
            already_seen[:, None], decay * batch_mean + (1.0 - decay) * old_mean, batch_mean
        )
        self.feature_variance[present] = torch.where(
            already_seen[:, None],
            decay * batch_variance + (1.0 - decay) * old_variance,
            batch_variance,
        )
        self.feature_seen.add_(counts)

    def _generate_hallucinated_features(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        seen = torch.where(self.feature_seen > 0)[0]
        if seen.numel() == 0 or self.sampled_classes <= 0 or self.sampled_features_per_class <= 0:
            return self.feature_mean.new_zeros((0, self.feature_mean.shape[1]), dtype=dtype), seen

        accuracy = self.fcbl.correct_class_ratio().detach()[seen].clamp(0.0, 1.0)
        probabilities = (1.0 - accuracy).clamp(min=1e-6)
        probabilities = probabilities / probabilities.sum()
        sampled = seen[
            torch.multinomial(probabilities, self.sampled_classes, replacement=True)
        ]
        repeated = sampled.repeat_interleave(self.sampled_features_per_class)
        means = self.feature_mean[repeated]
        standard_deviations = self.feature_variance[repeated].clamp(min=1e-6).sqrt()
        generated = means + standard_deviations * torch.randn_like(means)
        # TorchVision labels foreground classes from 1; zero is background.
        labels = repeated + 1
        return generated.to(dtype=dtype), labels

    def _jitter_ground_truth(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if boxes.numel() == 0 or self.statistics_boxes_per_gt <= 0:
            return boxes.new_zeros((0, 4)), labels.new_zeros((0,))
        boxes = boxes[: self.max_statistics_gt_per_image]
        labels = labels[: self.max_statistics_gt_per_image]
        repetitions = self.statistics_boxes_per_gt
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5

        shape = (boxes.shape[0], repetitions)
        jitter_x = (torch.rand(shape, device=boxes.device) - 0.5) * widths[:, None] / 3.0
        jitter_y = (torch.rand(shape, device=boxes.device) - 0.5) * heights[:, None] / 3.0
        scale_w = 0.85 + torch.rand(shape, device=boxes.device) * 0.30
        scale_h = 0.85 + torch.rand(shape, device=boxes.device) * 0.30
        new_widths = widths[:, None] * scale_w
        new_heights = heights[:, None] * scale_h
        new_centers_x = centers_x[:, None] + jitter_x
        new_centers_y = centers_y[:, None] + jitter_y
        jittered = torch.stack(
            (
                new_centers_x - new_widths * 0.5,
                new_centers_y - new_heights * 0.5,
                new_centers_x + new_widths * 0.5,
                new_centers_y + new_heights * 0.5,
            ),
            dim=-1,
        ).reshape(-1, 4)
        jittered = box_ops.clip_boxes_to_image(jittered, image_shape)
        keep = box_ops.remove_small_boxes(jittered, min_size=1.0)
        repeated_labels = labels.repeat_interleave(repetitions)
        return jittered[keep], repeated_labels[keep]

    @torch.no_grad()
    def _collect_feature_statistics(
        self,
        features: OrderedDict[str, torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        proposals: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for target, image_shape in zip(targets, image_shapes):
            boxes, image_labels = self._jitter_ground_truth(
                target["boxes"], target["labels"], image_shape
            )
            proposals.append(boxes)
            labels.append(image_labels)
        if not any(item.numel() for item in proposals):
            empty_embeddings = self.feature_mean.new_zeros((0, self.feature_mean.shape[1]))
            empty_labels = targets[0]["labels"].new_zeros((0,))
            self._update_feature_distribution(empty_embeddings, empty_labels)
            return
        pooled = self.box_roi_pool(features, proposals, image_shapes)
        embeddings = self.box_head(pooled)
        self._update_feature_distribution(embeddings, torch.cat(labels, dim=0))

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        number_of_classes = class_logits.shape[-1]
        boxes_per_image = [boxes.shape[0] for boxes in proposals]
        predicted_boxes = self.box_coder.decode(box_regression, proposals)
        predicted_scores = self.fcbl.probabilities(class_logits)
        box_lists = predicted_boxes.split(boxes_per_image, 0)
        score_lists = predicted_scores.split(boxes_per_image, 0)

        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for boxes, scores, image_shape in zip(box_lists, score_lists, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(number_of_classes, device=device).view(1, -1).expand_as(scores)
            boxes, scores, labels = boxes[:, 1:], scores[:, 1:], labels[:, 1:]
            boxes, scores, labels = boxes.reshape(-1, 4), scores.reshape(-1), labels.reshape(-1)
            keep = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_labels.append(labels[keep])
        return all_boxes, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            if targets is None:
                raise ValueError("targets are required while training")
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        pooled = self.box_roi_pool(features, proposals, image_shapes)
        embeddings = self.box_head(pooled)
        class_logits, box_regression = self.box_predictor(embeddings)
        result: list[dict[str, torch.Tensor]] = []
        losses: dict[str, torch.Tensor] = {}

        if self.training:
            assert labels is not None and regression_targets is not None and targets is not None
            labels_tensor = torch.cat(labels, dim=0)
            regression_tensor = torch.cat(regression_targets, dim=0)
            logits_for_classification = class_logits
            labels_for_classification = labels_tensor

            if self.fhm_enabled:
                self._collect_feature_statistics(features, image_shapes, targets)
                generated, generated_labels = self._generate_hallucinated_features(embeddings.dtype)
                if generated.numel():
                    generated_logits = self.box_predictor.cls_score(generated)
                    logits_for_classification = torch.cat((class_logits, generated_logits), dim=0)
                    labels_for_classification = torch.cat((labels_tensor, generated_labels), dim=0)

            if self.stage == "classifier":
                loss_classifier = self.fcbl(
                    logits_for_classification,
                    labels_for_classification,
                    reweight=self.current_epoch >= self.reweight_start_epoch,
                )
            else:
                loss_classifier = self.representation_loss(
                    logits_for_classification, labels_for_classification
                )
            loss_box_regression = _box_regression_loss(
                class_logits, box_regression, labels_tensor, regression_tensor
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_regression,
            }
        else:
            boxes, scores, detected_labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            result = [
                {"boxes": box, "labels": label, "scores": score}
                for box, label, score in zip(boxes, detected_labels, scores)
            ]
        return result, losses
