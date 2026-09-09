from __future__ import annotations

from typing import Iterable

from torch import nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn

from .roi_heads import BACLRoIHeads


def build_bacl_fasterrcnn(
    num_foreground_classes: int,
    stage: str = "representation",
    pretrained: bool = True,
    min_size: int | tuple[int, ...] = (640, 672, 704, 736, 768, 800),
    max_size: int = 1333,
    score_threshold: float = 1e-4,
    detections_per_image: int = 300,
    alpha: float = 0.85,
    probability_threshold: float = 0.7,
    feature_decay: float = 0.1,
    sampled_classes: int = 8,
    sampled_features_per_class: int = 12,
    statistics_boxes_per_gt: int = 16,
    fhm_start_epoch: int = 0,
    reweight_start_epoch: int = 0,
) -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=score_threshold,
        box_nms_thresh=0.5,
        box_detections_per_img=detections_per_image,
    )
    old_head = model.roi_heads
    input_features = old_head.box_predictor.cls_score.in_features
    old_head.box_predictor = FastRCNNPredictor(input_features, num_foreground_classes + 1)
    model.roi_heads = BACLRoIHeads(
        old_head.box_roi_pool,
        old_head.box_head,
        old_head.box_predictor,
        old_head.proposal_matcher.high_threshold,
        old_head.proposal_matcher.low_threshold,
        old_head.fg_bg_sampler.batch_size_per_image,
        old_head.fg_bg_sampler.positive_fraction,
        old_head.box_coder.weights,
        score_threshold,
        old_head.nms_thresh,
        detections_per_image,
        num_foreground_classes=num_foreground_classes,
        stage=stage,
        alpha=alpha,
        probability_threshold=probability_threshold,
        feature_decay=feature_decay,
        sampled_classes=sampled_classes,
        sampled_features_per_class=sampled_features_per_class,
        statistics_boxes_per_gt=statistics_boxes_per_gt,
        fhm_start_epoch=fhm_start_epoch,
        reweight_start_epoch=reweight_start_epoch,
    )
    return model


def freeze_for_classifier_stage(model: nn.Module) -> list[str]:
    """Match the original BACL stage-2 policy: train RPN and box predictors."""
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith("rpn.") or name.startswith("roi_heads.box_predictor.")
        parameter.requires_grad = enabled
        if enabled:
            trainable.append(name)
    return trainable


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
