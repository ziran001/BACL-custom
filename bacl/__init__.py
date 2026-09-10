"""Modern BACL components for TorchVision Faster R-CNN."""

from .data import LvisDetectionDataset, YoloDetectionDataset, build_detection_dataset, collate_fn, resolve_dataset_config
from .model import build_bacl_fasterrcnn, freeze_for_classifier_stage

__all__ = [
    "YoloDetectionDataset",
    "LvisDetectionDataset",
    "build_detection_dataset",
    "build_bacl_fasterrcnn",
    "collate_fn",
    "freeze_for_classifier_stage",
    "resolve_dataset_config",
]
