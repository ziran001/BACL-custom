from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from bacl.data import add_dataset_arguments, build_detection_dataset, collate_fn, resolve_dataset_config
from bacl.model import build_bacl_fasterrcnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real-data BACL train/inference step")
    add_dataset_arguments(parser)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    config = resolve_dataset_config(args.data, args.data_format)
    dataset = build_detection_dataset(config, split="train", augment=False)
    loader = DataLoader(Subset(dataset, [0]), batch_size=1, collate_fn=collate_fn)
    images, targets = next(iter(loader))
    images = [image.to(device) for image in images]
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    model = build_bacl_fasterrcnn(
        len(config.class_names),
        stage="classifier",
        pretrained=False,
        min_size=256,
        max_size=384,
        detections_per_image=20,
        sampled_classes=2,
        sampled_features_per_class=2,
        statistics_boxes_per_gt=2,
    ).to(device)
    model.train()
    model.roi_heads.set_epoch(0)
    losses = model(images, targets)
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite smoke-test loss")
    total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite smoke-test gradient")
    print("training_losses", {key: float(value.detach()) for key, value in losses.items()})

    model.eval()
    with torch.inference_mode():
        predictions = model(images)
    print(
        "inference",
        {
            "detections": len(predictions[0]["boxes"]),
            "box_shape": tuple(predictions[0]["boxes"].shape),
            "device": str(device),
        },
    )


if __name__ == "__main__":
    main()
