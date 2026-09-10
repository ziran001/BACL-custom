from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

from bacl.data import (
    add_dataset_arguments, build_detection_dataset, check_checkpoint_dataset,
    collate_fn, resolve_dataset_config,
)
from bacl.engine import evaluate_map50
from bacl.model import build_bacl_fasterrcnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BACL checkpoint at IoU=0.50")
    add_dataset_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default="evaluation.json")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.01)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = resolve_dataset_config(args.data, args.data_format)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    check_checkpoint_dataset(payload, config)
    model = build_bacl_fasterrcnn(
        len(config.class_names),
        stage=str(payload.get("stage", "classifier")),
        pretrained=False,
        score_threshold=args.score_threshold,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    dataset = build_detection_dataset(config, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    metrics = evaluate_map50(
        model,
        loader,
        device,
        len(config.class_names),
        max_batches=args.max_batches,
    )
    metrics["class_names"] = list(config.class_names)
    metrics["dataset_format"] = config.dataset_format
    metrics["category_ids"] = list(config.category_ids)
    metrics["metric"] = "bbox mAP@0.50 (101-point); not official LVIS AP@0.50:0.95"
    metrics["per_class"] = [
        {
            "name": name,
            "ap50": ap,
            "ground_truth_boxes": count,
        }
        for name, ap, count in zip(
            config.class_names,
            metrics.pop("per_class_ap50"),
            metrics.pop("ground_truth_count"),
        )
    ]
    output = json.dumps(metrics, indent=2, ensure_ascii=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
