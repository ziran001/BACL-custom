from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

from bacl.data import YoloDetectionDataset, collate_fn, resolve_dataset_config
from bacl.engine import evaluate_map50
from bacl.model import build_bacl_fasterrcnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BACL checkpoint at IoU=0.50")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default="evaluation.json")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.01)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = resolve_dataset_config(args.data)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_names = tuple(payload["class_names"])
    if checkpoint_names != config.class_names:
        raise ValueError("Checkpoint class names/order do not match the dataset")
    model = build_bacl_fasterrcnn(
        len(config.class_names),
        stage=str(payload.get("stage", "classifier")),
        pretrained=False,
        score_threshold=args.score_threshold,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    dataset = YoloDetectionDataset(config, split=args.split)
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
    Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
