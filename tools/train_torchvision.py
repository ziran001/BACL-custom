from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from bacl.data import (
    add_dataset_arguments, build_detection_dataset, check_checkpoint_dataset,
    class_balanced_dataset, collate_fn, resolve_dataset_config,
)
from bacl.engine import evaluate_map50, save_checkpoint, train_one_epoch
from bacl.model import build_bacl_fasterrcnn, freeze_for_classifier_stage, trainable_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train BACL Faster R-CNN on LVIS or YOLO data")
    add_dataset_arguments(parser)
    parser.add_argument(
        "--repeat-threshold", type=float, default=0.0,
        help="LVIS class-balanced repeat threshold; 0 disables repeating (YOLO is unchanged)",
    )
    parser.add_argument("--stage", required=True, choices=["representation", "classifier"])
    parser.add_argument("--checkpoint", default=None, help="Weights used to initialize this stage")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-process batch size")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=500,
        help="Linear learning-rate warmup iterations in epoch 1; set to 0 to disable",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.001,
        help="Initial learning rate as a fraction of the target rate during warmup",
    )
    parser.add_argument("--min-sizes", default="640,672,704,736,768,800")
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--score-threshold", type=float, default=1e-4)
    parser.add_argument("--detections-per-image", type=int, default=300)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=10.0,
        help="Maximum gradient norm; set to 0 to disable clipping",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--probability-threshold", type=float, default=0.7)
    parser.add_argument("--feature-decay", type=float, default=0.1)
    parser.add_argument("--sampled-classes", type=int, default=8)
    parser.add_argument("--sampled-features-per-class", type=int, default=12)
    parser.add_argument("--statistics-boxes-per-gt", type=int, default=16)
    parser.add_argument("--fhm-start-epoch", type=int, default=0)
    parser.add_argument("--reweight-start-epoch", type=int, default=0)
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed BACL training currently requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be non-negative")
    if not 0.0 < args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be in (0, 1]")
    if not math.isfinite(args.repeat_threshold) or args.repeat_threshold < 0:
        raise ValueError("--repeat-threshold must be finite and non-negative")
    rank, world_size, local_rank = distributed_context()
    main_process = rank == 0
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    config = resolve_dataset_config(args.data, args.data_format)
    train_dataset = build_detection_dataset(config, split="train", augment=True)
    val_dataset = build_detection_dataset(config, split="val", augment=False)
    original_train_size = len(train_dataset)
    if config.dataset_format == "lvis" and args.repeat_threshold > 0:
        train_dataset = class_balanced_dataset(train_dataset, args.repeat_threshold)
    if main_process:
        print(f"dataset: {config.dataset_format}, root: {config.root}, "
              f"classes: {len(config.class_names)}, train images: {original_train_size}, "
              f"samples/epoch: {len(train_dataset)}, val images: {len(val_dataset)}")
    dataset_metadata = {
        "dataset_format": config.dataset_format,
        "category_ids": list(config.category_ids),
        "class_names": list(config.class_names),
        "root": str(config.root),
        "train_split": str(config.train_split),
        "val_split": str(config.val_split),
    }
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else RandomSampler(train_dataset)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=SequentialSampler(val_dataset),
        num_workers=max(0, min(args.workers, 2)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    min_sizes = tuple(int(value) for value in args.min_sizes.split(",") if value.strip())
    if not min_sizes:
        raise ValueError("--min-sizes must contain at least one integer")
    model = build_bacl_fasterrcnn(
        len(config.class_names),
        stage=args.stage,
        pretrained=not args.no_pretrained and args.checkpoint is None,
        min_size=min_sizes,
        max_size=args.max_size,
        score_threshold=args.score_threshold,
        detections_per_image=args.detections_per_image,
        alpha=args.alpha,
        probability_threshold=args.probability_threshold,
        feature_decay=args.feature_decay,
        sampled_classes=args.sampled_classes,
        sampled_features_per_class=args.sampled_features_per_class,
        statistics_boxes_per_gt=args.statistics_boxes_per_gt,
        fhm_start_epoch=args.fhm_start_epoch,
        reweight_start_epoch=args.reweight_start_epoch,
    )
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        check_checkpoint_dataset(payload, config)
        model.load_state_dict(payload["model"], strict=True)
    elif args.stage == "classifier":
        raise ValueError("Classifier stage requires --checkpoint from the representation stage")

    if args.stage == "classifier":
        trainable = freeze_for_classifier_stage(model)
        if main_process:
            print(f"classifier stage trainable tensors: {len(trainable)}")
    model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    total_batch_size = args.batch_size * world_size
    learning_rate = args.lr if args.lr is not None else 0.02 * total_batch_size / 16.0
    optimizer = torch.optim.SGD(
        trainable_parameters(model),
        lr=learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    milestones = sorted({max(1, int(args.epochs * 8 / 12)), max(1, int(args.epochs * 11 / 12))})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    output_dir = Path(args.output)
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_args.json").write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output_dir / "dataset_config.json").write_text(
            json.dumps(dataset_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    history: list[dict[str, object]] = []
    best_map50 = -1.0
    for epoch in range(args.epochs):
        is_best = False
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            amp=not args.no_amp,
            grad_clip_norm=args.grad_clip_norm,
            warmup_iters=args.warmup_iters,
            warmup_ratio=args.warmup_ratio,
            max_batches=args.max_train_batches,
            show_progress=main_process,
        )
        scheduler.step()
        epoch_record: dict[str, object] = {"epoch": epoch + 1, "train": train_metrics}

        # Run full validation in a separate single-GPU pass for distributed jobs.
        if world_size == 1 and args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            validation = evaluate_map50(
                model,
                val_loader,
                device,
                len(config.class_names),
                max_batches=args.max_val_batches,
                show_progress=main_process,
            )
            epoch_record["validation"] = {
                "map50": validation["map50"],
                "classes_evaluated": validation["classes_evaluated"],
            }
            current_map50 = float(validation["map50"])
            if current_map50 > best_map50:
                best_map50 = current_map50
                is_best = True

        if main_process:
            history.append(epoch_record)
            print(json.dumps(epoch_record, ensure_ascii=False))
            save_checkpoint(
                output_dir / "last.pth",
                model,
                optimizer,
                epoch,
                args.stage,
                config.class_names,
                {"history": history, **dataset_metadata},
            )
            if is_best:
                save_checkpoint(
                    output_dir / "best.pth",
                    model,
                    optimizer,
                    epoch,
                    args.stage,
                    config.class_names,
                    {"history": history, "best_map50": best_map50, **dataset_metadata},
                )
            (output_dir / "metrics.json").write_text(
                json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
