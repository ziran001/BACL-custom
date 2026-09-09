from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.ops import box_iou
from tqdm import tqdm


def move_targets(targets: list[dict[str, torch.Tensor]], device: torch.device):
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool = True,
    grad_clip_norm: float | None = 10.0,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    model.train()
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.roi_heads.set_epoch(epoch)
    # BCE sums across all foreground channels, so use a conservative initial
    # scale instead of the generic 65536 default used by GradScaler.
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=1024.0, enabled=amp and device.type == "cuda"
    )
    totals: defaultdict[str, float] = defaultdict(float)
    number_of_batches = 0
    start = time.perf_counter()
    progress = tqdm(loader, desc=f"train epoch {epoch + 1}", disable=not show_progress)

    for batch_index, (images, targets) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            details = {name: float(value.detach().cpu()) for name, value in losses.items()}
            raise FloatingPointError(f"Non-finite training loss: {details}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        number_of_batches += 1
        totals["loss"] += float(loss.detach())
        for name, value in losses.items():
            totals[name] += float(value.detach())
        if show_progress:
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

    elapsed = time.perf_counter() - start
    if not number_of_batches:
        raise RuntimeError("Training loader produced no batches")
    result = {name: value / number_of_batches for name, value in totals.items()}
    result["seconds"] = elapsed
    result["batches"] = float(number_of_batches)
    return result


def _interpolated_ap(scores: list[float], true_positives: list[bool], gt_count: int) -> float:
    if gt_count == 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp = np.asarray(true_positives, dtype=np.float64)[order]
    fp = 1.0 - tp
    recall = np.cumsum(tp) / gt_count
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    values = []
    for threshold in np.linspace(0.0, 1.0, 101):
        candidates = precision[recall >= threshold]
        values.append(float(candidates.max()) if candidates.size else 0.0)
    return float(np.mean(values))


@torch.inference_mode()
def evaluate_map50(
    model: nn.Module,
    loader,
    device: torch.device,
    num_foreground_classes: int,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> dict[str, object]:
    model.eval()
    ground_truth_count = [0 for _ in range(num_foreground_classes)]
    scores: list[list[float]] = [[] for _ in range(num_foreground_classes)]
    true_positives: list[list[bool]] = [[] for _ in range(num_foreground_classes)]
    progress = tqdm(loader, desc="evaluate mAP@0.50", disable=not show_progress)

    for batch_index, (images, targets) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        device_images = [image.to(device, non_blocking=True) for image in images]
        outputs = model(device_images)
        for output, target in zip(outputs, targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            for label in gt_labels.tolist():
                ground_truth_count[label - 1] += 1

            predicted_boxes = output["boxes"].cpu()
            predicted_labels = output["labels"].cpu()
            predicted_scores = output["scores"].cpu()
            present_classes = torch.unique(predicted_labels)
            for label_tensor in present_classes:
                label = int(label_tensor)
                class_index = label - 1
                prediction_indices = torch.where(predicted_labels == label)[0]
                prediction_indices = prediction_indices[
                    torch.argsort(predicted_scores[prediction_indices], descending=True)
                ]
                target_indices = torch.where(gt_labels == label)[0]
                class_gt_boxes = gt_boxes[target_indices]
                matched = torch.zeros(len(class_gt_boxes), dtype=torch.bool)
                for prediction_index in prediction_indices:
                    score = float(predicted_scores[prediction_index])
                    is_true_positive = False
                    if len(class_gt_boxes):
                        overlaps = box_iou(
                            predicted_boxes[prediction_index].unsqueeze(0), class_gt_boxes
                        ).squeeze(0)
                        overlaps[matched] = -1.0
                        best_iou, best_index = overlaps.max(dim=0)
                        if float(best_iou) >= 0.5:
                            matched[best_index] = True
                            is_true_positive = True
                    scores[class_index].append(score)
                    true_positives[class_index].append(is_true_positive)

    per_class_ap = [
        _interpolated_ap(class_scores, class_tp, count)
        for class_scores, class_tp, count in zip(scores, true_positives, ground_truth_count)
    ]
    valid = [ap for ap, count in zip(per_class_ap, ground_truth_count) if count > 0]
    return {
        "map50": float(np.mean(valid)) if valid else 0.0,
        "per_class_ap50": per_class_ap,
        "ground_truth_count": ground_truth_count,
        "classes_evaluated": len(valid),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    stage: str,
    class_names: tuple[str, ...],
    extra: dict | None = None,
) -> None:
    unwrapped = model.module if hasattr(model, "module") else model
    payload = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "stage": stage,
        "class_names": list(class_names),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
