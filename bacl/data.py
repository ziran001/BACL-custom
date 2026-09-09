from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    train_split: Path
    val_split: Path
    test_split: Path
    class_names: tuple[str, ...]


def _read_names(raw: Any, root: Path) -> tuple[str, ...]:
    classes_file = root / "classes.txt"
    if classes_file.exists():
        names = tuple(
            line.strip()
            for line in classes_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
        if names:
            return names
    if isinstance(raw, dict):
        return tuple(str(raw[key]) for key in sorted(raw, key=lambda value: int(value)))
    if isinstance(raw, list):
        return tuple(str(value) for value in raw)
    raise ValueError(f"No class names found in {classes_file} or dataset YAML")


def resolve_dataset_config(data: str | Path) -> DatasetConfig:
    supplied = Path(data).expanduser()
    yaml_path = supplied / "dataset.yaml" if supplied.is_dir() else supplied
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
    yaml_parent = yaml_path.resolve().parent
    configured_root = Path(str(raw.get("path", yaml_parent))).expanduser()
    if not configured_root.is_absolute():
        configured_root = yaml_parent / configured_root

    # A copied dataset.yaml often still contains the old machine's absolute path.
    # Prefer the YAML's directory when it contains the actual split/image files.
    root = configured_root if configured_root.exists() else yaml_parent
    if (yaml_parent / "images").exists() and (yaml_parent / str(raw.get("train", "train.txt"))).exists():
        root = yaml_parent
    root = root.resolve()

    def split_path(key: str, default: str) -> Path:
        value = Path(str(raw.get(key, default)))
        return value if value.is_absolute() else root / value

    names = _read_names(raw.get("names"), root)
    declared = int(raw.get("nc", len(names)))
    if declared != len(names):
        raise ValueError(f"dataset.yaml declares nc={declared}, but {len(names)} names were found")

    return DatasetConfig(
        root=root,
        train_split=split_path("train", "train.txt"),
        val_split=split_path("val", "val.txt"),
        test_split=split_path("test", "test.txt"),
        class_names=names,
    )


def _label_path(root: Path, image_path: Path) -> Path:
    try:
        relative = image_path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image {image_path} is outside dataset root {root}") from exc
    parts = list(relative.parts)
    try:
        image_index = next(index for index, part in enumerate(parts) if part.lower() == "images")
    except StopIteration as exc:
        raise ValueError(f"Image path must contain an 'images' directory: {image_path}") from exc
    parts[image_index] = "labels"
    return (root / Path(*parts)).with_suffix(".txt")


class YoloDetectionDataset(Dataset):
    """Read standard YOLO ``class cx cy width height`` detection labels."""

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        self.config = config
        self.root = config.root
        self.class_names = config.class_names
        self.num_classes = len(self.class_names)
        self.augment = augment
        split_file = {
            "train": config.train_split,
            "val": config.val_split,
            "test": config.test_split,
        }[split]
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        self.image_paths = []
        for raw_line in split_file.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = self.root / line.replace("\\", "/").removeprefix("./")
            self.image_paths.append(candidate.resolve())
        if not self.image_paths:
            raise ValueError(f"No images listed in {split_file}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_target(self, index: int, width: int, height: int) -> dict[str, torch.Tensor]:
        image_path = self.image_paths[index]
        label_path = _label_path(self.root, image_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

        boxes: list[list[float]] = []
        labels: list[int] = []
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            fields = raw_line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 5 fields, got {len(fields)}")
            class_id = int(fields[0])
            if not 0 <= class_id < self.num_classes:
                raise ValueError(
                    f"{label_path}:{line_number}: class {class_id} outside [0, {self.num_classes - 1}]"
                )
            cx, cy, box_width, box_height = (float(value) for value in fields[1:])
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"{label_path}:{line_number}: box width/height must be positive")
            x1 = max(0.0, min(float(width), (cx - box_width / 2.0) * width))
            y1 = max(0.0, min(float(height), (cy - box_height / 2.0) * height))
            x2 = max(0.0, min(float(width), (cx + box_width / 2.0) * width))
            y2 = max(0.0, min(float(height), (cy + box_height / 2.0) * height))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{label_path}:{line_number}: box is empty after clipping")
            boxes.append([x1, y1, x2, y2])
            # TorchVision reserves 0 for background.
            labels.append(class_id + 1)

        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (
            (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])
            if len(box_tensor)
            else torch.zeros(0, dtype=torch.float32)
        )
        return {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(label_tensor), dtype=torch.int64),
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.image_paths[index]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as exc:
            raise OSError(f"Failed to decode image {image_path}: {exc}") from exc
        width, height = image.size
        target = self._read_target(index, width, height)
        image_tensor = TF.pil_to_tensor(image).float().div_(255.0)

        if self.augment and torch.rand(()) < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            boxes = target["boxes"]
            if len(boxes):
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1
        return image_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
