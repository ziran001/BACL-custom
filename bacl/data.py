from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    train_split: Path
    val_split: Path
    test_split: Path | None
    class_names: tuple[str, ...]
    dataset_format: str = "yolo"
    category_ids: tuple[int, ...] = ()
    image_root: Path | None = None
    exhaustive: bool = False


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


def _read_lvis(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict) or any(
        not isinstance(raw.get(key), list) for key in ("images", "annotations", "categories")
    ):
        raise ValueError(f"{path}: expected images, annotations and categories arrays")
    return raw


def _integer(value: Any, context: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{context}: expected an integer >= {minimum}, got {value!r}")
    return value


def _categories(raw: dict, path: Path) -> tuple[tuple[int, ...], tuple[str, ...]]:
    names = {}
    for category in raw["categories"]:
        category_id = _integer(category.get("id"), f"{path}: category id")
        name = category.get("name")
        if category_id in names:
            raise ValueError(f"{path}: duplicate category id {category_id}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: category {category_id} has no name")
        names[category_id] = name
    if not names:
        raise ValueError(f"{path}: categories must not be empty")
    ids = tuple(sorted(names))
    return ids, tuple(names[key] for key in ids)


def add_dataset_arguments(parser) -> None:
    parser.add_argument(
        "--data", default="/root/autodl-tmp/datasets",
        help="Dataset root or YAML config (default: /root/autodl-tmp/datasets)",
    )
    parser.add_argument("--data-format", choices=("auto", "lvis", "yolo"), default="auto")


def _find_lvis_split(root: Path, split: str) -> Path | None:
    candidates = [
        folder / name
        for folder in (root / "annotations", root)
        for name in (f"lvis_v1_{split}.json", f"lvis_{split}.json")
        if (folder / name).is_file()
    ]
    if len(candidates) > 1:
        raise ValueError(f"Multiple LVIS {split} files found; select one in a YAML config: {candidates}")
    return candidates[0] if candidates else None


def resolve_dataset_config(data: str | Path, dataset_format: str = "auto") -> DatasetConfig:
    if dataset_format not in {"auto", "yolo", "lvis"}:
        raise ValueError(f"Unknown dataset format: {dataset_format}")
    supplied = Path(data).expanduser()
    if supplied.is_dir() and dataset_format != "yolo":
        lvis_yaml = supplied / "dataset_lvis.yaml"
        if lvis_yaml.is_file():
            return resolve_dataset_config(lvis_yaml, dataset_format)
        train_json = _find_lvis_split(supplied, "train")
        if train_json is not None:
            ids, names = _categories(_read_lvis(train_json), train_json)
            return DatasetConfig(
                root=supplied.resolve(), train_split=train_json.resolve(),
                val_split=(_find_lvis_split(supplied, "val") or
                           supplied / "annotations/lvis_v1_val.json").resolve(),
                test_split=_find_lvis_split(supplied.resolve(), "test"),
                class_names=names, dataset_format="lvis", category_ids=ids,
            )
    yaml_path = supplied / "dataset.yaml" if supplied.is_dir() else supplied
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: expected a YAML mapping")
    selected_format = dataset_format if dataset_format != "auto" else str(raw.get("format", ""))
    if not selected_format:
        selected_format = "lvis" if str(raw.get("train", "")).endswith(".json") else "yolo"
    if selected_format not in {"lvis", "yolo"}:
        raise ValueError(f"{yaml_path}: unsupported format {selected_format!r}")
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
        value = Path(str(raw.get(key, default)).replace("\\", "/"))
        return value if value.is_absolute() else root / value

    if selected_format == "lvis":
        train_json = split_path("train", "annotations/lvis_v1_train.json")
        val_json = split_path("val", "annotations/lvis_v1_val.json")
        test_json = split_path("test", "annotations/lvis_v1_test.json")
        category_source = next((p for p in (train_json, val_json, test_json) if p.is_file()), train_json)
        ids, names = _categories(_read_lvis(category_source), category_source)
        image_root = Path(str(raw.get("image_root", ".")).replace("\\", "/"))
        if not image_root.is_absolute():
            image_root = root / image_root
        exhaustive = raw.get("exhaustive", False)
        if type(exhaustive) is not bool:
            raise ValueError(f"{yaml_path}: exhaustive must be true or false")
        return DatasetConfig(
            root=root, train_split=train_json, val_split=val_json,
            test_split=test_json if "test" in raw or test_json.is_file() else None,
            class_names=names, dataset_format="lvis", category_ids=ids,
            image_root=image_root.resolve(), exhaustive=exhaustive,
        )

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


class DetectionDataset(Dataset):
    def __len__(self) -> int:
        return len(self.image_paths)

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


class YoloDetectionDataset(DetectionDataset):
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
        if split_file is None or not split_file.exists():
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
            if not all(math.isfinite(v) for v in (cx, cy, box_width, box_height)):
                raise ValueError(f"{label_path}:{line_number}: non-finite box coordinate")
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


def _lvis_image_path(root: Path, image: dict) -> Path:
    # Custom exports use file_name; official LVIS v1 uses coco_url.
    name = image.get("file_name")
    if not name:
        url = str(image.get("coco_url", ""))
        parts = unquote(urlparse(url).path).strip("/").split("/")
        if len(parts) < 2 or not parts[-1]:
            raise ValueError(f"Image {image['id']}: missing file_name or usable coco_url")
        name = "/".join(parts[-2:])
    normalized = str(name).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate.resolve()
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ValueError(f"Image {image['id']}: stale Windows path {name!r}; use relative file_name")
    return (root / candidate).resolve()


class LvisDetectionDataset(DetectionDataset):
    """LVIS bounding boxes with stable category IDs and federated eval metadata."""

    def __init__(self, config: DatasetConfig, split: str = "train", augment: bool = False):
        self.config = config
        self.root = config.root
        self.class_names = config.class_names
        self.num_classes = len(self.class_names)
        self.augment = augment
        self.annotation_file = {"train": config.train_split, "val": config.val_split,
                                "test": config.test_split}[split]
        if self.annotation_file is None:
            raise FileNotFoundError(f"No {split} split configured; use --split val if no test set exists")
        raw = _read_lvis(self.annotation_file)
        ids, names = _categories(raw, self.annotation_file)
        if (ids, names) != (config.category_ids, config.class_names):
            raise ValueError(f"{self.annotation_file}: category IDs/names do not match the dataset config")
        self.category_id_to_label = {cid: i + 1 for i, cid in enumerate(ids)}
        self.label_to_category_id = {label: cid for cid, label in self.category_id_to_label.items()}
        self.images = raw["images"]
        self.annotations = defaultdict(list)
        self.image_paths = []
        image_ids = set()
        for image in self.images:
            image_id = _integer(image.get("id"), f"{self.annotation_file}: image id")
            if image_id in image_ids:
                raise ValueError(f"{self.annotation_file}: duplicate image id {image_id}")
            image_ids.add(image_id)
            for dimension in ("width", "height"):
                _integer(image.get(dimension), f"Image {image_id}: {dimension}", minimum=1)
            self.image_paths.append(_lvis_image_path(config.image_root or config.root, image))
        if not self.images:
            raise ValueError(f"{self.annotation_file}: no images")
        annotation_ids = set()
        for ann in raw["annotations"]:
            context = f"{self.annotation_file}: annotation {ann.get('id')}"
            ann_id = _integer(ann.get("id"), context, minimum=1)
            if ann_id in annotation_ids:
                raise ValueError(f"{context}: duplicate annotation id")
            annotation_ids.add(ann_id)
            image_id = _integer(ann.get("image_id"), context)
            category_id = _integer(ann.get("category_id"), context)
            if image_id not in image_ids or category_id not in self.category_id_to_label:
                raise ValueError(f"{context}: unknown image_id or category_id")
            if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                raise ValueError(f"{context}: crowd/ignore boxes are not supported by this training port")
            bbox = ann.get("bbox")
            if (not isinstance(bbox, list) or len(bbox) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox) or
                bbox[2] <= 0 or bbox[3] <= 0):
                raise ValueError(f"{context}: bbox must be finite pixel [x, y, width, height] with positive size")
            self.annotations[image_id].append(ann)

        all_ids = set(ids)
        for image in self.images:
            image_id = image["id"]
            present = {ann["category_id"] for ann in self.annotations[image_id]}
            for key in ("neg_category_ids", "not_exhaustive_category_ids"):
                if key not in image:
                    if not config.exhaustive:
                        raise ValueError(
                            f"Image {image_id}: missing {key}; only for fully annotated custom "
                            "data, set exhaustive: true in the YAML config"
                        )
                    image[key] = sorted(all_ids - present) if key == "neg_category_ids" else []
                values = image[key]
                if (not isinstance(values, list) or
                    any(type(v) is not int or v not in all_ids for v in values)):
                    raise ValueError(f"Image {image_id}: invalid {key}")
            if set(image["neg_category_ids"]) & (present | set(image["not_exhaustive_category_ids"])):
                raise ValueError(f"Image {image_id}: conflicting positive/negative/non-exhaustive categories")

    def category_labels(self, index: int) -> set[int]:
        return {self.category_id_to_label[ann["category_id"]]
                for ann in self.annotations[self.images[index]["id"]]}

    def _read_target(self, index: int, width: int, height: int) -> dict[str, torch.Tensor]:
        image = self.images[index]
        if (width, height) != (image["width"], image["height"]):
            raise ValueError(f"Image {image['id']}: decoded size {width}x{height} differs from JSON "
                             f"{image['width']}x{image['height']}")
        boxes, labels = [], []
        for ann in self.annotations[image["id"]]:
            x, y, w, h = ann["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Image {image['id']}: annotation {ann['id']} is empty after clipping")
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_label[ann["category_id"]])
        tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image["id"], dtype=torch.int64),
            "area": (tensor[:, 2] - tensor[:, 0]) * (tensor[:, 3] - tensor[:, 1]),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        for key in ("neg_category_ids", "not_exhaustive_category_ids"):
            target[key] = torch.tensor([self.category_id_to_label[cid] for cid in image[key]],
                                       dtype=torch.int64)
        return target


def build_detection_dataset(config: DatasetConfig, split: str = "train", augment: bool = False):
    dataset_type = LvisDetectionDataset if config.dataset_format == "lvis" else YoloDetectionDataset
    return dataset_type(config, split=split, augment=augment)


def check_checkpoint_dataset(payload: dict, config: DatasetConfig) -> None:
    if tuple(payload.get("class_names", ())) != config.class_names:
        raise ValueError("Checkpoint class names/order do not match the dataset")
    if (payload.get("dataset_format") == "lvis" and config.dataset_format == "lvis" and
        tuple(payload.get("category_ids", ())) != config.category_ids):
        raise ValueError("Checkpoint category IDs/order do not match the LVIS dataset")


def class_balanced_dataset(dataset: LvisDetectionDataset, threshold: float) -> Subset:
    """MMDetection ClassBalancedDataset rule: ceil(max(1, sqrt(t / f_c)))."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Repeat threshold must be finite and positive")
    labels_per_image = [dataset.category_labels(i) for i in range(len(dataset))]
    counts = Counter(label for labels in labels_per_image for label in labels)
    repeats = {label: max(1.0, math.sqrt(threshold * len(dataset) / count))
               for label, count in counts.items()}
    indices = []
    for index, labels in enumerate(labels_per_image):
        factor = max((repeats[label] for label in labels), default=1.0)
        indices.extend([index] * math.ceil(factor))
    return Subset(dataset, indices)


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
