from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from bacl.data import add_dataset_arguments, build_detection_dataset, resolve_dataset_config


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an LVIS or YOLO detection dataset for BACL")
    add_dataset_arguments(parser)
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_dataset_config(args.data, args.data_format)
    report: dict[str, object] = {
        "root": str(config.root),
        "num_classes": len(config.class_names),
        "dataset_format": config.dataset_format,
        "category_ids": list(config.category_ids),
        "class_names": list(config.class_names),
        "splits": {},
    }
    errors: list[str] = []
    total_distribution = Counter()
    image_owners: dict[Path, str] = {}

    for split in ("train", "val", "test"):
        if split == "test" and config.test_split is None:
            report["splits"][split] = {"skipped": "No test split configured"}
            continue
        try:
            dataset = build_detection_dataset(config, split=split)
        except Exception as exc:
            errors.append(f"{split}: {exc}")
            report["splits"][split] = {"error": str(exc)}
            continue
        distribution = Counter()
        image_distribution = Counter()
        box_count = 0
        for index, image_path in enumerate(tqdm(dataset.image_paths, desc=f"validate {split}")):
            try:
                if image_path in image_owners:
                    raise ValueError(f"Duplicate image path; already listed in {image_owners[image_path]}")
                image_owners[image_path] = split
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify()
                # ``verify`` checks the container structure but does not decode
                # pixel data. Reopen and load the image so truncated files are
                # caught before a long training run reaches them.
                with Image.open(image_path) as image:
                    image.load()
                target = dataset._read_target(index, width, height)
                box_count += len(target["labels"])
                distribution.update((target["labels"] - 1).tolist())
                image_distribution.update(set((target["labels"] - 1).tolist()))
            except Exception as exc:  # report every bad source item before failing
                errors.append(f"{split}: {image_path}: {exc}")
        total_distribution.update(distribution)
        counts = [distribution.get(index, 0) for index in range(len(config.class_names))]
        report["splits"][split] = {
            "images": len(dataset),
            "boxes": box_count,
            "classes_present": sum(count > 0 for count in counts),
            "min_boxes_per_present_class": min((count for count in counts if count > 0), default=0),
            "max_boxes_per_class": max(counts, default=0),
            "class_box_counts": counts,
            "class_image_counts": [image_distribution.get(i, 0) for i in range(len(config.class_names))],
        }

    all_counts = [total_distribution.get(index, 0) for index in range(len(config.class_names))]
    report["all_splits"] = {
        "boxes": sum(all_counts),
        "classes_present": sum(count > 0 for count in all_counts),
        "missing_classes": [
            config.class_names[index] for index, count in enumerate(all_counts) if count == 0
        ],
        "head_to_tail_ratio": (
            max(all_counts) / min(count for count in all_counts if count > 0)
            if any(all_counts)
            else None
        ),
    }
    report["errors"] = errors
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(f"Dataset validation failed with {len(errors)} error(s)")


if __name__ == "__main__":
    main()
