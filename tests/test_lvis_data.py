from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from bacl.data import (
    LvisDetectionDataset, YoloDetectionDataset, build_detection_dataset,
    check_checkpoint_dataset, class_balanced_dataset, resolve_dataset_config,
)
from bacl.engine import evaluate_map50


def make_fixture(root: Path) -> dict:
    """Small exhaustive LVIS export with deliberately unsorted, sparse IDs."""
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    payload = {
        "categories": [{"id": 42, "name": "squid", "frequency": "r"},
                       {"id": 7, "name": "snail", "frequency": "c"}],
        "images": [], "annotations": [],
    }
    for split in ("train", "val", "test"):
        raw = copy.deepcopy(payload)
        for index in range(2):
            relative = f"images/{split}/{index}.jpg"
            image_path = root / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (80, 64), (50, 100, 150)).save(image_path)
            raw["images"].append({
                "id": 100 + index, "file_name": relative, "width": 80, "height": 64,
                "neg_category_ids": [7] if index == 0 else [7, 42],
                "not_exhaustive_category_ids": [],
            })
        raw["annotations"] = [{"id": 1, "image_id": 100, "category_id": 42,
                               "bbox": [10, 5, 20, 30], "area": 600, "segmentation": []}]
        (root / "annotations" / f"lvis_v1_{split}.json").write_text(json.dumps(raw), encoding="utf-8")
    return json.loads((root / "annotations/lvis_v1_train.json").read_text())


class LvisDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = make_fixture(self.root)
        self.config = resolve_dataset_config(self.root)

    def write_train(self, raw):
        (self.root / "annotations/lvis_v1_train.json").write_text(json.dumps(raw), encoding="utf-8")

    def test_sparse_mapping_pixel_boxes_and_empty_images(self):
        dataset = build_detection_dataset(self.config)
        self.assertIsInstance(dataset, LvisDetectionDataset)
        self.assertEqual(self.config.category_ids, (7, 42))
        self.assertEqual(self.config.class_names, ("snail", "squid"))
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 64, 80))
        self.assertEqual(target["boxes"].tolist(), [[10, 5, 30, 35]])
        self.assertEqual(target["labels"].tolist(), [2])
        self.assertEqual(target["image_id"].item(), 100)
        self.assertEqual(target["neg_category_ids"].tolist(), [1])
        self.assertEqual(tuple(dataset[1][1]["boxes"].shape), (0, 4))
        self.assertEqual(dataset.label_to_category_id, {1: 7, 2: 42})

    def test_flip_does_not_mutate_cached_boxes(self):
        dataset = build_detection_dataset(self.config, augment=True)
        with patch("bacl.data.torch.rand", return_value=torch.tensor(0.0)):
            self.assertEqual(dataset[0][1]["boxes"].tolist(), [[50, 5, 70, 35]])
        dataset.augment = False
        self.assertEqual(dataset[0][1]["boxes"].tolist(), [[10, 5, 30, 35]])

    def test_reordered_categories_allowed_but_changed_mapping_rejected(self):
        val_file = self.root / "annotations/lvis_v1_val.json"
        raw = json.loads(val_file.read_text())
        raw["categories"].reverse()
        val_file.write_text(json.dumps(raw))
        build_detection_dataset(self.config, "val")
        raw["categories"][0]["name"] = "different"
        val_file.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "category IDs/names"):
            build_detection_dataset(self.config, "val")

    def test_duplicate_ids_or_dangling_annotations_fail(self):
        changes = [
            lambda r: r["images"].append(copy.deepcopy(r["images"][0])),
            lambda r: r["annotations"].append(copy.deepcopy(r["annotations"][0])),
            lambda r: r["annotations"][0].update(image_id=999),
            lambda r: r["annotations"][0].update(category_id=999),
        ]
        for change in changes:
            with self.subTest(change=change):
                raw = copy.deepcopy(self.raw)
                change(raw)
                self.write_train(raw)
                with self.assertRaises(ValueError):
                    build_detection_dataset(self.config)

    def test_bad_boxes_and_dimensions_fail(self):
        for box in ([0, 0, float("nan"), 1], [0, 0, 0, 1], [81, 0, 5, 5]):
            with self.subTest(box=box):
                raw = copy.deepcopy(self.raw)
                raw["annotations"][0]["bbox"] = box
                self.write_train(raw)
                with self.assertRaises(ValueError):
                    build_detection_dataset(self.config)[0]
        self.write_train(self.raw)
        with self.assertRaisesRegex(ValueError, "decoded size"):
            build_detection_dataset(self.config)._read_target(0, 100, 100)

    def test_missing_metadata_requires_explicit_exhaustive_opt_in(self):
        for image in self.raw["images"]:
            image.pop("neg_category_ids")
            image.pop("not_exhaustive_category_ids")
        self.write_train(self.raw)
        with self.assertRaisesRegex(ValueError, "exhaustive: true"):
            build_detection_dataset(self.config)
        dataset = build_detection_dataset(replace(self.config, exhaustive=True))
        self.assertEqual(dataset[0][1]["neg_category_ids"].tolist(), [1])
        self.assertEqual(dataset[1][1]["neg_category_ids"].tolist(), [1, 2])

    def test_metadata_conflicts_fail(self):
        self.raw["images"][0]["neg_category_ids"] = [42]
        self.write_train(self.raw)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            build_detection_dataset(self.config)

    def test_lvis_preferred_over_stale_yolo_yaml_and_yolo_still_supported(self):
        (self.root / "dataset.yaml").write_text("names: [snail, squid]\nnc: 2\ntrain: train.txt\n")
        (self.root / "train.txt").write_text("images/train/0.jpg\n")
        (self.root / "labels/train").mkdir(parents=True)
        (self.root / "labels/train/0.txt").write_text("1 0.25 0.3125 0.25 0.46875\n")
        self.assertEqual(resolve_dataset_config(self.root).dataset_format, "lvis")
        yolo = build_detection_dataset(resolve_dataset_config(self.root, "yolo"))
        self.assertIsInstance(yolo, YoloDetectionDataset)
        self.assertEqual(yolo[0][1]["boxes"].tolist(), [[10, 5, 30, 35]])

    def test_yaml_image_root_and_coco_url(self):
        for image in self.raw["images"]:
            image["file_name"] = image["file_name"].removeprefix("images/")
        self.write_train(self.raw)
        config_file = self.root / "dataset_lvis.yaml"
        config_file.write_text("format: lvis\npath: .\nimage_root: images\n")
        dataset = build_detection_dataset(resolve_dataset_config(self.root))
        self.assertEqual(dataset[0][0].shape[-1], 80)
        self.raw["images"][0].pop("file_name")
        self.raw["images"][0]["coco_url"] = "http://images.cocodataset.org/train/0.jpg"
        self.write_train(self.raw)
        dataset = build_detection_dataset(resolve_dataset_config(config_file))
        self.assertEqual(dataset[0][0].shape[-1], 80)

    def test_optional_test_split_and_explicit_missing_test_error(self):
        (self.root / "annotations/lvis_v1_test.json").unlink()
        config = resolve_dataset_config(self.root)
        self.assertIsNone(config.test_split)
        with self.assertRaisesRegex(FileNotFoundError, "No test split"):
            build_detection_dataset(config, "test")
        yaml = self.root / "dataset_lvis.yaml"
        yaml.write_text("format: lvis\npath: .\ntest: missing.json\n")
        self.assertIsNotNone(resolve_dataset_config(yaml).test_split)
        with self.assertRaises(FileNotFoundError):
            build_detection_dataset(resolve_dataset_config(yaml), "test")

    def test_class_balancing_keeps_empty_images_and_repeats_rare_images(self):
        dataset = build_detection_dataset(self.config)
        self.assertEqual(class_balanced_dataset(dataset, 1).indices, [0, 0, 1])
        self.assertEqual(class_balanced_dataset(dataset, 0.001).indices, [0, 1])

    def test_checkpoint_mapping_guard_and_legacy_checkpoint(self):
        payload = {"class_names": list(self.config.class_names)}
        check_checkpoint_dataset(payload, self.config)
        payload.update(dataset_format="lvis", category_ids=[7, 42])
        check_checkpoint_dataset(payload, self.config)
        payload["category_ids"] = [42, 7]
        with self.assertRaisesRegex(ValueError, "category IDs"):
            check_checkpoint_dataset(payload, self.config)

    def test_validator_records_decode_errors_and_split_overlap(self):
        from tools.validate_dataset import main

        (self.root / "images/train/1.jpg").write_bytes(b"not an image")
        val_file = self.root / "annotations/lvis_v1_val.json"
        raw = json.loads(val_file.read_text())
        raw["images"][0]["file_name"] = "images/train/0.jpg"
        val_file.write_text(json.dumps(raw))
        output = self.root / "reports/validation.json"
        with patch("sys.argv", ["validate_dataset", "--data", str(self.root), "--output", str(output)]):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                main()
        report = json.loads(output.read_text())
        self.assertEqual(len(report["errors"]), 2)
        self.assertTrue(any("Duplicate image path" in error for error in report["errors"]))
        self.assertEqual(report["splits"]["test"]["boxes"], 1)

    def test_duplicate_category_ids_and_ambiguous_auto_discovery_fail(self):
        raw = copy.deepcopy(self.raw)
        raw["categories"].append(copy.deepcopy(raw["categories"][0]))
        self.write_train(raw)
        with self.assertRaisesRegex(ValueError, "duplicate category id"):
            resolve_dataset_config(self.root)
        self.write_train(self.raw)
        (self.root / "annotations/lvis_train.json").write_text(json.dumps(self.raw))
        with self.assertRaisesRegex(ValueError, "Multiple LVIS train"):
            resolve_dataset_config(self.root)


class FixedDetector(torch.nn.Module):
    def forward(self, images):
        return [{"boxes": torch.tensor([[50., 40., 60., 50.], [10., 5., 30., 35.]]),
                 "labels": torch.tensor([2, 2]), "scores": torch.tensor([0.99, 0.9])}
                for _ in images]


class FederatedEvaluationTests(unittest.TestCase):
    def test_non_exhaustive_and_unknown_predictions_are_ignored(self):
        positive = {"boxes": torch.tensor([[10., 5., 30., 35.]]), "labels": torch.tensor([2]),
                    "neg_category_ids": torch.tensor([1]),
                    "not_exhaustive_category_ids": torch.tensor([2])}
        unknown = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long),
                   "neg_category_ids": torch.tensor([1]),
                   "not_exhaustive_category_ids": torch.empty(0, dtype=torch.long)}
        images = [torch.zeros(3, 64, 80), torch.zeros(3, 64, 80)]
        metrics = evaluate_map50(FixedDetector(), [(images, [positive, unknown])],
                                 torch.device("cpu"), 2, show_progress=False)
        self.assertAlmostEqual(metrics["map50"], 1.0)
        positive["not_exhaustive_category_ids"] = torch.empty(0, dtype=torch.long)
        unknown["neg_category_ids"] = torch.tensor([1, 2])
        metrics = evaluate_map50(FixedDetector(), [(images, [positive, unknown])],
                                 torch.device("cpu"), 2, show_progress=False)
        self.assertLess(metrics["map50"], 0.5)


if __name__ == "__main__":
    unittest.main()
