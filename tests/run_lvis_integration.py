"""Opt-in LEGACY TorchVision CLI check; does not test the original backend."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.test_lvis_data import make_fixture


def main():
    with tempfile.TemporaryDirectory(prefix="bacl-lvis-integration-") as directory:
        root = Path(directory)
        make_fixture(root)

        def run(module, *args):
            command = [sys.executable, "-m", module, "--backend", "torchvision",
                       "--data", str(root), *map(str, args)]
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, check=True)

        report_file = root / "report.json"
        run("tools.validate_dataset", "--output", report_file)
        report = json.loads(report_file.read_text(encoding="utf-8"))
        assert not report["errors"], report
        assert report["category_ids"] == [7, 42]
        run("tools.smoke_test")
        common = ["--epochs", "1", "--batch-size", "2", "--workers", "0",
                  "--min-sizes", "128", "--max-size", "192", "--no-pretrained",
                  "--max-train-batches", "1", "--max-val-batches", "1",
                  "--statistics-boxes-per-gt", "2", "--sampled-classes", "2",
                  "--sampled-features-per-class", "2"]
        stage1 = root / "representation"
        stage2 = root / "classifier"
        run("tools.train", "--stage", "representation", "--output", stage1, *common)
        run("tools.train", "--stage", "classifier", "--output", stage2,
            "--checkpoint", stage1 / "last.pth", *common)
        output = root / "results/test.json"
        run("tools.evaluate", "--checkpoint", stage2 / "last.pth", "--split", "test",
            "--workers", "0", "--output", output)
        metrics = json.loads(output.read_text(encoding="utf-8"))
        assert metrics["category_ids"] == [7, 42]
        assert metrics["dataset_format"] == "lvis"
        assert 0 <= metrics["map50"] <= 1
        assert sum(item["ground_truth_boxes"] for item in metrics["per_class"]) == 1
        print("LVIS integration check passed", flush=True)


if __name__ == "__main__":
    main()
