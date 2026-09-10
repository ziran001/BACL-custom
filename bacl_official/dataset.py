"""Adapt metadata and report storage; delegate scoring to upstream unchanged."""
import json
from pathlib import Path
from lvis import LVIS
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.lvis import LVISV1Dataset

from .data import image_filename


@DATASETS.register_module()
class CustomLVISV1Dataset(LVISV1Dataset):
    def __init__(self, *args, expected_category_ids=None, metrics_output=None, **kwargs):
        self.expected_category_ids = tuple(expected_category_ids or ())
        self.metrics_output = metrics_output
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        self.coco = LVIS(ann_file)
        self.cat_ids = sorted(self.coco.get_cat_ids())
        if self.expected_category_ids and tuple(self.cat_ids) != self.expected_category_ids:
            raise ValueError('LVIS category IDs differ from the configured training mapping')
        names = tuple(self.coco.cats[cid]['name'] for cid in self.cat_ids)
        if names != tuple(self.CLASSES):
            raise ValueError('LVIS category names/order differ from the configured training mapping')
        self.cat2label = {cid: index for index, cid in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        infos = []
        for iid in self.img_ids:
            info = dict(self.coco.load_imgs([iid])[0])
            info['filename'] = image_filename(info)
            infos.append(info)
        return infos

    def evaluate(self, results, *args, **kwargs):
        metrics = super().evaluate(results, *args, **kwargs)
        if self.metrics_output:
            output = Path(self.metrics_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'backend': 'official',
                'metric': 'LVIS bbox AP@0.50:0.95 (mmlvis; original BACL evaluator)',
                'category_ids': self.cat_ids,
                'class_names': list(self.CLASSES),
                'metrics': dict(metrics),
            }
            output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        return metrics
