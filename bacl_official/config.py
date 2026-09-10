"""Load the original configs, then apply only explicit dataset/runtime overrides."""
import math
from pathlib import Path

from .provenance import UPSTREAM_ROOT, verify_upstream


CONFIGS = {
    'representation': 'bacl_representation_faster_rcnn_r50_fpn_1x_lvis_v1.py',
    'classifier': 'bacl_classifier_faster_rcnn_r50_fpn_mstrain_1x_lvis_v1.py',
}


def build_config(spec, stage, output, checkpoint=None, batch_size=2, workers=2,
                 world_size=1, epochs=12, lr=None, bbox_only=True, test_split='test'):
    verify_upstream()
    if batch_size < 1 or world_size < 1 or workers < 0 or epochs < 1:
        raise ValueError('Batch size/world size/epochs must be positive; workers must be nonnegative')
    if lr is not None and (not math.isfinite(lr) or lr <= 0):
        raise ValueError('Learning rate must be finite and positive')
    from mmcv import Config
    cfg = Config.fromfile(str(UPSTREAM_ROOT / 'configs' / 'bacl' / CONFIGS[stage]))
    original_lr = cfg.optimizer.lr
    count = len(spec.classes)
    cfg.model.roi_head.bbox_head.num_classes = count
    cfg.model.roi_head.bbox_head.loss_cls.num_classes = count
    cfg.custom_imports = dict(imports=['bacl_official.dataset'], allow_failed_imports=False)
    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = workers
    for split in ('train', 'val', 'test'):
        ds = cfg.data[split]
        if ds['type'] == 'MultiImageMixDataset':
            ds = ds['dataset']
        source_split = test_split if split == 'test' else split
        # Training does not require a test JSON; unused slots may point to val.
        if source_split not in spec.annotations:
            source_split = 'val' if 'val' in spec.annotations else test_split
        ds['type'] = 'CustomLVISV1Dataset'
        ds['ann_file'] = spec.annotations[source_split].as_posix()
        ds['img_prefix'] = spec.image_root.as_posix() + '/'
        ds['classes'] = spec.classes
        ds['expected_category_ids'] = spec.category_ids
    if stage == 'representation' and bbox_only:
        for operation in cfg.data.train.dataset.pipeline:
            if operation['type'] == 'LoadAnnotations':
                operation['with_mask'] = False
            if operation['type'] == 'RandomCrop':
                operation['recompute_bbox'] = False
        cfg.data.train.pipeline = [p for p in cfg.data.train.pipeline if p['type'] != 'CopyPaste']
    cfg.work_dir = Path(output).resolve().as_posix()
    cfg.total_epochs = epochs
    cfg.optimizer.lr = lr if lr is not None else original_lr * batch_size * world_size / 16.0
    cfg.load_from = Path(checkpoint).resolve().as_posix() if checkpoint else None
    if stage == 'classifier' and not checkpoint:
        raise ValueError('Classifier stage requires an original-backend representation checkpoint')
    # The original save_best=auto selects bbox_AP first; make that choice explicit.
    cfg.evaluation.save_best = 'bbox_AP'
    cfg.evaluation.rule = 'greater'
    return cfg
