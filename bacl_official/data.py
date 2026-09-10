"""Read custom LVIS metadata without importing TorchVision or MMDetection."""
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_DATA = '/root/autodl-tmp/datasets'


@dataclass
class DatasetSpec:
    root: Path
    image_root: Path
    annotations: dict
    category_ids: tuple
    classes: tuple
    splits: dict


def category_table(raw):
    cats = raw.get('categories')
    if not isinstance(cats, list) or not cats:
        raise ValueError('LVIS categories must be a nonempty list')
    table = {}
    for cat in cats:
        cid, name = cat.get('id'), cat.get('name')
        if type(cid) is not int or cid < 0 or cid in table:
            raise ValueError('Invalid or duplicate category ID: {!r}'.format(cid))
        if not isinstance(name, str) or not name:
            raise ValueError('Missing category name: {}'.format(cid))
        if cat.get('frequency') not in ('r', 'c', 'f'):
            raise ValueError('Category {} needs LVIS frequency r/c/f for official evaluation'.format(cid))
        table[cid] = (name, cat['frequency'])
    return table


def image_filename(image):
    name = image.get('file_name')
    if not name:
        parts = urlparse(str(image.get('coco_url', ''))).path.strip('/').split('/')
        if len(parts) < 2 or not parts[-1]:
            raise ValueError('Image {} has neither file_name nor usable coco_url'.format(image.get('id')))
        name = '/'.join(parts[-2:])
    if not isinstance(name, str):
        raise ValueError('file_name must be a string')
    normalized = name.replace('\\', '/')
    if len(normalized) > 1 and normalized[1] == ':' and not Path(normalized).is_absolute():
        raise ValueError('Stale Windows image path: {}; use relative file_name'.format(name))
    return normalized


def load_dataset_spec(data=DEFAULT_DATA, needed=('train', 'val', 'test')):
    supplied = Path(data).expanduser().resolve()
    if supplied.is_dir():
        root = supplied
        config_file = root / 'dataset_lvis.yaml'
        raw_config = {}
        if config_file.is_file():
            supplied = config_file
    if not supplied.is_dir():
        import yaml
        raw_config = yaml.safe_load(supplied.read_text(encoding='utf-8-sig')) or {}
        if raw_config.get('format', 'lvis') != 'lvis':
            raise ValueError('The official backend requires LVIS JSON; use --backend torchvision for YOLO')
        root = Path(str(raw_config.get('path', supplied.parent))).expanduser()
        if not root.is_absolute():
            root = supplied.parent / root
    root = root.resolve()
    image_root = Path(str(raw_config.get('image_root', '.')))
    if not image_root.is_absolute():
        image_root = root / image_root
    annotations, splits, reference = {}, {}, None
    # Always derive class order/frequency from train, including standalone evaluation.
    required_splits = tuple(dict.fromkeys(('train',) + tuple(needed)))
    for split in required_splits:
        filename = raw_config.get(split, 'annotations/lvis_v1_{}.json'.format(split))
        path = Path(str(filename))
        if not path.is_absolute():
            path = root / path
        with path.open(encoding='utf-8-sig') as stream:
            raw = json.load(stream)
        table = category_table(raw)
        if reference is None:
            reference = table
        elif table != reference:
            raise ValueError('{} category IDs, names or frequency bins differ from train'.format(path))
        if any(not isinstance(raw.get(k), list) for k in ('images', 'annotations')):
            raise ValueError('{} must contain images and annotations arrays'.format(path))
        annotations[split], splits[split] = path.resolve(), raw
    ids = tuple(sorted(reference))
    return DatasetSpec(root, image_root.resolve(), annotations, ids,
                       tuple(reference[cid][0] for cid in ids), splits)


def inspect_dataset(spec, decode_images=False, require_masks=False, check_files=True):
    """Validate IDs/boxes/federated metadata and optionally fully decode images."""
    errors, owners, summaries = [], {}, {}
    all_categories = set(spec.category_ids)
    for split, raw in spec.splits.items():
        images, annotations = {}, set()
        boxes_per_class, images_per_class = Counter(), Counter()
        positives = {}
        mask_count = 0
        for image in raw['images']:
            iid = image.get('id')
            if type(iid) is not int or iid < 0 or iid in images:
                errors.append('{}: invalid/duplicate image id {}'.format(split, iid))
                continue
            images[iid] = image
            positives[iid] = set()
            try:
                for dimension in ('width', 'height'):
                    if type(image.get(dimension)) is not int or image[dimension] <= 0:
                        raise ValueError('invalid image dimensions')
                path = (spec.image_root / image_filename(image)).resolve()
                if path in owners:
                    raise ValueError('image path already listed in {}'.format(owners[path]))
                owners[path] = split
                if check_files and not path.is_file():
                    raise FileNotFoundError(str(path))
                if decode_images:
                    from PIL import Image
                    with Image.open(path) as source:
                        source.load()
                        if source.size != (image['width'], image['height']):
                            raise ValueError('decoded dimensions differ from JSON')
                for key in ('neg_category_ids', 'not_exhaustive_category_ids'):
                    values = image.get(key)
                    if not isinstance(values, list) or any(type(v) is not int or v not in all_categories for v in values):
                        raise ValueError('missing or invalid {}'.format(key))
            except (OSError, ValueError) as exc:
                errors.append('{} image {}: {}'.format(split, iid, exc))
        for ann in raw['annotations']:
            aid, iid, cid = ann.get('id'), ann.get('image_id'), ann.get('category_id')
            try:
                if type(aid) is not int or aid <= 0 or aid in annotations:
                    raise ValueError('invalid/duplicate annotation id')
                annotations.add(aid)
                if type(iid) is not int or iid not in images or type(cid) is not int or cid not in all_categories:
                    raise ValueError('unknown image/category ID')
                bbox = ann.get('bbox')
                if not isinstance(bbox, list) or len(bbox) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox):
                    raise ValueError('invalid pixel xywh bbox')
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    raise ValueError('nonpositive bbox size')
                image = images[iid]
                if min(x + w, image['width']) <= max(x, 0) or min(y + h, image['height']) <= max(y, 0):
                    raise ValueError('bbox outside image')
                area = ann.get('area')
                if type(area) not in (int, float) or not math.isfinite(area) or area <= 0:
                    raise ValueError('invalid area')
                positives[iid].add(cid)
                boxes_per_class[cid] += 1
                mask = ann.get('segmentation')
                has_mask = bool(mask) and isinstance(mask, (dict, list))
                mask_count += int(has_mask)
                if require_masks and not has_mask:
                    raise ValueError('original CopyPaste requires real instance masks')
            except (TypeError, KeyError, ValueError) as exc:
                errors.append('{} annotation {}: {}'.format(split, aid, exc))
        for iid, positive in positives.items():
            images_per_class.update(positive)
            image = images[iid]
            negatives = image.get('neg_category_ids', [])
            non_exhaustive = image.get('not_exhaustive_category_ids', [])
            if (isinstance(negatives, list) and isinstance(non_exhaustive, list)
                    and all(type(v) is int for v in negatives + non_exhaustive)):
                if set(negatives) & (positive | set(non_exhaustive)):
                    errors.append('{} image {}: conflicting LVIS category metadata'.format(split, iid))
        summaries[split] = {
            'images': len(raw['images']), 'annotations': len(raw['annotations']),
            'annotations_with_masks': mask_count,
            'class_box_counts': [boxes_per_class[cid] for cid in spec.category_ids],
            'class_image_counts': [images_per_class[cid] for cid in spec.category_ids],
            'annotation_sha256': hashlib.sha256(spec.annotations[split].read_bytes()).hexdigest(),
        }
    frequencies = Counter(cat['frequency'] for cat in spec.splits['train']['categories'])
    return {'root': str(spec.root), 'image_root': str(spec.image_root),
            'num_classes': len(spec.classes), 'category_ids': list(spec.category_ids),
            'class_names': list(spec.classes), 'frequency_category_counts': dict(frequencies),
            'splits': summaries, 'errors': errors}
