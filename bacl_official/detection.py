"""Batch inference for arbitrary images with the original BACL detector."""
import argparse
import json
import math
from pathlib import Path

from .config import build_config
from .data import DEFAULT_DATA, load_dataset_spec
from .provenance import PROJECT_ROOT, activate_upstream, verify_upstream
from .runtime import check_checkpoint, check_runtime


IMAGE_SUFFIXES = frozenset(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
DEFAULT_INPUT = '/root/autodl-tmp/test'
DEFAULT_OUTPUT = '/root/autodl-tmp/test_results'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'runs' / 'official_classifier' / 'best_bbox_AP.pth'


def find_images(input_path):
    """Return a deterministic image list and the root used for relative outputs."""
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError('Unsupported image extension: {}'.format(input_path.suffix))
        return [input_path], input_path.parent
    if not input_path.is_dir():
        raise ValueError('Input is neither an image nor a directory: {}'.format(input_path))
    images = sorted((path for path in input_path.rglob('*')
                     if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                    key=lambda path: path.as_posix().lower())
    if not images:
        raise ValueError('No supported images found under {}'.format(input_path))
    return images, input_path


def validate_output(input_path, output_path):
    """Prevent annotated files from replacing or being rediscovered as inputs."""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    input_root = input_path if input_path.is_dir() else input_path.parent
    if (output_path == input_root or input_root in output_path.parents
            or output_path in input_root.parents):
        raise ValueError('Input and output directories must not overlap')
    return output_path


def detection_records(result, classes, category_ids, score_threshold):
    """Convert MMDetection per-class arrays to portable JSON records."""
    if isinstance(result, tuple):
        result = result[0]
    if len(result) != len(classes) or len(classes) != len(category_ids):
        raise ValueError('Model outputs, class names and LVIS category IDs differ in length')
    records = []
    for label_index, boxes in enumerate(result):
        for row in boxes:
            values = row.tolist() if hasattr(row, 'tolist') else list(row)
            if len(values) < 5:
                raise ValueError('Detection row must contain x1, y1, x2, y2 and score')
            x1, y1, x2, y2, score = (float(value) for value in values[:5])
            if score <= score_threshold:
                continue
            records.append({
                'category_id': int(category_ids[label_index]),
                'label_index': label_index,
                'class_name': classes[label_index],
                'score': score,
                'bbox_xyxy': [x1, y1, x2, y2],
                'bbox_xywh': [x1, y1, x2 - x1, y2 - y1],
            })
    records.sort(key=lambda item: item['score'], reverse=True)
    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the original BACL classifier checkpoint on arbitrary images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help='Image file or directory; directories are scanned recursively')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Directory for annotated images and detections.json')
    parser.add_argument('--data', default=DEFAULT_DATA,
                        help='Training dataset root used to recover the exact category mapping')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT),
                        help='Completed stage-2 classifier checkpoint')
    parser.add_argument('--score-thr', type=float, default=.3,
                        help='Only draw and record detections above this confidence')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    if not math.isfinite(args.score_thr) or not 0 <= args.score_thr <= 1:
        raise ValueError('--score-thr must be a finite value between 0 and 1')
    images, input_root = find_images(args.input)
    output_root = validate_output(args.input, args.output)
    checkpoint = Path(args.checkpoint).expanduser().resolve()

    # The training JSON is the authority for non-contiguous LVIS IDs and class order.
    spec = load_dataset_spec(args.data, needed=('train', 'val'))
    runtime = check_runtime()
    check_checkpoint(checkpoint, spec, stage='classifier', resume=True)
    cfg = build_config(spec, 'classifier', output_root, checkpoint=checkpoint,
                       workers=0, test_split='val')

    activate_upstream()
    from mmdet.apis import inference_detector, init_detector
    model = init_detector(cfg, str(checkpoint), device=args.device)
    if tuple(model.CLASSES) != spec.classes:
        raise ValueError('Loaded model class names/order differ from the training JSON')

    output_root.mkdir(parents=True, exist_ok=True)
    image_reports = []
    total_detections = 0
    for index, image_path in enumerate(images, 1):
        relative = image_path.relative_to(input_root)
        output_file = output_root / relative
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result = inference_detector(model, str(image_path))
        records = detection_records(result, spec.classes, spec.category_ids, args.score_thr)
        model.show_result(str(image_path), result, score_thr=args.score_thr,
                          show=False, out_file=str(output_file))
        total_detections += len(records)
        image_reports.append({
            'input_file': relative.as_posix(),
            'output_file': relative.as_posix(),
            'detection_count': len(records),
            'detections': records,
        })
        print('[{}/{}] {}: {} detections'.format(
            index, len(images), relative.as_posix(), len(records)), flush=True)

    report = {
        'backend': 'official',
        'model_stage': 'classifier',
        'checkpoint': checkpoint.as_posix(),
        'dataset_root': spec.root.as_posix(),
        'input': Path(args.input).expanduser().resolve().as_posix(),
        'output': output_root.as_posix(),
        'score_threshold': args.score_thr,
        'image_count': len(images),
        'detection_count': total_detections,
        'category_ids': list(spec.category_ids),
        'class_names': list(spec.classes),
        'upstream': verify_upstream(),
        'runtime': runtime['runtime'],
        'images': image_reports,
    }
    result_file = output_root / 'detections.json'
    result_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                           encoding='utf-8')
    print('Done: annotated images are in {}'.format(output_root), flush=True)
    print('JSON results: {}'.format(result_file), flush=True)


if __name__ == '__main__':
    main()
