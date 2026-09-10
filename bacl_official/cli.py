"""Dataset/runtime adapters around the unmodified upstream entrypoints."""
import argparse
import json
import os
from pathlib import Path
import pprint
import runpy
import sys

from .config import build_config
from .data import DEFAULT_DATA, inspect_dataset, load_dataset_spec
from .provenance import UPSTREAM_ROOT, activate_upstream, verify_upstream


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def dump_config(cfg, path):
    # Old MMCV uses locale-dependent file I/O. ASCII-escaped Python literals
    # preserve Chinese paths/classes on both Windows and Linux without patching it.
    text = '\n'.join('{} = {}'.format(key, pprint.pformat(value, width=100))
                     for key, value in cfg.items()) + '\n'
    Path(path).write_text(text.encode('ascii', 'backslashreplace').decode('ascii'), encoding='ascii')


def dataset_args(parser):
    parser.add_argument('--data', default=DEFAULT_DATA, help='Dataset root or LVIS YAML')
    parser.add_argument('--data-format', choices=['lvis'], default='lvis')
    parser.add_argument('--with-masks', action='store_true',
                        help='Restore original CopyPaste; requires real instance masks (not box polygons)')


def checked_data(args, needed):
    spec = load_dataset_spec(args.data, needed=needed)
    report = inspect_dataset(spec, require_masks=args.with_masks)
    if report['errors']:
        raise ValueError('Dataset preflight failed:\n' + '\n'.join(report['errors'][:20]))
    return spec, report


def summary(report):
    return {**{key: value for key, value in report.items() if key != 'splits'},
        'splits': {split: {key: value for key, value in stats.items()
                           if not key.startswith('class_')}
                   for split, stats in report['splits'].items()}}


def run_upstream(script, arguments):
    activate_upstream()
    previous = sys.argv
    try:
        sys.argv = [str(UPSTREAM_ROOT / 'tools' / script)] + list(map(str, arguments))
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        sys.argv = previous
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def train():
    parser = argparse.ArgumentParser(description='Original BACL Faster R-CNN R50 LVIS v1 training')
    dataset_args(parser)
    parser.add_argument('--stage', required=True, choices=['representation', 'classifier'])
    parser.add_argument('--output', required=True, help='New directory, separate from legacy runs')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--checkpoint', help='Original representation checkpoint for stage 2')
    group.add_argument('--resume-from', help='Resume the SAME stage with optimizer/epoch state')
    parser.add_argument('--batch-size', type=int, default=2, help='Per-GPU batch size')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=12, help='Original schedule is 12; steps stay at 8,11')
    parser.add_argument('--lr', type=float, default=None, help='Default .02 * global batch / 16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true', help='Validate data/source and export config only')
    args = parser.parse_args()
    spec, report = checked_data(args, ('train', 'val'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    cfg = build_config(spec, args.stage, args.output,
                       checkpoint=args.checkpoint or args.resume_from,
                       batch_size=args.batch_size, workers=args.workers,
                       world_size=world_size, epochs=args.epochs, lr=args.lr,
                       bbox_only=not args.with_masks, test_split='val')
    if args.resume_from:
        cfg.load_from = None
        cfg.resume_from = Path(args.resume_from).resolve().as_posix()
    output = Path(cfg.work_dir)
    rank = int(os.environ.get('RANK', '0'))
    output.mkdir(parents=True, exist_ok=True)
    # Refuse accidental reuse of a legacy or completed experiment directory.
    if not args.dry_run and not args.resume_from and any(output.glob('*.pth')):
        raise ValueError('Output already contains checkpoints; use a new directory or --resume-from')
    config_file = output / 'resolved_config_rank{}.py'.format(rank)
    dump_config(cfg, config_file)
    provenance = {'backend': 'official', 'upstream': verify_upstream(), 'stage': args.stage,
                  'bbox_only': not args.with_masks, 'dataset': summary(report),
                  'global_batch_size': args.batch_size * world_size,
                  'learning_rate': cfg.optimizer.lr, 'epochs': args.epochs,
                  'best_checkpoint_metric': 'bbox_AP', 'dry_run': args.dry_run}
    if rank == 0:
        print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        if rank == 0:
            write_json(output / 'dry_run_provenance.json', provenance)
        print('DRY RUN ONLY: no model, checkpoint or CUDA training has been tested.', flush=True)
        return
    from .runtime import check_checkpoint, check_runtime, prepare_distributed
    provenance.update(check_runtime())
    if args.checkpoint or args.resume_from:
        check_checkpoint(args.checkpoint or args.resume_from, spec, args.stage,
                         resume=bool(args.resume_from))
    if rank == 0:
        write_json(output / 'provenance.json', provenance)
    prepare_distributed()
    run_upstream('train.py', [config_file, '--work-dir', cfg.work_dir,
                             '--launcher', 'pytorch', '--seed', args.seed])


def evaluate():
    parser = argparse.ArgumentParser(description='Original BACL LVIS bbox AP@0.50:0.95 evaluation')
    dataset_args(parser)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--stage', choices=['representation', 'classifier'], default='classifier',
                        help='Architecture used to produce the checkpoint')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--output', default='evaluation.json', help='Official metrics JSON')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    spec, report = checked_data(args, (args.split,))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = build_config(spec, args.stage, output.parent, checkpoint=args.checkpoint,
                       workers=args.workers, bbox_only=not args.with_masks, test_split=args.split)
    cfg.data.test.metrics_output = output.as_posix()
    rank = int(os.environ.get('RANK', '0'))
    config_file = output.with_suffix('.config.py' if rank == 0 else '.rank{}.config.py'.format(rank))
    dump_config(cfg, config_file)
    if args.dry_run:
        write_json(output.with_suffix('.dry-run.json'), {'upstream': verify_upstream(),
                   'dataset': summary(report), 'config': str(config_file), 'dry_run': True})
        print('DRY RUN ONLY: wrote {}; no inference or metrics were produced.'.format(config_file))
        return
    from .runtime import check_checkpoint, check_runtime, prepare_distributed
    runtime = check_runtime()
    check_checkpoint(args.checkpoint, spec, args.stage, resume=True)
    # No pretrained download: original test.py sets model.pretrained=None.
    prepare_distributed()
    run_upstream('test.py', [config_file, Path(args.checkpoint).resolve(), '--eval', 'bbox',
                            '--launcher', 'pytorch'])
    if int(os.environ.get('RANK', '0')) == 0:
        write_json(output.with_suffix('.provenance.json'), {
            **runtime, 'checkpoint': str(Path(args.checkpoint).resolve()),
            'stage': args.stage, 'split': args.split, 'dataset': summary(report)})


def validate_dataset():
    parser = argparse.ArgumentParser(description='Validate unchanged custom LVIS JSON and images')
    dataset_args(parser)
    parser.add_argument('--output')
    parser.add_argument('--skip-decode', action='store_true', help='Metadata/path checks only (faster)')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    args = parser.parse_args()
    spec = load_dataset_spec(args.data, needed=args.splits)
    report = inspect_dataset(spec, decode_images=not args.skip_decode, require_masks=args.with_masks)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(summary(report), ensure_ascii=False, indent=2))
    if report['errors']:
        raise SystemExit(1)


def smoke_test():
    parser = argparse.ArgumentParser(description='Original BACL source/data/config and CUDA kernel preflight')
    dataset_args(parser)
    parser.add_argument('--config-only', action='store_true', help='Do not import native ops or test GPU kernels')
    args = parser.parse_args()
    spec, report = checked_data(args, ('train', 'val'))
    for stage in ('representation', 'classifier'):
        cfg = build_config(spec, stage, '.', checkpoint='not_loaded.pth' if stage == 'classifier' else None,
                           bbox_only=not args.with_masks, test_split='val')
        print('{}: {} / {}, classes={}, weight_decay={}, best={}'.format(
            stage, cfg.model.roi_head.type, cfg.model.roi_head.bbox_head.loss_cls.type,
            cfg.model.roi_head.bbox_head.num_classes, cfg.optimizer.weight_decay,
            cfg.evaluation.save_best))
    result = {'upstream': verify_upstream(), 'dataset': summary(report)}
    if not args.config_only:
        from .runtime import check_runtime
        result.update(check_runtime())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('Preflight passed; this is NOT an end-to-end model training test.')
