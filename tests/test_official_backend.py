"""CPU audit: exact upstream bytes/configs, custom LVIS metadata, real LVIS AP.

These tests do not claim to exercise MMDetection's compiled model/CUDA runner.
Use requirements-audit.txt in an isolated environment.
"""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from mmcv import Config
from lvis import LVIS, LVISResults, LVISEval

from bacl_official.config import CONFIGS, build_config
from bacl_official.data import image_filename, inspect_dataset, load_dataset_spec
from bacl_official.detection import (detection_records, find_images, main as detect_main,
                                     validate_output)
from bacl_official.provenance import PROJECT_ROOT, UPSTREAM_ROOT, verify_upstream
from bacl_official.runtime import check_checkpoint, prepare_distributed


def fixture(root):
    (root / 'annotations').mkdir()
    (root / 'images').mkdir()
    categories = [dict(id=42, name='clam', frequency='f'), dict(id=7, name='snail', frequency='c')]
    for index, split in enumerate(('train', 'val', 'test'), 1):
        name = 'images/{}.jpg'.format(split)
        Image.new('RGB', (64, 64), 'white').save(root / name)
        data = dict(categories=categories, images=[dict(
            id=index, file_name=name, coco_url='', width=64, height=64,
            neg_category_ids=[42], not_exhaustive_category_ids=[])],
            annotations=[dict(id=index, image_id=index, category_id=7,
                              bbox=[5, 5, 20, 20], area=400, iscrowd=0, segmentation=[])])
        (root / 'annotations/lvis_v1_{}.json'.format(split)).write_text(json.dumps(data), encoding='utf-8')


class OfficialAudit(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='bacl-official-audit-')
        self.root = Path(self.directory.name)
        fixture(self.root)
        self.spec = load_dataset_spec(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def config(self, stage, **kwargs):
        return build_config(self.spec, stage, self.root / 'run',
                            checkpoint='representation.pth' if stage == 'classifier' else None, **kwargs)

    def test_upstream_284_blobs_match_locked_git_commit(self):
        result = verify_upstream()
        self.assertEqual(result['commit'], '65536623361587286d99c70133b886792a84463d')
        self.assertEqual(result['verified_files'], 284)

    def test_source_files_parse_on_python38(self):
        for folder in ('bacl_official',):
            for path in (PROJECT_ROOT / folder).glob('*.py'):
                ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))

    def test_noncontiguous_category_ids_and_bbox_json_unchanged(self):
        before = {p: p.read_bytes() for p in self.spec.annotations.values()}
        report = inspect_dataset(self.spec, decode_images=True)
        self.assertFalse(report['errors'], report)
        self.assertEqual(self.spec.category_ids, (7, 42))
        self.assertEqual(self.spec.classes, ('snail', 'clam'))
        self.assertEqual(report['splits']['train']['annotations_with_masks'], 0)
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))

    def test_masks_are_required_only_when_explicitly_enabled(self):
        report = inspect_dataset(self.spec, require_masks=True)
        self.assertEqual(len(report['errors']), 3)
        self.assertIn('real instance masks', report['errors'][0])

    def test_federated_metadata_conflicts_are_rejected(self):
        self.spec.splits['val']['images'][0]['neg_category_ids'] = [7]
        self.assertTrue(inspect_dataset(self.spec)['errors'])
        self.spec.splits['val']['images'][0]['neg_category_ids'] = [{}]
        self.assertTrue(inspect_dataset(self.spec)['errors'])

    def test_category_frequency_mismatch_is_rejected(self):
        path = self.spec.annotations['val']
        raw = json.loads(path.read_text())
        raw['categories'][0]['frequency'] = 'r'
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, 'frequency bins differ'):
            load_dataset_spec(self.root)

    def test_invalid_boxes_and_missing_files_are_reported(self):
        self.spec.splits['train']['annotations'][0]['bbox'][2] = -1
        self.spec.splits['val']['images'][0]['file_name'] = 'missing.jpg'
        errors = inspect_dataset(self.spec)['errors']
        self.assertEqual(len(errors), 2, errors)

    def test_image_path_resolution(self):
        self.assertEqual(image_filename({'file_name': r'images\snail.jpg'}), 'images/snail.jpg')
        self.assertEqual(image_filename({'coco_url': 'http://images.cocodataset.org/train2017/1.jpg'}),
                         'train2017/1.jpg')
        with self.assertRaises(ValueError):
            image_filename({'coco_url': ''})

    def test_original_model_and_optimizer_have_only_required_overrides(self):
        for stage in CONFIGS:
            original = Config.fromfile(str(UPSTREAM_ROOT / 'configs/bacl' / CONFIGS[stage]))
            adapted = self.config(stage)
            expected = copy.deepcopy(original.model)
            expected.roi_head.bbox_head.num_classes = 2
            expected.roi_head.bbox_head.loss_cls.num_classes = 2
            self.assertEqual(adapted.model, expected)
            self.assertEqual(adapted.optimizer.weight_decay, 5e-5 if stage == 'representation' else 1e-4)
            self.assertEqual(adapted.optimizer.lr, .0025)
            self.assertEqual(adapted.optimizer_config, original.optimizer_config)
            self.assertIsNone(adapted.optimizer_config.grad_clip)
            self.assertEqual(adapted.lr_config, original.lr_config)
            self.assertEqual(adapted.lr_config.step, [8, 11])
            self.assertEqual(adapted.total_epochs, 12)
            self.assertEqual(adapted.evaluation.save_best, 'bbox_AP')
            self.assertEqual(adapted.model.rpn_head.loss_bbox.type, 'L1Loss')
            self.assertEqual(adapted.model.roi_head.bbox_head.loss_bbox.type, 'L1Loss')
            self.assertEqual(adapted.model.pretrained, 'torchvision://resnet50')
            self.assertEqual(adapted.model.test_cfg.rcnn.score_thr, .0001)
            self.assertEqual(adapted.model.test_cfg.rcnn.max_per_img, 300)

    def test_bbox_adaptation_only_removes_mask_dependent_operations(self):
        original = Config.fromfile(str(UPSTREAM_ROOT / 'configs/bacl' / CONFIGS['representation']))
        cfg = self.config('representation')
        expected = copy.deepcopy(original.data.train.dataset.pipeline)
        expected[1]['with_mask'] = False
        expected[3]['recompute_bbox'] = False
        self.assertEqual(cfg.data.train.dataset.pipeline, expected)
        self.assertEqual(cfg.data.train.pipeline, original.data.train.pipeline[1:])
        masked = self.config('representation', bbox_only=False)
        self.assertEqual(masked.data.train.dataset.pipeline, original.data.train.dataset.pipeline)
        self.assertEqual(masked.data.train.pipeline, original.data.train.pipeline)
        self.assertEqual(self.config('classifier').data.train.type, 'CustomLVISV1Dataset')

    def test_lr_scaling_and_original_fhm_freeze_settings(self):
        cfg = self.config('classifier', world_size=8)
        self.assertEqual(cfg.optimizer.lr, .02)
        self.assertEqual(cfg.selectp, 1)
        self.assertEqual(cfg.custom_hooks, [dict(type='ReweightHook', step=1)])
        self.assertEqual(cfg.model.roi_head.bbox_head.fhm_cfg,
                         dict(decay_ratio=.1, sampled_num_classes=8, sampled_num_features=12))

    def test_invalid_cli_overrides_and_missing_stage1_checkpoint(self):
        with self.assertRaises(ValueError):
            build_config(self.spec, 'classifier', '.')
        with self.assertRaises(ValueError):
            self.config('representation', batch_size=0)
        with self.assertRaises(ValueError):
            self.config('representation', lr=float('nan'))

    def test_resolved_config_roundtrips_without_native_imports(self):
        path = self.root / 'resolved.py'
        cfg = self.config('representation')
        cfg.dump(str(path))
        loaded = Config.fromfile(str(path), import_custom_modules=False)
        self.assertEqual(loaded.data.train.dataset.ann_file,
                         self.spec.annotations['train'].as_posix())
        self.assertEqual(tuple(loaded.data.train.dataset.classes), self.spec.classes)

    def evaluator(self, box):
        gt = LVIS(str(self.spec.annotations['val']))
        predictions = [dict(image_id=2, category_id=7, bbox=box, score=.9)]
        evaluator = LVISEval(gt, LVISResults(gt, predictions), 'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return evaluator

    def test_real_lvis_evaluator_uses_ten_iou_thresholds(self):
        evaluator = self.evaluator([5, 5, 20, 20])
        np.testing.assert_allclose(evaluator.params.iou_thrs, np.arange(.5, 1., .05))
        metrics = evaluator.get_results()
        self.assertAlmostEqual(metrics['AP'], 1.)
        self.assertAlmostEqual(metrics['AP50'], 1.)
        self.assertEqual(metrics['APr'], -1.)  # No rare categories: never invent a value.
        self.assertEqual(evaluator.params.max_dets, 300)

    def test_ap_is_not_an_ap50_alias(self):
        metrics = self.evaluator([5, 5, 12, 20]).get_results()  # IoU=.6
        self.assertAlmostEqual(metrics['AP50'], 1.)
        self.assertLess(metrics['AP'], .4)
        self.assertEqual(metrics['AP75'], 0.)

    def test_unchanged_upstream_evaluate_method_with_real_lvis_api(self):
        # Execute the original method AST with only storage/JSON conversion
        # plumbing supplied below. This tests original scoring, not native model loading.
        import itertools
        import logging
        from collections import OrderedDict
        tree = ast.parse((UPSTREAM_ROOT / 'mmdet/datasets/lvis.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LVISV05Dataset')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        import os.path as osp
        namespace = dict(np=np, tempfile=tempfile, logging=logging, itertools=itertools, osp=osp,
                         OrderedDict=OrderedDict, print_log=lambda *a, **k: None)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<unchanged upstream evaluate>', 'exec'), namespace)
        outer = self

        class StorageOnly:
            coco = LVIS(str(outer.spec.annotations['val']))
            img_ids = [2]
            cat_ids = [7, 42]

            def __len__(self):
                return 1

            def results2json(self, results, prefix):
                path = str(prefix) + '.bbox.json'
                Path(path).write_text(json.dumps(results))
                return {'bbox': path}

        result = namespace['evaluate'](StorageOnly(), [dict(image_id=2, category_id=7,
                                        bbox=[5, 5, 12, 20], score=.9)], metric='bbox')
        self.assertEqual(result['bbox_AP50'], 1.)
        self.assertLess(result['bbox_AP'], .4)
        self.assertEqual(result['bbox_APr'], -1.)

    def test_default_cli_dry_runs_original_backend(self):
        output = self.root / 'dry-run'
        result = subprocess.run([sys.executable, '-m', 'tools.train', '--stage', 'representation',
                                 '--data', str(self.root), '--output', str(output), '--dry-run'],
                                cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        provenance = json.loads((output / 'dry_run_provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['backend'], 'official')
        self.assertEqual(provenance['best_checkpoint_metric'], 'bbox_AP')
        self.assertFalse(list(output.glob('*.pth')))

    def test_classifier_and_evaluation_dry_runs(self):
        output = self.root / 'stage2-dry'
        result = subprocess.run([sys.executable, '-m', 'tools.train', '--stage', 'classifier',
                                 '--data', str(self.root), '--output', str(output),
                                 '--checkpoint', 'not_loaded.pth', '--dry-run'],
                                cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = Config.fromfile(str(output / 'resolved_config_rank0.py'), import_custom_modules=False)
        self.assertEqual(cfg.model.roi_head.type, 'FhmRoIHead')
        metrics = self.root / 'eval.json'
        result = subprocess.run([sys.executable, '-m', 'tools.evaluate', '--data', str(self.root),
                                 '--checkpoint', 'not_loaded.pth', '--output', str(metrics), '--dry-run'],
                                cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(metrics.exists())
        cfg = Config.fromfile(str(metrics.with_suffix('.config.py')), import_custom_modules=False)
        self.assertEqual(cfg.data.test.ann_file, self.spec.annotations['test'].as_posix())
        self.assertEqual(cfg.data.test.metrics_output, metrics.as_posix())

    def test_checkpoint_guards_without_unpickling_untrusted_files(self):
        checkpoint = self.root / 'model.pth'
        checkpoint.touch()
        payload = {'model': {}}
        fake_torch = SimpleNamespace(load=lambda *args, **kwargs: payload)
        with patch.dict(sys.modules, {'torch': fake_torch}):
            with self.assertRaisesRegex(ValueError, 'state_dict missing'):
                check_checkpoint(checkpoint, self.spec)
            payload.clear()
            payload.update(state_dict={'roi_head.bbox_head.fc_cls.weight': np.zeros((3, 4))},
                           meta={'CLASSES': self.spec.classes})
            self.assertIsNone(check_checkpoint(checkpoint, self.spec))
            payload['meta']['CLASSES'] = ('clam', 'snail')
            with self.assertRaisesRegex(ValueError, 'names/order'):
                check_checkpoint(checkpoint, self.spec)
            payload['meta']['CLASSES'] = self.spec.classes
            (self.root / 'provenance.json').write_text(json.dumps({
                'stage': 'classifier', 'dataset': {'category_ids': list(self.spec.category_ids)}}))
            with self.assertRaisesRegex(ValueError, 'Expected representation'):
                check_checkpoint(checkpoint, self.spec, stage='classifier')
            self.assertEqual(check_checkpoint(checkpoint, self.spec, stage='classifier', resume=True),
                             'classifier')

    def checkpoint_with_config(self, stage):
        checkpoint = self.root / 'model.pth'
        checkpoint.touch()
        cfg = self.config(stage)
        # Metadata inspection must not import dataset/model registration modules.
        cfg.custom_imports = dict(imports=['bacl_nonexistent_audit_module'], allow_failed_imports=False)
        cfg.checkpoint_note = '软体数据集 / {{ fileDirname }}'
        payload = dict(state_dict={'roi_head.bbox_head.fc_cls.weight': np.zeros((3, 4))},
                       meta={'CLASSES': self.spec.classes, 'config': cfg.pretty_text})
        return checkpoint, cfg, payload

    def test_checkpoint_config_metadata_works_with_pinned_mmcv(self):
        original_fromfile = Config.fromfile
        for stage in ('representation', 'classifier'):
            with self.subTest(stage=stage):
                checkpoint, cfg, payload = self.checkpoint_with_config(stage)
                before = checkpoint.read_bytes()
                fake_torch = SimpleNamespace(load=lambda *args, **kwargs: payload)

                def read_config(filename, **kwargs):
                    parsed = original_fromfile(filename, **kwargs)
                    self.assertEqual(parsed.checkpoint_note, cfg.checkpoint_note)
                    return parsed

                with patch.dict(sys.modules, {'torch': fake_torch}), \
                        patch.object(Config, 'fromfile', side_effect=read_config) as parse:
                    # Stage 2 starts with stage 1, or resumes with its own checkpoint.
                    self.assertEqual(check_checkpoint(checkpoint, self.spec, stage='classifier',
                                     resume=(stage == 'classifier')), stage)
                parse.assert_called_once()
                self.assertFalse(parse.call_args.kwargs['import_custom_modules'])
                self.assertFalse(parse.call_args.kwargs['use_predefined_variables'])
                self.assertFalse(Path(parse.call_args.args[0]).exists())
                self.assertEqual(checkpoint.read_bytes(), before)

    def test_checkpoint_config_rejects_category_order_mismatch(self):
        for stage in ('representation', 'classifier'):
            with self.subTest(stage=stage):
                checkpoint, cfg, payload = self.checkpoint_with_config(stage)
                dataset = cfg.data.train.dataset if stage == 'representation' else cfg.data.train
                dataset.expected_category_ids = list(reversed(self.spec.category_ids))
                payload['meta']['config'] = cfg.pretty_text
                fake_torch = SimpleNamespace(load=lambda *args, **kwargs: payload)
                with patch.dict(sys.modules, {'torch': fake_torch}):
                    with self.assertRaisesRegex(ValueError, 'config category IDs'):
                        check_checkpoint(checkpoint, self.spec)

    def test_checkpoint_config_rejects_wrong_training_stage(self):
        for stage, resume in (('representation', True), ('classifier', False)):
            with self.subTest(stage=stage, resume=resume):
                checkpoint, _, payload = self.checkpoint_with_config(stage)
                fake_torch = SimpleNamespace(load=lambda *args, **kwargs: payload)
                expected = 'classifier' if resume else 'representation'
                with patch.dict(sys.modules, {'torch': fake_torch}):
                    with self.assertRaisesRegex(ValueError, 'Expected {} checkpoint'.format(expected)):
                        check_checkpoint(checkpoint, self.spec, stage='classifier', resume=resume)

    def test_arbitrary_image_discovery_and_safe_output(self):
        source = self.root / 'arbitrary-input'
        (source / 'nested').mkdir(parents=True)
        Image.new('RGB', (8, 8)).save(source / 'B.PNG')
        Image.new('RGB', (8, 8)).save(source / 'nested/a.jpg')
        (source / 'ignore.txt').write_text('not an image')
        images, root = find_images(source)
        self.assertEqual(root, source.resolve())
        self.assertEqual([path.name for path in images], ['B.PNG', 'a.jpg'])
        self.assertEqual(find_images(source / 'B.PNG')[0], [(source / 'B.PNG').resolve()])
        self.assertEqual(validate_output(source, self.root / 'results'),
                         (self.root / 'results').resolve())
        with self.assertRaisesRegex(ValueError, 'must not overlap'):
            validate_output(source, source / 'results')
        with self.assertRaisesRegex(ValueError, 'must not overlap'):
            validate_output(source / 'nested', source)
        with self.assertRaisesRegex(ValueError, 'Unsupported image'):
            find_images(source / 'ignore.txt')

    def test_detection_records_keep_lvis_ids_and_threshold(self):
        result = [np.array([[1, 2, 11, 22, .9], [3, 4, 8, 9, .3]]),
                  np.array([[5, 6, 7, 10, .7]])]
        records = detection_records(result, self.spec.classes, self.spec.category_ids, .3)
        self.assertEqual([item['category_id'] for item in records], [7, 42])
        self.assertEqual([item['class_name'] for item in records], ['snail', 'clam'])
        self.assertEqual(records[0]['bbox_xyxy'], [1., 2., 11., 22.])
        self.assertEqual(records[0]['bbox_xywh'], [1., 2., 10., 20.])
        self.assertAlmostEqual(records[1]['score'], .7)

    def test_detection_cli_help_needs_no_native_cuda_import(self):
        result = subprocess.run([sys.executable, '-m', 'tools.detect', '--help'],
                                cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('/root/autodl-tmp/test', result.stdout)

    def test_detection_writes_annotated_images_and_json(self):
        source = self.root / 'incoming'
        output = self.root / 'detections'
        source.mkdir()
        Image.new('RGB', (8, 8)).save(source / 'one.jpg')
        checkpoint = self.root / 'classifier.pth'
        checkpoint.touch()
        result = [np.array([[1, 2, 7, 6, .8]]), np.empty((0, 5))]

        class FakeModel:
            CLASSES = self.spec.classes

            def show_result(self, image, detections, **kwargs):
                self.assert_show_args(image, detections, kwargs)

            def assert_show_args(self, image, detections, kwargs):
                self_outer.assertEqual(Path(image), source / 'one.jpg')
                self_outer.assertIs(detections, result)
                self_outer.assertEqual(kwargs['score_thr'], .3)
                self_outer.assertEqual(kwargs['thickness'], 2)
                self_outer.assertEqual(kwargs['font_size'], 10)
                Image.new('RGB', (8, 8)).save(kwargs['out_file'])

        self_outer = self
        fake_apis = ModuleType('mmdet.apis')
        fake_apis.init_detector = lambda *args, **kwargs: FakeModel()
        fake_apis.inference_detector = lambda *args, **kwargs: result
        fake_mmdet = ModuleType('mmdet')
        fake_mmdet.apis = fake_apis
        argv = ['detect.py', '--input', str(source), '--output', str(output),
                '--data', str(self.root), '--checkpoint', str(checkpoint)]
        with patch.object(sys, 'argv', argv), \
                patch('bacl_official.detection.check_runtime',
                      return_value={'runtime': {'gpu': 'audit'}}), \
                patch('bacl_official.detection.check_checkpoint'), \
                patch.dict(sys.modules, {'mmdet': fake_mmdet, 'mmdet.apis': fake_apis}):
            detect_main()
        report = json.loads((output / 'detections.json').read_text(encoding='utf-8'))
        self.assertTrue((output / 'one.jpg').is_file())
        self.assertEqual(report['image_count'], 1)
        self.assertEqual(report['detection_count'], 1)
        self.assertEqual(report['line_width'], 2)
        self.assertEqual(report['font_size'], 10)
        self.assertEqual(report['images'][0]['detections'][0]['category_id'], 7)

    def test_single_gpu_also_gets_distributed_environment(self):
        import os
        with patch.dict(os.environ, {}, clear=True):
            prepare_distributed()
            self.assertEqual(os.environ['WORLD_SIZE'], '1')
            self.assertEqual(os.environ['RANK'], '0')
            self.assertEqual(os.environ['LOCAL_RANK'], '0')
            self.assertGreater(int(os.environ['MASTER_PORT']), 0)
        with patch.dict(os.environ, {'WORLD_SIZE': '2'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Incomplete distributed'):
                prepare_distributed()


if __name__ == '__main__':
    unittest.main()
