"""Fail clearly on incompatible runtimes/checkpoints; never fall back to a port."""
import json
import os
from pathlib import Path
import socket
import sys

from .provenance import UPSTREAM_ROOT, activate_upstream


def check_runtime():
    provenance = activate_upstream()
    try:
        import torch
        import torchvision
        import mmcv
        import lvis
        import pycocotools
        import numpy as np
        if tuple(int(v) for v in np.__version__.split('.')[:2]) >= (1, 24):
            raise RuntimeError('mmlvis 10.5.3 needs numpy<1.24 (reference pin: 1.23.5)')
        if mmcv.__version__ != '1.2.7':
            raise RuntimeError('Strict reference environment requires mmcv-full==1.2.7')
        if lvis.__version__ != '10.5.3':
            raise RuntimeError('Install mmlvis==10.5.3, not the incompatible lvis package')
        if getattr(pycocotools, '__version__', '') not in ('12.0.2', '12.0.3'):
            raise RuntimeError('Install mmpycocotools==12.0.3, not stock pycocotools')
        from mmcv.ops import nms, RoIAlign  # compiled extensions are mandatory
        import mmdet
        if Path(mmdet.__file__).resolve().parent != (UPSTREAM_ROOT / 'mmdet').resolve():
            raise RuntimeError('Another mmdet installation shadowed the locked BACL source')
        from mmdet.models import build_detector  # verify all registered BACL modules import
        if not torch.cuda.is_available():
            raise RuntimeError('Original BACL requires CUDA; FHM uses CUDA tensors and NCCL')
        if not torch.distributed.is_nccl_available():
            raise RuntimeError('Original classifier training requires NCCL (use Linux)')
        # Exercise the actual GPU kernels, not merely Python imports.
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
        torch.cuda.set_device(device)
        boxes = torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.]], device=device)
        scores = torch.tensor([.9, .8], device=device)
        _, kept = nms(boxes, scores, .5)
        assert kept.numel() == 1
        features = torch.ones((1, 1, 8, 8), device=device, requires_grad=True)
        rois = torch.tensor([[0., 0., 0., 4., 4.]], device=device)
        RoIAlign(2, spatial_scale=1., sampling_ratio=0)(features, rois).sum().backward()
        torch.cuda.synchronize()
        versions = {'python': sys.version.split()[0], 'torch': torch.__version__,
                    'torchvision': torchvision.__version__, 'mmcv': mmcv.__version__,
                    'mmlvis': lvis.__version__, 'mmpycocotools': pycocotools.__version__,
                    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device)}
        return {'upstream': provenance, 'runtime': versions, 'cuda_ops': 'passed'}
    except Exception as exc:
        raise RuntimeError('Original BACL runtime check failed: {}\n'
                           'Use the isolated Linux environment in docs/UPSTREAM_ALIGNMENT.md. '
                           'No TorchVision fallback was performed.'.format(exc)) from exc


def prepare_distributed():
    """Even one GPU must initialize DDP: original FCBL calls all_reduce."""
    if 'WORLD_SIZE' in os.environ:
        required = ('RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT')
        missing = [key for key in required if key not in os.environ]
        if missing:
            raise ValueError('Incomplete distributed environment: {}'.format(', '.join(missing)))
        return
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    os.environ.update(RANK='0', WORLD_SIZE='1', LOCAL_RANK='0',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))


def check_checkpoint(path, spec, stage=None, resume=False):
    """Only load trusted user checkpoints; reject incompatible port/category order."""
    import torch
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    # PyTorch 1.7 has no weights_only argument. User-supplied checkpoints must be trusted.
    payload = torch.load(str(path), map_location='cpu')
    if not isinstance(payload, dict) or 'state_dict' not in payload:
        raise ValueError('Not an original MMDetection/BACL checkpoint (state_dict missing). '
                         'TorchVision last.pth/best.pth cannot be reused; retrain representation.')
    names = payload.get('meta', {}).get('CLASSES')
    if names is None or tuple(names) != spec.classes:
        raise ValueError('Checkpoint class names/order do not match the LVIS training JSON')
    state = payload['state_dict']
    weight = state.get('roi_head.bbox_head.fc_cls.weight')
    if weight is None:
        weight = state.get('module.roi_head.bbox_head.fc_cls.weight')
    if weight is None or weight.shape[0] != len(spec.classes) + 1:
        raise ValueError('Checkpoint classifier shape differs from original BACL K+1 outputs')
    sidecar = path.parent / 'provenance.json'
    checkpoint_stage = None
    if sidecar.is_file():
        saved = json.loads(sidecar.read_text(encoding='utf-8'))
        if tuple(saved.get('dataset', {}).get('category_ids', ())) != spec.category_ids:
            raise ValueError('Checkpoint category IDs differ from the LVIS training JSON')
        checkpoint_stage = saved.get('stage')
    # Upstream checkpoints carry the resolved config in their metadata as well.
    config_text = payload.get('meta', {}).get('config')
    if config_text:
        from mmcv import Config
        saved_cfg = Config.fromstring(config_text, '.py')
        saved_ds = saved_cfg.data.train
        if saved_ds.type == 'MultiImageMixDataset':
            saved_ds = saved_ds.dataset
        saved_ids = saved_ds.get('expected_category_ids')
        if saved_ids is not None and tuple(saved_ids) != spec.category_ids:
            raise ValueError('Checkpoint config category IDs differ from the LVIS training JSON')
        checkpoint_stage = ('classifier' if saved_cfg.model.roi_head.type == 'FhmRoIHead'
                            else 'representation')
    if stage and checkpoint_stage:
        expected = stage if resume else 'representation'
        if checkpoint_stage != expected:
            raise ValueError('Expected {} checkpoint, got {}'.format(expected, checkpoint_stage))
    return checkpoint_stage
