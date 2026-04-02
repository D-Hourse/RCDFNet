# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import logging
import os
import os.path as osp
import sys
import types
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from rcdfnet.apis.test import single_gpu_test
from rcdfnet.datasets import build_dataloader, build_dataset
from rcdfnet.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor


def _normalize_loaded_outputs(outputs):
    """Normalize outputs for dataset.format_results compatibility."""
    if not isinstance(outputs, list) or len(outputs) == 0:
        return outputs

    def _unwrap(v):
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
            return v[0], True
        return v, False

    normalized, changed = [], False
    for item in outputs:
        if isinstance(item, dict):
            new_item = dict(item)
            for k, v in item.items():
                new_item[k], c = _unwrap(v)
                changed = changed or c
            normalized.append(new_item)
        else:
            new_item, c = _unwrap(item)
            normalized.append(new_item)
            changed = changed or c

    if changed:
        print('Normalized loaded results structure for evaluation compatibility.')
    return normalized


def _try_import_vod_evaluation(standalone_root):
    def _install_k3d_stub_if_needed(exc):
        if isinstance(exc, ModuleNotFoundError) and getattr(exc, 'name', '') == 'k3d':
            if 'k3d' not in sys.modules:
                sys.modules['k3d'] = types.ModuleType('k3d')
            return True
        return False

    # Prefer local environment package; fallback to known source-tree locations.
    try:
        from vod.evaluation import Evaluation  # type: ignore
        return Evaluation
    except Exception as e:
        if _install_k3d_stub_if_needed(e):
            from vod.evaluation import Evaluation  # type: ignore
            return Evaluation

        vod_repos = [
            osp.abspath(osp.join(standalone_root, 'view_of_delft_dataset_main')),
            osp.abspath(osp.join(standalone_root, '..', 'view_of_delft_dataset_main')),
            osp.abspath(osp.join(os.getcwd(), 'view_of_delft_dataset_main')),
        ]
        for vod_repo in vod_repos:
            if osp.isdir(vod_repo) and vod_repo not in sys.path:
                sys.path.insert(0, vod_repo)
        try:
            from vod.evaluation import Evaluation  # type: ignore
            return Evaluation
        except Exception as e2:
            if _install_k3d_stub_if_needed(e2):
                from vod.evaluation import Evaluation  # type: ignore
                return Evaluation
            raise ModuleNotFoundError(
                "No module named 'vod'. Checked installed packages and local paths: "
                + ', '.join(vod_repos)
            ) from e2


def _infer_data_root(cfg):
    if cfg.get('data_root', None):
        return cfg.data_root
    if isinstance(cfg.data.test, dict) and cfg.data.test.get('data_root', None):
        return cfg.data.test.data_root
    if isinstance(cfg.data.test, list) and len(cfg.data.test) > 0 and cfg.data.test[0].get('data_root', None):
        return cfg.data.test[0].data_root
    return None


def _build_dataset_and_loader(cfg, distributed):
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    return dataset, data_loader


def _list_checkpoints(checkpoint_dir, pattern):
    ckpts = sorted(glob.glob(osp.join(checkpoint_dir, pattern)))
    return [p for p in ckpts if osp.isfile(p)]


def _ckpt_tag(ckpt_path):
    base = osp.basename(ckpt_path)
    if base.endswith('.pth'):
        base = base[:-4]
    return base.replace(' ', '_')


def _run_inference_from_checkpoint(cfg, dataset, data_loader, distributed, args, checkpoint_path):
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    return outputs


def _build_eval_kwargs(cfg, args, extra_kwargs):
    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval, **extra_kwargs))
    return eval_kwargs


def _vod_eval_and_dump(cfg, args, submission_prefix, source_desc):
    standalone_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    Evaluation = _try_import_vod_evaluation(standalone_root)

    data_root = _infer_data_root(cfg)
    label_dir = args.label_dir or (osp.join(data_root, 'testing', 'label_2') if data_root else None)
    if label_dir is None:
        raise RuntimeError('Cannot infer label directory, please pass --label-dir')

    evaluation = Evaluation(test_annotation_file=label_dir)
    results = evaluation.evaluate(result_path=submission_prefix + 'pts_bbox', current_class=[0, 1, 2])

    entire_map = (
        results['entire_area']['Car_3d_all'] +
        results['entire_area']['Pedestrian_3d_all'] +
        results['entire_area']['Cyclist_3d_all']) / 3
    roi_map = (
        results['roi']['Car_3d_all'] +
        results['roi']['Pedestrian_3d_all'] +
        results['roi']['Cyclist_3d_all']) / 3

    msg = (
        'Results:\n'
        'Entire annotated area:\n'
        f"Car: {results['entire_area']['Car_3d_all']}\n"
        f"Pedestrian: {results['entire_area']['Pedestrian_3d_all']}\n"
        f"Cyclist: {results['entire_area']['Cyclist_3d_all']}\n"
        f"mAP: {entire_map}\n"
        'Driving corridor area:\n'
        f"Car: {results['roi']['Car_3d_all']}\n"
        f"Pedestrian: {results['roi']['Pedestrian_3d_all']}\n"
        f"Cyclist: {results['roi']['Cyclist_3d_all']}\n"
        f"mAP: {roi_map}\n"
    )
    print(msg)

    out_txt = submission_prefix + '_allresult.txt'
    with open(out_txt, 'w') as f:
        f.write(source_desc + '\n')
        f.write(msg)
    print(f'VoD official eval summary saved to {out_txt}')


def _post_process_outputs(outputs, args, cfg, dataset, source_desc, suffix=None):
    outputs = _normalize_loaded_outputs(outputs)

    out_path = args.out
    if out_path and suffix:
        if out_path.endswith(('.pkl', '.pickle')):
            stem, ext = osp.splitext(out_path)
            out_path = f'{stem}_{suffix}{ext}'
        else:
            out_path = osp.join(out_path, f'{suffix}.pkl')

    if out_path:
        mmcv.mkdir_or_exist(osp.dirname(out_path) if osp.dirname(out_path) else '.')
        print(f'writing results to {out_path}')
        mmcv.dump(outputs, out_path)

    kwargs = {} if args.eval_options is None else dict(args.eval_options)
    if suffix:
        for k in ['pklfile_prefix', 'submission_prefix']:
            if k in kwargs:
                kwargs[k] = f"{kwargs[k]}_{suffix}"

    for k in ['pklfile_prefix', 'submission_prefix']:
        if k in kwargs and kwargs[k]:
            prefix_dir = osp.dirname(kwargs[k])
            if prefix_dir:
                mmcv.mkdir_or_exist(prefix_dir)

    if args.format_only:
        dataset.format_results(outputs, **kwargs)

    if args.eval:
        print(dataset.evaluate(outputs, **_build_eval_kwargs(cfg, args, kwargs)))

    # Official VoD evaluation based on formatted txt result.
    # Requires submission_prefix in --eval-options and formatted result files.
    if 'submission_prefix' in kwargs:
        _vod_eval_and_dump(cfg, args, kwargs['submission_prefix'], source_desc)


def parse_args():
    parser = argparse.ArgumentParser(description='RCDFNet-Standalone test/eval/VoD-eval')
    parser.add_argument('--config', required=True, help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file path, or checkpoint directory for batch mode')
    parser.add_argument('--checkpoint-dir', help='directory that contains multiple checkpoints to run sequentially')
    parser.add_argument('--checkpoint-pattern', default='*.pth', help='glob pattern for checkpoints inside --checkpoint-dir')
    parser.add_argument('--result-pkl', help='existing inference pkl to evaluate directly')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='Fuse conv and bn for speed')
    parser.add_argument('--format-only', action='store_true', help='Only format output results')
    parser.add_argument('--eval', type=str, nargs='+', help='dataset evaluate metrics')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory to save visualized results')
    parser.add_argument('--gpu-collect', action='store_true', help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp directory for multi-gpu result collection')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='set deterministic CUDNN backend')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config settings')
    parser.add_argument('--options', nargs='+', action=DictAction, help='deprecated, use --eval-options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='kwargs for dataset.evaluate/format_results')
    parser.add_argument('--label-dir', help='VoD GT label directory (default: <data_root>/testing/label_2)')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError('--options and --eval-options cannot be both specified')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options

    input_modes = sum([
        args.result_pkl is not None,
        args.checkpoint is not None,
        args.checkpoint_dir is not None
    ])
    if input_modes != 1:
        raise ValueError('Exactly one of --result-pkl / --checkpoint / --checkpoint-dir must be provided')

    # Single mode expects a pkl file; batch mode allows output directory prefix.
    batch_mode = args.checkpoint_dir is not None or (
        args.checkpoint is not None and osp.isdir(args.checkpoint)
    )
    if batch_mode:
        return args

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('In single-checkpoint mode, --out must be a .pkl/.pickle file.')

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset, data_loader = _build_dataset_and_loader(cfg, distributed)
    rank, _ = get_dist_info()

    # Backward-compatible behavior: allow passing a checkpoint directory via --checkpoint.
    if args.checkpoint and osp.isdir(args.checkpoint) and args.checkpoint_dir is None:
        args.checkpoint_dir = args.checkpoint
        print(f'--checkpoint points to a directory, switching to batch mode: {args.checkpoint_dir}')

    # Batch mode: run all checkpoints under a directory.
    if args.checkpoint_dir:
        ckpts = _list_checkpoints(args.checkpoint_dir, args.checkpoint_pattern)
        if len(ckpts) == 0:
            raise FileNotFoundError(
                f'No checkpoint matched in {args.checkpoint_dir} with pattern {args.checkpoint_pattern}')

        print(f'Found {len(ckpts)} checkpoints for batch inference.')
        for idx, ckpt_path in enumerate(ckpts, start=1):
            tag = _ckpt_tag(ckpt_path)
            print(f'\n[{idx}/{len(ckpts)}] Running checkpoint: {ckpt_path}')
            outputs = _run_inference_from_checkpoint(
                cfg, dataset, data_loader, distributed, args, ckpt_path)

            if rank == 0:
                _post_process_outputs(
                    outputs=outputs,
                    args=args,
                    cfg=cfg,
                    dataset=dataset,
                    source_desc=ckpt_path,
                    suffix=tag)
        return

    if args.result_pkl:
        outputs = mmcv.load(args.result_pkl)
        if not isinstance(outputs, list):
            raise TypeError(f'Loaded result pkl must be a list, got {type(outputs)}')
        print(f'Loaded existing results: {args.result_pkl} (len={len(outputs)})')
    else:
        outputs = _run_inference_from_checkpoint(
            cfg, dataset, data_loader, distributed, args, args.checkpoint)

    if rank != 0:
        return

    source_desc = args.result_pkl if args.result_pkl else args.checkpoint
    _post_process_outputs(
        outputs=outputs,
        args=args,
        cfg=cfg,
        dataset=dataset,
        source_desc=source_desc,
        suffix=None)


if __name__ == '__main__':
    main()
