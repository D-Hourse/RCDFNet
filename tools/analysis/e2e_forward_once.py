#!/usr/bin/env python3
"""Minimal E2E forward script for Nsight Compute profiling.

Usage example:
  CUDA_VISIBLE_DEVICES=2 /usr/local/cuda/bin/ncu \
    --target-processes all \
    --profile-from-start off \
    --nvtx --nvtx-include E2E_FORWARD \
    --set full \
    --export research_outputs/ncu_e2e \
    /opt/anaconda3/bin/conda run -p /home/chengpeifeng/.conda/envs/RCDFNet --no-capture-output \
    python tools/analysis/e2e_forward_once.py \
    --config work_dirs/RCDFNet_VoD_BMA_cross_attention_deformable/RCDFNet_BMA_vod_deformableattention.py \
    --checkpoint work_dirs/RCDFNet_VoD_BMA_cross_attention_deformable/checkpoints_final/epoch_8_final.pth
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rcdfnet.datasets import build_dataloader, build_dataset  # noqa: E402
from rcdfnet.models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal E2E forward once for NCU")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-warmup", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test])
    else:
        raise TypeError("cfg.data.test must be dict or list")

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)
    model.CLASSES = getattr(dataset, "CLASSES", None)

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    batch = next(iter(data_loader))

    with torch.no_grad():
        for _ in range(max(0, args.num_warmup)):
            _ = model(return_loss=False, rescale=True, **batch)
        torch.cuda.synchronize()

        torch.cuda.nvtx.range_push("E2E_FORWARD")
        t0 = time.time()
        outputs = model(return_loss=False, rescale=True, **batch)
        torch.cuda.synchronize()
        t1 = time.time()
        torch.cuda.nvtx.range_pop()

    out_type = type(outputs).__name__
    out_len = len(outputs) if isinstance(outputs, list) else -1
    print("E2E_FORWARD_DONE")
    print(f"latency_sec={t1 - t0:.6f}")
    print(f"output_type={out_type}")
    print(f"output_len={out_len}")


if __name__ == "__main__":
    main()
