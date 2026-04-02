#!/usr/bin/env python3
"""Strict(er) end-to-end FLOPs profiling with custom-op correction.

This script measures e2e forward FLOPs using torch.profiler (operator-level),
then adds theoretical FLOPs for custom CUDA ops that are typically not counted:
- MultiScaleDeformableAttnFunction_fp32/fp16
- bev_pool_v2

Output includes a range:
- total_flops_lower: profiler + custom_lower
- total_flops_upper: profiler + custom_upper (includes bilinear interp overhead)

Example (physical GPU2):
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python tools/analysis/e2e_theoretical_flops_strict.py \
    --config work_dirs/RCDFNet_VoD_BMA_cross_attention_deformable/RCDFNet_BMA_vod_deformableattention.py \
    --checkpoint work_dirs/RCDFNet_VoD_BMA_cross_attention_deformable/checkpoints_final/epoch_8_final.pth \
    --num-batches 3 \
    --out research_outputs/e2e_strict_gpu2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rcdfnet.datasets import build_dataloader, build_dataset  # noqa: E402
from rcdfnet.models import build_model  # noqa: E402


@dataclass
class BatchStat:
    batch_index: int
    latency_sec: float
    profiler_flops: float
    custom_lower_flops: float
    custom_upper_flops: float
    total_lower_flops: float
    total_upper_flops: float


class CustomOpFlopsTracer:
    """Patch custom ops and accumulate theoretical FLOPs during forward."""

    def __init__(self) -> None:
        self.custom_lower_flops = 0.0
        self.custom_upper_flops = 0.0
        self._patched = False
        self._orig = {}

    def reset(self) -> None:
        self.custom_lower_flops = 0.0
        self.custom_upper_flops = 0.0

    def _count_ms_deform(self, value: torch.Tensor, attention_weights: torch.Tensor) -> None:
        # value: [B, num_keys, heads, c_per_head]
        # attention_weights: [B, num_query, heads, levels, points]
        b, nq, h, l, p = attention_weights.shape
        c = value.shape[-1]
        n = float(b * nq * h * l * p * c)

        # lower: weighted accumulation only (mul+add)
        lower = 2.0 * n
        # upper: weighted accumulation + bilinear interpolation overhead (approx)
        # assume extra ~8 FLOPs per sampled channel (4 mul + 4 add, approximate)
        upper = lower + 8.0 * n

        self.custom_lower_flops += lower
        self.custom_upper_flops += upper

    def _count_bev_pool(self, feat: torch.Tensor, ranks_bev: torch.Tensor) -> None:
        # feat shape in this project callsite: [B, N, H, W, C]
        # ranks_bev counts valid projected points.
        c = int(feat.shape[-1])
        m = int(ranks_bev.numel())
        n = float(m * c)

        # each contribution approx mul+add
        flops = 2.0 * n
        self.custom_lower_flops += flops
        self.custom_upper_flops += flops

    def install(self) -> None:
        if self._patched:
            return

        import rcdfnet.models.RCDFNet.multi_scale_deformable_attn_function as msda_rcdfnet
        import rcdfnet.models.custum_detectors.multi_scale_deformable_attn_function as msda_cpf
        import ops.bev_pool_v2.bev_pool as bev_pool_mod
        import rcdfnet.models.RCDFNet.ViewTransform as vt_mod

        # Patch MSDeformAttn fp32/fp16 (RCDFNet)
        for mod_name, cls_name in [
            ("rcdfnet_msda_fp32", "MultiScaleDeformableAttnFunction_fp32"),
            ("rcdfnet_msda_fp16", "MultiScaleDeformableAttnFunction_fp16"),
        ]:
            cls = getattr(msda_rcdfnet, cls_name)
            orig_forward = cls.forward
            self._orig[mod_name] = orig_forward

            def _wrap_forward(ctx, value, value_spatial_shapes, value_level_start_index,
                              sampling_locations, attention_weights, im2col_step,
                              _orig=orig_forward, _self=self):
                _self._count_ms_deform(value, attention_weights)
                return _orig(ctx, value, value_spatial_shapes, value_level_start_index,
                             sampling_locations, attention_weights, im2col_step)

            cls.forward = staticmethod(_wrap_forward)

        # Patch MSDeformAttn fp32/fp16 (cpf_detectors mirror)
        for mod_name, cls_name in [
            ("cpf_msda_fp32", "MultiScaleDeformableAttnFunction_fp32"),
            ("cpf_msda_fp16", "MultiScaleDeformableAttnFunction_fp16"),
        ]:
            cls = getattr(msda_cpf, cls_name)
            orig_forward = cls.forward
            self._orig[mod_name] = orig_forward

            def _wrap_forward(ctx, value, value_spatial_shapes, value_level_start_index,
                              sampling_locations, attention_weights, im2col_step,
                              _orig=orig_forward, _self=self):
                _self._count_ms_deform(value, attention_weights)
                return _orig(ctx, value, value_spatial_shapes, value_level_start_index,
                             sampling_locations, attention_weights, im2col_step)

            cls.forward = staticmethod(_wrap_forward)

        # Patch bev_pool_v2
        orig_bev_pool = bev_pool_mod.bev_pool_v2
        self._orig["bev_pool_v2"] = orig_bev_pool

        def _wrap_bev_pool(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                           bev_feat_shape, interval_starts, interval_lengths,
                           _orig=orig_bev_pool, _self=self):
            _self._count_bev_pool(feat, ranks_bev)
            return _orig(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                         bev_feat_shape, interval_starts, interval_lengths)

        bev_pool_mod.bev_pool_v2 = _wrap_bev_pool
        # ViewTransform imports `bev_pool_v2` into module namespace; patch alias too
        vt_mod.bev_pool_v2 = _wrap_bev_pool

        self._patched = True

    def uninstall(self) -> None:
        if not self._patched:
            return

        import rcdfnet.models.RCDFNet.multi_scale_deformable_attn_function as msda_rcdfnet
        import rcdfnet.models.custum_detectors.multi_scale_deformable_attn_function as msda_cpf
        import ops.bev_pool_v2.bev_pool as bev_pool_mod
        import rcdfnet.models.RCDFNet.ViewTransform as vt_mod

        msda_rcdfnet.MultiScaleDeformableAttnFunction_fp32.forward = self._orig["rcdfnet_msda_fp32"]
        msda_rcdfnet.MultiScaleDeformableAttnFunction_fp16.forward = self._orig["rcdfnet_msda_fp16"]
        msda_cpf.MultiScaleDeformableAttnFunction_fp32.forward = self._orig["cpf_msda_fp32"]
        msda_cpf.MultiScaleDeformableAttnFunction_fp16.forward = self._orig["cpf_msda_fp16"]

        bev_pool_mod.bev_pool_v2 = self._orig["bev_pool_v2"]
        vt_mod.bev_pool_v2 = self._orig["bev_pool_v2"]

        self._patched = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict e2e theoretical FLOPs with custom-op correction")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--out", default="research_outputs/e2e_strict_gpu2.json")
    return parser.parse_args()


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _sum_profiler_flops(prof: torch.profiler.profile) -> float:
    total = 0.0
    for ev in prof.key_averages():
        f = getattr(ev, "flops", 0)
        if f is not None:
            total += float(f)
    return total


def _output_summary(outputs: Any) -> Dict[str, Any]:
    out = {"type": type(outputs).__name__}
    if isinstance(outputs, list):
        out["len"] = len(outputs)
        if outputs:
            out["first_type"] = type(outputs[0]).__name__
            if isinstance(outputs[0], dict):
                out["first_keys"] = list(outputs[0].keys())
    return out


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

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

    model_total_params = int(sum(p.numel() for p in model.module.parameters()))
    model_trainable_params = int(sum(p.numel() for p in model.module.parameters() if p.requires_grad))
    model_fp32_size_mb = model_total_params * 4 / (1024 ** 2)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    checkpoint_state_dict_total = int(sum(v.numel() for v in state.values() if torch.is_tensor(v)))
    checkpoint_file_size_mb = os.path.getsize(args.checkpoint) / (1024 ** 2)

    tracer = CustomOpFlopsTracer()
    tracer.install()

    stats: List[BatchStat] = []
    first_output = None

    try:
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if i >= args.num_batches:
                    break

                tracer.reset()
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    with_flops=True,
                ) as prof:
                    outputs = model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                t1 = time.time()

                profiler_flops = _sum_profiler_flops(prof)
                c_lo = tracer.custom_lower_flops
                c_hi = tracer.custom_upper_flops
                total_lo = profiler_flops + c_lo
                total_hi = profiler_flops + c_hi

                stats.append(
                    BatchStat(
                        batch_index=i,
                        latency_sec=t1 - t0,
                        profiler_flops=profiler_flops,
                        custom_lower_flops=c_lo,
                        custom_upper_flops=c_hi,
                        total_lower_flops=total_lo,
                        total_upper_flops=total_hi,
                    )
                )

                if first_output is None:
                    first_output = _output_summary(outputs)
    finally:
        tracer.uninstall()

    if not stats:
        raise RuntimeError("No batches executed")

    avg_latency = float(sum(s.latency_sec for s in stats) / len(stats))
    avg_profiler_flops = float(sum(s.profiler_flops for s in stats) / len(stats))
    avg_custom_lo = float(sum(s.custom_lower_flops for s in stats) / len(stats))
    avg_custom_hi = float(sum(s.custom_upper_flops for s in stats) / len(stats))
    avg_total_lo = float(sum(s.total_lower_flops for s in stats) / len(stats))
    avg_total_hi = float(sum(s.total_upper_flops for s in stats) / len(stats))

    result = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": "cuda:0",
        "num_batches": len(stats),
        "status": "success",
        "notes": {
            "definition": "FLOPs are forward-pass theoretical operation counts.",
            "profiler_part": "torch.profiler operator FLOPs",
            "custom_part": "manual correction for ms_deform_attn and bev_pool_v2",
            "range_meaning": "lower/upper differ by bilinear interpolation overhead assumption in ms_deform_attn",
        },
        "size": {
            "model_total_params": model_total_params,
            "model_trainable_params": model_trainable_params,
            "model_fp32_size_mb": model_fp32_size_mb,
            "checkpoint_state_dict_total": checkpoint_state_dict_total,
            "checkpoint_file_size_mb": checkpoint_file_size_mb,
        },
        "e2e_output_summary": first_output,
        "avg_latency_sec": avg_latency,
        "flops": {
            "avg_profiler_flops": avg_profiler_flops,
            "avg_profiler_gflops": avg_profiler_flops / 1e9,
            "avg_custom_lower_flops": avg_custom_lo,
            "avg_custom_upper_flops": avg_custom_hi,
            "avg_total_lower_flops": avg_total_lo,
            "avg_total_upper_flops": avg_total_hi,
            "avg_total_lower_gflops": avg_total_lo / 1e9,
            "avg_total_upper_gflops": avg_total_hi / 1e9,
        },
        "batch_stats": [asdict(s) for s in stats],
    }

    _ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== Strict E2E FLOPs Done ===")
    print(f"Output JSON: {os.path.abspath(args.out)}")
    print(f"Batches: {len(stats)}")
    print(f"Avg latency: {avg_latency:.4f} sec")
    print(f"Avg total lower: {avg_total_lo/1e9:.3f} GFLOPs")
    print(f"Avg total upper: {avg_total_hi/1e9:.3f} GFLOPs")


if __name__ == "__main__":
    main()
