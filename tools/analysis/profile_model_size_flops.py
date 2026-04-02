#!/usr/bin/env python3
"""Profile RCDFNet model size and FLOPs on a single GPU.

Usage (GPU2 physical card):
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python tools/analysis/profile_model_size_flops.py \
      --config work_dirs/RCDFNet_VoD_BMA_cross_attention_deformable/RCDFNet_BMA_vod_deformableattention.py \
      --checkpoint work_dirs/RCDFNet_VoD_BMA_cross_attention_deformable/checkpoints_final/epoch_8_final.pth \
      --out research_outputs/profile_gpu2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from thop import profile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rcdfnet.models import build_model  # noqa: E402


@dataclass
class ModuleProfile:
    name: str
    macs: Optional[float]
    flops: Optional[float]
    params: Optional[float]
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile model size and FLOPs")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--out", default="research_outputs/profile_gpu2.json", help="Output JSON path")
    parser.add_argument("--device", default="cuda:0", help="Torch device, default cuda:0")
    return parser.parse_args()


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def count_checkpoint_params(ckpt_path: str) -> Dict[str, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    total = int(sum(v.numel() for v in state.values() if torch.is_tensor(v)))

    prefixes = [
        "img_backbone",
        "img_neck",
        "pts_voxel_encoder",
        "pts_middle_encoder",
        "pts_backbone",
        "pts_neck",
        "point_Fusion",
        "pts_pred_context",
        "BEVencoder",
        "cross_attention",
        "BMA",
        "pts_bbox_head",
    ]
    by_prefix = {}
    for p in prefixes:
        by_prefix[p] = int(
            sum(v.numel() for k, v in state.items() if k.startswith(p) and torch.is_tensor(v))
        )

    return {"total": total, "by_prefix": by_prefix}


def estimate_cross_attention_flops(bev_hw=(160, 160), in_img=80, in_pts=80, embed=256, heads=4, modalities=2, points=4, layers=6):
    nq = bev_hw[0] * bev_hw[1]
    mac_value_img = nq * in_img * embed
    mac_value_pts = nq * in_pts * embed
    mac_offsets = nq * embed * (modalities * heads * points * 2)
    mac_attn_w = nq * embed * (modalities * heads * points)
    mac_ffn = nq * embed * (embed * 2) + nq * (embed * 2) * embed
    mac_per_layer = mac_value_img + mac_value_pts + mac_offsets + mac_attn_w + mac_ffn
    mac_input_proj = nq * (in_img + in_pts) * embed
    mac_sampling = nq * heads * modalities * points * (embed // heads)
    mac_total = mac_input_proj + layers * (mac_per_layer + mac_sampling)
    return float(mac_total), float(mac_total * 2)


def run_profile(name: str, module: torch.nn.Module, inputs: tuple) -> ModuleProfile:
    try:
        macs, params = profile(module, inputs=inputs, verbose=False)
        return ModuleProfile(
            name=name,
            macs=float(macs),
            flops=float(macs * 2),
            params=float(params),
            status="ok",
        )
    except Exception as e:  # noqa: BLE001
        return ModuleProfile(name=name, macs=None, flops=None, params=None, status=f"fail: {type(e).__name__}: {e}")


def build_and_load_model(cfg_path: str, ckpt_path: str, device: torch.device):
    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)
    model.to(device)
    model.eval()
    return cfg, model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and "cuda" in args.device:
        raise RuntimeError("CUDA is not available. Please run on a GPU environment.")

    device = torch.device(args.device)
    cfg, model = build_and_load_model(args.config, args.checkpoint, device)

    # Model-level sizes
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model_size_fp32_mb = total_params * 4 / (1024 ** 2)
    checkpoint_size_mb = os.path.getsize(args.checkpoint) / (1024 ** 2)
    ckpt_info = count_checkpoint_params(args.checkpoint)

    # Input shapes
    input_h, input_w = cfg.data_config["input_size"]
    bev_h = int((cfg.grid_config["y"][1] - cfg.grid_config["y"][0]) / cfg.grid_config["y"][2])
    bev_w = int((cfg.grid_config["x"][1] - cfg.grid_config["x"][0]) / cfg.grid_config["x"][2])

    # Wrappers
    class ImgBranch(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            f = self.m.img_backbone(x)
            y = self.m.img_neck(f)
            return y[0] if isinstance(y, (list, tuple)) else y

    class PtsBranch(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            y = self.m.pts_backbone(x)
            y = self.m.pts_neck(y)
            return y[0] if isinstance(y, (list, tuple)) else y

    class CrossWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.c = m.cross_attention

        def forward(self, a, b):
            return self.c(a, b)

    class BMAWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.b = m.BMA

        def forward(self, x):
            b = x.shape[0]
            mask = torch.zeros((b, bev_h, bev_w), dtype=x.dtype, device=x.device)
            y, _ = self.b(mask, x)
            return y

    class HeadWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.h = m.pts_bbox_head

        def forward(self, x):
            outs = self.h([x])
            if isinstance(outs, (list, tuple)):
                o0 = outs[0]
                if isinstance(o0, (list, tuple)):
                    return o0[0]
                return o0
            return outs

    profs = []

    with torch.no_grad():
        # one warmup inference on key branch tensors
        _ = model.img_backbone(torch.randn(1, 3, input_h, input_w, device=device))

        img = torch.randn(1, 3, input_h, input_w, device=device)
        profs.append(run_profile("img_backbone+img_neck", ImgBranch(model).to(device).eval(), (img,)))

        pts = torch.randn(1, 64, bev_h * 2, bev_w * 2, device=device)
        profs.append(run_profile("pts_backbone+pts_neck", PtsBranch(model).to(device).eval(), (pts,)))

        a = torch.randn(1, 80, bev_h, bev_w, device=device)
        b = torch.randn(1, 80, bev_h, bev_w, device=device)
        cross_prof = run_profile("cross_attention", CrossWrap(model).to(device).eval(), (a, b))
        profs.append(cross_prof)

        x_bma = torch.randn(1, 256, bev_h, bev_w, device=device)
        profs.append(run_profile("BMA", BMAWrap(model).to(device).eval(), (x_bma,)))

        x_head = torch.randn(1, 256, bev_h, bev_w, device=device)
        profs.append(run_profile("pts_bbox_head", HeadWrap(model).to(device).eval(), (x_head,)))

        if hasattr(model, "pts_pred_context") and model.pts_pred_context is not None:
            x_ppc = torch.randn(1, 384, bev_h, bev_w, device=device)
            profs.append(run_profile("pts_pred_context", model.pts_pred_context.to(device).eval(), (x_ppc,)))

    measured_flops = sum(p.flops for p in profs if p.flops is not None)

    est_cross_macs, est_cross_flops = estimate_cross_attention_flops(
        bev_hw=(bev_h, bev_w), in_img=80, in_pts=80, embed=256, heads=4, modalities=2, points=4, layers=6
    )

    final_flops = measured_flops
    if all(p.name != "cross_attention" or p.flops is None for p in profs):
        final_flops += est_cross_flops

    result = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "input": {
            "image": [1, 3, input_h, input_w],
            "bev_hw": [bev_h, bev_w],
            "batch_size": 1,
            "flops_definition": "FLOPs = 2 * MACs",
        },
        "params": {
            "model_total": total_params,
            "model_trainable": trainable_params,
            "model_fp32_size_mb": model_size_fp32_mb,
            "checkpoint_state_dict_total": ckpt_info["total"],
            "checkpoint_file_size_mb": checkpoint_size_mb,
            "checkpoint_by_prefix": ckpt_info["by_prefix"],
        },
        "module_profiles": [asdict(p) for p in profs],
        "cross_attention_estimate": {
            "macs": est_cross_macs,
            "flops": est_cross_flops,
        },
        "summary": {
            "measured_flops": measured_flops,
            "final_flops_with_estimate_if_needed": final_flops,
            "measured_gflops": measured_flops / 1e9,
            "final_gflops_with_estimate_if_needed": final_flops / 1e9,
        },
    }

    _ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== Profiling Done ===")
    print(f"Output JSON: {os.path.abspath(args.out)}")
    print(f"Model total params: {total_params:,}")
    print(f"Model trainable params: {trainable_params:,}")
    print(f"Checkpoint state_dict params: {ckpt_info['total']:,}")
    print(f"Measured FLOPs: {measured_flops / 1e9:.3f} GFLOPs")
    print(f"Final FLOPs (with estimate-if-needed): {final_flops / 1e9:.3f} GFLOPs")


if __name__ == "__main__":
    main()
