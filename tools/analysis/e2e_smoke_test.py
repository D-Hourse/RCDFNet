#!/usr/bin/env python3
"""End-to-end smoke test for RCDFNet inference pipeline.

Runs: config -> dataset -> dataloader -> model load -> forward(return_loss=False)
for a small number of batches.

Example (use physical GPU2):
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python tools/analysis/e2e_smoke_test.py \
    --config work_dirs/RCDFNet_VoD_BMA_cross_attention_deformable/RCDFNet_BMA_vod_deformableattention.py \
    --checkpoint work_dirs/RCDFNet_VoD_BMA_cross_attention_deformable/checkpoints_final/epoch_8_final.pth \
    --num-batches 1 \
    --out research_outputs/e2e_smoke_gpu2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from thop import profile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rcdfnet.datasets import build_dataloader, build_dataset  # noqa: E402
from rcdfnet.models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RCDFNet end-to-end smoke inference test")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file")
    parser.add_argument("--num-batches", type=int, default=1, help="How many batches to run")
    parser.add_argument("--out", default="research_outputs/e2e_smoke_gpu2.json", help="Output JSON path")
    return parser.parse_args()


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def count_checkpoint_params(ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    total = int(sum(v.numel() for v in state.values() if torch.is_tensor(v)))
    return {"total": total}


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


def run_profile(name: str, module: torch.nn.Module, inputs: tuple) -> Dict[str, Any]:
    try:
        macs, params = profile(module, inputs=inputs, verbose=False)
        return {
            "name": name,
            "macs": float(macs),
            "flops": float(macs * 2),
            "params": float(params),
            "status": "ok",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "name": name,
            "macs": None,
            "flops": None,
            "params": None,
            "status": f"fail: {type(e).__name__}: {e}",
        }


def _summarize_output(outputs: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"type": type(outputs).__name__}
    if isinstance(outputs, list):
        summary["len"] = len(outputs)
        if len(outputs) > 0:
            summary["first_type"] = type(outputs[0]).__name__
            if isinstance(outputs[0], dict):
                summary["first_keys"] = list(outputs[0].keys())
    elif isinstance(outputs, dict):
        summary["keys"] = list(outputs.keys())
    return summary


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this end-to-end smoke test.")

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    # test dataset mode
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

    # size stats
    model_total_params = int(sum(p.numel() for p in model.parameters()))
    model_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model_fp32_size_mb = model_total_params * 4 / (1024 ** 2)
    checkpoint_file_size_mb = os.path.getsize(args.checkpoint) / (1024 ** 2)
    checkpoint_state = count_checkpoint_params(args.checkpoint)

    # shape from cfg
    input_h, input_w = cfg.data_config["input_size"]
    bev_h = int((cfg.grid_config["y"][1] - cfg.grid_config["y"][0]) / cfg.grid_config["y"][2])
    bev_w = int((cfg.grid_config["x"][1] - cfg.grid_config["x"][0]) / cfg.grid_config["x"][2])

    # module flops profiles (same口径 as Research.md: FLOPs = 2*MACs)
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

    profs: List[Dict[str, Any]] = []
    with torch.no_grad():
        img = torch.randn(1, 3, input_h, input_w, device="cuda")
        profs.append(run_profile("img_backbone+img_neck", ImgBranch(model).cuda().eval(), (img,)))

        pts = torch.randn(1, 64, bev_h * 2, bev_w * 2, device="cuda")
        profs.append(run_profile("pts_backbone+pts_neck", PtsBranch(model).cuda().eval(), (pts,)))

        a = torch.randn(1, 80, bev_h, bev_w, device="cuda")
        b = torch.randn(1, 80, bev_h, bev_w, device="cuda")
        profs.append(run_profile("cross_attention", CrossWrap(model).cuda().eval(), (a, b)))

        x_bma = torch.randn(1, 256, bev_h, bev_w, device="cuda")
        profs.append(run_profile("BMA", BMAWrap(model).cuda().eval(), (x_bma,)))

        x_head = torch.randn(1, 256, bev_h, bev_w, device="cuda")
        profs.append(run_profile("pts_bbox_head", HeadWrap(model).cuda().eval(), (x_head,)))

        if hasattr(model, "pts_pred_context") and model.pts_pred_context is not None:
            x_ppc = torch.randn(1, 384, bev_h, bev_w, device="cuda")
            profs.append(run_profile("pts_pred_context", model.pts_pred_context.cuda().eval(), (x_ppc,)))

    measured_flops = float(sum(x["flops"] for x in profs if x["flops"] is not None))
    est_cross_macs, est_cross_flops = estimate_cross_attention_flops(bev_hw=(bev_h, bev_w))
    cross_ok = any((x["name"] == "cross_attention" and x["flops"] is not None) for x in profs)
    final_flops = measured_flops if cross_ok else measured_flops + est_cross_flops

    # Important: thop may register temporary hooks into modules.
    # Rebuild a clean model for e2e inference to avoid hook-side effects.
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)
    model.CLASSES = getattr(dataset, "CLASSES", None)

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    run_logs: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= args.num_batches:
                break

            torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            t1 = time.time()

            run_logs.append(
                {
                    "batch_index": i,
                    "elapsed_sec": t1 - t0,
                    "output_summary": _summarize_output(outputs),
                }
            )

    if len(run_logs) == 0:
        raise RuntimeError("No batch executed. Please check dataset/dataloader.")

    avg_latency = sum(x["elapsed_sec"] for x in run_logs) / len(run_logs)

    result = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "num_batches": len(run_logs),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": "cuda:0",
        "status": "success",
        "avg_latency_sec": avg_latency,
        "batch_logs": run_logs,
        "size": {
            "model_total_params": model_total_params,
            "model_trainable_params": model_trainable_params,
            "model_fp32_size_mb": model_fp32_size_mb,
            "checkpoint_state_dict_total": checkpoint_state["total"],
            "checkpoint_file_size_mb": checkpoint_file_size_mb,
        },
        "flops": {
            "definition": "FLOPs = 2 * MACs",
            "module_profiles": profs,
            "measured_flops": measured_flops,
            "measured_gflops": measured_flops / 1e9,
            "cross_attention_estimate": {
                "macs": est_cross_macs,
                "flops": est_cross_flops,
            },
            "final_flops_with_estimate_if_needed": final_flops,
            "final_gflops_with_estimate_if_needed": final_flops / 1e9,
        },
    }

    _ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== E2E Smoke Test Done ===")
    print(f"Output JSON: {os.path.abspath(args.out)}")
    print(f"Batches run: {len(run_logs)}")
    print(f"Average latency: {avg_latency:.4f} sec")
    print(f"Model total params: {model_total_params:,}")
    print(f"Measured FLOPs: {measured_flops / 1e9:.3f} GFLOPs")
    print(f"Final FLOPs (with estimate-if-needed): {final_flops / 1e9:.3f} GFLOPs")


if __name__ == "__main__":
    main()
