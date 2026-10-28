"""Probe A-1: does the model actually use the thermal modality, and how fast does it
degrade when the thermal image is corrupted?

Variants (applied to the thermal tensor, already in [0,1]):
    full         no change                              (upper bound)
    t_zero       all zeros      -> is thermal used at all?
    t_mean       constant 0.5   -> neutral control (no temperature structure)
    t_rgb        thermal := RGB -> modality-identity test
    blur2/blur5  gaussian blur  -> low sensor resolution / defocus
    lowc50       contrast x0.5  -> weak temperature difference
    quant8       posterise to 8 levels -> low-bit sensor
    shift4       translate 4 px -> RGB/T misalignment
    noise10      gaussian noise sigma=0.10
    drop30       random 30% blocks zeroed -> sensor dropout

Usage:
    python analysis/robustness.py --datasets VT821 --limit 400 --step 2
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.get_data import test_dataset
from metric.metric import Emeasure, MAE, Smeasure, WeightedFmeasure, Fmeasure
from option import get_args_parser
from sam2.sam2_model.build_backbone_ours import Model

ROOT = Path(__file__).resolve().parents[1]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _blur(t, sigma):
    ksize = int(2 * 3 * sigma + 1)
    x = torch.arange(ksize, dtype=torch.float32, device=t.device) - ksize // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2 = (g[:, None] * g[None, :])[None, None].repeat(t.shape[1], 1, 1, 1)
    return F.conv2d(t, k2, padding=ksize // 2, groups=t.shape[1])


def _block_dropout(t, ratio, seed=0):
    g = torch.Generator(device=t.device).manual_seed(seed)
    b, c, h, w = t.shape
    small = (torch.rand(b, 1, 16, 16, generator=g, device=t.device) > ratio).float()
    mask = F.interpolate(small, size=(h, w), mode="nearest")
    return t * mask


def make_variant(name, t, rgb):
    if name == "full":
        return t
    if name == "t_zero":
        return torch.zeros_like(t)
    if name == "t_mean":
        return torch.full_like(t, 0.5)
    if name == "t_rgb":
        return (rgb * STD.to(rgb.device) + MEAN.to(rgb.device)).clamp(0, 1)
    if name.startswith("blur"):
        return _blur(t, float(name[4:]))
    if name == "lowc50":
        return 0.5 * (t - 0.5) + 0.5
    if name == "quant8":
        return torch.round(t * 7) / 7
    if name.startswith("shift"):
        p = int(name[5:])
        return torch.roll(t, shifts=(p, p), dims=(2, 3))
    if name.startswith("noise"):
        s = float(name[5:]) / 100.0
        return (t + torch.randn_like(t) * s).clamp(0, 1)
    if name.startswith("drop"):
        return _block_dropout(t, float(name[4:]) / 100.0)
    raise ValueError(f"unknown variant {name}")


@torch.inference_mode()
def run_variant(model, loader, gt_dir, variant, step, limit):
    sm, mae, em, wfm, fm = Smeasure(), MAE(), Emeasure(), WeightedFmeasure(), Fmeasure()
    loader.index = 0
    n = 0
    for idx in range(loader.size):
        image, modality, gt_hw, name = loader.load_data_for_inference()
        if idx % step:
            continue

        image = image.cuda(non_blocking=True)
        modality = modality.cuda(non_blocking=True)
        t = make_variant(variant, modality, image)

        res = model(image, t)
        if isinstance(res, (tuple, list)):
            res = res[0]
        res = F.interpolate(res, size=gt_hw, mode="bilinear", align_corners=False).sigmoid()
        res = res.float().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred = np.round(res * 255).astype(np.uint8)

        gt_path = Path(gt_dir) / name
        if not gt_path.exists():
            gt_path = Path(gt_dir) / (Path(name).stem + ".jpg")
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(gt_path)

        sm.step(pred, gt); mae.step(pred, gt); em.step(pred, gt); wfm.step(pred, gt); fm.step(pred, gt)
        n += 1
        if limit and n >= limit:
            break

    return {
        "n": n,
        "S": float(sm.get_results()["sm"]),
        "MAE": float(mae.get_results()["mae"]),
        "E": float(em.get_results()["em"]["adp"]),
        "Fw": float(wfm.get_results()["wfm"]),
        "Fb": float(fm.get_results()["fm"]["adp"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", default="res_checkpoints/final_pth/best_epoch_rgbt_final.pth")
    ap.add_argument("--datasets", nargs="+", default=["VT821"])
    ap.add_argument("--step", type=int, default=2, help="take every Nth image")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--json", default="results/robustness.json")
    ap.add_argument("--variants", nargs="+", default=None)
    a = ap.parse_args()

    variants = a.variants or ["full", "t_zero", "t_mean", "t_rgb",
                              "blur2", "blur5", "lowc50", "quant8",
                              "shift4", "noise10", "drop30"]

    argv = sys.argv
    sys.argv = [argv[0]]
    opt = get_args_parser()
    sys.argv = argv

    model = Model(opt, verbose=False)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad_(False)
    model.cuda().eval()
    print(f"ckpt={a.checkpoint} size={opt.input_size} step={a.step} limit={a.limit}")

    out = {}
    for ds in a.datasets:
        im = ROOT / "datasets" / "RGB-T" / ds / "RGB"
        gt = ROOT / "datasets" / "RGB-T" / ds / "GT"
        th = ROOT / "datasets" / "RGB-T" / ds / "T"
        loader = test_dataset(str(im), str(gt), modality_root=str(th), testsize=opt.input_size)

        out[ds] = {}
        base = None
        for v in variants:
            r = run_variant(model, loader, gt, v, a.step, a.limit)
            if base is None:
                base = r
            out[ds][v] = r
            d = "" if v == "full" else f"   dS={r['S']-base['S']:+.4f}  dMAE={r['MAE']-base['MAE']:+.4f}"
            print(f"[{ds}] {v:9s} n={r['n']:4d}  S={r['S']:.4f} MAE={r['MAE']:.4f} "
                  f"E={r['E']:.4f} Fb={r['Fb']:.4f}{d}", flush=True)

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved ->", a.json)


if __name__ == "__main__":
    main()
