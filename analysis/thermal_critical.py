"""Probe A-3: which images actually benefit from the thermal modality?

For every image we score it twice — with the real thermal input (`full`) and with the
thermal tensor zeroed (`t_zero`) — and look at the per-image difference dS.

Three things come out of this:
  1. how many images are "thermal-critical" (dS > 0.05) vs "thermal-harmful" (dS < 0),
  2. the *oracle per-image switch* upper bound: mean(max(S_full, S_zero)) - mean(S_full),
     i.e. the most any trust-routing module could gain by gating modalities alone,
  3. correlations between dS and cheap scene descriptors (thermal brightness/contrast,
     RGB-thermal similarity, object area) so we know what the agent could key on.

Usage:
    python analysis/thermal_critical.py --datasets VT821
"""
import argparse
import csv
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
from metric.metric import Smeasure
from option import get_args_parser
from sam2.sam2_model.build_backbone_ours import Model

ROOT = Path(__file__).resolve().parents[1]


def sm(pred, gt):
    m = Smeasure()
    m.step(pred=pred, gt=gt)
    return float(m.get_results()["sm"])


@torch.inference_mode()
def score_pair(model, loader, gt_dir):
    rows = []
    loader.index = 0
    for _ in range(loader.size):
        image, modality, gt_hw, name = loader.load_data_for_inference()
        image_c = image.cuda(non_blocking=True)
        mod_c = modality.cuda(non_blocking=True)

        preds = {}
        for tag, t in (("full", mod_c), ("zero", torch.zeros_like(mod_c))):
            r = model(image_c, t)
            if isinstance(r, (tuple, list)):
                r = r[0]
            r = F.interpolate(r, size=gt_hw, mode="bilinear", align_corners=False).sigmoid()
            r = r.float().cpu().numpy().squeeze()
            r = (r - r.min()) / (r.max() - r.min() + 1e-8)
            preds[tag] = np.round(r * 255).astype(np.uint8)

        gt_path = Path(gt_dir) / name
        if not gt_path.exists():
            gt_path = Path(gt_dir) / (Path(name).stem + ".jpg")
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        s_full, s_zero = sm(preds["full"], gt), sm(preds["zero"], gt)

        # cheap scene descriptors
        t_img = mod_c[0].float().cpu().numpy()            # (3,H,W) in [0,1]
        rgb_img = (image_c[0].float().cpu().numpy() * np.array([0.229, 0.224, 0.225])[:, None, None]
                   + np.array([0.485, 0.456, 0.406])[:, None, None]).clip(0, 1)
        t_gray = t_img.mean(0)
        rgb_gray = rgb_img.mean(0)
        t_flat, r_flat = t_gray.ravel(), rgb_gray.ravel()
        corr = float(np.corrcoef(t_flat, r_flat)[0, 1]) if t_flat.std() > 1e-6 else 0.0

        rows.append({
            "name": Path(name).stem,
            "S_full": s_full,
            "S_zero": s_zero,
            "dS": s_full - s_zero,
            "area": float((gt > 128).mean()),
            "t_mean": float(t_gray.mean()),
            "t_std": float(t_gray.std()),
            "rgb_t_corr": corr,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", default="res_checkpoints/final_pth/best_epoch_rgbt_final.pth")
    ap.add_argument("--datasets", nargs="+", default=["VT821"])
    a = ap.parse_args()

    argv = sys.argv
    sys.argv = [argv[0]]
    opt = get_args_parser()
    sys.argv = argv

    model = Model(opt, verbose=False)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad_(False)
    model.cuda().eval()

    summary = {}
    for ds in a.datasets:
        im = ROOT / "datasets" / "RGB-T" / ds / "RGB"
        gt = ROOT / "datasets" / "RGB-T" / ds / "GT"
        th = ROOT / "datasets" / "RGB-T" / ds / "T"
        loader = test_dataset(str(im), str(gt), modality_root=str(th), testsize=opt.input_size)

        rows = score_pair(model, loader, gt)
        rows.sort(key=lambda r: -r["dS"])

        csv_path = Path(f"results/thermal_critical_{ds.replace('/', '_')}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        s_full = np.array([r["S_full"] for r in rows])
        s_zero = np.array([r["S_zero"] for r in rows])
        d = np.array([r["dS"] for r in rows])
        oracle = np.maximum(s_full, s_zero)

        print(f"\n=== {ds}  n={len(rows)} ===")
        print(f"mean S  full={s_full.mean():.4f}  zero={s_zero.mean():.4f}  "
              f"oracle-switch={oracle.mean():.4f}  (headroom {oracle.mean()-s_full.mean():+.4f})")
        for thr in (0.01, 0.05, 0.10, 0.20):
            print(f"  dS > {thr:4.2f}: {int((d > thr).sum()):4d} imgs ({(d > thr).mean()*100:4.1f}%)  "
                  f"mean dS={d[d > thr].mean() if (d > thr).any() else 0:+.4f}")
        print(f"  dS < 0    : {int((d < 0).sum()):4d} imgs ({(d < 0).mean()*100:4.1f}%)  "
              f"mean dS={d[d < 0].mean() if (d < 0).any() else 0:+.4f}")

        if d.std() > 1e-9:
            corr = {k: float(np.corrcoef(d, np.array([r[k] for r in rows]))[0, 1])
                    for k in ("area", "t_mean", "t_std", "rgb_t_corr")}
            print("  corr(dS, .): " + "  ".join(f"{k}={v:+.3f}" for k, v in corr.items()))
        else:
            corr = {}

        summary[ds] = {
            "n": len(rows),
            "S_full": float(s_full.mean()),
            "S_zero": float(s_zero.mean()),
            "S_oracle_switch": float(oracle.mean()),
            "headroom": float(oracle.mean() - s_full.mean()),
            "n_dS_gt_0.05": int((d > 0.05).sum()),
            "n_dS_negative": int((d < 0).sum()),
            "corr": corr,
            "top20": [r["name"] for r in rows[:20]],
        }
        print(f"  csv -> {csv_path}")
        print("  top-5 thermal-critical:", ", ".join(r["name"] for r in rows[:5]))
        print("  worst-5 (thermal hurts):", ", ".join(r["name"] for r in rows[-5:]))

    Path("results/thermal_critical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nsummary -> results/thermal_critical_summary.json")


if __name__ == "__main__":
    main()
