"""Probe A-2: where does the error actually live?

Per-image metrics + object-size / confidence statistics, then bucket the error mass.
The point is to decide whether the next module should target small objects, low-confidence
regions, or simply "more data" — before designing anything.

Outputs a per-image CSV and a bucketed summary.

Usage:
    python analysis/hard_subset.py --datasets VT821 --out results/hard_vt821.csv
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
from option import get_args_parser
from sam2.sam2_model.build_backbone_ours import Model

ROOT = Path(__file__).resolve().parents[1]


def smeasure(pred: np.ndarray, gt: np.ndarray, alpha=0.5) -> float:
    """S-measure on uint8 maps (same formulation as metric/metric.py)."""
    from metric.metric import Smeasure
    m = Smeasure(alpha)
    m.step(pred=pred, gt=gt)
    return float(m.get_results()["sm"])


@torch.inference_mode()
def collect(model, loader, gt_dir):
    rows = []
    loader.index = 0
    for _ in range(loader.size):
        image, modality, gt_hw, name = loader.load_data_for_inference()
        image = image.cuda(non_blocking=True)
        modality = modality.cuda(non_blocking=True)

        res = model(image, modality)
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
        gtb = gt > 128

        inter = np.logical_and(pred > 128, gtb).sum()
        union = np.logical_or(pred > 128, gtb).sum()
        iou = float(inter / union) if union else 1.0

        # compute every statistic BEFORE any metric object touches the arrays
        pf = pred.astype(np.float64) / 255.0
        gf = gt.astype(np.float64) / 255.0
        mae_val = float(np.mean(np.abs(pf - gf)))
        p = pf
        entropy = float(np.mean(-(p * np.log(p + 1e-8) + (1 - p) * np.log(1 - p + 1e-8))))
        margin = float(np.mean(np.abs(p - 0.5)))
        area = float(gtb.mean())
        s_val = smeasure(pred, gt)

        rows.append({
            "name": Path(name).stem,
            "S": s_val,
            "MAE": mae_val,
            "IoU": iou,
            "area": area,
            "entropy": entropy,
            "margin": margin,
            "h": int(gt.shape[0]),
            "w": int(gt.shape[1]),
        })
    return rows


def summarise(rows, key, edges, label):
    print(f"\n--- bucketed by {label} ---")
    print(f"{'bucket':>18} {'n':>5} {'S':>7} {'MAE':>8} {'IoU':>7} {'err%':>6}")
    total_err = sum(r["MAE"] for r in rows)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        if not sel:
            continue
        err = sum(r["MAE"] for r in sel)
        rec = {
            "range": f"[{lo:g},{hi:g})", "n": len(sel),
            "S": sum(r["S"] for r in sel) / len(sel),
            "MAE": err / len(sel),
            "IoU": sum(r["IoU"] for r in sel) / len(sel),
            "err_share": 100.0 * err / total_err,
            "img_share": 100.0 * len(sel) / len(rows),
        }
        out.append(rec)
        print(f"{rec['range']:>18} {rec['n']:5d} {rec['S']:7.4f} {rec['MAE']:8.4f} "
              f"{rec['IoU']:7.4f} {rec['err_share']:5.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", default="res_checkpoints/final_pth/best_epoch_rgbt_final.pth")
    ap.add_argument("--datasets", nargs="+", default=["VT821"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
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
        loader.size = min(loader.size, a.limit) if a.limit else loader.size

        rows = collect(model, loader, gt)
        rows.sort(key=lambda r: r["MAE"], reverse=True)

        csv_path = Path(a.out or f"results/hard_{ds.replace('/', '_')}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        total_err = sum(r["MAE"] for r in rows)
        top10 = rows[: max(1, len(rows) // 10)]
        print(f"\n=== {ds}  n={len(rows)} ===")
        print(f"mean S={np.mean([r['S'] for r in rows]):.4f}  "
              f"MAE={np.mean([r['MAE'] for r in rows]):.4f}")
        print(f"worst 10% images hold {100.0*sum(r['MAE'] for r in top10)/total_err:.1f}% of total error")

        summary[ds] = {
            "n": len(rows),
            "S": float(np.mean([r["S"] for r in rows])),
            "MAE": float(np.mean([r["MAE"] for r in rows])),
            "worst10_err_share": float(100.0 * sum(r["MAE"] for r in top10) / total_err),
            "by_area": summarise(rows, "area", [0, 0.005, 0.01, 0.03, 0.10, 1.01], "object area ratio"),
            "by_entropy": summarise(rows, "entropy", [0, 0.15, 0.25, 0.35, 0.45, 1.01], "pred entropy"),
        }
        print(f"csv -> {csv_path}")

    out_json = Path("results/hard_summary.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary -> {out_json}")


if __name__ == "__main__":
    main()
