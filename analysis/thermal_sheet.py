"""Render a contact sheet for the most thermal-critical images.

Columns: RGB | Thermal | pred (full) | pred (thermal zeroed)
Rows   : top-N images ranked by dS = S_full - S_zero
"""
import argparse
import csv
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


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/thermal_critical_VT821.csv")
    ap.add_argument("--dataset", default="VT821")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--reverse", action="store_true", help="worst (thermal hurts) instead of best")
    ap.add_argument("--out", default="results/thermal_cases.png")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    rows.sort(key=lambda r: float(r["dS"]), reverse=not a.reverse)
    picks = [r["name"] for r in rows[: a.top]]

    argv = sys.argv
    sys.argv = [argv[0]]
    opt = get_args_parser()
    sys.argv = argv

    model = Model(opt, verbose=False)
    model.load_state_dict(torch.load("res_checkpoints/final_pth/best_epoch_rgbt_final.pth", map_location="cpu"))
    for p in model.parameters():
        p.requires_grad_(False)
    model.cuda().eval()

    loader = test_dataset(str(ROOT / "datasets/RGB-T" / a.dataset / "RGB"),
                          str(ROOT / "datasets/RGB-T" / a.dataset / "GT"),
                          modality_root=str(ROOT / "datasets/RGB-T" / a.dataset / "T"),
                          testsize=opt.input_size)

    tiles = []
    for _ in range(loader.size):
        image, modality, gt_hw, name = loader.load_data_for_inference()
        stem = Path(name).stem
        if stem not in picks:
            continue
        img_c, mod_c = image.cuda(), modality.cuda()
        outs = {}
        for tag, t in (("full", mod_c), ("zero", torch.zeros_like(mod_c))):
            r = model(img_c, t)[0]
            r = F.interpolate(r, size=gt_hw, mode="bilinear", align_corners=False).sigmoid()
            r = r.float().cpu().numpy().squeeze()
            outs[tag] = np.round((r - r.min()) / (r.max() - r.min() + 1e-8) * 255).astype(np.uint8)

        rgb_vis = cv2.cvtColor((image[0].permute(1, 2, 0).numpy() * [0.229, 0.224, 0.225]
                                + [0.485, 0.456, 0.406]).clip(0, 1).astype(np.float32), cv2.COLOR_RGB2BGR)
        t_vis = cv2.cvtColor((modality[0, 0].numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        h, w = outs["full"].shape
        rgb_vis = cv2.resize(rgb_vis, (w, h))
        t_vis = cv2.resize(t_vis, (w, h))
        d = [float(r["dS"]) for r in rows if r["name"] == stem][0]
        line = [rgb_vis, t_vis, cv2.cvtColor(outs["full"], cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(outs["zero"], cv2.COLOR_GRAY2BGR)]
        # images have different resolutions -> normalise every tile
        line = [cv2.resize(x, (320, 240)) for x in line]
        line = [cv2.copyMakeBorder(x, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for x in line]
        cv2.putText(line[0], f"{stem}  dS={d:+.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(np.hstack(line))

    sheet = np.vstack(tiles)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print("saved", a.out, sheet.shape)


if __name__ == "__main__":
    main()
