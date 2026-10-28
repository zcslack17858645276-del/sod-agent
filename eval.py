"""Batch evaluation wrapper around the *existing* repo pipeline.

Nothing about the metric is re-implemented here: inference matches
`sam2/tools/output_maps.py` line for line (same `test_dataset`, same `np.round`,
same `./pred/<root>/` layout), and scoring is delegated to
`metric/test_metric.evaluation_metric` — the very function `metric/metric_model.py`
calls. With no flags this must reproduce the original numbers exactly.

The only additions are convenience switches needed for A/B experiments:
    --datasets   pick which test sets to run (default: the three RGB-T sets)
    --limit      stop after N images (debugging)
    --tta        horizontal-flip test-time augmentation (averaged probabilities)
    --json       dump the returned dict to a json file

Usage:
    python eval.py -c res_checkpoints/final_pth/best_epoch_rgbt_final.pth
    python eval.py -c <ckpt> --datasets VT821 --tta --json results/x.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.get_data import test_dataset
from metric.test_metric import evaluation_metric
from option import get_args_parser
from sam2.sam2_model.build_backbone_ours import Model

ROOT = Path(__file__).resolve().parent
DEFAULT_SETS = ["VT5000/Test", "VT1000", "VT821"]


def load_model(ckpt: str):
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        opt = get_args_parser()
    finally:
        sys.argv = argv
    model = Model(opt, verbose=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad_(False)
    model.cuda().eval()
    return model, opt


@torch.inference_mode()
def infer_to_dir(model, loader, out_dir, tta=False, limit=0):
    """Exactly `output_maps.py`: forward -> interpolate -> sigmoid -> minmax -> round -> png."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for _ in tqdm(range(loader.size), desc=os.path.basename(out_dir.rstrip("/")), ncols=80):
        image, modality, gt_hw, name = loader.load_data_for_inference()
        image = image.cuda(non_blocking=True)
        modality = modality.cuda(non_blocking=True)

        res = model(image, modality)
        if isinstance(res, (tuple, list)):
            res = res[0]
        res = F.interpolate(res, size=gt_hw, mode="bilinear", align_corners=False).sigmoid()

        if tta:
            res2 = model(image.flip(-1), modality.flip(-1))
            if isinstance(res2, (tuple, list)):
                res2 = res2[0]
            res2 = F.interpolate(res2, size=gt_hw, mode="bilinear", align_corners=False).sigmoid().flip(-1)
            res = 0.5 * (res + res2)

        res = res.float().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(os.path.join(out_dir, name), np.round(res * 255).astype(np.uint8))
        n += 1
        if limit and n >= limit:
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", default="res_checkpoints/final_pth/best_epoch_rgbt_final.pth")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--pred-dir", default="pred")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    model, opt = load_model(a.checkpoint)
    print(f"ckpt={a.checkpoint}  size={opt.input_size}  tta={a.tta}")

    out = {"checkpoint": a.checkpoint, "size": opt.input_size, "tta": a.tta, "results": {}}
    for root in a.datasets:
        im_dir = ROOT / "datasets" / "RGB-T" / root / "RGB"
        gt_dir = ROOT / "datasets" / "RGB-T" / root / "GT"
        t_dir = ROOT / "datasets" / "RGB-T" / root / "T"
        if not im_dir.is_dir():
            print(f"[skip] {root}: {im_dir} not found")
            continue

        loader = test_dataset(str(im_dir), str(gt_dir), modality_root=str(t_dir), testsize=opt.input_size)
        pred_dir = ROOT / a.pred_dir / root
        n = infer_to_dir(model, loader, str(pred_dir), tta=a.tta, limit=a.limit)

        print(f"\n=== {root} ({n} imgs) ===")
        r = evaluation_metric(str(gt_dir) + "/", str(pred_dir))
        out["results"][root] = r

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved -> {a.json}")


if __name__ == "__main__":
    main()
