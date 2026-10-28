"""Standalone evaluation for a trained checkpoint: S-measure / MAE / E-measure / Fw-beta.

This is intentionally separate from `train.py::test()`, which stays as the cheap
per-epoch S-measure probe used for model selection.  Use this file when you want the
full four-metric table for a finished checkpoint.

Usage
-----
    python eval.py --checkpoint res_checkpoints/final_pth/best_epoch_rgbt_final.pth
    python eval.py -c <ckpt> --datasets VT5000 VT1000 VT821 --tta
    python eval.py -c <ckpt> --limit 200 --save-pred preds/   # quick sanity check

Outputs a table and, with --json, a machine-readable dump.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset_config import get_test_sets
from datasets.get_data import get_test_loader
from metric.metric import Emeasure, MAE, Smeasure, WeightedFmeasure, Fmeasure
from option import get_args_parser
from sam2.sam2_model.build_backbone_ours import Model


def build_model(opt, checkpoint: str, device):
    # option.py uses argparse.parse_args(); feed it a clean argv so eval flags never leak in.
    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    try:
        cfg = get_args_parser()
    finally:
        sys.argv = old_argv
    cfg.input_size = opt.size

    model = Model(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and not any(k.startswith("vmamba") for k in state):
        state = state["model"]
    # load before .to(device): SAMFusion uses nn.LazyConv2d, whose parameters only
    # materialise when the state dict (or a forward pass) provides their shape.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def forward_prob(model, image, modality, out_hw, tta: bool):
    logits = model(image, modality)[0]
    probs = []
    for flip in ([False, True] if tta else [False]):
        x, m = (image.flip(-1), modality.flip(-1)) if flip else (image, modality)
        out = model(x, m)[0] if flip else logits
        if flip:
            out = out.flip(-1)
        out = F.interpolate(out, size=out_hw, mode="bilinear", align_corners=False)
        p = out.sigmoid()
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        probs.append(p)
    return torch.stack(probs, 0).mean(0)


def evaluate(model, loader, device, size, tta, save_dir=None, limit=None, verbose=False):
    sm, mae, em, wfm, fm = Smeasure(), MAE(), Emeasure(), WeightedFmeasure(), Fmeasure()
    n = 0
    t0 = time.time()
    for pack in tqdm(loader, desc="eval", ncols=80, disable=not verbose):
        image, _, modality, gt, name = pack
        image = image.to(device, non_blocking=True)
        modality = modality.to(device, non_blocking=True)
        out_hw = (gt.shape[-2], gt.shape[-1])

        prob = forward_prob(model, image, modality, out_hw, tta)
        # EXACTLY match sam2/tools/output_maps.py: np.round, not truncation.
        # The E/F metrics are surprisingly sensitive to this +/-1/255 quantisation
        # (the adaptive threshold inside metric/utils.prepare_data amplifies it),
        # so round vs trunc changes adpF by ~1 point. Keep it identical.
        pred_np = np.round(prob[0, 0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

        gt_255 = np.asarray(gt[0].cpu().numpy() * 255.0).astype(np.uint8)
        sm.step(pred_np, gt_255)
        mae.step(pred_np, gt_255)
        em.step(pred_np, gt_255)
        wfm.step(pred_np, gt_255)
        fm.step(pred_np, gt_255)

        if save_dir:
            from PIL import Image

            Path(save_dir).mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred_np).save(os.path.join(save_dir, name[0]))
        n += 1
        if limit and n >= limit:
            break

    res = {
        "n": n,
        "S": float(sm.get_results()["sm"]),
        "MAE": float(mae.get_results()["mae"]),
        "E": float(em.get_results()["em"]["adp"]),
        "Fw": float(wfm.get_results()["wfm"]),
        "Fb": float(fm.get_results()["fm"]["adp"]),
        "seconds": round(time.time() - t0, 1),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", required=True)
    ap.add_argument("--datasets", nargs="*", default=None, help="default: TEST_DATASETS in dataset_config.py")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (debug)")
    ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
    ap.add_argument("--save-pred", default=None, help="directory to dump predicted masks")
    ap.add_argument("--json", default=None, help="dump results to this json file")
    ap.add_argument("-v", "--verbose", action="store_true", help="show the progress bar")
    opt = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={opt.checkpoint}  size={opt.size}  tta={opt.tta}")
    model = build_model(opt, opt.checkpoint, device)

    test_sets = get_test_sets(opt.datasets)
    results = {}
    for ts in test_sets:
        loader = get_test_loader(
            str(ts["im_root"]), str(ts["gt_root"]),
            modality_root=str(ts["modality_root"]),
            testsize=opt.size, batchsize=1, num_workers=4,
        )
        save_dir = os.path.join(opt.save_pred, ts["name"]) if opt.save_pred else None
        r = evaluate(model, loader, device, opt.size, opt.tta,
                     save_dir=save_dir, limit=opt.limit or None, verbose=opt.verbose)
        results[ts["name"]] = r
        print(f"[{ts['name']}] S={r['S']:.4f} MAE={r['MAE']:.4f} E={r['E']:.4f} "
              f"Fw={r['Fw']:.4f} Fb={r['Fb']:.4f}  ({r['n']} imgs, {r['seconds']}s)")

    if opt.json:
        Path(opt.json).parent.mkdir(parents=True, exist_ok=True)
        with open(opt.json, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": opt.checkpoint, "size": opt.size, "tta": opt.tta,
                       "results": results}, f, indent=2, ensure_ascii=False)
        print(f"saved -> {opt.json}")


if __name__ == "__main__":
    main()
