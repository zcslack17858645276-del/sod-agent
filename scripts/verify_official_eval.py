"""Reproduce the *official* evaluation path end-to-end on a chosen dataset:

    test_dataset (load_data_for_inference) -> model -> PNG maps -> evaluation_metric

This is the exact pipeline of sam2/tools/output_maps.py + metric/metric_model.py,
wrapped so we can run it on one dataset and compare against eval.py.

Usage:
    python scripts/verify_official_eval.py --datasets VT821
    python scripts/verify_official_eval.py --datasets VT5000/Test --limit 100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.get_data import test_dataset
from metric.test_metric import evaluation_metric
from option import get_args_parser
from sam2.sam2_model.build_backbone_ours import Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", default="res_checkpoints/final_pth/best_epoch_rgbt_final.pth")
    ap.add_argument("--datasets", nargs="+", default=["VT821"])
    ap.add_argument("--pred-dir", default="pred")
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
    print(f"ckpt loaded: {a.checkpoint}")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for root in a.datasets:
        im_dir = os.path.join(base, "datasets", "RGB-T", root, "RGB")
        gt_dir = os.path.join(base, "datasets", "RGB-T", root, "GT")
        t_dir = os.path.join(base, "datasets", "RGB-T", root, "T")
        loader = test_dataset(im_dir, gt_dir, modality_root=t_dir, testsize=opt.input_size)

        out_dir = os.path.join(a.pred_dir, root)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for _ in tqdm(range(loader.size), desc=root, ncols=80):
            image, modality, gt_hw, name = loader.load_data_for_inference()
            image = image.cuda(non_blocking=True)
            modality = modality.cuda(non_blocking=True)
            with torch.inference_mode():
                res = model(image, modality)
            if isinstance(res, (tuple, list)):
                res = res[0]
            res = F.interpolate(res, size=gt_hw, mode="bilinear", align_corners=False)
            res = res.sigmoid().float().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = np.round(res * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, name), res)
            n += 1
            if a.limit and n >= a.limit:
                break

        print(f"\n=== official metric on {root} ({n} images) ===")
        evaluation_metric(gt_dir + "/", out_dir)


if __name__ == "__main__":
    main()
