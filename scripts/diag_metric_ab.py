"""Metric A/B: same GT, predictions from the official path (np.round) vs eval.py (truncate)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from metric.metric import Emeasure, Fmeasure, Smeasure, MAE, WeightedFmeasure

gt_dir = "datasets/RGB-T/VT821/GT"
names = sorted(os.listdir(gt_dir))


def run(pred_dir):
    em, fm, sm, mm, wm = Emeasure(), Fmeasure(), Smeasure(), MAE(), WeightedFmeasure()
    idx = {os.path.splitext(f)[0]: f for f in os.listdir(pred_dir)}
    n = 0
    for g in names:
        stem = os.path.splitext(g)[0]
        if stem not in idx:
            continue
        gt = cv2.imread(os.path.join(gt_dir, g), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_dir, idx[stem]), cv2.IMREAD_GRAYSCALE)
        em.step(pred, gt); fm.step(pred, gt); sm.step(pred, gt); mm.step(pred, gt); wm.step(pred, gt)
        n += 1
    return (n, sm.get_results()["sm"], mm.get_results()["mae"],
            em.get_results()["em"]["adp"], fm.get_results()["fm"]["adp"], wm.get_results()["wfm"])


for tag, d in (("round  (official)", "pred_check/VT821"), ("trunc  (eval.py)", "pred_mine/VT821")):
    n, s, mae, e, f, w = run(d)
    print(f"{tag}: n={n} S={s:.4f} MAE={mae:.4f} adpE={e:.4f} adpF={f:.4f} Fw={w:.4f}")

# how flat are the maps? (adaptive threshold sensitivity)
import glob
means = []
for p in sorted(glob.glob("pred_check/VT821/*.png"))[:200]:
    x = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    means.append(x.mean())
print(f"pred mean over 200 imgs: avg={np.mean(means):.3f} max={np.max(means):.3f}")
