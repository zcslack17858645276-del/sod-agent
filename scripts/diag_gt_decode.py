"""Diagnose the E/F discrepancy: same predictions, GT decoded by cv2 vs PIL."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from PIL import Image
from metric.metric import Emeasure, WeightedFmeasure, Smeasure, MAE, Fmeasure

gt_dir = "datasets/RGB-T/VT821/GT"
pred_dir = "pred_check/VT821"

preds = {os.path.splitext(f)[0]: f for f in os.listdir(pred_dir) if f.lower().endswith(".png")}
names = sorted(os.listdir(gt_dir))


def run(gt_reader):
    em, fm, sm, mm = Emeasure(), Fmeasure(), Smeasure(), MAE()
    n = 0
    for g in names:
        stem = os.path.splitext(g)[0]
        p = preds.get(stem)
        if p is None:
            continue
        pred = cv2.imread(os.path.join(pred_dir, p), cv2.IMREAD_GRAYSCALE)
        gt = gt_reader(os.path.join(gt_dir, g))
        if pred is None or gt is None:
            continue
        em.step(pred, gt); fm.step(pred, gt); sm.step(pred, gt); mm.step(pred, gt)
        n += 1
    return n, sm.get_results()["sm"], mm.get_results()["mae"], em.get_results()["em"]["adp"], fm.get_results()["fm"]["adp"]


def gt_cv2(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE)


def gt_pil(p):
    return np.array(Image.open(p).convert("L"))


for tag, fn in (("cv2", gt_cv2), ("PIL", gt_pil)):
    n, s, mae, e, f = run(fn)
    print(f"{tag:4s} n={n}  S={s:.4f} MAE={mae:.4f} adpE={e:.4f} adpF={f:.4f}")

# how different are the two GT decodings?
g = names[0]
a, b = gt_cv2(os.path.join(gt_dir, g)), gt_pil(os.path.join(gt_dir, g))
d = (a.astype(int) - b.astype(int))
print(f"GT decode diff on {g}: max|d|={np.abs(d).max()}  mean|d|={np.abs(d).mean():.3f}  "
      f"bin-flip(pixels crossing 128)={np.count_nonzero((a > 128) != (b > 128))}")
