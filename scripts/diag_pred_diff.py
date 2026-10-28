"""Pixel-level diff between the two prediction dumps."""
import os
import numpy as np
import cv2

a_dir, b_dir = "pred_check/VT821", "pred_mine/VT821"
fa = {os.path.splitext(f)[0]: f for f in os.listdir(a_dir)}
fb = {os.path.splitext(f)[0]: f for f in os.listdir(b_dir)}
common = sorted(set(fa) & set(fb))
print(f"files: official={len(fa)} mine={len(fb)} common={len(common)}")

ndiff, tot, maxd, signed = 0, 0, 0, []
worst = []
for k in common:
    x = cv2.imread(os.path.join(a_dir, fa[k]), cv2.IMREAD_GRAYSCALE).astype(np.int16)
    y = cv2.imread(os.path.join(b_dir, fb[k]), cv2.IMREAD_GRAYSCALE).astype(np.int16)
    if x.shape != y.shape:
        print("shape mismatch", k, x.shape, y.shape); continue
    d = y - x
    ndiff += int((d != 0).sum()); tot += d.size
    maxd = max(maxd, int(np.abs(d).max()))
    signed.append(float(d.mean()))
    worst.append((float(np.abs(d).mean()), k))

signed = np.array(signed)
print(f"differing pixels: {ndiff}/{tot} = {100.0*ndiff/tot:.2f}%")
print(f"max |diff| = {maxd}, mean signed diff = {signed.mean():+.3f}")
worst.sort(reverse=True)
print("worst 5 by mean|d|:", [(f"{v:.2f}", k) for v, k in worst[:5]])
