"""Minimal smoke test: build the model, run one forward pass, report shapes / params / memory.

Usage:
    python scripts/smoke_test.py [--size 384] [--batch 1]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from option import get_args_parser
from sam2.sam2_model.build_backbone_ours import Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=1)
    a, _ = ap.parse_known_args()

    sys.argv = ["smoke"]
    cfg = get_args_parser()

    t0 = time.time()
    model = Model(cfg).to("cuda").eval()
    print(f"[build] {time.time() - t0:.1f}s")

    rgb = torch.randn(a.batch, 3, a.size, a.size, device="cuda")
    thm = torch.randn(a.batch, 3, a.size, a.size, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model(rgb, thm)
    torch.cuda.synchronize()
    print(f"[forward] {time.time() - t0:.2f}s  peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    # SAMFusion uses nn.LazyConv2d: params only materialise after the first forward.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[params] total={total/1e6:.1f}M trainable={trainable/1e6:.1f}M")

    names = ["sam_out", "vssm", "vssm_pred", "vssm_det"]
    for n, o in zip(names, out):
        if isinstance(o, (list, tuple)):
            print(f"  {n}: " + ", ".join(str(tuple(x.shape)) for x in o))
        else:
            print(f"  {n}: {tuple(o.shape)}")


if __name__ == "__main__":
    main()
