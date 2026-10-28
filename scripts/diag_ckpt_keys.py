"""Why do eval.py and the official pipeline disagree? Check checkpoint/model key alignment."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from option import get_args_parser
from sam2.sam2_model.build_backbone_ours import Model

argv = sys.argv
sys.argv = [argv[0]]
opt = get_args_parser()
sys.argv = argv

ck = torch.load("res_checkpoints/final_pth/best_epoch_rgbt_final.pth", map_location="cpu", weights_only=False)
print("type:", type(ck).__name__, "| n keys:", len(ck))
print("first keys:", list(ck.keys())[:3])
print("has 'model' wrapper:", isinstance(ck, dict) and "model" in ck and not any(
    k.startswith("vmamba") for k in ck))

model = Model(opt, verbose=False)
mk = set(model.state_dict().keys())
print("\nmodel keys:", len(mk))
for tag, state in (("raw", ck), ("unwrapped", ck.get("model", ck) if isinstance(ck, dict) else ck)):
    sk = set(state.keys())
    print(f"\n[{tag}] ckpt keys: {len(sk)}")
    print("  missing in ckpt:", sorted(mk - sk)[:8])
    print("  extra in ckpt :", sorted(sk - mk)[:8])
    try:
        model.load_state_dict(state, strict=True)
        print("  strict load: OK")
    except Exception as e:
        print("  strict load FAILED:", str(e).split("\n")[0][:160])
