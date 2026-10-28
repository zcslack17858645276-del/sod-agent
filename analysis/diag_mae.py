"""Why is per-image MAE ~= object area? Reproduce on 3 images with instrumentation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, torch, torch.nn.functional as F
from datasets.get_data import test_dataset
from option import get_args_parser
from sam2.sam2_model.build_backbone_ours import Model
from metric.metric import Smeasure

argv = sys.argv; sys.argv = [argv[0]]
opt = get_args_parser(); sys.argv = argv

m = Model(opt, verbose=False)
m.load_state_dict(torch.load("res_checkpoints/final_pth/best_epoch_rgbt_final.pth", map_location="cpu"))
m.cuda().eval()

loader = test_dataset("datasets/RGB-T/VT821/RGB", "datasets/RGB-T/VT821/GT",
                      modality_root="datasets/RGB-T/VT821/T", testsize=opt.input_size)
loader.index = 809  # image 810

with torch.inference_mode():
    image, modality, gt_hw, name = loader.load_data_for_inference()
    res = m(image.cuda(), modality.cuda())[0]
    res = F.interpolate(res, size=gt_hw, mode="bilinear", align_corners=False).sigmoid()
    res = res.float().cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    pred = np.round(res * 255).astype(np.uint8)

gt = cv2.imread(f"datasets/RGB-T/VT821/GT/{name}", cv2.IMREAD_GRAYSCALE)
if gt is None:
    gt = cv2.imread(f"datasets/RGB-T/VT821/GT/{os.path.splitext(name)[0]}.jpg", cv2.IMREAD_GRAYSCALE)

print("name:", name, "pred", pred.shape, pred.dtype, "gt", gt.shape, gt.dtype)
print("pred min/max/mean:", pred.min(), pred.max(), pred.mean())
print("gt   min/max/mean:", gt.min(), gt.max(), gt.mean())
mae_before = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean() / 255.0
print("MAE before smeasure:", mae_before)

s = Smeasure(); s.step(pred=pred, gt=gt); print("S:", s.get_results()["sm"])
mae_after = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean() / 255.0
print("MAE after  smeasure:", mae_after)
print("pred unchanged?", np.array_equal(pred, np.round(res * 255).astype(np.uint8)))
