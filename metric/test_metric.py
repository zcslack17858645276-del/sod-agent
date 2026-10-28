from tqdm import tqdm
import os
import sys
import cv2
from metric.metric import Fmeasure, WeightedFmeasure, Smeasure, Emeasure, MAE 
from metric.fmeasurev2 import FmeasureV2, FmeasureHandler, PrecisionHandler, RecallHandler, FPRHandler, IOUHandler, DICEHandler, SpecificityHandler, BERHandler, OverallAccuracyHandler, KappaHandler

def evaluation_metric(gt_root, pred_root):
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    _MAE = MAE()

    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
    overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)

    FMv2 = FmeasureV2(
        metric_handlers={
            "fm": FmeasureHandler(**sample_gray, beta=0.3),
            "f1": FmeasureHandler(**sample_gray, beta=0.1),
            "pre": PrecisionHandler(**sample_gray),
            "rec": RecallHandler(**sample_gray),
            "fpr": FPRHandler(**sample_gray),
            "iou": IOUHandler(**sample_gray),
            "dice": DICEHandler(**sample_gray),
            "spec": SpecificityHandler(**sample_gray),
            "ber": BERHandler(**sample_gray),
            "oa": OverallAccuracyHandler(**sample_gray),
            "kappa": KappaHandler(**sample_gray),

            "sample_bifm": FmeasureHandler(**sample_bin, beta=0.3),
            "sample_bif1": FmeasureHandler(**sample_bin, beta=1),
            "sample_bipre": PrecisionHandler(**sample_bin),
            "sample_birec": RecallHandler(**sample_bin),

            "overall_bifm": FmeasureHandler(**overall_bin, beta=0.3),
            "overall_bif1": FmeasureHandler(**overall_bin, beta=1),
            "overall_bipre": PrecisionHandler(**overall_bin),
            "overall_birec": RecallHandler(**overall_bin),
        }
    )

    print(f"\n📊 Evaluating:")
    print(f"GT   : {gt_root}")
    print(f"PRED : {pred_root}")

    #mask_name_list = sorted(os.listdir(gt_root))
    mask_name_list = sorted(os.listdir(gt_root))

    pred_name_map = {}
    for pred_name in os.listdir(pred_root):
        pred_path = os.path.join(pred_root, pred_name)
        if not os.path.isfile(pred_path):
            continue
        stem, _ = os.path.splitext(pred_name)
        if stem and stem not in pred_name_map:
            pred_name_map[stem] = pred_name

    skipped = 0

    # ====== 进度条 ======
    for mask_name in tqdm(mask_name_list, desc="Metric", ncols=100):
        mask_path = os.path.join(gt_root, mask_name)
        mask_stem, _ = os.path.splitext(mask_name)
        pred_name = pred_name_map.get(mask_stem)
        if pred_name is None:
            skipped += 1
            continue
        pred_path = os.path.join(pred_root, pred_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if mask is None or pred is None:
            skipped += 1
            continue

        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        _MAE.step(pred=pred, gt=mask)
        FMv2.step(pred=pred, gt=mask)

    # ====== 汇总 ======
    fm = FM.get_results()["fm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = _MAE.get_results()["mae"]

    curr_results = {
        "Smeasure": sm,
        "maxFm": fm["curve"].max(),
        "maxEm": em["curve"].max(),
        "MAE": mae,
        "adpEm": em["adp"],
        "adpFm": fm["adp"],
    }

    # ====== 简洁输出 ======
    if skipped > 0:
        print(f"Skipped {skipped} samples due to missing/unreadable pairs.")
    print("\n📌 Results:")
    print(" | ".join([f"{k}: {v:.3f}" for k, v in curr_results.items()]))

    return curr_results