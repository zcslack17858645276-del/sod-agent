"""
RGB-T / RGB-D 显著性图推理脚本。

性能说明（重要）：
    本脚本会被 AutoDL 镜像的默认环境变量拖慢 4~5 倍：
        /etc/profile.d/autodl.env.sh 里 export OMP_NUM_THREADS=0 / MKL_NUM_THREADS=0
    OMP_NUM_THREADS=0 对 libgomp 是非法值（启动时会打印
    "libgomp: Invalid value for environment variable OMP_NUM_THREADS"），
    libgomp 会忽略它并按 CPU 核数开线程（本机 192 核 -> torch 拿到 96 线程）。
    海量线程在小张量算子上疯狂争抢 CPU，CPU 侧的 torchvision/Numpy 变换
    会慢 40 倍（实测 15ms -> 594ms/张）。

    因此这里在 import torch 之前强制把线程数设成合理值。
    注意：必须在 import torch 之前设置，否则 libgomp 已经初始化就无效了。
"""

import os

# --- 必须在 import torch 之前 ---
_num_threads = os.environ.get("SAM2_NUM_THREADS", "8")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    # 0 / 空 / 非法值都会导致 libgomp 报错并回退到按核数开线程
    if os.environ.get(_var) in (None, "", "0"):
        os.environ[_var] = _num_threads
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cv2
from sam2.sam2_model.build_backbone_ours import Model
from option import get_args_parser

from datasets.get_data import test_dataset
import queue
import threading
import time

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

opt = get_args_parser()

# ==========================================================
# 参数统计函数
# ==========================================================
def count_params(model):
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    # 按照训练阶段的冻结策略统计
    # SAM image encoder 冻结，其余参数视为可训练
    trainable_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "image_encoder" not in name
    )

    return total_params, trainable_params

# ====== 模型加载 ======
model = Model(opt, verbose=False)
#model.load_state_dict(torch.load(opt.res_checkpoint))
model.load_state_dict(torch.load('final_pth/best_epoch_rgbt_final.pth'))
for param in model.parameters():
    param.requires_grad_(False)
model.cuda()
model.eval()

# ==========================================================
# 参数统计
# ==========================================================
total_params, trainable_params = count_params(model)

print("\n====== Model Parameters ======")
print(
    f"Total Params: "
    f"{total_params / 1e6:.3f} M"
)
print(
    f"Trainable Params: "
    f"{trainable_params / 1e6:.3f} M"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

test_rgbd_roots = [
    #'DES',
    'DUT-RGBD',
    #'LFSD',
    'NJU2K',
    'NLPR',
    'SIP',
    'STERE',
]

test_rgbt_roots = [
    #'VT5000/Train',
    'VT5000/Test',
    'VT1000',
    'VT821',
    #"VT-IMAG"
    #'UVT20K/Train',
    #'UVT20K/Test',
    #'UVT2000'
    #'unaligned_dataset/VT821_unalign',
    #'unaligned_dataset/VT1000_unalign',
    #'unaligned_dataset/VT5000-Test_unalign',
]

task = ['RGB-D']
task = ['RGB-T']

# 第二模态目录名：RGB-T 用 T（热红外），RGB-D 用 depth
MODALITY_DIR = {'RGB-D': 'depth', 'RGB-T': 'T'}


def prefetch(loader, depth=8):
    """后台线程预取下一批数据，避免 CPU 读取/变换与 GPU 前向互相阻塞。

    数据加载和模型推理是串行的：CPU 解码图片时 GPU 空转，GPU 推理时 CPU 空转。
    用一个后台线程把数据准备好后放进队列，主线程取数据直接就能送进 GPU。
    """
    q = queue.Queue(maxsize=depth)

    def producer():
        for i in range(loader.size):
            q.put(loader.load_data_for_inference())
        q.put(None)  # 结束哨兵

    threading.Thread(target=producer, daemon=True).start()
    return q

# ====== 开始评估 ======
for t in task:
    print(f"\n🎯 Starting evaluation for task: {t}")

    if t == 'SOD':
        test_roots = test_sod_roots
    elif t == 'RGB-D':
        test_roots = test_rgbd_roots
    else:
        test_roots = test_rgbt_roots
    for root in test_roots:
        print(f"\n🚀 Testing on dataset: {root}")

        test_loader = test_dataset(
            os.path.join(BASE_DIR, 'datasets', t, root, 'RGB'),
            os.path.join(BASE_DIR, 'datasets', t, root, 'GT'),
            modality_root=os.path.join(BASE_DIR, 'datasets', t, root, MODALITY_DIR[t]),
            testsize=opt.input_size
        )

        save_path = os.path.join(opt.pred_dir, root)
        os.makedirs(save_path, exist_ok=True)

        total_time = 0
        total_images = 0

        data_queue = prefetch(test_loader)

        # ====== 进度条 ======
        wall_start = time.perf_counter()
        gpu_events = []
        for i in tqdm(range(test_loader.size), desc=f"{root}", ncols=100):
            item = data_queue.get()
            if item is None:
                break
            image, modality, gt_hw, name = item

            image = image.cuda(non_blocking=True)
            modality = modality.cuda(non_blocking=True)

            # 用 CUDA event 计时：只统计 GPU 前向的耗时，
            # 且不需要每张图都 torch.cuda.synchronize()（那会打断 CPU/GPU 流水）
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode():
                res = model(image, modality)
            end.record()
            gpu_events.append((start, end))

            if isinstance(res, (tuple, list)):
                    res = res[0]
            res = F.interpolate(res, size=gt_hw, mode='bilinear', align_corners=False)
            res = res.sigmoid().float().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            res = np.round(res * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, name), res)
            total_images += 1

        # 统一同步一次再读 event 耗时（否则 event 还没完成，读不出来）
        torch.cuda.synchronize()
        gpu_time = sum(s.elapsed_time(e) for s, e in gpu_events) / 1000.0

        # 端到端吞吐（含数据读取、CPU 后处理、写盘），这才是实际速度
        total_time = time.perf_counter() - wall_start

        fps = total_images / total_time
        print(f"✅ Dataset {root}: FPS = {fps:.2f}  "
              f"(GPU forward: {total_images / gpu_time:.2f} FPS)")

    print(f"\n🎉 All datasets for task {t} have been evaluated!")

print("\n🎉 All Test Done!")
