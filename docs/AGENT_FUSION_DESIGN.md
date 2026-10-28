# ATF：Agent-guided Trust Fusion 设计与实施文档

> 项目：`D:\workplace\Ours\my_sam_project-main`（RGB-T 伪装目标检测 / 视频显著性）
> 目标：在多模态融合环节引入 LLM Agent，形成 **A（离线蒸馏，零推理开销）+ B（在线难例闭环）** 双档方案
> 版本：v1.0 · 2026-10-28
> 状态：设计定稿待评审 → 进入实现

---

## 0. 阅读指引

| 章节 | 内容 | 读者 |
|---|---|---|
| §1 | 现有模型的代码级解剖（数据流、融合机制、损失、问题清单） | 必读，所有后续设计都建立在此之上 |
| §2 | 环境方案（WSL + conda + sm_120 适配） | 复现前必读 |
| §3 | **Agent 融合设计（核心）** | 论文主体 |
| §4 | 训练 / 评测协议与消融矩阵 | 实验执行 |
| §5 | 工程实施计划（里程碑、文件清单、验收标准） | 排期 |
| §6 | 风险登记表 | 决策 |

---

## 1. 现状摸底（代码级）

### 1.1 仓库结构

```
my_sam_project-main/
├── train.py                      # 训练 + 测试主脚本（321 行）
├── option.py                     # argparse 配置（input_size=384, bs=4, epochs=30）
├── dataset_config.py             # 数据集注册表（RGB-T: VT5000/UVT20K/VT1000/VT821 + unaligned 版本）
├── model_structure.txt           # 64KB 模型结构 dump
├── datasets/
│   ├── get_data.py               # Dataset / DataLoader（RGB-T：RGB + T）
│   └── RGB-T/{VT5000, VT1000, VT821}
├── checkpoints/
│   ├── sam2_hiera_base_plus.pt   # 323 MB，SAM2-B+ 预训练
│   └── vmamba_chekcpoints/vssmsmall_dp03_ckpt_epoch_238.pth  # 177 MB
├── res_checkpoints/final_pth/    # 三个已训练权重（各 ~689 MB ≈ 172M 参数 fp32）
│   ├── best_epoch_rgbt_final.pth
│   ├── best_epoch_rgbd_final.pth
│   └── best_epoch_rgbd_train2.pth
├── sam2/
│   ├── modeling/                 # 原生 SAM2（Hiera 主干 + FPN neck + mask decoder）
│   └── sam2_model/
│       ├── build_backbone_ours.py        # ★ 顶层 Model，改造后的前向
│       └── Module/
│           ├── GRCM.py                   # SAMFusion = HFusion（SAM↔VMamba 适配器）
│           ├── HeterogeneousFusionAggregationModule.py  # HFAM（超图式多节点注意力融合）
│           ├── CPD.py                    # CPD_VMamba 解码器（RFB + HA + 双头）
│           ├── ProgressiveRefine.py      # SAM 输出精修
│           ├── HolisticAttention.py      # HA
│           └── ConvBNReLU.py
├── Vmamba/models/
│   ├── Vmamba_encoder_dual.py            # 双模态 VSSM 封装
│   └── encoders/
│       ├── dual_vmamba.py                # ★ RGBXMamba：共享 VSSM + 4×HFAM 融合
│       ├── vmamba.py                     # VSSM 主体（含 selective_scan CUDA 依赖）
│       └── selective_scan/               # ★ 必须编译的 CUDA 扩展（cp310）
└── metric/metric.py              # S-measure / MAE / E-measure / Fβ / weighted-Fβ
```

**环境实况（本机，2026-10-28 实测）**

| 项 | 值 |
|---|---|
| WSL | Ubuntu 2（WSL2），20 核，15 GB RAM，4 GB swap，磁盘可用 859 GB |
| GPU | NVIDIA RTX 5060 Ti，16311 MiB，驱动 581.57，**compute capability (12, 0) = sm_120** |
| conda | `~/miniconda3`（conda 25.9.1），已有 envs: base / GPTSoVits / brainocr / **dformer** / env_name / rfdetr / ticket_env / yolo_v5 / yolov5_new |
| `dformer` env | torch 2.1.2+cu121 → **不可用**（`no kernel image is available`，只编到 sm_90） |
| `rfdetr` env | torch 2.9.1+cu128 → 可用，但缺 timm / hydra / selective_scan |
| 数据集 | 仅 RGB-T：VT5000（2500 train / 2500 test）、VT1000（1000）、VT821（821）。**RGB-D 目录不存在** |

### 1.2 数据流与张量形状（输入 384×384）

```
RGB  ──► ImageNet normalize ──┐
                              ├──► VSSM-S（共享权重，跑两遍）──► [96²,48²,24²,12²] × {96,192,384,768}
Thermal ──► /255, L→RGB ──────┘                                      │
                                                          HFAM ×4（跨模态融合）★融合点1
                                                                     ▼
                                                        fused_vssm = [96²,48²,24²,12²]
                                                                     │
RGB ──► Hiera-B+ trunk（冻结）── 逐 block 前向 ──► stage_ends 处注入 ◄──┘
                                                    SAMFusion ×4 ★融合点2
                                                                     ▼
                                             FPN neck（3 级，256ch，stride 4/8/16）
                                                                     ▼
                              mask decoder（零 prompt / 零 PE）──► upscaled_embedding 96²×32
                                                                     ▼
                              ProgressiveRefine ──► sam_out 384²×1        ← 主输出
                                                                     ▼
                       vssm[1..3] ──► CPD_VMamba ──► detection 384²×1 / attention 384²×1
```

| 张量 | 形状（384 输入） |
|---|---|
| VSSM stage 输出 | 96×96×96 / 48×48×192 / 24×24×384 / 12×12×768 |
| Hiera stage 输出 | 96×96×112 / 48×48×224 / 24×24×448 / 12×12×896 |
| FPN（backbone_fpn） | 3 级 × 256ch，stride 4 / 8 / 16 |
| image_embed | 24×24×256 |
| low_res_masks / upscaled_embedding | 96×96 |
| sam_out / detection / attention | 384×384×1 |

> `stage_ends` 对 `hiera_b+` 为 `[1, 4, 20, 23]`（blocks 共 24 个，depths=[2,3,16,3]）；`scalp=1` 丢弃 stride-32 层，故 FPN 只有 3 级。

### 1.3 融合机制解剖（三处，全部是"隐式、逐像素、无监督"的）

#### 融合点 1：`dual_vmamba.RGBXMamba` —— 跨模态 VSSM 级融合

```python
outs_rgb = self.vssm_r(x_rgb)      # 同一套 VSSM 权重跑 RGB
outs_e   = self.vssm_r(x_e)        # 再跑 Thermal  ← 权重共享
out_fused = self.modality_mamba4([outs_rgb[0], outs_e[0]])   # 4 个 stage 各一个 HFAM
```

#### 融合点 2：`GRCM.SAMFusion` —— SAM 主干的"增量注入"

```python
x2_aligned = align_conv(x2)                    # VMamba 通道 → SAM 通道（LazyConv2d）
fused      = hga([x1, x2_aligned])             # HFAM
delta      = out_conv(fused)                   # 零初始化 → 初期 delta≈0，不破坏 SAM 预训练
```

注入位置在 `build_backbone_ours.Model.forward` 中：每个 stage 结束时
`rgb = rgb + delta + mlp(norm(rgb + delta))`。

#### 融合点 3：`CPD_VMamba` —— 解码端的双头聚合

RFB 提多尺度 → `aggregation` 金字塔聚合 → HA（高斯软注意力）→ 第二个 aggregation 出 `detection`；
第一遍的 `attention` 单独输出。两个头分别以 0.2 / 0.5 权重参与损失。

#### HFAM 的内部机制（**这是 Agent 要介入的地方**）

对 `N` 个模态特征（此处 N=2）：

```
H0      = MLP( 0.5·(mean_k E(x_k) + max_k E(x_k)) )        # 增强后的"hub"节点
spokes  = [x_1, ..., x_N]
for l in 1..L:                                            # L=3
    W_H0   = SW(H0);  W_Hk = SW(Hk)
    logit_k= attn([W_H0, W_Hk])         # 1×1 conv → (B,1,H,W)
    α      = softmax_k(logit)            # k = 0..N 共 N+1 个节点
    H0     = H0 + ReLU( Σ_k α_k · W_H_k )
```

**关键洞察**：HFAM 已经在逐像素地、隐式地学"该信谁"——`α_rgb(r)` 与 `α_t(r)` 就是一张**逐像素模态信任图**，只不过它是纯数据驱动的、没有语义解释、也不是区域一致的。这给了我们三样东西：

1. 一个天然的注入接口（在 softmax 前加偏置）；
2. 一个可解释性抓手（可视化 α 就能看出模型在哪信谁）；
3. 一个与 Agent 输出对照的基线（Agent 的 trust 应该与 α 的分布相关，这是最好的 sanity check）。

### 1.4 训练 / 损失 / 评测现状

| 项 | 现状 |
|---|---|
| 损失 | `L = structure_loss(sam_out, gt) + 0.2·structure_loss(det) + 0.5·structure_loss(prediction)` |
| `structure_loss` | weighted BCE（kernel 31, weight = 1+5·\|avgpool−gt\|）+ weighted IoU |
| 优化器 | AdamW，4 组 lr：adapter 1e-4 / vmamba **1e-5** / fusion 1e-4 / decoder 1e-4 |
| 调度 | 5 epoch warmup + cosine，共 30 epoch |
| 冻结 | `setup()` 冻结 `self.model.*` 中所有不含 `sam_mask_decoder` 的参数 → **Hiera 主干与 patch_embed 冻结**，只训 mask decoder |
| 输入 | 训练时喂的是 `aug_image`（颜色抖动后的图），干净图 `image` 被丢弃 |
| 评测 | `test()` 只用 **S-measure**，逐 epoch 在 VT5000 上评，取均值作 best |
| 输出 | `sam_out` 是唯一被评测的输出，`detection` / `attention` 只参与损失 |

### 1.5 已发现的问题与机会（按优先级）

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| P1 | `image_pe = torch.zeros(...)`，mask decoder 的 **位置编码被完全置零** | decoder 没有显式空间先验，只能靠 image embedding 自带的位置信息；可能是精度天花板之一 | 作为**独立小实验**（不改主线）：改用 `self.model.sam_prompt_encoder.get_dense_pe()`，预期 +0.3~0.8 S-measure |
| P2 | 只报 S-measure | 论文无法与其他方法对齐（缺 MAE / E / Fwβ） | 评测脚本换全套 4 指标（`metric/metric.py` 已实现 5 个类） |
| P3 | 训练喂 `aug_image`，测试喂原图 | train/test 分布不一致 | 建议 `p=0.5` 概率用增强图，或干脆只喂原图 |
| P4 | Thermal 用 `L→RGB` 复制三通道且**不做 ImageNet 归一化**，与 RGB 预处理不一致 | 两路输入统计量不同，共享 VSSM 时更明显 | 对 T 单独做一次标准化（均值/方差来自训练集），或统一归一化 |
| P5 | `stage3_adapter_every` 默认 0 → stage3 内层 adapter 全部闲置 | 模块闲置，白占显存 | 保持 0，作为消融项 |
| P6 | 无 git 仓库 | 改动无法回溯，实验不可复现 | ✅ 已 `git init` 并推送 `sod-agent` |
| P9 | `option.py` 的 `vmamba_pretrained_model` 指向 `checkpoints/vmamba_checkpoints/`，实际目录是 `checkpoints/vmamba_chekcpoints/`（**拼写不一致**） | 现在重新训练会**静默丢掉了 ImageNet 预训练的 VSSM**，从随机初始化开始（只打印一行 "Failed loading checkpoint"） | **必须修**：改默认路径或重命名目录。已验证发布的 `best_epoch_rgbt_final.pth` 里 VSSM 权重与预训练权重 **cos=0.9997~1.0000**，说明当初训练时是加载成功的，只是现在路径坏了 |
| P7 | 无 RGB-D 数据 | "D 作为考虑"只能停留在设计层 | 见 §5 阶段 5（可选） |
| P8 | `test()` 里 `(res - res.min())/(res.max()-res.min())` 逐图 min-max | 破坏阈值语义，但对 rank-based 指标影响小 | 保留，但记录 |

---

## 2. 运行环境方案

### 2.1 约束推导

- GPU 是 **RTX 5060 Ti (sm_120, Blackwell)** → 必须 torch ≥ 2.7 + cu128。现有 `dformer` 环境（torch 2.1.2+cu121）**彻底不可用**（实测 `no kernel image is available`）。
- `Vmamba/models/encoders/vmamba.py:35` 是**无保护**的 `import selective_scan_cuda_core` → 必须从 `Vmamba/models/encoders/selective_scan` 源码编译，且要针对 sm_120 重新编译（仓库里 `.so` 是 cp310 + 老架构）。
- 原 wheel `mamba_ssm-2.2.4+cu12torch2.2...whl` 对应 torch 2.2，**不能用**（ABI 不匹配）。
- WSL 内存只有 15 GB → 训练 batch 需要保守（GPU 16 GB 也不宽裕）。

### 2.2 方案：conda env `atf`（**已建成**）

| 组件 | 版本 | 理由 |
|---|---|---|
| Python | 3.10.19 | 与 `selective_scan/build/lib.linux-x86_64-cpython-310` 一致 |
| PyTorch | **2.9.1 + cu128** | 实测 sm_120 可跑 matmul |
| torchvision | 0.24.1 + cu128 | 配套 |
| numpy | 1.26.4 | 老代码（`metric.py`）兼容性 |
| timm | **0.9.16** | 代码用 `timm.models.layers.{DropPath,to_2tuple,trunc_normal_}`，timm 1.x 已移除 |
| hydra-core / omegaconf | 1.3.2 / 2.3.0 | `build_sam2` 用 hydra compose |
| 其他 | einops, fvcore, opencv-python-headless(4.10), scikit-image, scipy, matplotlib, openai, pydantic | 已验证可 import |
| CUDA toolkit | 需 **12.8** 的 nvcc（系统现有 13.0，见 §2.4 的坑） | 编译 selective_scan |

### 2.3 实际安装路径（踩坑记录，脚本见 `scripts/setup_env.sh`）

三条路都试过，最终走第 3 条：

1. `conda create -n atf python=3.10` → **失败**：`repo.anaconda.com` 与 `conda.anaconda.org` 的 repodata 均超时（本地 pkgs 缓存里其实有 python-3.10）。
2. `pip install torch==2.9.x --index-url …/cu128` → **失败**：索引元数据端点 120 s 超时；wheel 本体直连可达但 `nvidia-*` 依赖（~2.5 GB）在 PyPI 上停滞，走阿里云镜像也只有 ~200 KB/s。
3. **`conda create -n atf --clone rfdetr`**（`rfdetr` 已有 torch 2.9.1+cu128 全栈并在 sm_120 上验证可用）→ 秒级完成，再 `pip -i 阿里云` 补 timm/hydra/fvcore 等小包。**这是本机最快的可行路径。**

```bash
conda create -y -n atf --clone rfdetr        # 复用已验证的 cu128 torch 栈
conda activate atf
pip install -i https://mirrors.aliyun.com/pypi/simple/ \
  "numpy==1.26.4" "timm==0.9.16" einops fvcore \
  "hydra-core==1.3.2" "omegaconf==2.3.0" scikit-image openai pydantic
python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability(0))"
```

> 代价：env 里残留 rfdetr 的无关包（ultralytics 等），但 `docs/env_versions.txt` 已完整记录，可复现性不受影响。

### 2.4 编译 selective_scan（**已完成，2026-10-28**）

实际跑通的路径（**注意：pip 的 nvidia-cuda-nvcc-cu12 里没有 nvcc 二进制，只有 ptxas，所以合成 CUDA_HOME 这条路走不通**）：

```bash
# 1) 通过 Clash 代理装 CUDA 12.8 的 nvcc + cudart/cuSPARSE 头文件（不需要 4GB 全套）
export http_proxy=http://127.0.0.1:7890 https_proxy=$http_proxy
curl -fsSL -o /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb && apt-get update
apt-get install -y cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8 \
                   libcusparse-dev-12-8 libcublas-dev-12-8 libcusolver-dev-12-8

# 2) 关键补丁：setup.py 原本硬编码 sm_70/80/90，必须加 sm_120
#    Vmamba/models/encoders/selective_scan/setup.py
#    if bare_metal_version >= Version("12.8"):
#        cc_flag += ["-gencode", "arch=compute_120,code=sm_120"]

# 3) 编译
conda activate atf
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8
cd Vmamba/models/encoders/selective_scan && pip install -e . --no-build-isolation --no-deps
python -c "import torch, selective_scan_cuda_core; print('ok')"   # 必须先 import torch
```

**踩坑记录**
| 坑 | 现象 | 解法 |
|---|---|---|
| 系统只有 CUDA 13.0 | `CUDA version (13.0) mismatches ... (12.8)` | apt 装 12.8 |
| `pip nvidia-cuda-nvcc-cu12` | 只有 `bin/ptxas`，**没有 nvcc** | 放弃合成 CUDA_HOME |
| 缺 `cusparse.h` | `CUDAContextLight.h` 编译失败 | 装 `libcusparse-dev-12-8` |
| setup.py 硬编码架构 | 编译出 sm_70/80/90，运行时 `no kernel image is available` | 打补丁加 sm_120 |
| 无 ninja | 回退 distutils，编译极慢 | `pip install ninja` |

**冒烟测试结果**（`scripts/smoke_test.py`，384×384，batch=1）：
`selective_scan OK` → 172.1M 参数 → 前向 **0.69 s**，峰值显存 **1.37 GiB**，四路输出形状与 §1.2 完全一致。

### 2.5 回退方案（如果编译失败）

| 方案 | 做法 | 代价 |
|---|---|---|
| R1 | 把 `vmamba.py` 的 `SelectiveScan` 改为调用 `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`（pip 装 `mamba-ssm`，其对新 torch 支持更好） | 改 ~15 行；需再编译 mamba_ssm |
| R2 | 用 Triton 重写 selective scan（社区有成熟实现） | 1~2 天 |
| R3 | 退化为 `torch.compile` 友好的纯 PyTorch chunked scan | 速度掉 3~5×，只用于正确性验证 |

> **判定标准**：能跑通 `python sam2/sam2_model/build_backbone_ours.py` 且显存 < 12 GB、单步 < 0.6 s，即视为环境合格。

### 2.6 数据 I/O

`/mnt/d` 走 DrvFs，小文件随机读很慢（2500×3 张图 × 30 epoch）。建议：

```bash
mkdir -p ~/data && rsync -a --info=progress2 /mnt/d/workplace/Ours/my_sam_project-main/datasets/ ~/data/datasets/
export SAM2_DATASET_ROOT=$HOME/data/datasets      # dataset_config.py 已支持该环境变量
```

---

## 3. Agent 融合设计（核心）

### 3.1 三条设计原则

1. **Agent 只做"区域级、离散、可解释"的决策**，不产生像素级掩码、不产生连续权重场。它输出的是一张**低分辨率信任图 + 每区一个算子标签**，由网络负责把它渲染成像素级、可微的调制。
2. **Agent 必须在决策环里，而不是在描述环里**。它要能"试运行融合 → 看结果 → 改主意"（`run_fusion` 工具），否则就是给图片打分的伪 Agent。
3. **Agent 不参与训练循环**。所有 LLM 调用都在离线批处理（A 档）或推理期难例（B 档），训练图里永远只有 router 那一层 40k 参数的卷积。

### 3.2 问题重述

> 融合的真实困难不是"怎么加权"，而是：**在哪一块区域、信谁、注多少、边界交给谁**。

- **信谁**（trust）：热成像在"热背景/同温目标"失效，RGB 在"完美伪装"失效——**失效是局部的**，全局标量门控表达不了。
- **注多少**（gain）：SAM 主干是纯 RGB 预训练的，VMamba 融合特征携带热信息。热信息越关键，增量注入应越强；两模态都糊的时候应该**少注入**、退回 SAM 先验。
- **边界交给谁**：热成像分辨率低、边缘糊；RGB 的边界更锐但可能被伪装骗。经验规则是"**位置信热、边界信 RGB**"，这条规则 LLM 有先验，CNN 没有。

### 3.3 决策空间的形式化

Agent（或 router）对每个区域 `r` 输出两个标量：

| 符号 | 含义 | 范围 | 用在哪 |
|---|---|---|---|
| `a(r)` | **thermal trust**：该区域热成像的可信度 | [0,1] | 融合点 1（HFAM logit 偏置） |
| `g(r)` | **injection gain**：VMamba 增量注入强度 | [0,1] | 融合点 2（SAM 适配器门控） |

加一个离散标签用于可解释性与日志：

```
op ∈ {thermal_dominant, rgb_dominant, balanced, rgb_only, thermal_only}
boundary_from ∈ {rgb, thermal, fused}
```

渲成张量：

```
A ∈ [0,1]^{B×1×48×48}      # stride 8，384 输入 → 48×48
G ∈ [0,1]^{B×1×48×48}
```

> 为什么是 48×48（stride 8）而不是 24×24 或逐像素？
> 384 输入下 48×48 ≈ 8×8 像素一格，正好对应"物体的一部分"这种区域粒度；再细就超出 MLLM 的空间分辨能力，再粗就失去"分区"的意义。**作为超参消融 {24, 48, 96}**。

### 3.4 系统总体结构

```
┌─────────────────────────── 离线（A 档，一次性） ───────────────────────────┐
│  训练集子集 → [当前模型前向] → 分歧图 → 提案 K 区 → Agent 辩论/裁决        │
│                              ↓                                            │
│                     信任图 A*, G* 缓存 (.npy)                             │
│                              ↓                                            │
│              Router(f_rgb, f_t) --蒸馏--> 40k 参数，推理零开销             │
└───────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── 在线（B 档，难例） ─────────────────────────────┐
│  pred0 = model(rgb, t)                                                    │
│  s0 = Critic(rgb, t, pred0)          ← 无参考质量评估器（0.5M 参数）        │
│  if s0 < τ_c (≈10% 图像):                                                 │
│      A,G = Agent(rgb, t, 统计, 分歧图, 提案)   ← LLM 调用                  │
│      pred1 = model(rgb, t, trust=(A,G))       ← 同权重，第二次前向          │
│      s1 = Critic(rgb, t, pred1)                                           │
│      pred = pred1 if s1 > s0 + ε else pred0                               │
└───────────────────────────────────────────────────────────────────────────┘
```

**A 档是主表，B 档是卖点。** 论文里 A 档报"零开销 +X"，B 档报"额外 0.35 s 覆盖 9% 图像 +Y"。

### 3.5 两个注入点（最小侵入改造）

#### 注入点 P1：HFAM 的可加 logit 偏置

改造 `HeterogeneousFusionAggregationModule.HFAM.forward(feats, bias=None)`，`bias` 形状 `(B, N, H, W)`（N=2，对应 rgb / thermal）：

```
for l in 1..L:
    logit_k = attn([W_H0, W_Hk])                 # (B,1,H,W)
    if bias is not None:
        logit_0  += λ · log(0.5)                 # hub：中性
        logit_1  += λ · log(1 - a + ε)           # rgb
        logit_2  += λ · log(a + ε)               # thermal
    α = softmax_k(logit)
    ...
```

- `λ` 是**可学习标量，初始化 0** → 第 0 步与现权重逐比特一致，可无损 warm start。
- `a` 从 48×48 双线性上采样到该 stage 分辨率。
- `ε = 1e-3` 防止 log(0)。
- 当 `a = 0.5` 时三项偏置相等 → 等价于无偏置，设计上自洽。

#### 注入点 P2：SAM 适配器的增益门控

改造 `GRCM.SAMFusion.forward(x1, x2, stage_idx=None, gain=None)`：

```python
delta = self.out_conv(fused_feat)
g = 1.0 if gain is None else (0.3 + 0.7 * gain)   # floor 0.3：任何区域都保留保底注入
return delta * g
```

- `gain` 同样从 48×48 上采样；初始 `g ≈ 1`（网络输出 bias 初始化为 +6 → sigmoid≈0.9975），warm start 几乎无损。
- 0.3 的下限是为了防止某区域被完全切断导致梯度消失。

#### 为什么只改这两处，不改解码器

- P1 管"跨模态选谁"，P2 管"跨模态信息注多少"，二者已覆盖融合的全部自由度；
- 解码端 CPD 是单模态聚合，加 Agent 信息收益小、改动大；
- **两处加起来的可训练参数 ≈ 0**（只有 λ 和 gate 的 bias），风险最低。

### 3.6 监督信号：怎么让 trust 不是玄学

#### 3.6.1 先加两个"单模态辅助头"（关键前置）

现有模型**没有单模态分支**，无法知道"哪个模态在哪个区域对"。新增（每分支 ~25k 参数）：

```
f_rgb_s2 = outs_rgb[2]   # 24×24×384
f_t_s2   = outs_e[2]
head: Conv3×3(384→64) → ReLU → Conv1×1(64→1) → 上采样至 384² → logits
```

为此给 `VmambaModel.forward` 加参数 `return_unimodal=True`，顺带返回 `outs_rgb` / `outs_e`。

**这两个头一次投入，三处收益**：
1. 提供 oracle trust 的监督信号；
2. 提供 `L_noharm` 的"best single"参照；
3. 天然就是论文里的单模态基线（RGB-only / T-only），省掉两次单独训练。

#### 3.6.2 Oracle Routing（免人工的信任标签）

对每个像素 `r`：

```
L_rgb(r) = BCE(p_rgb(r), y(r))
L_t(r)   = BCE(p_t(r),   y(r))
a*(r)    = σ( (L_rgb(r) − L_t(r)) / τ_o ),   τ_o = 0.1
a*       = BoxBlur15×15(a*)  →  avgpool 到 48×48
```

含义：RGB 错而热成像对的地方，`a*→1`。这就是 MoE 里的 oracle routing。

**注意**：oracle 依赖两个 aux 头的预测质量，训练早期不可靠。做法——**用 EMA 版本的 aux 头算 oracle**，并且只在前 5 epoch 之后启用 `L_trust`。

#### 3.6.3 损失

```
L = L_sam
  + 0.2·L_det + 0.5·L_pred              # 现有
  + 0.10·(L_aux_rgb + L_aux_t)          # 单模态头
  + 0.20·L_trust                        # 蒸馏（BCE + Dice on A）
  + 0.10·L_noharm                       # 融合不得劣于最优单模态
```

其中

```
L_noharm = mean_r  ReLU( L_fused(r) − min(L_rgb(r), L_t(r)) + 0.05 )
L_trust  = BCE(A, A*) + Dice(A, A*)
```

`L_noharm` 是治"融合反噬"的排序约束，也是审稿人最容易认可的一条设计。

#### 3.6.4 Router（A 档的产物）

```
输入: [f_rgb_s2(384), f_t_s2(384), |f_rgb_s2 − f_t_s2|(384)]  @ stride 16
结构: Conv3×3(1152→64) → BN → GELU → Conv3×3(64→64) → BN → GELU → Conv1×1(64→2)
输出: (A, G) @ 24×24  →  双线性上采样到 48×48
参数: ≈ 75k   耗时: < 2 ms（相对于 4 s 的单图推理可忽略）
```

训练：`L_router = BCE(A, A_agent) + 0.5·BCE(G, G_agent) + 0.1·TV(A)`（TV 保证区域平滑）。

### 3.7 Critic：无参考质量评估器

**没有它，B 档的闭环就是装饰**（测试时没有 GT，Agent 无法知道自己改对没有）。

```
输入: concat(rgb, t, pred)  @ 192×192，7 通道
结构: Conv7×7/2(7→32) → [Conv3×3 → BN → ReLU → MaxPool] ×4 → GAP → FC(→1) → σ
参数: ≈ 0.55 M
训练: L1 回归真实 IoU（训练集 + 现权重前向即可产出，无需额外标注，一天内可完成）
指标: 与真实 IoU 的 SRCC（目标 > 0.85）、PLCC
```

三处用途：**触发条件**（`s0 < τ_c`）、**接受判据**（`s1 > s0 + ε`）、**早停**。

### 3.8 Agent 的四个模块

#### 3.8.1 State：喂什么给 LLM

| 证据 | 形式 | 说明 |
|---|---|---|
| 数值统计表 | 文本 | 每区的对比度 / 梯度能量 / 局部熵 / 跨模态一致性 / 单模态置信度 |
| 分歧图 | 伪彩色 PNG | `|p_rgb − p_t|`，让模型"看见哪里吵起来了" |
| 单模态预测 | 两张伪彩色 PNG | 各自的显著性图，半透明叠加在原图上 |
| 局部裁剪 | 并排 PNG | RGB / T 的放大裁剪（这是绕开 MLLM 空间定位弱的关键） |

统计表示例（程序生成，不经过 LLM）：

```
Region#2  box=[312,180,470,356]  area=158x176
  RGB     : contrast 0.41  grad 0.18  entropy 6.9  (texture ≈ background)
  Thermal : dT(in,out)=6.2C  edge_sharpness 0.31
  Depth   : n/a
  Agreement: 0.42  (high disagreement)
  Aux pred: RGB IoU_proxy 0.51   T IoU_proxy 0.78
```

#### 3.8.2 Proposal：把"整图判断"降成"3~5 个局部判断"

```
dis  = |p_rgb − p_t|                    # 或 |f_rgb − f_t| 的通道均值
dis  = BoxBlur(dis, 9)
regions = top-K connected components of (dis > quantile(dis, 0.85))
K ≤ 5, 每区外扩 15% 作为 crop
```

好处：LLM 调用量 O(5)；每区可给放大图；失败的 case 能在论文里画出来。

#### 3.8.3 Tools：没有工具就不是 Agent

| 工具 | 签名 | 作用 |
|---|---|---|
| `run_fusion` | `(region, a, g) → map_stats` | **试运行**：真的把 `(A,G)` 注入跑一次前向（用小分辨率或单区近似），返回新预测的统计量 |
| `zoom` | `(region, modality, scale) → image` | 主动取证据 |
| `compare` | `(map_a, map_b, region) → {IoU, boundary_F}` | 比较两次融合 |
| `query_prior` | `(scene_tag) → text` | 检索场景先验（沙地/雪地/水下/夜间/室内热源） |
| `recall` | `(region_feat) → [cases]` | 从缓存里检索相似区域的历史裁决（可选） |

> `run_fusion` 是本设计与"LLM 打分"的分水岭。实现上不必真的跑全模型：把 `(A,G)` 限制在该区、其余区沿用 router 的输出，做一次 192×192 的轻量前向即可（约 0.4 s）。

#### 3.8.4 Debate：只在"高分歧 + 低分"区域开

```
Affirmative(RGB)  : "边界清晰，热成像在此区 ΔT 仅 0.8°C，不可信"
Negative(Thermal) : "RGB 与背景纹理同化，热成像显示目标整体连通，位置可信"
Judge             : → {"trust_thermal":0.35, "inject_gain":0.7, "boundary_from":"rgb", ...}
```

三人、固定顺序、**裁判只输出 JSON**（不输出自由文本）。轮数上限写死为 2 轮（论文必须注明）。

### 3.9 Prompt 与输出契约

**System / Meta Prompt（英文，Qwen2.5-VL 可直接用）**

```text
You are an expert in RGB-T (visible–thermal) camouflaged/salient object detection,
acting as a fusion director. For each proposed region you must decide how much to
trust the thermal modality and how strongly to inject cross-modal features.

Domain priors you must apply:
- Thermal gives LOCATION but not BOUNDARY: low resolution, soft edges.
- Thermal FAILS when: background is hot (sun-heated rock/sand, fire, engines),
  target and background are iso-thermal, or the scene is indoors with heaters.
- RGB gives TEXTURE and BOUNDARY, but FAILS when camouflage is near-perfect.
- Depth (if present) gives structure, but FAILS on glass, water, mirrors, and far range.

Rules:
- Never agree just to be agreeable; argue from the evidence given.
- Ground every claim in the numeric statistics or in what you can see in the crops.
- Output ONLY the JSON object, no prose.

Output schema:
{"regions":[{"id":int,"box":[x1,y1,x2,y2],
             "trust_thermal":float in [0,1],
             "inject_gain":float in [0,1],
             "boundary_from":"rgb"|"thermal"|"fused",
             "op":"thermal_dominant"|"rgb_dominant"|"balanced"|"rgb_only"|"thermal_only",
             "evidence":str (<=30 words),
             "confidence":float in [0,1]}],
 "global":{"trust_thermal":float,"inject_gain":float}}
```

**Json Schema（供程序校验，pydantic）**

```python
class RegionDecision(BaseModel):
    id: int
    box: tuple[float, float, float, float]
    trust_thermal: float = Field(ge=0, le=1)
    inject_gain: float = Field(ge=0, le=1)
    boundary_from: Literal["rgb", "thermal", "fused"] = "rgb"
    op: Literal["thermal_dominant","rgb_dominant","balanced","rgb_only","thermal_only"]
    evidence: str
    confidence: float = Field(ge=0, le=1)

class ImageDecision(BaseModel):
    regions: list[RegionDecision]
    global_: dict = Field(alias="global")
```

**校验与兜底**：解析失败 → 重试 1 次 → 仍失败则回退到 router 输出；所有原始响应落盘 `cache/raw/{image_id}.json`。

### 3.10 A 档流水线（离线蒸馏，零推理开销）

```
输入：训练集（或按分歧度采样 top 40%，约 1000 张）
for img in subset:
    with torch.no_grad():
        pred, aux, f = model(img, t, return_aux=True)
    dis      = |aux_rgb − aux_t|
    regions  = propose(dis, K=3)
    state    = build_state(img, t, aux, regions)      # 统计 + 3 张图
    decision = agent.decide(state)                     # 1 次 LLM（含辩论则 6 次）
    A, G     = render(decision, size=48)               # 区域 → 稠密
    np.save(f"cache/trust/{img_id}.npy", stack([A, G]).half())

# 蒸馏（10 epoch，冻结其他所有参数）
for batch in loader:
    A_pred, G_pred = router(f_rgb_s2, f_t_s2)
    loss = BCE(A_pred, A_gt) + 0.5*BCE(G_pred, G_gt) + 0.1*TV(A_pred)
```

**产出**：`router` 权重（~300 KB）+ 一张"LLM 标注的信任图数据集"（本身可作为论文附录的贡献）。

**成本控制**：K=3、无辩论 → 3 次调用/图；1000 张 → 3000 次调用。用 Qwen2.5-VL-3B 批处理或 API，一晚跑完。

### 3.11 B 档流水线（在线难例闭环）

```python
def infer(rgb, t):
    pred0, aux = model(rgb, t, trust=None)          # router 自动生效
    s0 = critic(rgb, t, pred0)
    if s0 >= TAU_C:                                  # 90% 的图像走这里
        return pred0

    regions = propose(|aux_rgb - aux_t|, K=3)
    dec     = agent.decide(build_state(rgb, t, aux, regions))
    A, G    = render(dec, 48)
    pred1   = model(rgb, t, trust=(A, G))
    s1      = critic(rgb, t, pred1)
    return pred1 if s1 > s0 + 1e-3 else pred0
```

**预算**：单张触发图像 +1 次全模型前向（~4 s 中约 +0.35 s，因为可复用 VSSM/Hiera 的特征缓存）+ 3~6 次 LLM 调用。

### 3.12 与现有代码的接口（最终形态）

```python
# build_backbone_ours.Model.forward
def forward(self, rgb, modality=None, trust=None, return_intermediates=False):
    ...
    vssm = self.vmamba(rgb, modality, trust=trust)      # P1 在 dual_vmamba 内部生效
    ...
    adapter_output = self.stage_adapters[i](rgb_nchw, vssm[i], i,
                                            gain=gain_i)  # P2
```

`trust` 为 `None` 时自动使用 router 输出 → **训练与 A 档推理走同一条路径，不需要分支**。

---

## 4. 训练与评测协议

### 4.1 四阶段训练计划

| 阶段 | 初始化 | 可训练 | 数据 | 目标 | 预计耗时（5060 Ti） |
|---|---|---|---|---|---|
| **S0 复现** | `best_epoch_rgbt_final.pth` | 现状 | VT5000 2500 | 对齐已报告 S-measure | 30 ep × ~12 min ≈ 6 h |
| **S1 接口** | S0 | +aux 头 +router +λ+gate | 同上 | 指标不掉（ΔS ≥ −0.2） | 10 ep ≈ 2 h |
| **S2 蒸馏** | S1 | 只训 router | 1000 张缓存 | `L_router` 收敛 | 10 ep ≈ 20 min |
| **S3 端到端** | S2 | 全部（除冻结主干） | 2500 | 主结果 | 20 ep ≈ 4 h |

> S1 的验收是**接口无损**：`λ=0`、`gate≈1` 时输出与 S0 的 L1 差 < 1e-4。

### 4.2 消融矩阵（论文 Table 2 的骨架）

| # | 配置 | 目的 |
|---|---|---|
| 1 | Baseline（复现） | 下界 |
| 2 | +aux 头（不加 trust） | 控制变量：确认增益不是来自辅助监督 |
| 3 | +router（端到端随机初始化） | 证明"数据驱动也能学，但不如 LLM 先验" |
| 4 | **+router（A 档蒸馏）** | **主结果（零开销）** |
| 5 | +oracle trust（用 GT 算 a\*） | 上界 |
| 6 | **+Agent 在线（B 档）** | **主结果（难例）** |
| 7 | B 档去掉 debate（单 Agent） | 辩论的净贡献 |
| 8 | B 档去掉 `run_fusion` 工具 | 工具调用的净贡献 |
| 9 | 超参：K ∈ {1,3,5}、分辨率 ∈ {24,48,96}、τ_c、τ_o | 敏感性 |
| 10 | 退化鲁棒性：T 加噪/降分辨率/温度压缩、RGB 亮度扰动 | 实用性 |

### 4.3 基线实测（2026-10-28，`best_epoch_rgbt_final.pth`，384，无 TTA）

**与官方口径完全一致**（`output_maps.py` → PNG → `evaluation_metric`；已用 `scripts/verify_official_eval.py` 逐数据集核对）：

| 数据集 | S↑ | MAE↓ | E(adp)↑ | Fw↑ | Fβ(adp)↑ | 耗时 |
|---|---|---|---|---|---|---|
| VT5000 (2500) | **0.9347** | 0.0184 | 0.9655 | 0.9076 | 0.9106 | 365 s |
| VT1000 (1000) | **0.9544** | 0.0124 | 0.9773 | 0.9394 | 0.9337 | 149 s |
| VT821 (821) | **0.9462** | 0.0137 | 0.9681 | 0.9211 | 0.9162 | 123 s |

> ⚠️ **评测陷阱（已踩，务必记住）**：`metric/utils.prepare_data` 会对 pred 再做一次 min-max，并给 E-measure 用 `2*mean` 自适应阈值。结果是 **uint8 量化时用 `np.round` 还是截断，会让 adpE 差 ~0.6、adpF 差 ~1.1**（S/MAE 几乎不变）。
> 实测（VT821，同一权重、同一 GT，仅差 ±1 灰度）：`round → adpE 0.9681 / adpF 0.9161`，`truncate → adpE 0.9621 / adpF 0.9048`。
> **结论：必须 `np.round(res*255)`，与 `output_maps.py` 一模一样；任何自写评测都要先和官方脚本对齐到小数点后三位。**

**关于 TTA**：早先用截断口径跑出的 TTA 表（E/Fβ 大涨）已被证实是量化伪影，**整表作废，需用 `--tta` + 新口径重跑**；S/MAE 的相对增益仍可信（+0.12~0.24 S）。

**这张表改变了后续所有判断，务必先看：**

1. **模型已经很强**。VT5000 S=0.9347、E=0.9655，属于 RGB-T SOD 第一梯队。**"提分辨率再涨 1~1.5 个 S"不成立**——VT5000 的 S 天花板约 0.945，可用空间只剩 1 个点。所有"杠杆"的真实预期收益都应重估为 **+0.2 ~ 0.5 S**。
2. **短板在 MAE 与边界**：VT821 明显低于 VT1000；Fβ 与 Fw 接近说明细长/小目标仍掉分。**要提分应打边界与难例，而不是打平均 S。**
3. **VT5000 是三者中最低的**，因为它规模最大、以视频帧为主、小目标最多；VT1000 的 0.9544 可视为该权重的乐观上界。
4. **风险提示（审稿人会问）**：`train.py` 每个 epoch 在**测试集**上评 S 并存 best，属于 *test-set model selection*。补救方式（三选一）：(a) 从 VT5000-Train 切 10% 做 val 重新选 epoch；(b) 报告固定最后一轮；(c) 明确声明，并用未参与选择的 VT1000/VT821 作旁证。

### 4.3b 探针 A：模型到底用不用热红外？（2026-10-29，`analysis/robustness.py`）

把热红外张量整体置零（等于"没有热模态"）后重测：

| 数据集 | 完整 | 去掉热红外 | ΔS | ΔMAE |
|---|---|---|---|---|
| VT5000/Test (2500, 全量) | 0.9347 | **0.9262** | **−0.0085** | +0.0026 |
| VT821 (400 抽样) | 0.9425 | 0.9323 | −0.0102 | +0.0033 |

**整个热红外模态只值 0.85~1.0 个 S-measure 点。** 三个对照（全零 / 常数 0.5 / 用 RGB 冒充 T）结果几乎相同（0.9323 / 0.9321 / 0.9305），说明模型对"T 里有什么"基本不敏感——它更接近一个**带装饰性热分支的 RGB 显著目标检测器**。

退化曲线（VT821，400 张）：

| 退化 | ΔS | 解读 |
|---|---|---|
| blur σ=2 | −0.0055 | 轻微模糊尚可 |
| **blur σ=5** | **−0.0104** | **掉到和"没有 T"一样多** → 只用到 T 的最低频信息 |
| contrast ×0.5 | −0.0012 | 温差变小几乎无影响（因为本来也没用多少） |
| quantize 8 bit | −0.0052 | 量化损失明显 |
| **shift 4px（错位）** | **−0.0030** | **错位几乎不疼** → 进一步证明跨模态对齐/融合没在工作 |
| noise σ=0.10 | −0.0036 | — |
| dropout 30% | −0.0031 | — |

> 结论：**当前融合是"弱融合"**。这不是坏消息，而是把论文的靶子换了——见 §4.3c。

### 4.3c 探针 A：错误集中在哪？（`analysis/hard_subset.py`，VT821 全量 821 张）

| 指标 | 值 |
|---|---|
| mean S / MAE | 0.9462 / 0.0133 |
| **最差 10% 图像占总误差** | **51.6%** |
| 小目标（面积 < 3%，123 张 = 15%） | S = 0.866 ~ 0.918（大目标是 0.954） |
| 大目标（面积 > 10%，391 张） | S = 0.954，但因为面积大占了 60.8% 的 MAE |

**两条硬结论：**
1. **误差极度集中**——修好最差的 10% 图像，等于修好一半的 MAE。这正是 Agent 该打的地方（难例闭环），而不是去优化平均 S。
2. **小目标质量最差但误差占比小**（MAE 按面积加权）。所以论文里"难例提升"要用 **S/E/F 或子集指标**来报，不能用全集 MAE。

**由此改写论文的指标设计（重要）**：

| 指标 | 现状 | 目标 | 说明 |
|---|---|---|---|
| 全集 S / MAE | 0.9347 / 0.0184 | +0.2~0.4 | 空间有限 |
| **热依赖度 ΔS(T→0)** | **0.0085** | **≥ 0.030** | 证明融合真的在起作用 |
| **最差 10% 子集 S** | ~0.80（待测） | +0.05 | 证明 Agent 救的是难例 |
| 退化鲁棒性（blur5 / shift4） | −0.010 / −0.003 | 减半 | 实用性 |

> 结论：**Agent 的目标不是"再涨 0.3 个平均 S"，而是"让热红外真正被用起来 + 救回最差的那 10%"。** 这两条都可量化、都可被审稿人验证，而且现在的基线给了巨大的对照空间（0.0085 → 0.03 是 3.5 倍）。

### 4.3d 探针 A：逐图热依赖（`analysis/thermal_critical.py`，VT821 821 张）

对每张图分别用「真实 T」和「T 置零」评分，得到逐图 ΔS：

| 统计 | 值 |
|---|---|
| mean S：full / zero | 0.9462 / 0.9354 |
| **逐图 oracle 切换（max(full, zero)）** | **0.9491 → 只比 full 高 +0.0029** |
| ΔS > 0.05 | 48 张（5.8%），平均 **+0.19** |
| ΔS > 0.20 | 17 张（2.1%），平均 **+0.36** |
| **ΔS < 0（T 反而有害）** | **381 张（46.4%），平均 −0.006** |
| corr(ΔS, 面积/热图均值/热图对比度/RGB-T 相关) | +0.02 / −0.02 / +0.03 / +0.01（**全部 ≈ 0**） |

**三条硬结论，直接决定了方案取舍：**

1. **"逐图选模态"这条路已经死了。** 完美 oracle 只能拿 **+0.0029**——因为 `max()` 只回收"T 有害"的那 46%，而"T 有用"的 6% 本来就在 full 里。**任何 A 档 image-level router 的天花板 = +0.3 个 S 点，不值得做。**
2. **信任分配必须下沉到区域级**（文档 §3.3 的 48×48 信任图方向是对的），而且真正的价值不在"关掉 T"，而在"**在被 T 救的那 6% 里把 T 用好**"。
3. **廉价统计量完全无法预测 ΔS**（四个相关系数都 ≈0）。这意味着 router 不能靠"热图亮度/对比度/面积"这类浅层特征学出来，只能靠**语义**——这恰好是 LLM 的用武之地，也是把 Agent 留在方案里的唯一硬理由。

> **重新定位**：Agent 不应做"运行时逐图门控"，而应做 **① 区域级信任（P1/P2 注入）** 和 **② 离线挖掘"热关键样本"来重训融合**（agent-as-data-engine）。前者有 ~1 个点的上限，后者能直接抬高 ΔS(T→0)。

### 4.4 评测口径

- **训练期**：`train.py::test()` 保持只算 S-measure（快，只用于选 epoch），**不改**。
- **最终**：用根目录 `eval.py`（独立脚本，加载 .pth，出 S/MAE/E/Fw/Fβ 全套 + JSON），支持 `--tta`、`--limit`、`--save-pred`。
  ```bash
  python eval.py -c res_checkpoints/final_pth/best_epoch_rgbt_final.pth \
                 --datasets VT5000 VT1000 VT821 --json results/xxx.json
  ```

### 4.5 必须准备的对照数字

| 对照 | 来源 |
|---|---|
| RGB-only / T-only | 新增的 aux 头（免费） |
| 全局门控（learnable scalar per modality） | 加 5 行代码，证明"区域级"比"全局"强 |
| 逐像素 sigmoid 门控（无区域约束） | 证明"区域一致"比"逐像素"稳 |
| Oracle | 上界 |

---

## 5. 工程实施计划

### 5.1 里程碑

| M | 内容 | 交付物 | 验收 |
|---|---|---|---|
| **M0** | `git init` + 环境 + 编译 selective_scan + 跑通一次前向 | `requirements-atf.txt`、`ENV.md` | `python sam2/sam2_model/build_backbone_ours.py` 输出四种张量形状且显存 < 12 GB |
| **M1** | 复现基线（S0），补全 4 指标评测 | `eval.py`、基线表格 | VT5000 S-measure 与已有权重一致（±0.3） |
| **M2** | 注入接口 P1/P2 + router + aux 头（S1） | `agent/fusion_bias.py`、`agent/router.py`、`agent/aux_head.py` | 接口无损（L1 < 1e-4），指标不掉 |
| **M3** | Oracle + `L_trust` + `L_noharm`（S3） | `agent/oracle.py` | 消融 #5 出数，oracle 明显高于 router |
| **M4** | Agent 离线批处理 + 缓存（A 档） | `agent/{client,state,proposal,tools,debate,render}.py`、`cache/` | 1000 张跑完，JSON 解析成功率 > 98% |
| **M5** | 蒸馏（S2）+ 主结果 | `router` 权重 | 消融 #4 出数 |
| **M6** | Critic + 在线闭环（B 档） | `agent/critic.py`、`infer_agent.py` | 触发率 ≈ 10%，SRCC > 0.85 |
| **M7** | 论文材料：可视化、效率表、失败案例 | 图表 | — |
| **M8**（可选） | RGB-D 适配 | `datasets/RGB-D`、prompt 分支 | NJU2K/NLPR 上跑通 |

### 5.2 文件清单（新增）

```
agent/
├── __init__.py
├── schema.py        # pydantic 输出契约 + 校验/兜底
├── state.py         # 质量统计（对比度/梯度/熵/一致性）
├── proposal.py      # 分歧图 → 候选区
├── render.py        # 区域决策 → 48×48 (A,G) 稠密图（含羽化）
├── tools.py         # run_fusion / zoom / compare / query_prior
├── client.py        # Qwen2.5-VL 调用 + 重试 + 磁盘缓存
├── debate.py        # 辩手 + 裁判编排
├── oracle.py        # oracle routing（EMA aux 头）
├── router.py        # 75k 参数轻量 router
├── aux_head.py      # 单模态辅助头
├── critic.py        # 无参考质量评估器
├── cache.py         # .npy 缓存读写
└── prompts/
    ├── system.md
    ├── debater_aff.md
    ├── debater_neg.md
    └── judge.md

scripts/
├── setup_env.sh
├── build_selective_scan.sh
├── precompute_trust.py     # A 档批处理
├── train_router.py         # 蒸馏
├── train_critic.py
├── eval.py                 # 4 指标
└── infer_agent.py          # B 档推理

docs/
├── AGENT_FUSION_DESIGN.md  # 本文档
└── ENV.md
```

### 5.3 改动清单（现有文件）

| 文件 | 改动 | 风险 |
|---|---|---|
| `HeterogeneousFusionAggregationModule.py` | `HFAM.forward` 加 `bias=None` 参数 | 低（默认行为不变） |
| `GRCM.py` | `SAMFusion.forward` 加 `gain=None` | 低 |
| `dual_vmamba.py` | `RGBXMamba.forward(x_rgb, x_e, trust=None)` 把 `a` 下发给 4 个 HFAM | 中 |
| `Vmamba_encoder_dual.py` | 透传 `trust`，可选返回单模态特征 | 低 |
| `build_backbone_ours.py` | `Model.forward(..., trust=None)`；实例化 router/aux；`stage_adapters` 传 gain | 中 |
| `train.py` | 加 aux / trust / noharm 三项损失与权重 | 中 |
| `option.py` | 新增 `--use_agent`, `--tau_c`, `--trust_size`, `--w_trust`, `--w_noharm` 等 | 低 |
| `metric` 使用 | `test()` 扩展到 4 指标 | 低 |

### 5.4 命名与约定

- 权重文件：`res_checkpoints/atf/{stage}_{dataset}_{metric}_{epoch}.pth`
- 缓存：`cache/trust/{dataset}/{image_id}.npy`（float16，2×48×48）
- 日志：`runs/{exp_name}/`，配置与 git commit 一起落盘
- 所有 LLM 原始响应：`cache/raw/{image_id}.json`（**必须保留，便于复现与 rebuttal**）

---

## 6. 风险登记表

| R | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | `selective_scan` 无法在 torch 2.9 + sm_120 上编译 | **高** | 阻断 | 回退方案 R1/R2/R3（§2.5）；预留 1 天 |
| R2 | 16 GB 显存 + 15 GB 内存导致 OOM | 中 | 阻断 | batch 2 + 梯度累积；`torch.cuda.empty_cache()`；DataLoader worker=2 |
| R3 | Agent 输出 JSON 不稳定 | 中 | 噪声 | pydantic 校验 + 重试 1 次 + router 兜底 |
| R4 | LLM 成本超预算 | 中 | 延期 | 只跑 1000 张 + 无辩论档先出数；辩论只用于 top-10% 高分歧 |
| R5 | 增益不显著（< 0.3 S-measure） | **中高** | 论文失败 | 保留 oracle 上界作为"还有空间"的证据；退而做"LLM 标注的信任数据集 + 蒸馏"叙事情 |
| R6 | 审稿人质疑"就是加权，不是 Agent" | 中 | 拒稿 | 必须有 `run_fusion` 试运行 + 辩论消融（#7/#8） |
| R7 | 训练不稳定（加了 bias 后） | 低 | 返工 | λ 零初始化 + 5 epoch 后才开 `L_trust` |
| R8 | 与现有 SOTA 不可比（只有 VT5000 自测） | 高 | 拒稿 | 尽快补 VT1000/VT821 与 RGB-D（M8） |

---

## 7. 附：默认超参

| 参数 | 默认值 | 说明 |
|---|---|---|
| `trust_size` | 48 | 信任图分辨率 |
| `K` | 3 | 提案区数量（A 档）/ 3（B 档） |
| `tau_o` | 0.1 | oracle 温度 |
| `tau_c` | 0.75 | Critic 触发阈值 → 目标触发率 8~12% |
| `lambda_bias` | 0.0（可学习，初值 0） | HFAM 偏置强度 |
| `gate_floor` | 0.3 | 注入增益下限 |
| `w_aux` | 0.10 | 单模态头损失权重 |
| `w_trust` | 0.20 | 蒸馏损失权重 |
| `w_noharm` | 0.10 | 排序约束权重 |
| `debate_rounds` | 2 | 辩论轮数上限 |
| `llm` | Qwen2.5-VL-7B（离线）/ 3B（在线） | 与已有两篇参考文献一致 |

---

## 8. 一句话总结

**把"跨模态融合"从"网络自己学一个逐像素权重"重新表述为"Agent 做区域级的模态信任分配"，用 oracle routing 提供训练期监督、用 LLM 提供测试期的无 GT 推断，最后蒸馏成一个 75k 参数的 router，使推理零开销；难例再用无参考质量评估器触发一次 Agent 闭环。**
