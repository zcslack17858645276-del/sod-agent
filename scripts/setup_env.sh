#!/usr/bin/env bash
# Create the `atf` conda env for this project.
#
# 实测结论（本机 2026-10）：
#   1) defaults / conda-forge 的 repodata 都会超时 -> 不要指望在线 conda create
#   2) 从 PyPI 直连装 torch 的 nvidia-* 依赖会卡死；阿里云镜像也只有 ~200 KB/s
#   3) 最快路径：克隆本机已有的 rfdetr env（torch 2.9.1+cu128 已在 sm_120 验证可用）
#
# 用法：bash scripts/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONDA_BIN="$CONDA_DIR/bin/conda"
MIRROR="${MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "[1/4] create env atf"
conda env remove -y -n atf >/dev/null 2>&1 || true
if conda env list | grep -qE '^rfdetr\s'; then
  echo "  >> cloning rfdetr (has torch 2.9.1+cu128)"
  conda create -y -q -n atf --clone rfdetr
else
  echo "  >> no rfdetr to clone; falling back to offline create (needs cached python 3.10)"
  conda create -y -q -n atf --offline python=3.10 pip
fi
conda activate atf
python -V

echo "[2/4] project deps (aliyun mirror)"
pip install --no-cache-dir --timeout 60 --retries 3 -i "$MIRROR" --trusted-host mirrors.aliyun.com \
  "numpy==1.26.4" "timm==0.9.16" einops fvcore \
  "hydra-core==1.3.2" "omegaconf==2.3.0" \
  "opencv-python-headless" scikit-image scipy tqdm pyyaml matplotlib openai pydantic

echo "[3/4] verify"
python - <<'PY'
import torch, torchvision, timm
print("torch", torch.__version__, "tv", torchvision.__version__, "timm", timm.__version__)
print("cap", torch.cuda.get_device_capability(0), "avail", torch.cuda.is_available())
print("matmul ok:", float((torch.randn(512,512,device="cuda") @ torch.randn(512,512,device="cuda")).sum()))
PY

echo "[4/4] freeze versions"
pip freeze > "$REPO_ROOT/docs/env_versions.txt" || true
echo "done. next: bash scripts/build_selective_scan.sh 12.0"
