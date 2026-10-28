#!/usr/bin/env bash
# Build the VSSM `selective_scan_cuda_core` CUDA extension for the current GPU.
#
# Requires an active conda env with torch installed AND a CUDA toolkit providing nvcc:
#   conda install -y -c nvidia cuda-toolkit=12.8
#
# Usage:
#   bash scripts/build_selective_scan.sh [arch]
#   bash scripts/build_selective_scan.sh 12.0
set -euo pipefail

ARCH="${1:-12.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/Vmamba/models/encoders/selective_scan"

command -v nvcc >/dev/null 2>&1 || { echo "nvcc not found; run: conda install -y -c nvidia cuda-toolkit=12.8"; exit 1; }

export TORCH_CUDA_ARCH_LIST="$ARCH"
export MAX_JOBS="${MAX_JOBS:-8}"

cd "$SRC"
echo "building selective_scan for arch=$ARCH with $(nvcc --version | tail -1)"
pip install -e . --no-build-isolation

python - <<'PY'
import selective_scan_cuda_core as ext
print("selective_scan_cuda_core OK:", ext.__file__)
PY
