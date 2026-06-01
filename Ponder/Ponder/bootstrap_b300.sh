#!/usr/bin/env bash
# bootstrap_b300.sh — fresh Blackwell(B300/B200) 박스에서 PonderNet 본 학습 한 방 실행.
#
#   git clone -b Ponder https://github.com/34-123hs/AMOE.git
#   cd AMOE/Ponder
#   # (먼저 train.bin / val.bin 을 이 폴더에 올려둘 것 — 아래 '데이터' 항목 참고)
#   export WANDB_API_KEY=<key>        # 또는: python -m wandb login <key>
#   bash bootstrap_b300.sh
#
# 하는 일: 캐시를 작업폴더로(작은 /tmp 회피) → 의존성 설치 → Blackwell 커널/ bf16
# matmul 동작 검증 → 데이터 확인 → wandb 인증 → run_final.sh(본 학습) 실행.
# 모델/학습 수학은 H100 런과 100% 동일(bf16). Blackwell은 그냥 더 빠른 bf16일 뿐.

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY=$(command -v python3 || command -v python || echo /opt/conda/bin/python)
export PY
export TMPDIR="$HERE/tmp" TORCHINDUCTOR_CACHE_DIR="$HERE/.inductor_cache"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "[py] $PY"

echo "### 1) deps (Blackwell 은 torch>=2.12 + CUDA 12.8/13 wheel 필요) ###"
"$PY" -m pip install --quiet --disable-pip-version-check \
    "torch>=2.12" torchvision transformers accelerate tiktoken einops safetensors wandb \
    || { echo "FATAL: pip install 실패"; exit 1; }

echo "### 2) GPU / Blackwell 커널 검증 ###"
"$PY" - <<'PYEOF' || { echo "FATAL: GPU/torch 불일치 — Blackwell 커널 없음일 수 있음. torch nightly(cu130) 시도."; exit 1; }
import torch
assert torch.cuda.is_available(), "CUDA 안 보임"
cc = torch.cuda.get_device_capability(0)
print("GPU:", torch.cuda.get_device_name(0), "| sm", cc, "| torch", torch.__version__)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()          # Blackwell sm 커널이 없으면 여기서 에러
print("bf16 matmul OK")
PYEOF

echo "### 3) 데이터 확인 (train.bin 3.8G / val.bin 39M) ###"
{ [ -s train.bin ] && [ -s val.bin ]; } || {
  echo "FATAL: train.bin / val.bin 이 $HERE 에 없음.";
  echo "  → 현재 H100 박스에서 옮기거나(rsync/scp) 재토크나이즈해서 이 폴더에 두세요.";
  exit 1; }
ls -lhL train.bin val.bin

echo "### 4) wandb 인증 ###"
"$PY" -c "import wandb; print('wandb user:', wandb.Api().viewer.username)" \
    || { echo "FATAL: wandb 미인증 → $PY -m wandb login <KEY>  (또는 export WANDB_API_KEY)"; exit 1; }

echo "### 5) 본 학습 실행 (bf16 + compile, ~B300 기준 H100의 ~2.5-3.5배 속도) ###"
PY="$PY" bash run_final.sh
echo "DONE. tail -f $HERE/final_run.log"
