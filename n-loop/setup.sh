#!/usr/bin/env bash
# setup.sh — venv 생성 + 의존성 설치 + 검증.
#   bash setup.sh             # ./.venv 에 설치
#   bash setup.sh ../amoe-venv # 다른 경로에 설치
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV="${1:-.venv}"
PY="$VENV/bin/python"

# 1) venv (없으면 생성)
[ -x "$PY" ] || { echo "[setup] venv 생성: $VENV"; python3 -m venv "$VENV"; }
"$PY" -m pip install -U pip

# 2) 의존성 (핀 고정). torch는 CUDA 13.0 휠 — CPU/다른 CUDA면 이 줄만 환경에 맞게 교체.
"$PY" -m pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu130
"$PY" -m pip install \
  "transformers==5.9.0" \
  "accelerate==1.13.0" \
  "wandb==0.27.0" \
  "tiktoken==0.13.0" \
  "numpy==2.4.6" \
  einops
# muon은 로컬 muon.py로 제공 → pip 설치 안 함

# 3) 검증 (import + CUDA + 로컬 muon) — 레포 루트에서 실행해야 muon import 됨
"$PY" -c "import torch,transformers,accelerate,wandb,tiktoken,numpy,einops,muon; print('ok', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[done] '$VENV' 준비 완료 → 활성화: source $VENV/bin/activate"
