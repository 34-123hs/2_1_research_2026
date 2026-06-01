#!/usr/bin/env bash
# bootstrap_amoe.sh — 이미 등록된 sweep에 agent를 GPU당 1개씩 띄운다 (등록은 안 함).
#   등록(중앙에서 1회):  wandb sweep --project AMOE-SWEEP123 sweep.yaml  → ID를 .sweep_id에 저장
#   ./bootstrap_amoe.sh            # .sweep_id의 sweep에 join
#   ./bootstrap_amoe.sh <SWEEP_ID> # 특정 sweep에 join
set -euo pipefail

AMOE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$(dirname "$AMOE_DIR")/amoe-venv"
ENTITY="choijiwan1229-hansung-science-high-school"
PROJECT="AMOE-SWEEP123"
cd "$AMOE_DIR"

export PATH="$VENV/bin:$PATH"          # trial이 venv python을 쓰도록
mkdir -p logs

# sweep ID: 인자 > .sweep_id. (등록 기능 없음 — sweep은 여기서 중앙집중적으로 미리 등록)
if [ -n "${1:-}" ]; then
  SWEEP_ID="$1"
elif [ -s .sweep_id ]; then
  SWEEP_ID="$(cat .sweep_id)"
else
  echo "[error] sweep ID 없음. 먼저 등록 후 ID를 인자로 주거나 .sweep_id에 기록:" >&2
  echo "        wandb sweep --project $PROJECT sweep.yaml   # → ID를 .sweep_id에 저장" >&2
  exit 1
fi
echo "[sweep] $ENTITY/$PROJECT/$SWEEP_ID"

# GPU당 agent 1개, CUDA_VISIBLE_DEVICES로 핀
HOST="$(hostname)"
for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  CUDA_VISIBLE_DEVICES="$g" nohup wandb agent "$ENTITY/$PROJECT/$SWEEP_ID" \
        > "logs/agent_${HOST}_g${g}.log" 2>&1 &
  echo "[agent gpu=$g] PID $! → logs/agent_${HOST}_g${g}.log"
done
echo "[done] GPU당 1 agent"
