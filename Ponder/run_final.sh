#!/usr/bin/env bash
# run_final.sh — PonderNet 본 학습 (full 2.0B-token run, 1 epoch).
#
#   bash run_final.sh           # 자신이 있는 폴더에서 train.bin/val.bin 을 찾는다
#   PY=/path/to/python bash run_final.sh   # python 경로 직접 지정
#
# Sweep PounderNet/6wjjahq7 best config (eval/loss 3.886 @ 50M proxy) 를 그대로 쓰되
# max_size 만 50M → 2.0B 로 올린 최종 런. AMoE / Vanilla 와 동일 조건(2.0B, 1 epoch,
# bf16 + torch.compile, batch 8 · grad_accum 6 · block 768)으로 공정 비교용.
# 경로 독립: 스크립트가 있는 폴더(BASE)를 작업/캐시 루트로 쓴다 → 어느 박스든 동일 동작.

BASE="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-/opt/conda/bin/python}
LOG="$BASE/final_run.log"

set -u
cd "$BASE"
export TMPDIR="$BASE/tmp" TORCHINDUCTOR_CACHE_DIR="$BASE/.inductor_cache"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

# 로컬 GPU 점유 해제(이 박스에 sweep agent 가 있으면 종료 — 본 학습에 GPU 전부 양보)
pkill -9 -f "wandb agent" 2>/dev/null
sleep 3

setsid bash -c "export TMPDIR='$TMPDIR' TORCHINDUCTOR_CACHE_DIR='$TORCHINDUCTOR_CACHE_DIR'; exec $PY '$BASE/train_pondernet.py' \
  --project=PounderNet \
  --run_name='Final PonderNet' \
  --train_bin_path=train.bin \
  --val_bin_path=val.bin \
  --output_dir=final_outputs \
  --epochs=1 \
  --max_size=2000000815 \
  --max_val_size=200000 \
  --eval_interval=1000 \
  --warmup_steps=150 \
  --batch_size=8 \
  --grad_accum=6 \
  --block_size=768 \
  --dim=768 \
  --heads=12 \
  --dim_head=64 \
  --mlp_dim=3072 \
  --dropout=0 \
  --core_depth=12 \
  --ponder_steps=8 \
  --lambda_p=0.3 \
  --ponder_beta=0.01906962800737401 \
  --lr=0.004844764957248182 \
  --muon_lr=0.03162628063372118 \
  --compile --no-grad_checkpoint" > "$LOG" 2>&1 < /dev/null &

sleep 10
if pgrep -f "train_pondernet" >/dev/null; then
  echo "본 학습 시작됨 -> $LOG"
else
  echo "WARN: 프로세스 미탐지 — 로그 확인: $LOG"
fi
echo "tail -f $LOG"
