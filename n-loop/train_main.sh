#!/usr/bin/env bash
# train_main.sh — 비영속 박스용 메인 학습 (sweep/agent 아님, 단일 프로세스, 풀 코퍼스 1회).
#
# 0) 이 박스엔 처음에 코드가 없으므로 git clone이 가장 먼저다 (그게 이 스크립트를 가져온다):
#       git clone https://github.com/34-123hs/AMOE.git
#       cd AMOE
#       # train.bin / val.bin 를 이 디렉토리에 직접 넣기 (corpus는 git에 없음 — .bin은 gitignore)
#       # wandb 인증: WANDB_API_KEY 환경변수 또는 `wandb login`
# 1) 그 다음 이 스크립트 실행:
#       ./train_main.sh                              # 포그라운드
#       nohup ./train_main.sh > main.log 2>&1 &      # 백그라운드 (~18-23h)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"     # 레포 디렉토리

# 1) 최신 코드 sync (영속 볼륨이 없으니 git으로 동기화)
git pull --ff-only || echo "[warn] git pull 실패 — 현재 코드로 진행"

# 2) venv + 의존성 (레포 안 .venv, 없으면 생성)
VENV=".venv"; PY="$VENV/bin/python"
if [ ! -x "$PY" ] || ! "$PY" -c "import torch,wandb,transformers,tiktoken,numpy,einops" 2>/dev/null; then
  echo "[setup] venv 생성 + 의존성 설치"
  python3 -m venv "$VENV"
  "$PY" -m pip install -U pip
  "$PY" -m pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu130
  "$PY" -m pip install "transformers==5.9.0" "accelerate==1.13.0" "wandb==0.27.0" \
       "tiktoken==0.13.0" "numpy==2.4.6" einops
fi
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA 불가'; \
print(f'[setup] torch {torch.__version__} | {torch.cuda.get_device_name(0)}')"

# 3) corpus 확인 (사용자가 직접 넣음)
for f in train.bin val.bin; do
  [ -f "$f" ] || { echo "[error] $f 없음 — corpus를 이 디렉토리에 넣으세요" >&2; exit 1; }
done

# 4) 탐색 HP — sweep에서 추출한 값 (ponder_beta는 대조군에서 무효라 제외)
LR=0.0018411529824275833
MUON_LR=0.032273990315242855
ALPHA=0.005

# 5) 메인 학습 (단일 프로세스, 풀 코퍼스). wandb는 기존과 동일 프로젝트로 로깅.
"$PY" train_with_hooks.py \
  --project AMOE-SWEEP123 --run_name main \
  --train_bin_path train.bin --val_bin_path val.bin \
  --output_dir main_out \
  --max_size 2000000815 --max_val_size 200000 \
  --epochs 1 --warmup_steps 150 --eval_interval 1000 \
  --save_steps 1000 --resume 1 --max_grad_norm 1.0 \
  --block_size 768 --batch_size 48 --grad_accum 1 \
  --dim 768 --depth 12 --heads 12 --dim_head 64 --mlp_dim 3072 \
  --dropout 0 --ponder_steps 4 \
  --grad_checkpoint 0 --compile 0 \
  --lr "$LR" --muon_lr "$MUON_LR" --alpha "$ALPHA"
