#!/usr/bin/env bash
# bootstrap_h100.sh — one-shot setup + launch the PonderNet sweep on a fresh H100.
#
#   bash /root/baselines/bootstrap_h100.sh
#
# Assumes: the persistent /root volume (with /root/baselines code + ~/.netrc wandb
# login) is mounted, and the provider base image's conda python is at /opt/conda
# (ships torch+numpy). It installs the remaining deps, redirects all caches to /root
# (the /tmp tmpfs is tiny and ENOSPCs), wires the data, and launches a detached
# sweep agent that keeps producing trials.
#
# Reuses the existing sweep in /root/baselines/SWEEP_ID.txt; delete that file first
# to start a brand-new sweep from sweep.yaml.
# If wandb isn't logged in: `export WANDB_API_KEY=<key>` before running, or
# `/opt/conda/bin/wandb login <key>` once (persists in ~/.netrc on the volume).

PY=/opt/conda/bin/python
BASE=/root/baselines
ENTITY=choijiwan1229-hansung-science-high-school
PROJECT=PounderNet

set -u
[ -x "$PY" ] || { echo "FATAL: $PY not found (different base image?). Adjust PY=."; exit 1; }
[ -d "$BASE" ] || { echo "FATAL: $BASE missing — is the persistent volume mounted?"; exit 1; }
cd "$BASE"

echo "### 0) caches -> /root (avoid the small /tmp tmpfs ENOSPC) ###"
export TMPDIR=/root/tmp TORCHINDUCTOR_CACHE_DIR=/root/.inductor_cache
mkdir -p /root/tmp /root/.inductor_cache

echo "### 1) python deps (conda base has torch+numpy; add the rest) ###"
# torch>=2.12 enables torch.compile (~1.8x); torchvision must match torch or
# `from transformers import Trainer` breaks via the torchvision::nms op.
"$PY" -m pip install --quiet --disable-pip-version-check \
    "torch>=2.12" torchvision transformers accelerate tiktoken einops safetensors wandb \
    || { echo "FATAL: pip install failed"; exit 1; }
"$PY" -c "import torch,torchvision,transformers,wandb; from transformers import Trainer; \
print('env OK | torch', torch.__version__, '| cuda', torch.cuda.is_available(), \
'|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" \
    || { echo "FATAL: import check failed"; exit 1; }

echo "### 2) data symlinks (real r50k uint16 shards on the volume) ###"
[ -e train.bin ] || ln -sf /root/AMOE/train.bin train.bin
[ -e val.bin ]   || ln -sf /root/AMOE/val.bin   val.bin
ls -lh train.bin val.bin 2>/dev/null || { echo "FATAL: train.bin/val.bin not found"; exit 1; }

echo "### 3) wandb auth ###"
"$PY" -c "import wandb; print('wandb user:', wandb.Api().viewer.username)" \
    || { echo "FATAL: wandb not authenticated. Run: $PY -m wandb login <KEY>  (or export WANDB_API_KEY)"; exit 1; }

echo "### 4) sweep (reuse SWEEP_ID.txt, else create from sweep.yaml) ###"
if [ -s SWEEP_ID.txt ]; then
  SWEEP=$(cat SWEEP_ID.txt)
  echo "reusing existing sweep: $SWEEP"
else
  "$PY" -m wandb sweep --project "$PROJECT" sweep.yaml 2>sweep_create.log
  SWEEP=$(grep -oE "[A-Za-z0-9_-]+/$PROJECT/[A-Za-z0-9]+" sweep_create.log | tail -1)
  [ -n "$SWEEP" ] || { echo "FATAL: sweep creation failed"; tail -5 sweep_create.log; exit 1; }
  echo "$SWEEP" > SWEEP_ID.txt
  echo "created sweep: $SWEEP"
fi

echo "### 5) launch detached agent (continuous trials; output to /root) ###"
pkill -9 -f "wandb agent" 2>/dev/null
sleep 3
setsid bash -c "export TMPDIR=/root/tmp TORCHINDUCTOR_CACHE_DIR=/root/.inductor_cache; \
exec $PY -m wandb agent $SWEEP > $BASE/sweep_agent.log 2>&1" < /dev/null &
sleep 8
if pgrep -f "wandb agent" >/dev/null; then
  echo "agent UP -> $BASE/sweep_agent.log"
else
  echo "WARN: agent not detected; check $BASE/sweep_agent.log"
fi
echo "Dashboard: https://wandb.ai/$ENTITY/$PROJECT/sweeps/${SWEEP##*/}"
echo "DONE. Tail logs: tail -f $BASE/sweep_agent.log"
