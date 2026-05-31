#!/usr/bin/env bash
# setup_env.sh — restore Python deps after a container restart.
#
# Only /root persists across restarts on this box; the base conda python at
# /opt/conda already ships torch + numpy (CUDA build), but pip-installed extras
# (transformers, tiktoken, ...) live on the ephemeral container and vanish on
# restart. Re-run this to get the baseline runnable again.
set -e
PY=/opt/conda/bin/python

# torch>=2.12 is needed for torch.compile of the ponder hot path (~1.8x).
# The conda base ships torch 2.9.1; upgrade it here.
# torchvision must be upgraded together with torch (transformers imports it; an
# ABI-mismatched torchvision breaks `from transformers import Trainer`).
"$PY" -m pip install --quiet --disable-pip-version-check \
    "torch>=2.12" torchvision transformers accelerate tiktoken einops safetensors wandb

"$PY" -c "import torch, numpy, transformers, tiktoken, einops, safetensors, accelerate, wandb; \
print('env OK |', torch.__version__, '| cuda', torch.cuda.is_available())"

echo
echo "Use this interpreter for the baseline (NOT bare 'python'):"
echo "  $PY check_pondernet.py"
echo "  $PY train_pondernet.py --train_bin_path <train.bin> --val_bin_path <val.bin> ..."
