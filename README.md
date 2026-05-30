# AMoE LLM

A from-scratch decoder-only **AMoE** language model (PyTorch + HuggingFace Trainer + the Muon
optimizer). Its defining feature is that the FFN slot is an **Adaptive Mixture-of-Experts** block:
top-1 MoE routing (*which* expert) combined with PonderNet-style adaptive halting (*how long* —
per-token variable compute). See [`CLAUDE.md`](CLAUDE.md) for the full architecture and philosophy.

## Layout

**Model (single source of truth):**
- `model.py` — `RoPE, MoE, AMoE, Attention, Transformer, LLM` + `TiktokenHFWrapper` + `MemmapDataset`.
  Shared by training and inference; behavior differs only via `self.training` (training runs the AMoE
  loop a fixed `max_steps` and uses gradient checkpointing; inference breaks early once all tokens halt).

**Wrappers / tooling:**
- `train_custom.py` — main single-GPU training entry (HF `Trainer` + Muon).
- `train_with_hooks.py` — training with forward-hook diagnostics + Switch load-balance aux loss.
- `inference_custom.py` — checkpoint load + autoregressive generation.
- `stability_check.py` — one-batch forward+backward diagnostic (NaN / grad / router / halting).
- `optim.py` — `split_params` + `build_muon_optimizer` (Muon vs AdamW param partition).
- `config.py` — `add_base_args` (CLI args shared by the two training entries).
- `generate.py` — `sample_next_token` (temperature + top-k sampling).
- `diagnostics.py` — `switch_gate_stats` (pure load-balance / router-collapse / entropy metrics).
- `train_common.py` — `install_signal_handlers` + `init_wandb`.
- `muon.py` — the Muon optimizer library (local; imported as `from muon import ...`).
- `launch_agent.py` + `sweep.yaml` — W&B hyperparameter sweep driver.
- `tests/` — CPU smoke + unit tests.

## Install

```bash
pip install -r requirements.txt          # transformers, wandb, tiktoken, numpy, einops, torch
# muon is provided by the local muon.py — do NOT `pip install muon` (that's an unrelated package).
```

## Data format

Training reads pre-tokenized **`uint16` token shards** via `numpy.memmap`: a flat binary of
`r50k_base` (GPT-2 BPE) token ids, sliced into `block_size` windows by `MemmapDataset`. Provide
`train.bin` / `val.bin`. (For a CPU smoke test you can synthesize one: `python tests/make_tiny_bin.py
--path tiny.bin --n_tokens 20000`.)

## Usage

```bash
# Train (single-GPU; DDP intentionally unsupported)
python train_custom.py \
  --train_bin_path train.bin --val_bin_path val.bin \
  --project my-wandb-project --run_name my-run

# Train with hook diagnostics (router/halting heatmaps, balance aux loss)
python train_with_hooks.py --train_bin_path train.bin --val_bin_path val.bin --print_console

# Inference (arch flags must match the trained checkpoint)
python inference_custom.py --model_dir custom-llm-out --prompt "Hello" --max_new_tokens 100

# One-batch stability diagnostic
python stability_check.py --train_bin_path train.bin

# W&B sweep
wandb sweep --project <project> sweep.yaml
wandb agent <entity>/<project>/<sweep-id>
```

## Tests (CPU)

All tests run on CPU with tiny dims + synthetic data (no GPU needed):

```bash
pytest tests/ -q
```

Covers: train forward/backward, inference logits, the AMoE early-exit contract (fixed loop in train,
early break at inference), `MemmapDataset`, generation, the optimizer split, arg defaults, sampling,
and the diagnostic metrics.

## CPU vs GPU

Everything except scaled training runs on CPU (device falls back to CPU automatically), so refactors
and tests are verified cheaply on CPU. Reserve a GPU for: real-data training at scale, `fp16` speed,
and W&B sweeps.
