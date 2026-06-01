# Training Methodology (PonderNet baseline & comparison)

What we learned tuning training throughput for the AMoE-vs-baselines comparison, on a single
**H100 80GB**. Measurements are fwd+bwd wall-clock at the final config; "~h/epoch" assumes
**2.0B tokens** (1 epoch, `--max_size 2000000815`).

## Final config (shared)
`dim 768 · block_size 768 · batch_size 8 · grad_accum 6 · dropout 0 · lr 0.00296 · warmup 150 ·
epochs 1 · max_size ≈2.0B · eval_interval 200`. Optimizer: **Muon** (2D hidden weights) + **AdamW**
(embeddings/head/biases/norms); `muon_lr` default 0.02 is a *separate* LR knob.

- **Vanilla LLM**: depth 12.
- **PonderNet**: `core_depth 12` (weight-shared core) + `ponder_steps 8` (halting horizon; 162M params).
- **AMoE**: depth 12, experts as configured.

## Key findings (in order of impact)

### 1. `torch.compile` — the biggest win (~1.8×)
Compiling the ponder hot path took bf16 from **297 → 165 ms/step** (~27h → **~15h/epoch**) and also
*cut* peak VRAM (38.6 → 29.8 GB). Compile the inner step, not `forward`, to avoid graph-breaks in the
ponder loop / halting side-effect:
```python
model._ponder_step = torch.compile(model._ponder_step)   # needs torch >= 2.12
```
Now integrated into `train_pondernet.py` (`--compile`, default on). **Apply the same to Vanilla/AMoE**
for a fair comparison.

### 2. Precision: use **bf16** (not fp16, not fp8)
| precision | ms/step | tok/s | ~h/epoch | note |
|---|---:|---:|---:|---|
| fp32 | 1764 | 3.5K | ~160h | far too slow |
| **bf16** | 297 (eager) / **165 (compiled)** | 37K | **~15h** | use this |
| fp8 (torchao, compiled) | 196 | 31K | ~18h | **slower than bf16 (0.84×)** |

**fp8 is not worth it here.** The dominant op is the per-ponder-step full-vocab head (768→50257),
which stays bf16 (50257 not ÷16), so fp8 only accelerates the modest core matmuls while adding
cast/scaling overhead. fp8's only benefit was lower VRAM (23 vs 30 GB), not speed. (Also: torchao fp8
kernels need torch ≥ 2.11, and only help *with* `torch.compile` — in eager they are ~2× slower.)

### 3. Gradient checkpointing: **OFF** for this config
PonderNet checkpoints each ponder step (else the per-step full-vocab logits keep ~`ponder_steps`×
the activation graph). But at `ponder_steps 8`, bf16, batch 8, **ckpt OFF fits in 38.6 GB** and is
~1.37× faster than ckpt ON. So default **OFF** here (`--grad-checkpoint` to re-enable if VRAM-bound,
e.g. batch>16). ckpt OFF batch ceiling ≈ 16 (76 GB); batch 24 OOMs.

### 4. `ponder_steps` is the main PonderNet cost knob
Training runs **all** `ponder_steps` every step → cost ∝ ponder_steps. 10→8 cut ~46h→~37h (ckpt ON).
With λ_p=0.2 the prior mean is ~5 steps, so 8 is already generous; 6 would cut more with little
fidelity loss.

## Timing progression (PonderNet, this config)
`ponder10 · ckptON · eager` **~46h** → `ponder8 · ckptOFF · eager` **~27h** →
**`+ torch.compile` ≈ 15h/epoch** (≈3× total speedup). bf16 throughput ~37K tok/s (compute-bound;
bigger batch doesn't raise tok/s).

> **Bottom line: ~15 hours for the 1-epoch (2.0B-token) PonderNet run on one H100** (bf16 + compile +
> ckpt OFF + ponder_steps 8 + batch 8). Add a little for optimizer/eval/dataloader overhead.

## Hyperparameter sweep (wandb)
Sweep these (most impactful first), keeping architecture/compute fixed:
1. **`lr` × `muon_lr`** — two separate LRs; muon_lr governs the bulk (core) weights.
2. **`lambda_p`** — the PonderNet prior (mean steps = 1/λ_p); paper shows strong sensitivity. e.g. {0.1,0.2,0.3,0.5}.
3. **`ponder_beta`** (KL weight) — log-uniform 1e-3…1e-1.
4. (secondary) `weight_decay`, `warmup_steps`.
Use a small **proxy `--max_size`** per trial (e.g. 30–50M tokens) so trials are ~1h, with Bayes +
HyperBand. Sweep Vanilla/AMoE the same way (Vanilla: lr·muon_lr; AMoE: + lambda_p·ponder_beta).

## Loss fidelity (PonderNet)
Verified faithful to the paper (Pounder.pdf Eq.3, §2.3) — see `verify_vs_labml.py`:
- Reconstruction `Σ p_n·CE(ŷ_n,y)` — **numerically identical** to labml_nn's `ReconstructionLoss`.
- KL = **forward** `KL(p ‖ p_G)` with renormalized truncated-geometric prior (paper formula).
  (labml_nn's code does reverse KL + unnormalized prior — we follow the paper.)
- `p_n` sums to 1 via remaining-mass-to-last-step (paper §2.3 option b).

## Environment notes (important)
- Use **`/opt/conda/bin/python`** (ships torch+numpy), NOT bare `python` (resolves to /usr/bin without torch).
- The container resets and **pip extras vanish** (only `/root` persists). Restore with:
  `bash setup_env.sh` (installs torch≥2.12 + transformers/tiktoken/einops/… into conda).
- `torchvision 0.24.1 requires torch==2.9.1` conflict warning after the torch upgrade is **harmless**.

## Run commands
```bash
bash setup_env.sh        # restore deps (after a container reset)

# PonderNet baseline (final config, compile on, ckpt off, bf16 — all defaults)
/opt/conda/bin/python train_pondernet.py \
  --train_bin_path train.bin --val_bin_path val.bin \
  --dim 768 --core_depth 12 --ponder_steps 8 --heads 12 --dim_head 64 --mlp_dim 3072 \
  --block_size 768 --batch_size 8 --grad_accum 6 --lr 0.00296 \
  --max_size 2000000815 --warmup_steps 150 --eval_interval 200 \
  --project custom-llm --run_name "PonderNet baseline"

# E2E + timing check on any edited model
/opt/conda/bin/python check_pondernet.py --core_depth 12 --ponder_steps 8 --block_size 768
```
