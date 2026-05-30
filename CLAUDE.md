# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Custom decoder-only transformer LLM trained from scratch using PyTorch + HuggingFace Trainer, with the Muon optimizer and W&B logging/sweep support.

## Common Commands

```bash
# Install dependencies (muon must be installed separately — not in requirements.txt)
pip install -r requirements.txt
pip install muon  # or from source

# Train (single-GPU only; DDP intentionally not supported)
python train_custom.py \
  --train_bin_path train.bin --val_bin_path val.bin \
  --project my-wandb-project --run_name my-run

# W&B hyperparameter sweep
wandb sweep --project <project> sweep.yaml
wandb agent <entity>/<project>/<sweep-id>
```

## Architecture

Model classes live in `model.py` (shared by training and inference; behavior differs only via `self.training`). Standard pre-norm decoder-only transformer:

- `RoPE` — rotary positional embeddings; sin/cos buffers registered at init, applied per-layer in attention
- `Attention` — multi-head self-attention with RMSNorm pre-norm, causal mask via `scaled_dot_product_attention`
- `MoE` — top-1 routed mixture of experts. Each expert is RMSNorm → Linear(dim → 4·dim) → GELU → Linear(4·dim → dim) with dropout. Also produces a per-token `certainty` scalar via a sigmoid head used by `AMoE`
- `AMoE` — adaptive computation wrapper around `MoE`. Runs up to `max_steps` (default 10) MoE iterations per token; per-step `certainty` accumulates and tokens "halt" once cumulative certainty ≥ 1−eps. Returns `(sum_logit, halting_probs)`. The vertical loop runs a fixed `max_steps` in training, but at inference breaks early once every token has halted. Gradient checkpointing is applied per MoE call only when `self.training`
- `Transformer` — stacks `(Attention, AMoE)` pairs with residual connections, final RMSNorm. Surfaces per-layer `halting_probs` for the PonderNet loss
- `LLM` — embedding → dropout → Transformer → linear head. Computes `task_loss + ponder_beta * KL(halting || geometric_prior(lambda_p))` when `labels` are given (training); returns logits only otherwise (inference)

### Tokenizer

`TiktokenHFWrapper` wraps tiktoken's `r50k_base` (GPT-2 BPE, ~50k vocab) to satisfy the HuggingFace `PreTrainedTokenizer` interface so it works with `DataCollatorForLanguageModeling`.

### Data

`MemmapDataset` reads pre-tokenized binary files (`train.bin`, `val.bin`) via `numpy.memmap` with `dtype=uint16`. Tokens are sliced into fixed `block_size` windows. Data must be pre-processed into this format before training.

### Optimizer (Muon)

`create_muon_optimizer` splits parameters into two groups:
- **Muon** (`SingleDeviceMuonWithAuxAdam`): all 2D weight matrices except `embedding` and `mlp_head`
- **AdamW**: everything else (embeddings, biases, norms, head)

`--muon_lr` and `--lr` are separate knobs for these two groups.

### Training Budget

Training is token-budget-based: `max_steps = ceil(max_size / tokens_per_step)` where `tokens_per_step = batch_size × grad_accum × block_size`. The `--epochs` argument is passed to `TrainingArguments` but `max_steps` takes precedence via HuggingFace's step-based cutoff.

---

## Behavioral Guidelines

*Derived from [Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.*

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- Remove imports/variables/functions that YOUR changes made unused; leave pre-existing dead code alone.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```
