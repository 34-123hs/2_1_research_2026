"""
train_with_hooks.py  (dense custom-llm)

Train the dense decoder-only LLM (train.LLM) WITHOUT modifying the model code.
All add-ons are wired in via forward hooks + a Trainer subclass:

  • Per-layer activation diagnostics (forward hooks, no model edit):
      - block-output RMS  (FeedForward output + residual)
      - MLP dead-neuron %  (fraction of GELU activations <= 0; sparsity proxy)
  • Per-layer / global grad-norm diagnostics every logging_steps:
      - L2 grad norm per (Attention + FeedForward) layer, captured between
        backward and optimizer.step (accelerator.sync_gradients).
      - With --log_grad_detail: per-submodule grad norm (QKV / attn-out /
        MLP-in / MLP-out).
  • HF Trainer >=4.46 GA loss bug fix: LLM.forward has **kwargs. The dense LLM
    already sets accepts_loss_kwargs=False, but we also force
    model_accepts_loss_kwargs off in HookedTrainer.__init__ for safety/parity.

This is the dense counterpart of AMoE/train_with_hooks.py. Differences are
confined to model-specific bits: dense has no MoE gate / experts / AMoE halting
/ ponder / balance loss, so router/balance/halting metrics, the certainty
heatmap, and router-bias init are dropped; activation RMS + MLP dead% take the
place of router collapse / entropy.

Console:
  Every logging_steps:
    [Aux step=N] one-liner with the headline scalars (always)
    With --print_console: a full per-layer table.

wandb:
  Always: global scalars (aux/base_loss, grad_norm/global_l2,
          activation/rms_global, activation/mlp_dead_pct_global) and one image
          'diag/per_layer_bars' (per-layer activation RMS bar + grad-norm bar)
          refreshed each logging_steps.
  With --log_per_layer: also per-layer activation scalars.
  With --log_grad_detail: also per-submodule grad-norm scalars.
"""

import os
import math
import signal
import argparse
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
)
import tiktoken
import wandb
from muon import SingleDeviceMuonWithAuxAdam as MuonWithAuxAdam
from train import LLM, MemmapDataset


# ============================================================
# Hooks: capture per-layer activation diagnostics (detached)
# ============================================================

class HookCollector:
    """
    Forward hooks per layer collect detached diagnostic scalars (dense LLM has
    no aux loss, so no grad needs to stay alive):
      • act_rms  : RMS of each block output (FeedForward output + residual)
      • mlp_dead : fraction of GELU activations <= 0 (dead-neuron / sparsity)

    Dense uses no gradient checkpointing, so forward runs once — no backward
    recapture, hence no cap (cf. AMoE gate hook).
    """
    def __init__(self):
        self.act_rms = []    # list of float, per layer (forward order)
        self.mlp_dead = []   # list of float, per layer

    def clear(self):
        self.act_rms.clear()
        self.mlp_dead.clear()

    def _ff_hook(self, module, inputs, output):
        # block output = FeedForward(x) + x  (residual added outside the module)
        block_out = output + inputs[0]
        rms = block_out.float().pow(2).mean().sqrt()
        self.act_rms.append(float(rms.detach()))

    def _gelu_hook(self, module, inputs, output):
        dead = (output <= 0).float().mean()
        self.mlp_dead.append(float(dead.detach()))


def attach_hooks(model: LLM, collector: HookCollector):
    for _atten, ff in model.transformer.layers:
        ff.register_forward_hook(collector._ff_hook)
        ff.layers[2].register_forward_hook(collector._gelu_hook)  # GELU


# ============================================================
# Per-(layer) metrics from captured tensors
# ============================================================

def compute_metrics(collector: HookCollector,
                    depth: int,
                    log_per_layer: bool):
    """
    Returns:
      metrics          : dict[str, float]       — wandb-loggable scalars
      per_layer_table  : dict[str, list[float]] — for console rendering
    """
    metrics = {}
    act_rms = collector.act_rms
    mlp_dead = collector.mlp_dead

    metrics["debug/act_capture_count"] = len(act_rms)
    metrics["debug/expected_act_count"] = depth

    if len(act_rms) == 0:
        return metrics, {"act_rms": [], "mlp_dead": []}

    metrics["activation/rms_global"] = float(np.mean(act_rms))
    if mlp_dead:
        metrics["activation/mlp_dead_pct_global"] = float(np.mean(mlp_dead) * 100.0)

    if log_per_layer:
        for li in range(len(act_rms)):
            metrics[f"activation/rms/L{li}"] = act_rms[li]
        for li in range(len(mlp_dead)):
            metrics[f"activation/mlp_dead_pct/L{li}"] = mlp_dead[li] * 100.0

    per_layer_table = {
        "act_rms": list(act_rms),
        "mlp_dead": [d * 100.0 for d in mlp_dead],
    }
    return metrics, per_layer_table


# ============================================================
# wandb image: per-layer activation RMS bar + grad_norm bar
# ============================================================

_MATPLOTLIB_INITED = False

def _log_diag_image(step: int, act_rms, grad_norms):
    global _MATPLOTLIB_INITED
    if not _MATPLOTLIB_INITED:
        import matplotlib
        matplotlib.use("Agg")
        _MATPLOTLIB_INITED = True
    import matplotlib.pyplot as plt

    has_act = act_rms is not None and len(act_rms) > 0
    has_gn = grad_norms is not None and len(grad_norms) > 0
    if not (has_act or has_gn):
        return
    D = max(len(act_rms) if has_act else 0, len(grad_norms) if has_gn else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, max(3, D * 0.25)))
    ys = list(range(D))

    if has_act:
        ax1.barh(ys, act_rms[:D], color="tab:blue")
    ax1.set_yticks(ys)
    ax1.invert_yaxis()
    ax1.set_xlabel("activation RMS")
    ax1.set_ylabel("Layer (depth)")
    ax1.set_title(f"per-layer activation RMS @ step {step}")
    ax1.grid(True, axis="x", alpha=0.3)

    if has_gn:
        ax2.barh(ys, grad_norms[:D], color="tab:orange")
        ax2.set_xscale("log")
    ax2.set_yticks(ys)
    ax2.invert_yaxis()
    ax2.set_xlabel("grad_norm (L2)")
    ax2.set_title("per-layer grad_norm")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    wandb.log({"diag/per_layer_bars": wandb.Image(fig)}, step=step)
    plt.close(fig)


# ============================================================
# Console pretty-printer
# ============================================================

def print_console_report(step: int,
                         per_layer_table: dict,
                         grad_norms,
                         globals_: dict):
    print(f"\n========== [Console step={step}] ==========", flush=True)
    print(f"  act_rms={globals_.get('act_rms', float('nan')):.4f}  "
          f"mlp_dead%={globals_.get('mlp_dead', float('nan')):.2f}  "
          f"base_loss={globals_.get('base_loss', float('nan')):.4f}  "
          f"grad_norm={globals_.get('grad_norm', float('nan')):.3e}",
          flush=True)

    ar = per_layer_table.get("act_rms", [])
    md = per_layer_table.get("mlp_dead", [])
    gn = grad_norms or []
    depth = max(len(ar), len(md), len(gn))

    if depth > 0:
        print("\n  per-layer:", flush=True)
        print(f"  {'L':>3}  {'grad_norm':>10}  {'act_rms':>8}  "
              f"{'mlp_dead%':>9}", flush=True)
        for li in range(depth):
            def _pick(arr, i):
                return arr[i] if i < len(arr) else float("nan")
            print(f"  {li:>3}  {_pick(gn, li):>10.4e}  "
                  f"{_pick(ar, li):>8.4f}  {_pick(md, li):>9.2f}", flush=True)
    print("=" * 44, flush=True)


# ============================================================
# HookedTrainer
# ============================================================

class HookedTrainer(Trainer):
    def __init__(self, *args, collector: HookCollector, depth: int,
                 log_per_layer: bool, print_console: bool,
                 log_grad_detail: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # HF Trainer >=4.46 GA loss bug fix
        self.model_accepts_loss_kwargs = False

        self._collector = collector
        self._depth = depth
        self._log_per_layer = log_per_layer
        self._print_console = print_console
        self._log_grad_detail = log_grad_detail
        self._last_aux_metrics = {}
        self._last_per_layer = {}
        self._last_grad_norms = None   # list[float] len=depth
        self._last_grad_detail = None  # (qkv, attn_out, mlp_in, mlp_out)

    def _compute_layer_grad_norms(self):
        out = []
        for atten, ff in self.model.transformer.layers:
            sq = 0.0
            for p in atten.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().float().pow(2).sum())
            for p in ff.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().float().pow(2).sum())
            out.append(math.sqrt(sq))
        return out

    def _compute_grad_detail(self):
        """layer별 QKV / attn-out / MLP-in / MLP-out weight grad-norm (L2)."""
        def _gn(param):
            g = param.grad
            return (math.sqrt(float(g.detach().float().pow(2).sum()))
                    if g is not None else float("nan"))
        qkv, attn_out, mlp_in, mlp_out = [], [], [], []
        for atten, ff in self.model.transformer.layers:
            qkv.append(_gn(atten.to_qkv.weight))
            # to_out is Sequential(Linear, Dropout) unless Identity
            if isinstance(atten.to_out, nn.Sequential):
                attn_out.append(_gn(atten.to_out[0].weight))
            else:
                attn_out.append(float("nan"))
            mlp_in.append(_gn(ff.layers[1].weight))   # Linear dim->hidden
            mlp_out.append(_gn(ff.layers[4].weight))  # Linear hidden->dim
        return qkv, attn_out, mlp_in, mlp_out

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs,
                                     num_items_in_batch=num_items_in_batch)
        # 마지막 micro-batch (accumulation sync) 직후, optimizer.step 직전 캡쳐
        if self.accelerator.sync_gradients:
            self._last_grad_norms = self._compute_layer_grad_norms()
            if self._log_grad_detail:
                self._last_grad_detail = self._compute_grad_detail()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        self._collector.clear()
        outputs = model(**inputs)
        base_loss = outputs.loss  # task CE (dense: no aux loss)

        metrics, per_layer_table = compute_metrics(
            self._collector,
            depth=self._depth,
            log_per_layer=self._log_per_layer,
        )
        metrics["aux/base_loss"] = float(base_loss.detach())

        self._last_aux_metrics = metrics
        self._last_per_layer = per_layer_table

        return (base_loss, outputs) if return_outputs else base_loss

    def log(self, logs, *args, **kwargs):
        if self._last_aux_metrics:
            logs.update(self._last_aux_metrics)

            step   = self.state.global_step
            a_rms  = self._last_aux_metrics.get("activation/rms_global", float("nan"))
            m_dead = self._last_aux_metrics.get("activation/mlp_dead_pct_global", float("nan"))
            base   = self._last_aux_metrics.get("aux/base_loss", float("nan"))

            gn = self._last_grad_norms
            gn_total = math.sqrt(sum(g * g for g in gn)) if gn else float("nan")
            if gn:
                logs["grad_norm/global_l2"] = gn_total

            if self._log_grad_detail and self._last_grad_detail is not None:
                gd_qkv, gd_aout, gd_min, gd_mout = self._last_grad_detail
                for li, v in enumerate(gd_qkv):
                    logs[f"grad_norm/qkv/L{li}"] = v
                for li, v in enumerate(gd_aout):
                    logs[f"grad_norm/attn_out/L{li}"] = v
                for li, v in enumerate(gd_min):
                    logs[f"grad_norm/mlp_in/L{li}"] = v
                for li, v in enumerate(gd_mout):
                    logs[f"grad_norm/mlp_out/L{li}"] = v

            print(f"[Aux step={step}] act_rms={a_rms:.4f}  "
                  f"mlp_dead%={m_dead:.2f}  grad_norm={gn_total:.3e}", flush=True)

            if self._print_console:
                print_console_report(
                    step=step,
                    per_layer_table=self._last_per_layer,
                    grad_norms=gn,
                    globals_={
                        "act_rms": a_rms, "mlp_dead": m_dead,
                        "base_loss": base, "grad_norm": gn_total,
                    },
                )

            if wandb.run is not None:
                try:
                    _log_diag_image(step, self._last_per_layer.get("act_rms"), gn)
                except Exception as e:
                    print(f"[wandb diag image skip] {e}", flush=True)
        return super().log(logs, *args, **kwargs)


# ============================================================
# Boilerplate (signals, args, wandb, optimizer)
# ============================================================

def install_signal_handlers():
    def _handler(signum, frame):
        print(f"signal {signum} → cleanup", flush=True)
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=143, quiet=True)
        finally:
            os._exit(143)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def parse_args():
    p = argparse.ArgumentParser()

    # paths / wandb
    p.add_argument("--project", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--train_bin_path", default="train.bin")
    p.add_argument("--val_bin_path", default="val.bin")
    p.add_argument("--output_dir", default="hooks_outputs")

    # data + schedule
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_size", type=int, default=50_000_000)
    p.add_argument("--max_val_size", type=int, default=500_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=576)

    # model (dense: dim_head = dim // heads, mlp_dim = dim * 4 — 파생, 인자 없음)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--rope_base", type=int, default=10000)
    p.add_argument("--dropout", type=float, default=0.0)

    # Muon
    p.add_argument("--lr", type=float, default=3e-4,
                   help="AdamW (embedding/head/bias/norm) learning rate")
    p.add_argument("--muon_lr", type=float, default=0.02,
                   help="Muon (2D hidden weight) learning rate")
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="gradient clipping max-norm")

    # logging
    p.add_argument("--log_per_layer", action="store_true",
                   help="per-layer 활성 스칼라를 wandb에 기록")
    p.add_argument("--print_console", action="store_true",
                   help="매 logging_steps마다 콘솔에 per-layer 표")
    p.add_argument("--log_grad_detail", action="store_true",
                   help="layer별 QKV/attn_out/MLP-in/MLP-out grad_norm을 wandb 스칼라로 기록")

    return p.parse_args()


def init_wandb(args):
    wandb.init(project=args.project, name=args.run_name, config=vars(args),
               allow_val_change=True)
    # sweep override 반영
    for k, v in dict(wandb.config).items():
        if hasattr(args, k):
            setattr(args, k, v)
    print(f"args={vars(args)}")
    return args


def create_muon_optimizer(model, args):
    hidden, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embedding" not in name and "mlp_head" not in name:
            hidden.append(p)
        else:
            other.append(p)

    n_h = sum(p.numel() for p in hidden)
    n_o = sum(p.numel() for p in other)
    print(f"[Optimizer] Muon params={n_h:,}  Aux params={n_o:,}")

    return MuonWithAuxAdam([
        dict(params=hidden, lr=args.muon_lr, momentum=args.muon_momentum,
             weight_decay=args.weight_decay, use_muon=True),
        dict(params=other, lr=args.lr,
             weight_decay=args.weight_decay, use_muon=False),
    ])


# ============================================================
# Main
# ============================================================

def run_training(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert os.path.exists(args.train_bin_path), f"파일 없음: {args.train_bin_path}"
    assert os.path.exists(args.val_bin_path),   f"파일 없음: {args.val_bin_path}"

    enc = tiktoken.get_encoding("r50k_base")
    vocab_size = enc.n_vocab  # 50257

    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.dim * 4, heads=args.heads, dim_head=args.dim // args.heads,
        vocab_size=vocab_size, base=args.rope_base, dropout=args.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] params={n_params/1e6:.2f}M")
    wandb.run.summary["n_params_M"] = n_params / 1e6

    collector = HookCollector()
    attach_hooks(model, collector)
    print(f"[Hooks] depth={args.depth}  log_per_layer={args.log_per_layer}  "
          f"print_console={args.print_console}  log_grad_detail={args.log_grad_detail}")

    train_ds = MemmapDataset(args.train_bin_path, args.block_size)
    eval_ds  = MemmapDataset(args.val_bin_path,  args.block_size,
                             max_tokens=args.max_val_size)

    def collate_fn(examples):
        input_ids = torch.stack([e["input_ids"] for e in examples])
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    max_steps = max(1, math.ceil(args.max_size / tokens_per_step))
    print(f"[Budget] max_size={args.max_size:,} tokens → "
          f"max_steps={max_steps:,} (tokens/step={tokens_per_step:,})")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_interval,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="wandb",
        run_name=args.run_name,
        dataloader_pin_memory=True,
        seed=args.seed,
        max_steps=max_steps,
        max_grad_norm=args.max_grad_norm,
    )

    optimizer = create_muon_optimizer(model, args)

    trainer = HookedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        optimizers=(optimizer, None),
        collector=collector,
        depth=args.depth,
        log_per_layer=args.log_per_layer,
        print_console=args.print_console,
        log_grad_detail=args.log_grad_detail,
    )
    trainer.train()

    eval_metrics = trainer.evaluate()
    ppl = math.exp(eval_metrics["eval_loss"]) if eval_metrics["eval_loss"] < 20 else float("inf")
    print(f"[Eval] loss={eval_metrics['eval_loss']:.4f}  ppl={ppl:.2f}")
    wandb.log({"final/eval_loss": eval_metrics["eval_loss"],
               "final/perplexity": ppl})
    trainer.save_model(args.output_dir)
    wandb.finish()


def main():
    install_signal_handlers()
    args = init_wandb(parse_args())
    run_training(args)


if __name__ == "__main__":
    main()
