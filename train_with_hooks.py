"""
train_with_hooks.py

Train the custom decoder-only LLM (model.LLM) WITHOUT modifying
the model code. All add-ons are wired in via forward hooks + a Trainer subclass:

  • Switch Transformer load-balance auxiliary loss (computed from gate logits
    captured on MoE.gate; added to total loss with --balance_beta).
  • Per-layer diagnostics every logging_steps:
      - router collapse (argmax %), normalized entropy, balance contribution
      - AMoE halting distribution as a (depth × max_steps) certainty heatmap
      - L2 grad norm per (Attention + AMoE) layer, captured between backward
        and optimizer.step (accelerator.sync_gradients).
  • Optional pre-training router-bias init (--router_bias_init_mean / _std).
    Initializes MoE.gate.bias to N(mean, std) so the gate starts with a
    deliberate asymmetry. (A constant shift is shifted out by softmax; the
    randomness around the mean is what matters.)
  • HF Trainer ≥4.46 GA loss bug fix:
    LLM.forward has **kwargs → HF assumes model_accepts_loss_kwargs=True →
    skips dividing the loss by gradient_accumulation_steps for reporting,
    inflating logged train/loss by exactly grad_accum. We override
    HookedTrainer.__init__ to force the flag off.

Console:
  Every logging_steps:
    [Aux step=N] one-liner with the headline scalars (always)
    With --print_console: a full per-layer table + ASCII certainty heatmap.

wandb:
  Always: global scalars (balance, router/max_pct, router/entropy_norm,
          halting/mean_step, base_loss, total_loss, grad_norm/global_l2) and
          one image 'halting/cert_heatmap' (depth × AMoE_step heatmap + grad
          norm bar) refreshed each logging_steps.
  With --log_per_layer: also per-layer scalars (80+ keys — noisy by design).
"""

import os
import math
import signal
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import wandb
from model import LLM, TiktokenHFWrapper, MemmapDataset
from optim import build_muon_optimizer
from config import add_base_args


# ============================================================
# Model post-init: router bias
# ============================================================

def init_router_bias(model: LLM, mean: float, std: float):
    """
    각 layer의 MoE.gate.bias를 N(mean, std)로 재초기화. softmax는 constant shift
    에 불변이므로 mean만으로는 효과가 없고, 평균 주변 분산이 라우터 초기 비대칭을
    만든다. mean<0 + 양의 std → 평균적으론 약한 음수 logit으로 시작, 분산이
    expert간 우열을 정함.
    """
    if std <= 0 and mean == 0.0:
        return 0
    count = 0
    for atten, amoe in model.transformer.layers:
        b = amoe.moe.gate.bias
        with torch.no_grad():
            b.normal_(mean=mean, std=std)
        count += 1
    return count


# ============================================================
# Hooks: capture gate logits (grad alive) and AMoE halting probs
# ============================================================

class HookCollector:
    """
    Forward hooks per layer collect:
      • gate_logits : output of MoE.gate (nn.Linear) — [S, E] with grad alive
      • halting_probs : output[1] of AMoE — [T, B, N]

    MoE is wrapped in gradient_checkpoint(use_reentrant=False), so its forward
    is recomputed during backward — gate hook would fire again. A cap
    (depth × max_steps) makes backward-time recaptures no-ops.
    """
    def __init__(self):
        self.gate_logits = []      # list of [S, E]
        self.halting_probs = []    # list of [T, B, N]
        self._gate_cap = 10**9
        self._amoe_cap = 10**9

    def clear(self):
        self.gate_logits.clear()
        self.halting_probs.clear()

    def _gate_hook(self, module, inputs, output):
        if len(self.gate_logits) < self._gate_cap:
            self.gate_logits.append(output)

    def _amoe_hook(self, module, inputs, output):
        if len(self.halting_probs) < self._amoe_cap:
            _, hp = output
            self.halting_probs.append(hp)


def attach_hooks(model: LLM, collector: HookCollector):
    depth = len(model.transformer.layers)
    max_steps = model.transformer.layers[0][1].max_steps
    collector._gate_cap = depth * max_steps
    collector._amoe_cap = depth
    for _atten, amoe in model.transformer.layers:
        amoe.moe.gate.register_forward_hook(collector._gate_hook)
        amoe.register_forward_hook(collector._amoe_hook)


# ============================================================
# Aux loss + per-(layer, step) metrics from captured tensors
# ============================================================

def compute_aux_and_metrics(collector: HookCollector,
                            depth: int,
                            log_per_layer: bool):
    """
    Returns:
      balance_loss      : scalar tensor (grad alive) — Switch load-balance aux
      metrics           : dict[str, float]            — wandb-loggable scalars
      cert_matrix       : np.ndarray [depth, T] | None — mean step_cert per
                          (layer, AMoE_step)
      per_layer_table   : dict[str, list[float]]      — for console rendering
    """
    metrics = {}
    empty_table = {"balance": [], "max_pct": [], "ent_norm": [], "mean_step": []}

    gate_logits = collector.gate_logits
    n_gate = len(gate_logits)
    metrics["debug/gate_capture_count"] = n_gate
    metrics["debug/expected_gate_count"] = depth * collector._gate_cap // max(depth, 1)

    if n_gate == 0 or n_gate % depth != 0:
        # 캡처가 비정상이면 안전한 fallback: balance=0, 전체 평균 statistic만
        if n_gate == 0:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.zeros((), device=device), metrics, None, empty_table

        E = gate_logits[0].size(-1)
        bal, mxs, ents = [], [], []
        for gl in gate_logits:
            p = F.softmax(gl.float(), dim=-1)
            sel = p.argmax(dim=-1)
            f = F.one_hot(sel, num_classes=E).to(p.dtype).mean(dim=0)
            P = p.mean(dim=0)
            bal.append(E * (f * P).sum())
            mxs.append(f.max().detach())
            ents.append((-(p * p.clamp_min(1e-12).log()).sum(-1).mean()
                        / math.log(E)).detach())
        balance_loss = torch.stack(bal).mean()
        metrics["aux/balance_loss"] = float(balance_loss.detach())
        metrics["router/max_pct_global"] = float(torch.stack(mxs).mean())
        metrics["router/entropy_norm_global"] = float(torch.stack(ents).mean())
        return balance_loss, metrics, None, empty_table

    # 정상 경로: depth × step_count_per_layer 모양으로 walk
    E = gate_logits[0].size(-1)
    spl = n_gate // depth  # steps per layer
    per_layer_bal_t = []   # grad-alive scalar tensors
    per_layer_max = []
    per_layer_ent = []
    idx = 0
    for _li in range(depth):
        bal_t, mx_t, ent_t = [], [], []
        for _t in range(spl):
            gl = gate_logits[idx]; idx += 1
            p = F.softmax(gl.float(), dim=-1)
            sel = p.argmax(dim=-1)
            f = F.one_hot(sel, num_classes=E).to(p.dtype).mean(dim=0)
            P = p.mean(dim=0)
            bal_t.append(E * (f * P).sum())
            mx_t.append(f.max().detach())
            ent_t.append((-(p * p.clamp_min(1e-12).log()).sum(-1).mean()
                         / math.log(E)).detach())
        per_layer_bal_t.append(torch.stack(bal_t).mean())
        per_layer_max.append(float(torch.stack(mx_t).mean()))
        per_layer_ent.append(float(torch.stack(ent_t).mean()))

    balance_loss = torch.stack(per_layer_bal_t).mean()
    metrics["aux/balance_loss"] = float(balance_loss.detach())
    metrics["router/max_pct_global"] = float(np.mean(per_layer_max))
    metrics["router/entropy_norm_global"] = float(np.mean(per_layer_ent))

    # halting: certainty matrix + mean_step
    cert_matrix = None
    per_layer_mean_step = []
    hp_list = collector.halting_probs
    if hp_list:
        T = hp_list[0].size(0)
        cert_matrix = np.zeros((len(hp_list), T), dtype=np.float32)
        for li, hp in enumerate(hp_list):
            t_idx = torch.arange(1, T + 1, device=hp.device, dtype=hp.dtype)
            per_layer_mean_step.append(
                float((hp * t_idx.view(T, 1, 1)).sum(dim=0).mean().detach())
            )
            cert_matrix[li] = hp.mean(dim=(1, 2)).detach().float().cpu().numpy()
        metrics["halting/mean_step_global"] = float(np.mean(per_layer_mean_step))

    if log_per_layer:
        for li in range(depth):
            metrics[f"aux/balance/L{li}"] = float(per_layer_bal_t[li].detach())
            metrics[f"router/max_pct/L{li}"] = per_layer_max[li]
            metrics[f"router/entropy_norm/L{li}"] = per_layer_ent[li]
            if li < len(per_layer_mean_step):
                metrics[f"halting/mean_step/L{li}"] = per_layer_mean_step[li]

    per_layer_table = {
        "balance": [float(b.detach()) for b in per_layer_bal_t],
        "max_pct": per_layer_max,
        "ent_norm": per_layer_ent,
        "mean_step": per_layer_mean_step,
    }
    return balance_loss, metrics, cert_matrix, per_layer_table


# ============================================================
# wandb image: depth × AMoE_step heatmap + per-layer grad_norm bar
# ============================================================

_MATPLOTLIB_INITED = False

def _log_heatmap_image(step: int, cert_matrix, grad_norms=None):
    global _MATPLOTLIB_INITED
    if not _MATPLOTLIB_INITED:
        import matplotlib
        matplotlib.use("Agg")
        _MATPLOTLIB_INITED = True
    import matplotlib.pyplot as plt

    D, T = cert_matrix.shape
    has_gn = grad_norms is not None and len(grad_norms) > 0

    if has_gn:
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(max(6, T * 0.5 + 3), max(3, D * 0.25)),
            gridspec_kw={"width_ratios": [T, 4]},
        )
    else:
        fig, ax = plt.subplots(figsize=(max(4, T * 0.5), max(3, D * 0.25)))
        ax2 = None

    im = ax.imshow(cert_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("AMoE step")
    ax.set_ylabel("Layer (depth)")
    ax.set_xticks(range(T))
    ax.set_yticks(range(D))
    ax.set_title(f"Certainty heatmap @ step {step}")
    fig.colorbar(im, ax=ax, label="mean step_cert")

    if ax2 is not None:
        ys = list(range(D))
        ax2.barh(ys, grad_norms[:D], color="tab:orange")
        ax2.set_yticks(ys)
        ax2.invert_yaxis()
        ax2.set_xlabel("grad_norm (L2)")
        ax2.set_title("per-layer grad_norm")
        ax2.set_xscale("log")
        ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    wandb.log({"halting/cert_heatmap": wandb.Image(fig)}, step=step)
    plt.close(fig)


# ============================================================
# Console pretty-printer
# ============================================================

_BAR_CHARS = " ·░▒▓█"

def _bar_char(v: float) -> str:
    idx = max(0, min(len(_BAR_CHARS) - 1, int(v * len(_BAR_CHARS))))
    return _BAR_CHARS[idx]


def print_console_report(step: int,
                         cert_matrix,
                         per_layer_table: dict,
                         grad_norms,
                         globals_: dict):
    print(f"\n========== [Console step={step}] ==========", flush=True)
    print(f"  balance={globals_.get('balance', float('nan')):.4f}  "
          f"router_max={globals_.get('router_max', float('nan')):.3f}  "
          f"ent_norm={globals_.get('ent_norm', float('nan')):.3f}  "
          f"halt_mean_step={globals_.get('halt_step', float('nan')):.2f}  "
          f"base_loss={globals_.get('base_loss', float('nan')):.4f}  "
          f"total={globals_.get('total_loss', float('nan')):.4f}  "
          f"grad_norm={globals_.get('grad_norm', float('nan')):.3e}",
          flush=True)

    bal = per_layer_table.get("balance", [])
    mx  = per_layer_table.get("max_pct", [])
    en  = per_layer_table.get("ent_norm", [])
    ms  = per_layer_table.get("mean_step", [])
    gn  = grad_norms or []
    depth = max(len(bal), len(mx), len(en), len(ms), len(gn))

    if depth > 0:
        print("\n  per-layer:", flush=True)
        print(f"  {'L':>3}  {'grad_norm':>10}  {'bal':>6}  "
              f"{'max%':>6}  {'ent':>6}  {'h_step':>7}", flush=True)
        for li in range(depth):
            def _pick(arr, i):
                return arr[i] if i < len(arr) else float("nan")
            print(f"  {li:>3}  {_pick(gn, li):>10.4e}  {_pick(bal, li):>6.3f}  "
                  f"{_pick(mx, li):>6.3f}  {_pick(en, li):>6.3f}  "
                  f"{_pick(ms, li):>7.3f}", flush=True)

    if cert_matrix is not None:
        D, T = cert_matrix.shape
        print(f"\n  certainty heatmap [depth={D} × AMoE_step={T}]  "
              f"(mean step_cert per token; rows ~sum to 1):", flush=True)
        print("      " + " ".join(f" s{t:<3d}" for t in range(T)), flush=True)
        for li in range(D):
            row_vals = " ".join(f"{cert_matrix[li, t]:5.3f}" for t in range(T))
            row_bars = "".join(_bar_char(cert_matrix[li, t]) for t in range(T))
            print(f"  L{li:>2}  {row_vals}   {row_bars}", flush=True)
    print("=" * 44, flush=True)


# ============================================================
# HookedTrainer
# ============================================================

class HookedTrainer(Trainer):
    def __init__(self, *args, collector: HookCollector, depth: int,
                 balance_beta: float, log_per_layer: bool,
                 print_console: bool, log_grad_detail: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # HF Trainer >=4.46 GA loss bug fix
        self.model_accepts_loss_kwargs = False

        self._collector = collector
        self._depth = depth
        self._balance_beta = balance_beta
        self._log_per_layer = log_per_layer
        self._print_console = print_console
        self._log_grad_detail = log_grad_detail
        self._last_aux_metrics = {}
        self._last_cert_matrix = None
        self._last_per_layer = {}
        self._last_grad_norms = None  # list[float] len=depth
        self._last_grad_detail = None  # (qkv:list, experts:list[list])

    def _compute_layer_grad_norms(self):
        out = []
        for atten, amoe in self.model.transformer.layers:
            sq = 0.0
            for p in atten.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().float().pow(2).sum())
            for p in amoe.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().float().pow(2).sum())
            out.append(math.sqrt(sq))
        return out

    def _compute_grad_detail(self):
        """layer별 Attention QKV grad-norm + layer·전문가별 grad-norm (L2)."""
        qkv, experts = [], []
        for atten, amoe in self.model.transformer.layers:
            g = atten.to_qkv.weight.grad
            qkv.append(math.sqrt(float(g.detach().float().pow(2).sum()))
                       if g is not None else float("nan"))
            row = []
            for expert in amoe.moe.experts:
                sq = 0.0
                for prm in expert.parameters():
                    if prm.grad is not None:
                        sq += float(prm.grad.detach().float().pow(2).sum())
                row.append(math.sqrt(sq))
            experts.append(row)
        return qkv, experts

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
        base_loss = outputs.loss  # task + ponder_beta * ponder_kl (모델 내부)

        bal_loss, metrics, cert_matrix, per_layer_table = compute_aux_and_metrics(
            self._collector,
            depth=self._depth,
            log_per_layer=self._log_per_layer,
        )
        total = base_loss + self._balance_beta * bal_loss

        metrics["aux/base_loss"] = float(base_loss.detach())
        metrics["aux/total_loss"] = float(total.detach())

        self._last_aux_metrics = metrics
        self._last_cert_matrix = cert_matrix
        self._last_per_layer = per_layer_table

        return (total, outputs) if return_outputs else total

    def log(self, logs, *args, **kwargs):
        if self._last_aux_metrics:
            logs.update(self._last_aux_metrics)

            step  = self.state.global_step
            r_max = self._last_aux_metrics.get("router/max_pct_global", float("nan"))
            r_ent = self._last_aux_metrics.get("router/entropy_norm_global", float("nan"))
            h_ms  = self._last_aux_metrics.get("halting/mean_step_global", float("nan"))
            b_l   = self._last_aux_metrics.get("aux/balance_loss", float("nan"))
            base  = self._last_aux_metrics.get("aux/base_loss", float("nan"))
            tot   = self._last_aux_metrics.get("aux/total_loss", float("nan"))

            gn = self._last_grad_norms
            gn_total = math.sqrt(sum(g * g for g in gn)) if gn else float("nan")
            if gn:
                logs["grad_norm/global_l2"] = gn_total

            if self._log_grad_detail and self._last_grad_detail is not None:
                gd_qkv, gd_exp = self._last_grad_detail
                for li, v in enumerate(gd_qkv):
                    logs[f"grad_norm/qkv/L{li}"] = v
                for li, row in enumerate(gd_exp):
                    for e, v in enumerate(row):
                        logs[f"grad_norm/expert{e}/L{li}"] = v

            print(f"[Aux step={step}] balance={b_l:.4f}  router_max={r_max:.3f}  "
                  f"router_ent_norm={r_ent:.3f}  halt_mean_step={h_ms:.2f}  "
                  f"grad_norm={gn_total:.3e}", flush=True)

            if self._print_console:
                print_console_report(
                    step=step,
                    cert_matrix=self._last_cert_matrix,
                    per_layer_table=self._last_per_layer,
                    grad_norms=gn,
                    globals_={
                        "balance": b_l, "router_max": r_max, "ent_norm": r_ent,
                        "halt_step": h_ms, "base_loss": base, "total_loss": tot,
                        "grad_norm": gn_total,
                    },
                )

            if self._last_cert_matrix is not None and wandb.run is not None:
                try:
                    _log_heatmap_image(step, self._last_cert_matrix, gn)
                except Exception as e:
                    print(f"[wandb heatmap skip] {e}", flush=True)
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
    add_base_args(p, output_dir_default="hooks_outputs")

    # AMoE load-balance
    p.add_argument("--balance_beta", type=float, default=0.01,
                   help="Switch Transformer load-balance aux weight")

    # router bias init (post-init)
    p.add_argument("--router_bias_init_mean", type=float, default=-0.05,
                   help="MoE.gate.bias 초기화 평균 (collapse 완화 목적)")
    p.add_argument("--router_bias_init_std", type=float, default=0.02,
                   help="MoE.gate.bias 초기화 std (0이면 비활성)")

    # gradient clipping
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="gradient clipping max-norm (<=0이면 클리핑 비활성)")

    # logging
    p.add_argument("--log_per_layer", action="store_true",
                   help="per-layer 스칼라 80+개를 wandb에 기록 (산만함)")
    p.add_argument("--print_console", action="store_true",
                   help="매 logging_steps마다 콘솔에 per-layer 표 + heatmap")
    p.add_argument("--log_grad_detail", action="store_true",
                   help="layer별 QKV grad_norm + layer·전문가별 grad_norm을 wandb 스칼라로 기록")

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


# ============================================================
# Main
# ============================================================

def run_training(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert os.path.exists(args.train_bin_path), f"파일 없음: {args.train_bin_path}"
    assert os.path.exists(args.val_bin_path),   f"파일 없음: {args.val_bin_path}"

    tokenizer = TiktokenHFWrapper("r50k_base")

    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id,
        experts=args.experts, base=args.rope_base, dropout=args.dropout,
        ponder_beta=args.ponder_beta, lambda_p=args.lambda_p,
    )

    n_init = init_router_bias(model,
                              mean=args.router_bias_init_mean,
                              std=args.router_bias_init_std)
    print(f"[RouterBiasInit] applied to {n_init}/{args.depth} layers  "
          f"(mean={args.router_bias_init_mean}, std={args.router_bias_init_std})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] params={n_params/1e6:.2f}M")
    wandb.run.summary["n_params_M"] = n_params / 1e6

    collector = HookCollector()
    attach_hooks(model, collector)
    max_steps_amoe = model.transformer.layers[0][1].max_steps
    print(f"[Hooks] depth={args.depth}  max_steps={max_steps_amoe}  "
          f"balance_beta={args.balance_beta}  log_per_layer={args.log_per_layer}  "
          f"print_console={args.print_console}")

    train_ds = MemmapDataset(args.train_bin_path, args.block_size)
    eval_ds  = MemmapDataset(args.val_bin_path,  args.block_size,
                             max_tokens=args.max_val_size)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        run_name=args.run_name,
        dataloader_pin_memory=True,
        seed=args.seed,
        max_steps=max_steps,
        max_grad_norm=args.max_grad_norm,
    )

    optimizer = build_muon_optimizer(model, args)

    trainer = HookedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        optimizers=(optimizer, None),
        collector=collector,
        depth=args.depth,
        balance_beta=args.balance_beta,
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
