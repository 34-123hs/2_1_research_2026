"""
train_with_hooks.py

Train the custom decoder-only LLM (model.LLM) WITHOUT modifying
the model code. All add-ons are wired in via forward hooks + a Trainer subclass:

  • Switch Transformer load-balance auxiliary loss (computed from gate logits
    captured on MoE.gate; added to total loss with --balance_beta).
  • Per-layer diagnostics every logging_steps:
      - router collapse (argmax %), normalized entropy, balance contribution
      - expert dispatch heatmaps (by layer, and by recurrence step)
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
    With --print_console: a full per-layer table.

wandb:
  Always: global scalars (balance, router/max_pct, router/entropy_norm,
          base_loss, total_loss, grad_norm/global_l2) and expert-dispatch
          heatmaps (by layer + by recurrence step) refreshed each logging_steps.
  With --log_per_layer: also per-layer scalars (60+ keys — noisy by design).
"""

import os
import json
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
import wandb
from model import LLM, TiktokenHFWrapper, MemmapDataset
from optim import build_muon_optimizer
from config import add_base_args
from train_common import install_signal_handlers, init_wandb
from diagnostics import switch_gate_stats


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
# Hooks: capture gate logits (grad alive) per recurrence step
# ============================================================

class HookCollector:
    """
    Per-layer forward hooks tag every gate capture with its AMoE layer index and
    the recurrence-step index within that layer. compute_aux runs in compute_loss
    BEFORE backward, so the gradient-checkpoint recompute that fires these hooks
    again during backward only appends post-compute garbage (cleared next step)
    and never pollutes the metrics.
      • gate_logits : list of (layer, step, [S, E]) — each MoE.gate call
    """
    def __init__(self):
        self.gate_logits = []      # [(layer, step, [S, E])]
        self._layer_step = {}      # layer -> next recurrence-step index

    def clear(self):
        self.gate_logits.clear()
        self._layer_step = {}

    def _gate_hook(self, li):
        def hook(module, inputs, output):
            st = self._layer_step.get(li, 0)
            self.gate_logits.append((li, st, output))
            self._layer_step[li] = st + 1
        return hook


def attach_hooks(model: LLM, collector: HookCollector):
    for li, (_atten, amoe) in enumerate(model.transformer.layers):
        amoe.moe.gate.register_forward_hook(collector._gate_hook(li))


# ============================================================
# Aux loss + per-(layer, step) metrics from captured tensors
# ============================================================

def compute_aux_and_metrics(collector: HookCollector,
                            depth: int,
                            log_per_layer: bool):
    """
    Returns:
      balance_loss   : scalar tensor (grad alive) — Switch load-balance aux
      metrics        : dict[str, float]            — wandb scalars
      dispatch_layer : [depth, E]  | None — expert dispatch fraction per layer
      dispatch_step  : [maxstep, E]| None — expert dispatch fraction per recurrence step
                       (summed over layers; vertical-depth view)
      per_layer_table: dict[str, list[float]]      — for console rendering
    """
    metrics = {}
    empty_table = {"balance": [], "max_pct": [], "ent_norm": []}

    gate = collector.gate_logits   # [(layer, step, [S, E])]
    metrics["debug/gate_capture_count"] = len(gate)
    if not gate:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.zeros((), device=device), metrics, None, None, empty_table

    E = gate[0][2].size(-1)
    maxstep = 1 + max(st for _, st, _ in gate)

    per_layer_bal = [[] for _ in range(depth)]   # grad-alive balance tensors
    per_layer_max = [[] for _ in range(depth)]
    per_layer_ent = [[] for _ in range(depth)]
    layer_cnt = np.zeros((depth, E), dtype=np.float64)   # expert dispatch counts / layer
    layer_tot = np.zeros(depth, dtype=np.float64)
    step_cnt  = np.zeros((maxstep, E), dtype=np.float64)  # ...summed over layers / step
    step_tot  = np.zeros(maxstep, dtype=np.float64)

    for li, st, gl in gate:
        b, mx, ent = switch_gate_stats(gl)
        per_layer_bal[li].append(b)
        per_layer_max[li].append(float(mx))
        per_layer_ent[li].append(float(ent))
        sel = gl.float().argmax(dim=-1)
        cnt = torch.bincount(sel, minlength=E).double().cpu().numpy()
        n = gl.size(0)
        layer_cnt[li] += cnt; layer_tot[li] += n
        step_cnt[st]  += cnt; step_tot[st]  += n

    # balance loss: per-layer mean → global mean (grad alive)
    per_layer_bal_t = [torch.stack(b).mean() for b in per_layer_bal if b]
    balance_loss = torch.stack(per_layer_bal_t).mean()
    metrics["aux/balance_loss"] = float(balance_loss.detach())

    pl_bal = [float(torch.stack(b).mean().detach()) if b else float("nan") for b in per_layer_bal]
    pl_max = [float(np.mean(m)) if m else float("nan") for m in per_layer_max]
    pl_ent = [float(np.mean(e)) if e else float("nan") for e in per_layer_ent]
    metrics["router/max_pct_global"] = float(np.nanmean(pl_max))
    metrics["router/entropy_norm_global"] = float(np.nanmean(pl_ent))

    # expert dispatch fraction: by layer [depth, E] and by recurrence step [maxstep, E]
    dispatch_layer = layer_cnt / np.maximum(layer_tot[:, None], 1.0)
    dispatch_step  = step_cnt  / np.maximum(step_tot[:, None], 1.0)

    if log_per_layer:
        for li in range(depth):
            metrics[f"aux/balance/L{li}"] = pl_bal[li]
            metrics[f"router/max_pct/L{li}"] = pl_max[li]
            metrics[f"router/entropy_norm/L{li}"] = pl_ent[li]

    per_layer_table = {
        "balance": pl_bal,
        "max_pct": pl_max,
        "ent_norm": pl_ent,
    }
    return balance_loss, metrics, dispatch_layer, dispatch_step, per_layer_table


# ============================================================
# wandb images: expert-dispatch heatmaps
# ============================================================

_MATPLOTLIB_INITED = False


def _log_dispatch_heatmaps(step: int, dispatch_layer, dispatch_step):
    """Two W&B heatmaps: expert dispatch by layer, and by ponder step (layers summed)."""
    global _MATPLOTLIB_INITED
    if not _MATPLOTLIB_INITED:
        import matplotlib
        matplotlib.use("Agg")
        _MATPLOTLIB_INITED = True
    import matplotlib.pyplot as plt

    for name, mat, ylabel in (
        ("router/dispatch_by_layer", dispatch_layer, "Layer (depth)"),
        ("router/dispatch_by_step", dispatch_step, "AMoE step (vertical depth)"),
    ):
        if mat is None:
            continue
        R, E = mat.shape
        fig, ax = plt.subplots(figsize=(max(4, E * 0.7), max(3, R * 0.3)))
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xlabel("Expert")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(E))
        ax.set_yticks(range(R))
        ax.set_title(f"{name} @ step {step}")
        for i in range(R):
            for j in range(E):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="w" if mat[i, j] < 0.5 else "k", fontsize=7)
        fig.colorbar(im, ax=ax, label="dispatch fraction")
        fig.tight_layout()
        wandb.log({name: wandb.Image(fig)}, step=step)
        plt.close(fig)


# ============================================================
# Console pretty-printer
# ============================================================

def print_console_report(step: int,
                         per_layer_table: dict,
                         grad_norms,
                         globals_: dict):
    print(f"\n========== [Console step={step}] ==========", flush=True)
    print(f"  balance={globals_.get('balance', float('nan')):.4f}  "
          f"router_max={globals_.get('router_max', float('nan')):.3f}  "
          f"ent_norm={globals_.get('ent_norm', float('nan')):.3f}  "
          f"base_loss={globals_.get('base_loss', float('nan')):.4f}  "
          f"total={globals_.get('total_loss', float('nan')):.4f}  "
          f"grad_norm={globals_.get('grad_norm', float('nan')):.3e}",
          flush=True)

    bal = per_layer_table.get("balance", [])
    mx  = per_layer_table.get("max_pct", [])
    en  = per_layer_table.get("ent_norm", [])
    gn  = grad_norms or []
    depth = max(len(bal), len(mx), len(en), len(gn))

    if depth > 0:
        print("\n  per-layer:", flush=True)
        print(f"  {'L':>3}  {'grad_norm':>10}  {'bal':>6}  "
              f"{'max%':>6}  {'ent':>6}", flush=True)
        for li in range(depth):
            def _pick(arr, i):
                return arr[i] if i < len(arr) else float("nan")
            print(f"  {li:>3}  {_pick(gn, li):>10.4e}  {_pick(bal, li):>6.3f}  "
                  f"{_pick(mx, li):>6.3f}  {_pick(en, li):>6.3f}", flush=True)
    print("=" * 44, flush=True)


# ============================================================
# HookedTrainer
# ============================================================

class HookedTrainer(Trainer):
    def __init__(self, *args, collector: HookCollector, depth: int,
                 balance_beta: float, log_per_layer: bool,
                 print_console: bool, log_grad_detail: bool = False,
                 log_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        # HF Trainer >=4.46 GA loss bug fix
        self.model_accepts_loss_kwargs = False

        self._collector = collector
        self._depth = depth
        self._balance_beta = balance_beta
        self._log_per_layer = log_per_layer
        self._print_console = print_console
        self._log_grad_detail = log_grad_detail
        self._log_path = log_path
        self._last_aux_metrics = {}
        self._last_dispatch_layer = None
        self._last_dispatch_step = None
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
        """layer별 Attention QKV grad-norm + layer·전문가별 grad-norm (L2).
        전문가는 grouped Parameter expert1[E,D,H]/expert2[E,H,D]의 슬라이스 e로 본다."""
        qkv, experts = [], []
        for atten, amoe in self.model.transformer.layers:
            g = atten.to_qkv.weight.grad
            qkv.append(math.sqrt(float(g.detach().float().pow(2).sum()))
                       if g is not None else float("nan"))
            moe = amoe.moe
            g1, g2 = moe.expert1.grad, moe.expert2.grad
            row = []
            for e in range(moe.expert1.size(0)):
                sq = 0.0
                if g1 is not None:
                    sq += float(g1[e].detach().float().pow(2).sum())
                if g2 is not None:
                    sq += float(g2[e].detach().float().pow(2).sum())
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
        base_loss = outputs.loss  # task + alpha * LBL (모델 내부)

        (bal_loss, metrics, dispatch_layer, dispatch_step,
         per_layer_table) = compute_aux_and_metrics(
            self._collector,
            depth=self._depth,
            log_per_layer=self._log_per_layer,
        )
        total = base_loss  # model.py가 주는 loss가 유일한 학습 손실 (wrapper 추가 loss 없음)

        metrics["aux/base_loss"] = float(base_loss.detach())
        metrics["aux/total_loss"] = float(total.detach())

        self._last_aux_metrics = metrics
        self._last_dispatch_layer = dispatch_layer
        self._last_dispatch_step = dispatch_step
        self._last_per_layer = per_layer_table

        return (total, outputs) if return_outputs else total

    def log(self, logs, *args, **kwargs):
        if self._last_aux_metrics:
            logs.update(self._last_aux_metrics)

            step  = self.state.global_step
            r_max = self._last_aux_metrics.get("router/max_pct_global", float("nan"))
            r_ent = self._last_aux_metrics.get("router/entropy_norm_global", float("nan"))
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
                  f"router_ent_norm={r_ent:.3f}  "
                  f"grad_norm={gn_total:.3e}", flush=True)

            if self._print_console:
                print_console_report(
                    step=step,
                    per_layer_table=self._last_per_layer,
                    grad_norms=gn,
                    globals_={
                        "balance": b_l, "router_max": r_max, "ent_norm": r_ent,
                        "base_loss": base, "total_loss": tot,
                        "grad_norm": gn_total,
                    },
                )

            # B: expert-dispatch heatmaps (by layer + by recurrence step)
            if wandb.run is not None:
                try:
                    _log_dispatch_heatmaps(step, self._last_dispatch_layer,
                                           self._last_dispatch_step)
                except Exception as e:
                    print(f"[wandb dispatch heatmap skip] {e}", flush=True)

            # raw hooked data -> logs/<sweep>_log.jsonl (one record per logging step)
            if self._log_path is not None:
                rec = {
                    "step": int(step),
                    "run_id": (wandb.run.id if wandb.run is not None else None),
                    "metrics": self._last_aux_metrics,
                    "grad_norms": gn,
                    "dispatch_by_layer": (self._last_dispatch_layer.tolist()
                                          if self._last_dispatch_layer is not None else None),
                    "dispatch_by_step": (self._last_dispatch_step.tolist()
                                         if self._last_dispatch_step is not None else None),
                }
                try:
                    with open(self._log_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception as e:
                    print(f"[raw log skip] {e}", flush=True)
        return super().log(logs, *args, **kwargs)


# ============================================================
# Boilerplate (signals, args, wandb, optimizer)
# ============================================================

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


# ============================================================
# Main
# ============================================================

def run_training(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # TF32: fp32 matmul/cudnn 가속 (bf16 외 연산 속도↑, 정밀도 영향 미미)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True   # 입력 shape 고정 → 커널 autotune

    assert os.path.exists(args.train_bin_path), f"파일 없음: {args.train_bin_path}"
    assert os.path.exists(args.val_bin_path),   f"파일 없음: {args.val_bin_path}"

    tokenizer = TiktokenHFWrapper("r50k_base")

    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id,
        experts=args.experts, base=args.rope_base, dropout=args.dropout,
        ponder_beta=args.ponder_beta, lambda_p=args.lambda_p, alpha=args.alpha,
    )

    # AMoE의 기존 속성을 인스턴스 단위로 세팅 (model.py 원본 불변)
    #   ponder_steps → max_steps (수직 루프 횟수), grad_checkpoint → use_checkpoint
    for _atten, _amoe in model.transformer.layers:
        _amoe.max_steps = args.ponder_steps
        _amoe.use_checkpoint = bool(args.grad_checkpoint)

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

    # sweep(--resume=0): run마다 고유 output_dir(<run_id>) → 동시 agent 충돌 방지.
    # 메인 학습(--resume=1): main_out_ponder_<n> 로 고정 → n별 분리 + 박스 재시작 시 재개.
    if not args.resume:
        if wandb.run is not None:
            args.output_dir = os.path.join(args.output_dir, wandb.run.id)
    else:
        args.output_dir = f"{args.output_dir}_ponder_{args.ponder_steps}"

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
        save_strategy=("steps" if args.resume else "no"),  # 풀런(resume=1)만 저장, sweep은 저장 안 함
        save_steps=args.save_steps,
        save_total_limit=2,                                 # 마지막 2개만 유지
        bf16=torch.cuda.is_available(),
        report_to="wandb",
        run_name=args.run_name,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        seed=args.seed,
        max_steps=max_steps,
        max_grad_norm=args.max_grad_norm,
        torch_compile=bool(args.compile),
    )

    optimizer = build_muon_optimizer(model, args)

    # raw hooked-data log file: logs/<run_id>_log.jsonl (per-run → safe for concurrent agents)
    tag = (wandb.run.id if wandb.run is not None
           else (args.run_name or "run"))
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"{tag}_log.jsonl")
    print(f"[Hooks] raw hooked-data log -> {log_path}")

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
        log_path=log_path,
    )
    # --resume: 고정 output_dir에 남은 마지막 체크포인트가 있으면 거기서 재개 (없으면 새로).
    resume_ckpt = None
    if args.resume and os.path.isdir(args.output_dir):
        resume_ckpt = get_last_checkpoint(args.output_dir)
        if resume_ckpt:
            print(f"[Resume] {resume_ckpt} 에서 재개")
    trainer.train(resume_from_checkpoint=resume_ckpt)

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
