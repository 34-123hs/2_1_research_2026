"""
train_with_hooks.py

train_custom.py와 동일한 인터페이스의 학습 스크립트.
다른 점: custom_model_train.py(원본)를 건드리지 않고 forward hook으로
  (1) Switch Transformer load-balance aux loss를 계산해 total loss에 더하고
  (2) per-layer router/halting/balance 지표를 wandb와 콘솔에 찍는다.

캡처 대상:
  - 각 layer의 MoE.gate (nn.Linear): forward hook → gate_logits → softmax → gate_probs (grad 유지)
      → Switch balance: L_b = E * sum_i(f_i * P_i),  f = mean(one_hot(argmax)),  P = mean(softmax)
  - 각 layer의 AMoE: forward hook → output[1] = halting_probs [T, B, N]
      → halting/mean_step (가중 평균 스텝수)

지표 (조직화):
  per-step (logging_steps마다 wandb로):
    aux/balance_loss          : Switch balance loss (총합, layer-mean, max_steps 평균)
    aux/ponder_loss           : 모델 내부에서 더해진 KL ponder. 추정값으로 분리 로깅
                                (loss - task_loss) / (ponder_beta) 또는 hook으로 재구성
    router/max_pct_global     : 전체 layer/스텝 평균에서 top-1 expert 비율 (collapse alarm)
    router/entropy_norm_global: 전체 layer 평균 router entropy / log(E) ∈ [0,1]
    halting/mean_step_global  : 전체 layer 평균 expected step count

  per-layer (선택, --log_per_layer 켜면):
    router/max_pct/L{i}
    router/entropy_norm/L{i}
    halting/mean_step/L{i}
    aux/balance/L{i}

  console:
    train_custom.py의 출력 + 매 logging_steps마다 위 global 지표 한 줄 요약
"""

import os
import math
import signal
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import wandb
from muon import SingleDeviceMuonWithAuxAdam as MuonWithAuxAdam
from custom_model_train import LLM, MoE, AMoE, TiktokenHFWrapper, MemmapDataset


# ============================================================
# Hooks
# ============================================================

class HookCollector:
    """
    Forward hook으로 gate logits / halting probs를 캡쳐.
    forward마다 리스트가 채워지고, compute_loss 안에서 소비 + clear.

    주의: MoE는 gradient checkpoint(use_reentrant=False)로 감싸져 있어 backward
    중에 forward가 재실행됨 → gate hook이 그때도 fire. 캡(depth*max_steps)을
    걸어 backward 중 추가 캡처를 무시한다.
    """
    def __init__(self):
        self.gate_logits_per_layer = []    # list of [S, E] tensors (grad alive)
        self.halting_probs_per_layer = []  # list of [T, B, N] tensors
        self._gate_cap = 10**9             # set by attach_hooks
        self._amoe_cap = 10**9

    def clear(self):
        self.gate_logits_per_layer.clear()
        self.halting_probs_per_layer.clear()

    def make_gate_hook(self):
        def _hook(module, inputs, output):
            if len(self.gate_logits_per_layer) >= self._gate_cap:
                return
            self.gate_logits_per_layer.append(output)
        return _hook

    def make_amoe_hook(self):
        def _hook(module, inputs, output):
            if len(self.halting_probs_per_layer) >= self._amoe_cap:
                return
            _, hp = output
            self.halting_probs_per_layer.append(hp)
        return _hook


def attach_hooks(model: LLM, collector: HookCollector):
    """Transformer.layers의 각 AMoE.moe.gate(nn.Linear), AMoE에 forward hook 등록."""
    handles = []
    depth = len(model.transformer.layers)
    max_steps = model.transformer.layers[0][1].max_steps
    collector._gate_cap = depth * max_steps
    collector._amoe_cap = depth
    for atten, amoe in model.transformer.layers:
        h1 = amoe.moe.gate.register_forward_hook(collector.make_gate_hook())
        h2 = amoe.register_forward_hook(collector.make_amoe_hook())
        handles.extend([h1, h2])
    return handles


# ============================================================
# Aux loss + metrics from captured tensors
# ============================================================

def compute_aux_and_metrics(collector: HookCollector,
                            depth: int,
                            max_steps: int,
                            log_per_layer: bool):
    """
    Returns:
      balance_loss: scalar tensor (grad alive)
      metrics: dict[str, float]  (wandb-loggable)
    """
    metrics = {}

    # ---- balance loss ----
    gate_logits_list = collector.gate_logits_per_layer  # length = depth * max_steps (if no recompute)
    n_gate = len(gate_logits_list)

    if n_gate == 0:
        balance_loss = torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
        metrics["debug/gate_capture_count"] = 0
        return balance_loss, metrics

    # E from first
    E = gate_logits_list[0].size(-1)
    metrics["debug/gate_capture_count"] = n_gate
    metrics["debug/expected_gate_count"] = depth * max_steps

    # per-(layer,step) balance, router stats
    per_layer_bal = []          # E[f·P] sum * E per layer (avg over steps)
    per_layer_max_pct = []
    per_layer_entropy_norm = []

    # Walk layers
    # Order: stability_check 분석에 의하면 hook 순서는 layer0의 step0..N, layer1의 step0..N, ...
    # 정확히는 forward 중 layer.depth 만큼 AMoE가 순서대로 fire되고 그 안에서 step마다 gate.fire.
    step_count_per_layer = n_gate // depth if n_gate % depth == 0 else 0

    if step_count_per_layer == 0:
        # 비균등 — fallback: 전체를 단순 평균
        all_bal = []
        all_max = []
        all_ent = []
        for gl in gate_logits_list:
            p = F.softmax(gl.float(), dim=-1)
            sel = p.argmax(dim=-1)
            f = F.one_hot(sel, num_classes=E).to(p.dtype).mean(dim=0)
            P = p.mean(dim=0)
            all_bal.append(E * (f * P).sum())
            all_max.append(f.max().detach())
            ent = -(p * p.clamp_min(1e-12).log()).sum(dim=-1).mean()
            all_ent.append((ent / math.log(E)).detach())
        balance_loss = torch.stack(all_bal).mean()
        metrics["router/max_pct_global"] = float(torch.stack(all_max).mean())
        metrics["router/entropy_norm_global"] = float(torch.stack(all_ent).mean())
        metrics["aux/balance_loss"] = float(balance_loss.detach())
    else:
        idx = 0
        for li in range(depth):
            bal_steps = []
            max_steps_li = []
            ent_steps_li = []
            for _ in range(step_count_per_layer):
                gl = gate_logits_list[idx]; idx += 1
                p = F.softmax(gl.float(), dim=-1)
                sel = p.argmax(dim=-1)
                f = F.one_hot(sel, num_classes=E).to(p.dtype).mean(dim=0)
                P = p.mean(dim=0)
                bal_steps.append(E * (f * P).sum())
                max_steps_li.append(f.max().detach())
                ent = -(p * p.clamp_min(1e-12).log()).sum(dim=-1).mean()
                ent_steps_li.append((ent / math.log(E)).detach())
            bal_layer = torch.stack(bal_steps).mean()      # avg over steps; grad alive
            per_layer_bal.append(bal_layer)
            per_layer_max_pct.append(float(torch.stack(max_steps_li).mean()))
            per_layer_entropy_norm.append(float(torch.stack(ent_steps_li).mean()))

        balance_loss = torch.stack(per_layer_bal).mean()    # mean over layers
        metrics["aux/balance_loss"] = float(balance_loss.detach())
        metrics["router/max_pct_global"] = float(np.mean(per_layer_max_pct))
        metrics["router/entropy_norm_global"] = float(np.mean(per_layer_entropy_norm))

        if log_per_layer:
            for li in range(depth):
                metrics[f"aux/balance/L{li}"] = float(per_layer_bal[li].detach())
                metrics[f"router/max_pct/L{li}"] = per_layer_max_pct[li]
                metrics[f"router/entropy_norm/L{li}"] = per_layer_entropy_norm[li]

    # ---- halting stats ----
    hp_list = collector.halting_probs_per_layer  # [T, B, N] each
    if len(hp_list) > 0:
        # expected step count per token = sum_t (step_cert[t] * (t+1))
        per_layer_mean_step = []
        for li, hp in enumerate(hp_list):
            T = hp.size(0)
            t_idx = torch.arange(1, T + 1, device=hp.device, dtype=hp.dtype)
            mean_step = (hp * t_idx.view(T, 1, 1)).sum(dim=0).mean()
            per_layer_mean_step.append(float(mean_step.detach()))
            if log_per_layer:
                metrics[f"halting/mean_step/L{li}"] = per_layer_mean_step[-1]
        metrics["halting/mean_step_global"] = float(np.mean(per_layer_mean_step))

    return balance_loss, metrics


# ============================================================
# Custom Trainer
# ============================================================

class HookedTrainer(Trainer):
    def __init__(self, *args, collector: HookCollector, depth: int,
                 max_steps_amoe: int, balance_beta: float,
                 log_per_layer: bool, console_log_interval: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self._collector = collector
        self._depth = depth
        self._max_steps_amoe = max_steps_amoe
        self._balance_beta = balance_beta
        self._log_per_layer = log_per_layer
        self._console_interval = console_log_interval
        self._last_aux_metrics = {}

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        self._collector.clear()
        outputs = model(**inputs)
        base_loss = outputs.loss  # = task + ponder_beta * ponder_kl

        bal_loss, metrics = compute_aux_and_metrics(
            self._collector,
            depth=self._depth,
            max_steps=self._max_steps_amoe,
            log_per_layer=self._log_per_layer,
        )

        total = base_loss + self._balance_beta * bal_loss

        # task/ponder 분리 추정용: ponder_loss는 모델 내부 → 외부에서 분리 불가능.
        # 대안: hooks로 halting_probs 잡고 동일 공식으로 재계산
        metrics["aux/base_loss"] = float(base_loss.detach())
        metrics["aux/total_loss"] = float(total.detach())

        self._last_aux_metrics = metrics

        if return_outputs:
            return total, outputs
        return total

    def log(self, logs, *args, **kwargs):
        # HF Trainer가 logging_steps마다 부르는 훅에 우리 metric 추가
        if self._last_aux_metrics:
            logs.update(self._last_aux_metrics)
            # 콘솔 요약
            step = self.state.global_step
            r_max = self._last_aux_metrics.get("router/max_pct_global", float("nan"))
            r_ent = self._last_aux_metrics.get("router/entropy_norm_global", float("nan"))
            h_ms  = self._last_aux_metrics.get("halting/mean_step_global", float("nan"))
            b_l   = self._last_aux_metrics.get("aux/balance_loss", float("nan"))
            print(f"[Aux step={step}] balance={b_l:.4f}  router_max={r_max:.3f}  "
                  f"router_ent_norm={r_ent:.3f}  halt_mean_step={h_ms:.2f}",
                  flush=True)
        return super().log(logs, *args, **kwargs)


# ============================================================
# Boilerplate (train_custom.py 그대로)
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
    p.add_argument("--project", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--train_bin_path", default="train.bin")
    p.add_argument("--val_bin_path", default="val.bin")
    p.add_argument("--output_dir", default="custom-llm-out")
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--max_size", type=int, default=50_000_000)
    p.add_argument("--max_val_size", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=576)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=2048)
    p.add_argument("--rope_base", type=int, default=10000)
    p.add_argument("--dropout", type=float, default=0.0)

    # AMoE
    p.add_argument("--experts",      type=int,   default=4)
    p.add_argument("--ponder_beta",  type=float, default=0.01)
    p.add_argument("--lambda_p",     type=float, default=0.2)

    # NEW: hook-based balance loss
    p.add_argument("--balance_beta", type=float, default=0.01,
                   help="Switch Transformer load-balance aux weight (hook 기반)")
    p.add_argument("--log_per_layer", action="store_true",
                   help="layer별 router/halting/balance를 wandb에 모두 기록")

    # Muon
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=0.1)
    return p.parse_args()


def init_wandb(args):
    wandb.init(project=args.project, name=args.run_name, config=vars(args),
               allow_val_change=True)
    for k, v in dict(wandb.config).items():
        if hasattr(args, k):
            setattr(args, k, v)
    print(f"args={vars(args)}")
    return args


def create_muon_optimizer(model, args):
    hidden_matrix_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        use_muon = (
            p.ndim == 2
            and "embedding" not in name
            and "mlp_head" not in name
        )
        if use_muon:
            hidden_matrix_params.append(p)
        else:
            other_params.append(p)

    n_muon = sum(p.numel() for p in hidden_matrix_params)
    n_other = sum(p.numel() for p in other_params)
    print(f"[Optimizer] Muon params={n_muon:,}  Aux params={n_other:,}")

    param_groups = [
        dict(params=hidden_matrix_params, lr=args.muon_lr,
             momentum=args.muon_momentum, weight_decay=args.weight_decay,
             use_muon=True),
        dict(params=other_params, lr=args.lr,
             weight_decay=args.weight_decay, use_muon=False),
    ]
    return MuonWithAuxAdam(param_groups)


def run_training(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert os.path.exists(args.train_bin_path), f"파일 없음: {args.train_bin_path}"
    assert os.path.exists(args.val_bin_path), f"파일 없음: {args.val_bin_path}"

    tokenizer = TiktokenHFWrapper("r50k_base")

    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id,
        experts=args.experts,
        base=args.rope_base, dropout=args.dropout,
        ponder_beta=args.ponder_beta, lambda_p=args.lambda_p,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] params={n_params/1e6:.2f}M")
    wandb.run.summary["n_params_M"] = n_params / 1e6

    # ----- Hook 등록 (모델 코드 무수정) -----
    collector = HookCollector()
    attach_hooks(model, collector)
    sample_amoe = model.transformer.layers[0][1]
    max_steps_amoe = sample_amoe.max_steps
    print(f"[Hooks] attached. depth={args.depth}  max_steps={max_steps_amoe}  "
          f"balance_beta={args.balance_beta}  log_per_layer={args.log_per_layer}")

    train_ds = MemmapDataset(args.train_bin_path, args.block_size)
    eval_ds = MemmapDataset(args.val_bin_path, args.block_size,
                            max_tokens=args.max_val_size)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    max_steps = max(1, math.ceil(args.max_size / tokens_per_step))
    print(f"[Budget] max_size={args.max_size:,} tokens → max_steps={max_steps:,} "
          f"(tokens/step={tokens_per_step:,})")

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
    )

    optimizer = create_muon_optimizer(model, args)

    trainer = HookedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        optimizers=(optimizer, None),
        # custom
        collector=collector,
        depth=args.depth,
        max_steps_amoe=max_steps_amoe,
        balance_beta=args.balance_beta,
        log_per_layer=args.log_per_layer,
        console_log_interval=targs.logging_steps,
    )
    trainer.train()

    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
    print(f"[Eval] loss={metrics['eval_loss']:.4f}  ppl={ppl:.2f}")
    wandb.log({"final/eval_loss": metrics["eval_loss"], "final/perplexity": ppl})
    trainer.save_model(args.output_dir)
    wandb.finish()


def main():
    install_signal_handlers()
    args = init_wandb(parse_args())
    run_training(args)


if __name__ == "__main__":
    main()
