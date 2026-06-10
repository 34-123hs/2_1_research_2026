"""
eval_custom.py — 학습된 대조군(n-loop) 체크포인트(safetensors)를 val.bin에서 평가한다.

대조군 학습 loss = task_CE + alpha·LBL (halting/ponder 없음). 여기서도 동일하게:

    eval_loss = task_CE + alpha·LBL          # 대조군은 ponder 항이 없어 학습 loss와 동일

perplexity 는 순수 task_CE 기준 exp(task_CE) 로 보고한다(LBL은 NLL이 아님).

LLM.forward 는 (loss, logits) 만 돌려주고 LBL 을 따로 노출하지 않으므로,
forward 본문을 복제해 model.transformer(x) → (x, total_LBL) 에서 LBL 을 직접 꺼낸다. model.py 는 불변.

대조군 AMoE는 halting/조기 break 없이 항상 고정 max_steps 회 도므로(고정 N-loop),
2D_AMOE처럼 고정 스텝을 강제할 필요가 없다. 결과는 W&B(eval/* 스칼라)에 로깅된다.

  python eval_custom.py --ckpt main_out/model.safetensors --val_bin val.bin \
      --project AMOE-SWEEP123 --run_name my-eval
"""

import os
import math
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from model import LLM, TiktokenHFWrapper, MemmapDataset
from train_common import init_wandb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="safetensors(.safetensors) 또는 .pt/.bin 가중치 경로")
    p.add_argument("--val_bin", default="val.bin")
    p.add_argument("--block_size", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_val_size", type=int, default=0,
                   help="val.bin에서 사용할 토큰 상한 (0=전체)")
    # 아키텍처 (train_main.sh와 동일해야 state_dict가 로드됨)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=3072)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--rope_base", type=int, default=10000)
    p.add_argument("--ponder_steps", type=int, default=8,
                   help="고정 재귀 횟수 N = AMoE.max_steps (학습과 일치)")
    p.add_argument("--alpha", type=float, default=0.005,
                   help="보고 loss의 LBL 가중치 (train_main.sh와 일치)")
    # W&B (eval 전용 새 run)
    p.add_argument("--project", default="amoe-eval", help="W&B project")
    p.add_argument("--run_name", default=None, help="W&B run 이름 (None=자동)")
    return p.parse_args()


def load_weights(model, ckpt, device):
    """
    체크포인트 가중치를 model에 적재.
    input : model (nn.Module), ckpt (str) 경로, device
    output: 없음. (실패 시 load_state_dict가 shape 불일치를 명확히 raise → 아키텍처 플래그 점검)
    """
    if ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(ckpt, device=str(device))
    else:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)


@torch.no_grad()
def evaluate(model, loader, alpha, device):
    """
    대조군 eval. LLM.forward 본문을 복제해 LBL 을 직접 꺼낸다(transformer → (x, LBL)).
    input : model (eval 상태), loader (DataLoader), alpha (float), device
    output: dict(eval_loss, task_ce, avg_lbl, perplexity, n_tokens)
    """
    ce_sum_total = 0.0      # 모든 토큰 CE 합 (reduction="sum" 누적)
    tok_total    = 0        # ignore(-100) 아닌 토큰 수
    lbl_total    = 0.0      # per-batch LBL 합
    n_batches    = 0

    for batch in tqdm(loader, desc="eval"):
        ids = batch["input_ids"].to(device)                # [B, block]
        x = model.embedding(ids)                           # [B, block, D]
        x = model.dropout(x)                               # eval ⇒ no-op (fidelity용)
        x, total_LBL = model.transformer(x)                # [B,block,D], scalar (halting 없음)
        logits = model.mlp_head(x)                         # [B, block, V]

        shift_logits = logits[:, :-1, :].contiguous()      # [B, block-1, V]
        shift_labels = ids[:, 1:].contiguous()             # [B, block-1]
        ce_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),  # [B*(block-1), V]
            shift_labels.view(-1),                         # [B*(block-1)]
            ignore_index=-100, reduction="sum",
        )
        n_tok = int((shift_labels.view(-1) != -100).sum())

        ce_sum_total += float(ce_sum)
        tok_total    += n_tok
        lbl_total    += float(total_LBL)
        n_batches    += 1

    task_ce   = ce_sum_total / max(1, tok_total)           # 토큰가중 평균 CE
    avg_lbl   = lbl_total / max(1, n_batches)              # batch 평균 LBL
    eval_loss = task_ce + alpha * avg_lbl                  # 대조군 학습 loss와 동일
    ppl       = math.exp(task_ce) if task_ce < 20 else float("inf")
    return dict(eval_loss=eval_loss, task_ce=task_ce, avg_lbl=avg_lbl,
                perplexity=ppl, n_tokens=tok_total)


def main():
    args = parse_args()
    args = init_wandb(args)                                # eval 전용 새 run; vars(args)가 config로 기록(ckpt 포함)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    assert os.path.exists(args.ckpt), f"체크포인트 없음: {args.ckpt}"
    assert os.path.exists(args.val_bin), f"val.bin 없음: {args.val_bin}"

    tok = TiktokenHFWrapper("r50k_base")
    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tok.vocab_size, padding_idx=tok.pad_token_id,
        experts=args.experts, base=args.rope_base, dropout=0.0,
        alpha=args.alpha,
    )
    load_weights(model, args.ckpt, device)
    model.to(device).eval()
    # 학습과 동일하게: AMoE 고정 재귀 = ponder_steps, 추론이라 checkpoint off.
    # (대조군은 halting/조기 break이 없어 항상 max_steps 회 돈다 → 고정 스텝 강제 불필요)
    for _atten, amoe in model.transformer.layers:
        amoe.max_steps = args.ponder_steps
        amoe.use_checkpoint = False
    print(f"[load] {args.ckpt}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M  dim={args.dim} depth={args.depth} "
          f"experts={args.experts} recurrence_N={args.ponder_steps}")

    max_tokens = args.max_val_size if args.max_val_size > 0 else None
    ds = MemmapDataset(args.val_bin, args.block_size, max_tokens=max_tokens)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"[data] {args.val_bin} | blocks={len(ds)} | block_size={args.block_size} "
          f"batch_size={args.batch_size}")

    m = evaluate(model, loader, args.alpha, device)

    print("\n==================== EVAL (대조군, 학습 loss와 동일) ====================")
    print(f"eval_loss (task_CE + {args.alpha}·LBL) : {m['eval_loss']:.4f}")
    print(f"  task_CE                            : {m['task_ce']:.4f}")
    print(f"  avg LBL                            : {m['avg_lbl']:.4f}")
    print(f"perplexity (exp task_CE)             : {m['perplexity']:.2f}")
    print(f"eval tokens                          : {m['n_tokens']:,}")

    logs = {
        "eval/loss": m["eval_loss"],        # task_CE + alpha·LBL (= 대조군 학습 loss)
        "eval/task_ce": m["task_ce"],       # task_CE 단독
        "eval/avg_lbl": m["avg_lbl"],
        "eval/perplexity": m["perplexity"],
        "eval/n_tokens": m["n_tokens"],
    }
    wandb.log(logs)
    wandb.run.summary.update(logs)
    wandb.finish()


if __name__ == "__main__":
    main()
