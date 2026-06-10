"""
eval_compute.py — 학습된 대조군(n-loop) 체크포인트의 **효율 지표**를 val.bin(eval)에서 측정.

품질(task_CE·LBL·perplexity)은 eval_custom.py가 담당한다. 여기서는 효율 지표만 잰다:

    - 벽시계 시간 (time_total_s, tokens_per_s, ms_per_forward)
    - FLOPs (total, per token)  ← flops_count.py와 동일 측정(MAC×2)

대조군 AMoE는 halting 없이 항상 고정 N(=ponder_steps)회 도므로 "평균 ponder step"/"가변 스텝"
개념이 없다(N은 상수). FLOPs 계측(FlopCounterMode)은 op를 instrument 하느라 시간을 왜곡하므로,
**시간 측정과 FLOPs 측정을 서로 다른 패스로 분리**한다. 결과는 W&B(bench/* 스칼라)에 로깅된다.

  python eval_compute.py --ckpt main_out/model.safetensors --val_bin val.bin \
      --project AMOE-SWEEP123 --run_name my-bench
"""

import os
import time
import argparse

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm
import wandb

from model import LLM, TiktokenHFWrapper
from train_common import init_wandb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="main_out/model.safetensors",
                   help="safetensors 가중치 경로 (없으면 랜덤 init — 라우팅이 달라 비대표적)")
    p.add_argument("--val_bin", default="val.bin")
    p.add_argument("--block_size", type=int, default=768)
    p.add_argument("--max_tokens", type=int, default=0,
                   help="val.bin 앞에서부터 측정에 쓸 토큰 수, batch는 1 고정 (0=전체)")
    p.add_argument("--ponder_steps", type=int, default=8,
                   help="고정 재귀 횟수 N = AMoE.max_steps (학습과 일치)")
    # 아키텍처 (train_main.sh와 동일)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=3072)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--rope_base", type=int, default=10000)
    # W&B (eval 전용 새 run)
    p.add_argument("--project", default="amoe-eval", help="W&B project")
    p.add_argument("--run_name", default=None, help="W&B run 이름 (None=자동)")
    return p.parse_args()


def main():
    args = parse_args()
    args = init_wandb(args)                                # eval 전용 새 run; vars(args)가 config로 기록(ckpt 포함)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = TiktokenHFWrapper("r50k_base")

    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tok.vocab_size, padding_idx=tok.pad_token_id,
        experts=args.experts, base=args.rope_base, dropout=0.0,
    )

    # 학습과 동일하게: 각 AMoE 고정 재귀 = ponder_steps, 추론이라 checkpoint off.
    for _atten, amoe in model.transformer.layers:
        amoe.max_steps = args.ponder_steps
        amoe.use_checkpoint = False

    if os.path.exists(args.ckpt):
        from safetensors.torch import load_file
        sd = load_file(args.ckpt, device=str(device)) if args.ckpt.endswith(".safetensors") \
            else torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        print(f"[load] {args.ckpt}")
    else:
        print(f"[warn] ckpt 없음({args.ckpt}) → 랜덤 init로 측정. "
              f"라우팅이 달라 FLOPs 분포는 비대표적입니다.")

    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    # 데이터: val.bin 앞에서부터 max_tokens 토큰 → block_size 청크(마지막은 부분), batch=1
    raw = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    n_tok = len(raw) if args.max_tokens <= 0 else min(args.max_tokens, len(raw))
    data = np.asarray(raw[:n_tok]).astype(np.int64)
    chunks = [torch.from_numpy(data[i:i + args.block_size]).unsqueeze(0).to(device)
              for i in range(0, n_tok, args.block_size)]
    print(f"[data] {args.val_bin} | 앞에서부터 {n_tok:,} tok | block={args.block_size} batch=1 "
          f"| forwards={len(chunks)}")
    print(f"[model] params={n_params/1e6:.2f}M | recurrence N={args.ponder_steps} | device={device}")

    # ============ Pass 1: 시간 측정 (FlopCounter 없이) ============
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        # warmup: lazy init/cudnn autotune 등을 타이밍에서 제외
        for ch in chunks[:2]:
            model(input_ids=ch)
        sync()

        t0 = time.perf_counter()
        total_tokens = 0
        for ch in tqdm(chunks, desc="timing"):
            model(input_ids=ch)
            total_tokens += ch.numel()
        sync()
        time_total_s = time.perf_counter() - t0

    tokens_per_s   = total_tokens / max(1e-9, time_total_s)
    ms_per_forward = 1e3 * time_total_s / max(1, len(chunks))

    # ============ Pass 2: FLOPs ============
    fcm = FlopCounterMode(display=False)
    with torch.no_grad(), fcm:
        for ch in tqdm(chunks, desc="flops"):
            model(input_ids=ch)

    total_flops    = fcm.get_total_flops()          # MAC당 2 FLOP로 카운트
    flops_per_token = total_flops / max(1, total_tokens)

    print("\n==================== COMPUTE (eval, 대조군 고정 N) ====================")
    print(f"측정 토큰          : {total_tokens:,}")
    print(f"time_total         : {time_total_s:.4f} s")
    print(f"tokens / s         : {tokens_per_s:.1f}")
    print(f"ms / forward       : {ms_per_forward:.3f} (forward 1회 = 시퀀스 1개)")
    print(f"총 FLOPs(샘플)     : {total_flops:.3e}")
    print(f"FLOPs / token      : {flops_per_token:.3e}")
    print(f"고정 재귀 N        : {args.ponder_steps} (halting 없음 → 상수)")
    print("주: torch flop_counter는 matmul을 MAC×2로 카운트. sort/scatter/elementwise는 미포함.")

    logs = {
        "bench/time_total_s": time_total_s,
        "bench/tokens_per_s": tokens_per_s,
        "bench/ms_per_forward": ms_per_forward,
        "bench/flops_total": total_flops,
        "bench/flops_per_token": flops_per_token,
        "bench/recurrence_n": args.ponder_steps,
        "bench/n_tokens": total_tokens,
    }
    wandb.log(logs)
    wandb.run.summary.update(logs)
    wandb.finish()


if __name__ == "__main__":
    main()
