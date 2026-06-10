"""
flops_count.py — 학습 완료된 대조군 모델의 FLOPs를 val.bin(eval 모드)에서 측정.

대조군은 고정 N(=ponder_steps)회 MoE 재귀 + top-1 sparsity라, FLOPs는 N과 top-1 라우팅에
의해 결정된다. eval 모드에서 실데이터로 재면 top-1 sparsity가 반영된 실제 연산량이 측정된다.
torch.utils.flop_counter.FlopCounterMode 가 matmul FLOPs를 카운트한다.

  python flops_count.py --val_bin val.bin --ckpt main_out/model.safetensors
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from model import LLM, TiktokenHFWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="main_out/model.safetensors",
                   help="safetensors 가중치 경로 (없으면 랜덤 init — 숫자 비대표적)")
    p.add_argument("--val_bin", default="val.bin")
    p.add_argument("--block_size", type=int, default=768)
    p.add_argument("--max_tokens", type=int, default=100_000,
                   help="val.bin 앞에서부터 측정에 쓸 토큰 수 (batch는 1 고정)")
    p.add_argument("--ponder_steps", type=int, default=8, help="고정 재귀 횟수 N = AMoE.max_steps (학습과 일치)")
    # 아키텍처 (train_main.sh와 동일)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=3072)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--rope_base", type=int, default=10000)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TiktokenHFWrapper("r50k_base")

    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tok.vocab_size, padding_idx=tok.pad_token_id,
        experts=args.experts, base=args.rope_base, dropout=0.0,
    )

    # ★ 학습과 동일하게: 각 AMoE의 수직 재귀 횟수 = ponder_steps, 추론이라 checkpoint off
    for _atten, _amoe in model.transformer.layers:
        _amoe.max_steps = args.ponder_steps
        _amoe.use_checkpoint = False

    # 가중치 로드 (없으면 랜덤 init — 라우팅이 달라 FLOPs 분포가 비대표적)
    if os.path.exists(args.ckpt):
        from safetensors.torch import load_file
        sd = load_file(args.ckpt, device=device) if args.ckpt.endswith(".safetensors") \
            else torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd)
        print(f"[load] {args.ckpt}")
    else:
        print(f"[warn] ckpt 없음({args.ckpt}) → 랜덤 init로 측정. "
              f"라우팅이 달라 FLOPs 분포는 비대표적입니다.")

    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    # 데이터: val.bin 앞에서부터 정확히 max_tokens 토큰 → block_size 청크(마지막은 부분), batch=1
    raw = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    n_tok = min(args.max_tokens, len(raw))
    data = np.asarray(raw[:n_tok]).astype(np.int64)
    chunks = [data[i:i + args.block_size] for i in range(0, n_tok, args.block_size)]
    print(f"[data] {args.val_bin} | 앞에서부터 {n_tok:,} tok | block={args.block_size} batch=1 "
          f"| forwards={len(chunks)}")
    print(f"[model] params={n_params/1e6:.2f}M | recurrence N={args.ponder_steps} | device={device}")

    total_tokens = 0
    fcm = FlopCounterMode(display=False)
    with torch.no_grad(), fcm:
        for ch in chunks:
            x = torch.from_numpy(ch).unsqueeze(0).to(device)   # [1, len]
            model(input_ids=x)
            total_tokens += x.numel()

    total_flops = fcm.get_total_flops()        # torch flop_counter 기준(MAC당 2 FLOP로 카운트)
    fpt = total_flops / total_tokens
    val_total_tok = os.path.getsize(args.val_bin) // 2   # uint16

    print("\n==================== FLOPs (eval, 적응) ====================")
    print(f"측정 토큰          : {total_tokens:,}")
    print(f"총 FLOPs(샘플)     : {total_flops:.3e}")
    print(f"FLOPs / token      : {fpt:.3e}")
    print(f"FLOPs / forward    : {total_flops/max(1,len(chunks)):.3e} (forward 1회 = 시퀀스 1개)")
    print(f"val.bin 전체 외삽  : {fpt*val_total_tok:.3e}  (전체 {val_total_tok:,} tok 가정)")
    print("주: torch flop_counter는 matmul을 MAC×2(FLOP)로 카운트. sort/scatter/elementwise는 미포함(무시 가능).")

    # 모듈/op별 분해 (가능하면)
    try:
        counts = fcm.get_flop_counts()         # {module: {op: flops}}
        glob = counts.get("Global", {})
        if glob:
            print("\n---- op별 FLOPs (Global) ----")
            for op, fl in sorted(glob.items(), key=lambda kv: -kv[1]):
                print(f"  {str(op):40s} {fl:.3e}  ({100*fl/total_flops:.1f}%)")
    except Exception as e:
        print(f"[분해 생략] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
