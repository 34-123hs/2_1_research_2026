"""
measure_flops.py

학습된 PonderNet 모델의 '추론 FLOPs' 측정.
- val.bin 앞쪽 N 토큰(기본 100,000)을 block_size 윈도우로 잘라 batch_size=1 로 forward.
- torch.utils.flop_counter.FlopCounterMode 로 '실제 실행된 aten 연산'의 FLOPs 를 합산
  → PonderNet 의 per-token 조기 종료(halt)가 그대로 반영됨 (AMoE adaptive-compute 비교용).
- 추론 경로를 타도록 model.eval() + labels 없이 호출. torch.compile 은 끈다(dispatch 방해).
- 관례: FlopCounterMode 는 matmul 을 2·M·N·K FLOP 로 센다(곱셈+덧셈).
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from safetensors.torch import load_file

from data import TiktokenHFWrapper
from pondernet import PonderLLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="final_outputs/model.safetensors",
                   help="model.safetensors 경로 (또는 final_outputs 폴더)")
    p.add_argument("--val_bin_path", default="val.bin")
    p.add_argument("--n_tokens", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--block_size", type=int, default=768)
    # 아키텍처 — 학습과 동일해야 weight 가 로드됨 (final 학습 config 기본값)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=3072)
    p.add_argument("--core_depth", type=int, default=12)
    p.add_argument("--ponder_steps", type=int, default=8)
    p.add_argument("--rope_base", type=int, default=10000)
    p.add_argument("--lambda_p", type=float, default=0.3)
    p.add_argument("--ponder_beta", type=float, default=0.01906962800737401)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = TiktokenHFWrapper("r50k_base")

    model = PonderLLM(
        dim=args.dim, max_len=args.block_size, mlp_dim=args.mlp_dim,
        heads=args.heads, dim_head=args.dim_head,
        vocab_size=tok.vocab_size, padding_idx=tok.pad_token_id,
        core_depth=args.core_depth, base=args.rope_base, dropout=0.0,
        max_steps=args.ponder_steps, lambda_p=args.lambda_p, ponder_beta=args.ponder_beta,
        use_checkpoint=False,
    )

    # weight 로드 (폴더를 주면 model.safetensors 를 붙임)
    wpath = args.weights
    if os.path.isdir(wpath):
        wpath = os.path.join(wpath, "model.safetensors")
    sd = load_file(wpath)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load] WARN missing={len(missing)} unexpected={len(unexpected)}")
        if missing:    print("  missing   :", missing[:5])
        if unexpected: print("  unexpected:", unexpected[:5])
    model.to(device).eval()

    # 데이터: val.bin '앞에서부터' 정확히 n_tokens 개. block_size 로 자르고,
    # 마지막 자투리(<block_size)는 짧은 시퀀스 1개로 그대로 넣어 토큰 수를 정확히 맞춘다.
    data = np.memmap(args.val_bin_path, dtype=np.uint16, mode="r")
    n_tokens = min(args.n_tokens, len(data))
    bs = args.block_size
    arr = np.asarray(data[:n_tokens]).astype(np.int64)
    n_full = n_tokens // bs
    rem = n_tokens - n_full * bs
    print(f"[data] {n_tokens:,} tokens (앞에서부터) = {n_full} x {bs}"
          + (f" + 1 x {rem}" if rem else "") + f"  (batch={args.batch_size})")

    # FLOPs 측정 — 실행되는 연산을 그대로 카운트
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        full = arr[:n_full * bs].reshape(n_full, bs)
        for i in range(0, n_full, args.batch_size):
            xb = torch.from_numpy(full[i:i + args.batch_size]).to(device)
            model(input_ids=xb)
        if rem:  # 자투리 토큰 (batch=1, 가변 길이)
            xb = torch.from_numpy(arr[n_full * bs:].reshape(1, rem)).to(device)
            model(input_ids=xb)
    measured = n_tokens

    total = counter.get_total_flops()
    print("\n===== PonderNet inference FLOPs (val.bin, batch=1) =====")
    print(f"total FLOPs    : {total:,}   ({total/1e9:.3f} GFLOP, {total/1e12:.4f} TFLOP)")
    print(f"tokens         : {measured:,}")
    print(f"FLOPs / token  : {total/measured:,.0f}   ({total/measured/1e6:.3f} MFLOP/token)")

    # 연산별 분해 (어디서 FLOPs 가 나오는지)
    glob = counter.get_flop_counts().get("Global", {})
    if glob:
        print("\nper-op breakdown (Global):")
        for op, f in sorted(glob.items(), key=lambda kv: -kv[1]):
            print(f"  {str(op):42s} {f:>20,}  ({100*f/total:5.1f}%)")
    print("\n(추론 모드 — PonderNet per-token 조기 종료가 반영된 실제 연산량입니다.)")


if __name__ == "__main__":
    main()
