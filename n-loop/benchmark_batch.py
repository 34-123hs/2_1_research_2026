"""
benchmark_batch.py — find the optimal (batch_size, grad_accum) split for a
FIXED effective batch of 24.

Effective batch = batch_size * grad_accum = 24, so the candidates are:
    24x1, 12x2, 8x3, 6x4
Fewer grad_accum micro-steps => less accumulation overhead => faster, so the
optimum is the LARGEST batch_size that fits in memory. We measure peak memory
and throughput at the sweep's worst-case (largest) model so the chosen split is
safe for every trial.

Faithful to train_custom.py: fp16 autocast + GradScaler, Muon optimizer, and the
model's internal gradient checkpointing (active when model.training).
"""

import argparse
import time
import torch
import numpy as np
from model import LLM, TiktokenHFWrapper, MemmapDataset
from optim import build_muon_optimizer
from config import add_base_args

EFFECTIVE_BATCH = 24
CANDIDATES = [(24, 1), (12, 2), (8, 3), (6, 4)]  # (batch_size, grad_accum)


def make_args(batch_size, grad_accum, base):
    args = argparse.Namespace(**vars(base))
    args.batch_size = batch_size
    args.grad_accum = grad_accum
    return args


def bench_one(batch_size, grad_accum, base, n_eff_steps, warmup_eff_steps):
    """Run a few effective optimizer steps; return (tok_per_s, peak_gb) or None on OOM."""
    torch.manual_seed(base.seed)
    np.random.seed(base.seed)
    device = "cuda"

    tokenizer = TiktokenHFWrapper("r50k_base")
    model = LLM(
        dim=base.dim, depth=base.depth, max_len=base.block_size,
        mlp_dim=base.mlp_dim, heads=base.heads, dim_head=base.dim_head,
        vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id,
        experts=base.experts, base=base.rope_base, dropout=base.dropout,
        ponder_beta=base.ponder_beta, lambda_p=base.lambda_p,
    ).to(device)
    model.train()

    args = make_args(batch_size, grad_accum, base)
    optimizer = build_muon_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler()

    ds = MemmapDataset(base.train_bin_path, base.block_size)
    block = base.block_size

    def get_batch(step, micro):
        # deterministic contiguous blocks; labels = input_ids (model shifts internally for CE).
        idxs = [((step * grad_accum + micro) * batch_size + b) % len(ds)
                for b in range(batch_size)]
        xs = torch.stack([ds[i]["input_ids"] for i in idxs])
        return xs.to(device)

    torch.cuda.reset_peak_memory_stats(device)
    tokens_per_eff_step = batch_size * grad_accum * block

    try:
        t_start = None
        for step in range(warmup_eff_steps + n_eff_steps):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(grad_accum):
                x = get_batch(step, micro)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(input_ids=x, labels=x)
                    loss = out["loss"] if isinstance(out, dict) else out[0]
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            if step == warmup_eff_steps - 1:
                t_start = time.time()
        elapsed = time.time() - t_start
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    tok_per_s = (n_eff_steps * tokens_per_eff_step) / elapsed
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return tok_per_s, peak_gb


def main():
    p = argparse.ArgumentParser()
    add_base_args(p, output_dir_default="bench-out")
    p.add_argument("--n_eff_steps", type=int, default=8)
    p.add_argument("--warmup_eff_steps", type=int, default=3)
    base = p.parse_args()

    print(f"[Config] dim={base.dim} depth={base.depth} block={base.block_size} "
          f"experts={base.experts} | effective_batch={EFFECTIVE_BATCH} | fp16")
    print(f"{'batch x accum':>14} | {'peak mem (GB)':>13} | {'tok/s':>10} | note")
    print("-" * 60)

    results = []
    for bs, ga in CANDIDATES:
        assert bs * ga == EFFECTIVE_BATCH
        r = bench_one(bs, ga, base, base.n_eff_steps, base.warmup_eff_steps)
        if r is None:
            print(f"{f'{bs}x{ga}':>14} | {'OOM':>13} | {'-':>10} | does not fit")
            results.append((bs, ga, None, None))
        else:
            tok_s, peak = r
            print(f"{f'{bs}x{ga}':>14} | {peak:>13.2f} | {tok_s:>10,.0f} |")
            results.append((bs, ga, tok_s, peak))

    fit = [r for r in results if r[2] is not None]
    if fit:
        best = max(fit, key=lambda r: r[2])
        print("-" * 60)
        print(f"[Optimal] batch_size={best[0]} grad_accum={best[1]} "
              f"({best[2]:,.0f} tok/s, {best[3]:.2f} GB)")


if __name__ == "__main__":
    main()
