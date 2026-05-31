"""
check_pondernet.py

E2E compatibility + rough-timing checker for the PonderNet baseline (pondernet.py).
Self-contained (uses only local copied modules). Tiny synthetic data, no real
training. Same checks as AMOE/check_model.py plus a PonderNet-specific halting check.

WHAT IT CHECKS:
  1. construct      — PonderLLM(**train kwargs) builds; params + Muon/Aux split
  2. train-forward  — model(ids, labels) -> .loss (finite, grad) + .logits [B,N,V]
  3. backward       — finite grads flow
  4. optimizer-step — split_params + build_muon_optimizer -> step()
  5. infer-forward  — model(ids) -> .loss None, finite .logits
  6. halting        — per-token Σ_n p_n ≈ 1 (valid PonderNet halting distribution)
  7. hf-trainer     — real HF Trainer 3-step train+eval on synthetic data
  8. generation     — autoregressive decode loop
  9. timing         — fwd (eval) / fwd+bwd (train) ms, tok/s, peak VRAM

Usage:
  python check_pondernet.py                          # standard config, GPU if available
  python check_pondernet.py --ponder_steps 10 --core_depth 6
Exit code is non-zero if any non-timing check FAILS.
"""
import argparse
import os
import time
import traceback

import numpy as np
import torch

import pondernet as P
from data import TiktokenHFWrapper, MemmapDataset
from optim import split_params, build_muon_optimizer


class Results:
    def __init__(self):
        self.rows = []

    def record(self, name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
        self.rows.append(ok)

    def warn(self, name, detail=""):
        print(f"  [WARN] {name}" + (f"  — {detail}" if detail else ""))

    @property
    def n_fail(self):
        return self.rows.count(False)


def check(res, name, fn):
    try:
        res.record(name, True, fn() or "")
        return True
    except Exception as e:
        traceback.print_exc()
        res.record(name, False, f"{type(e).__name__}: {e}")
        return False


class _OptArgs:
    muon_lr = 0.02
    muon_momentum = 0.95
    lr = 3e-4
    weight_decay = 0.1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--core_depth", type=int, default=6)
    p.add_argument("--ponder_steps", type=int, default=10)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=2048)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--rope_base", type=int, default=10000)
    p.add_argument("--ponder_beta", type=float, default=0.01)
    p.add_argument("--lambda_p", type=float, default=0.2)
    p.add_argument("--time_reps", type=int, default=20)
    p.add_argument("--time_warmup", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build(args, tok, dev):
    """Construct PonderLLM with the exact kwargs train_pondernet.run_training uses."""
    return P.PonderLLM(
        dim=args.dim, max_len=args.block_size, mlp_dim=args.mlp_dim,
        heads=args.heads, dim_head=args.dim_head,
        vocab_size=tok.vocab_size, padding_idx=tok.pad_token_id,
        core_depth=args.core_depth, base=args.rope_base, dropout=0.0,
        max_steps=args.ponder_steps, lambda_p=args.lambda_p, ponder_beta=args.ponder_beta,
    ).to(dev)


@torch.no_grad()
def _generate(model, ids, max_new_tokens, top_k=50):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(ids).logits[:, -1, :]
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


def main():
    args = parse_args()
    dev = torch.device(args.device)
    B, N = args.batch_size, args.block_size
    res = Results()
    print("=" * 70)
    print(f"check_pondernet.py — device={dev}  dim={args.dim} core_depth={args.core_depth} "
          f"ponder_steps={args.ponder_steps} block={N} batch={B}")
    print("=" * 70)

    tok = TiktokenHFWrapper("r50k_base")
    V = tok.vocab_size
    st = {}

    def _construct():
        torch.manual_seed(0)
        m = build(args, tok, dev); st["m"] = m
        n = sum(p.numel() for p in m.parameters())
        mu, ax = split_params(m)
        return (f"params={n/1e6:.1f}M  Muon={sum(p.numel() for p in mu)/1e6:.1f}M  "
                f"Aux={sum(p.numel() for p in ax)/1e6:.1f}M  vocab={V}")
    if not check(res, "1. construct PonderLLM(**train kwargs)", _construct):
        print("\nconstruct failed — stopping."); raise SystemExit(1)
    m = st["m"]
    ids = lambda: torch.randint(1, V, (B, N), device=dev)

    def _train_fwd():
        m.train()
        out = m(input_ids=ids(), labels=ids())
        assert out.loss is not None and out.loss.ndim == 0 and torch.isfinite(out.loss), "bad .loss"
        assert out.loss.requires_grad, ".loss not differentiable"
        assert tuple(out.logits.shape) == (B, N, V), f"logits {tuple(out.logits.shape)}"
        return f"loss={float(out.loss.detach()):.3f}  logits={tuple(out.logits.shape)}"
    check(res, "2. train forward -> .loss + .logits", _train_fwd)

    def _backward():
        m.zero_grad(set_to_none=True)
        m(input_ids=ids(), labels=ids()).loss.backward()
        g = [p for p in m.parameters() if p.grad is not None]
        assert g and all(torch.isfinite(p.grad).all() for p in g), "missing/non-finite grad"
        return f"{len(g)} params got finite grads"
    check(res, "3. backward -> finite grads", _backward)

    def _opt():
        mu, ax = split_params(m)
        if not mu:
            res.warn("4. optimizer", "Muon group EMPTY")
        opt = build_muon_optimizer(m, _OptArgs())
        m.zero_grad(set_to_none=True)
        m(input_ids=ids(), labels=ids()).loss.backward(); opt.step()
        return f"{type(opt).__name__} groups={len(opt.param_groups)}"
    check(res, "4. optimizer split + step", _opt)

    def _infer():
        m.eval()
        with torch.no_grad():
            out = m(input_ids=ids())
        assert getattr(out, "loss", None) is None, ".loss should be None"
        assert torch.isfinite(out.logits).all() and tuple(out.logits.shape) == (B, N, V)
        return "logits finite, loss=None"
    check(res, "5. inference forward (no labels)", _infer)

    def _halting():
        m.train()
        with torch.no_grad():
            m(input_ids=ids(), labels=ids())
        s = m._last_halt_sum                          # [B, N] Σ_n p_n per token
        err = (s - 1.0).abs().max().item()
        assert err < 1e-4, f"halting Σp_n deviates from 1 by {err:.2e}"
        return f"per-token Σ p_n = 1 (max dev {err:.1e}) — valid halting distribution"
    check(res, "6. halting distribution sums to 1", _halting)

    def _hf_trainer():
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        tmp = "/tmp/_check_pondernet.bin"
        np.random.default_rng(0).integers(0, V, max(B * N * 6, 50000), dtype=np.uint16).tofile(tmp)
        ds = MemmapDataset(tmp, N)
        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
        targs = TrainingArguments(
            output_dir="/tmp/_check_pondernet_out",
            max_steps=3, per_device_train_batch_size=B, per_device_eval_batch_size=B,
            gradient_accumulation_steps=1, logging_steps=1,
            eval_strategy="no", save_strategy="no", report_to=[],
            fp16=(dev.type == "cuda"), dataloader_pin_memory=(dev.type == "cuda"),
            disable_tqdm=True,
        )
        fresh = build(args, tok, torch.device("cpu"))
        trainer = Trainer(model=fresh, args=targs, train_dataset=ds, eval_dataset=ds,
                          data_collator=collator,
                          optimizers=(build_muon_optimizer(fresh, _OptArgs()), None))
        trainer.model_accepts_loss_kwargs = False
        out = trainer.train()
        ev = trainer.evaluate()
        assert np.isfinite(out.training_loss) and np.isfinite(ev.get("eval_loss", float("nan")))
        os.remove(tmp)
        return f"3-step train_loss={out.training_loss:.3f}  eval_loss={ev['eval_loss']:.3f}"
    check(res, "7. HF Trainer 3-step train+eval", _hf_trainer)

    def _gen():
        out = _generate(m, torch.randint(1, V, (1, 4), device=dev), max_new_tokens=8)
        assert out.shape[1] == 12, f"generation shape {tuple(out.shape)}"
        return f"grew 4 -> {out.shape[1]} tokens; decoded: {tok.decode(out[0,4:].tolist())[:30]!r}"
    check(res, "8. generation", _gen)

    # timing
    print("-" * 70)
    print("Timing (rough):")
    _timing(m, args, dev, B, N, V)

    print("=" * 70)
    print(f"RESULT: {'ALL PASS' if res.n_fail == 0 else str(res.n_fail) + ' FAILED'} ({len(res.rows)} checks)")
    print("=" * 70)
    raise SystemExit(1 if res.n_fail else 0)


def _timing(m, args, dev, B, N, V):
    cuda = dev.type == "cuda"
    toks = B * N
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    x = torch.randint(1, V, (B, N), device=dev)

    def bench(fn):
        for _ in range(args.time_warmup):
            fn()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.time_reps):
            fn()
        if cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / args.time_reps

    def fwd():
        m.eval()
        with torch.no_grad():
            m(input_ids=x)

    def step():
        m.train(); m.zero_grad(set_to_none=True)
        m(input_ids=x, labels=x).loss.backward()

    tf, ts = bench(fwd), bench(step)
    peak = f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB" if cuda else "n/a"
    print(f"  device          : {torch.cuda.get_device_name(0) if cuda else 'CPU'}")
    print(f"  forward (eval)  : {tf*1e3:8.1f} ms/iter  ({toks/tf:,.0f} tok/s)")
    print(f"  fwd+bwd (train) : {ts*1e3:8.1f} ms/step  ({toks/ts:,.0f} tok/s)")
    print(f"  peak VRAM       : {peak}   (tokens/iter = {toks:,})")


if __name__ == "__main__":
    main()
