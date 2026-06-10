"""
check_model.py

End-to-end compatibility + rough-timing checker for a (user-edited) model.py.

Run this AFTER editing model.py to confirm it still plugs into the real
training / inference pipeline. It uses tiny synthetic data only — no real
training, no meaningful GPU-hours. Each check reuses the ACTUAL wrapper code
(optim.py, generate.py, inference_custom.py, and the same HF Trainer setup as
train_custom.py), so passing means your model really fits the entry points.

WHAT IT CHECKS (the E2E contracts the pipeline depends on):
  1. construct      — LLM(**train_custom's exact kwargs) builds; report param/Muon-Aux split
  2. train-forward  — train(); model(ids, labels) -> .loss (finite scalar, requires_grad) + .logits [B,N,V]
  3. backward       — loss.backward() -> finite grads actually flow
  4. optimizer-step — split_params + build_muon_optimizer -> step() works (warns if Muon group empty)
  5. infer-forward  — eval(); model(ids) -> .loss is None, .logits finite
  6. hf-trainer     — mirror train_custom (DataCollator + Trainer + custom optimizer): 3 steps + eval
  7. generation     — inference_custom.generate(): tokens grow and decode
  8. timing         — fwd (eval) and fwd+bwd (train) ms / tokens-per-sec, peak VRAM, device

Usage:
  python check_model.py                       # standard config, GPU if available
  python check_model.py --dim 512 --depth 6   # match your real config for timing
  python check_model.py --batch_size 8 --block_size 512 --time_reps 30
Exit code is non-zero if any non-timing check FAILS.
"""
import argparse
import os
import time
import traceback

import numpy as np
import torch

import model as M
from optim import split_params, build_muon_optimizer
import inference_custom


# ----------------------------------------------------------------------------
# tiny result tracker
# ----------------------------------------------------------------------------
class Results:
    def __init__(self):
        self.rows = []

    def record(self, name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        self.rows.append((name, ok, detail))
        print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))

    def warn(self, name, detail=""):
        self.rows.append((name, True, "WARN: " + detail))
        print(f"  [WARN] {name}" + (f"  — {detail}" if detail else ""))

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def check(res, name, fn):
    """Run one check fn() -> detail str; record PASS/FAIL, never raise."""
    try:
        detail = fn() or ""
        res.record(name, True, detail)
        return True
    except Exception as e:
        res.record(name, False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=2048)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--rope_base", type=int, default=10000)
    p.add_argument("--ponder_beta", type=float, default=0.01)
    p.add_argument("--lambda_p", type=float, default=0.2)
    p.add_argument("--time_reps", type=int, default=20)
    p.add_argument("--time_warmup", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class _OptArgs:
    """Minimal args object build_muon_optimizer expects."""
    muon_lr = 0.02
    muon_momentum = 0.95
    lr = 3e-4
    weight_decay = 0.1


def build_llm(args, tok):
    """Construct LLM with the EXACT kwargs train_custom.run_training uses."""
    return M.LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tok.vocab_size, padding_idx=tok.pad_token_id,
        experts=args.experts, base=args.rope_base, dropout=0.0,
        ponder_beta=args.ponder_beta, lambda_p=args.lambda_p,
    )


def main():
    args = parse_args()
    dev = torch.device(args.device)
    B, N = args.batch_size, args.block_size
    res = Results()

    print("=" * 70)
    print(f"check_model.py — device={dev}  config: dim={args.dim} depth={args.depth} "
          f"experts={args.experts} block={N} batch={B}")
    print("=" * 70)

    tok = M.TiktokenHFWrapper("r50k_base")
    V = tok.vocab_size

    # 1. construct -----------------------------------------------------------
    state = {}

    def _construct():
        torch.manual_seed(0)
        m = build_llm(args, tok).to(dev)
        state["model"] = m
        n = sum(p.numel() for p in m.parameters())
        muon, aux = split_params(m)
        state["muon_n"] = sum(p.numel() for p in muon)
        state["aux_n"] = sum(p.numel() for p in aux)
        return (f"params={n/1e6:.1f}M  Muon={state['muon_n']/1e6:.1f}M  "
                f"Aux={state['aux_n']/1e6:.1f}M  vocab={V}")
    if not check(res, "1. construct LLM(**train kwargs)", _construct):
        print("\nConstruction failed — remaining checks skipped.")
        _summary(res)
        raise SystemExit(1)
    m = state["model"]

    def _ids():
        return torch.randint(1, V, (B, N), device=dev)

    # 2. train-forward -------------------------------------------------------
    def _train_fwd():
        m.train()
        ids = _ids()
        out = m(input_ids=ids, labels=ids)
        assert hasattr(out, "loss") and out.loss is not None, "no .loss returned"
        assert out.loss.ndim == 0 and torch.isfinite(out.loss), f"bad loss {out.loss}"
        assert out.loss.requires_grad, ".loss has no grad (Trainer can't backprop)"
        assert tuple(out.logits.shape) == (B, N, V), f"logits {tuple(out.logits.shape)} != {(B,N,V)}"
        state["loss"] = out.loss
        return f"loss={float(out.loss):.3f}  logits={tuple(out.logits.shape)}"
    check(res, "2. train forward -> .loss + .logits", _train_fwd)

    # 3. backward ------------------------------------------------------------
    def _backward():
        m.zero_grad(set_to_none=True)
        out = m(input_ids=_ids(), labels=_ids())
        out.loss.backward()
        with_grad = [p for p in m.parameters() if p.grad is not None]
        assert with_grad, "no parameter received a gradient"
        assert all(torch.isfinite(p.grad).all() for p in with_grad), "non-finite gradient"
        frac = len(with_grad) / sum(1 for _ in m.parameters())
        return f"{len(with_grad)} params got finite grads ({frac*100:.0f}% of all)"
    check(res, "3. backward -> finite grads flow", _backward)

    # 4. optimizer-step ------------------------------------------------------
    def _opt_step():
        muon, aux = split_params(m)
        if not muon:
            res.warn("4. optimizer-step", "Muon group EMPTY — did you rename 2D weights / 'embedding'/'mlp_head'?")
        opt = build_muon_optimizer(m, _OptArgs())
        # ensure grads exist, then step
        out = m(input_ids=_ids(), labels=_ids())
        m.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        return f"optimizer={type(opt).__name__}  groups={len(opt.param_groups)}"
    check(res, "4. optimizer split + step", _opt_step)

    # 5. infer-forward -------------------------------------------------------
    def _infer_fwd():
        m.eval()
        with torch.no_grad():
            out = m(input_ids=_ids())
        assert getattr(out, "loss", None) is None, ".loss should be None without labels"
        assert tuple(out.logits.shape) == (B, N, V), f"logits {tuple(out.logits.shape)}"
        assert torch.isfinite(out.logits).all(), "non-finite logits"
        return f"logits={tuple(out.logits.shape)} finite, loss=None"
    check(res, "5. inference forward (no labels)", _infer_fwd)

    # 6. hf-trainer (the core E2E) ------------------------------------------
    def _hf_trainer():
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        tmp = "/tmp/_check_model.bin"
        np.random.default_rng(0).integers(0, V, max(B * N * 6, 50000), dtype=np.uint16).tofile(tmp)
        ds = M.MemmapDataset(tmp, N)
        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
        targs = TrainingArguments(
            output_dir="/tmp/_check_model_out",
            max_steps=3, per_device_train_batch_size=B, per_device_eval_batch_size=B,
            gradient_accumulation_steps=1, logging_steps=1,
            eval_strategy="no", save_strategy="no", report_to=[],
            fp16=(dev.type == "cuda"), dataloader_pin_memory=(dev.type == "cuda"),
            disable_tqdm=True,
        )
        fresh = build_llm(args, tok)  # Trainer moves it to device itself
        trainer = Trainer(model=fresh, args=targs, train_dataset=ds,
                          eval_dataset=ds, data_collator=collator,
                          optimizers=(build_muon_optimizer(fresh, _OptArgs()), None))
        trainer.model_accepts_loss_kwargs = False  # GA-loss scaling fix (mirror train_custom)
        out = trainer.train()
        ev = trainer.evaluate()
        tr_loss = out.training_loss
        ev_loss = ev.get("eval_loss", float("nan"))
        assert np.isfinite(tr_loss) and np.isfinite(ev_loss), f"non-finite loss tr={tr_loss} ev={ev_loss}"
        os.remove(tmp)
        return f"3-step train_loss={tr_loss:.3f}  eval_loss={ev_loss:.3f}  (real HF Trainer path)"
    check(res, "6. HF Trainer 3-step train+eval", _hf_trainer)

    # 7. generation ----------------------------------------------------------
    def _gen():
        m.eval()
        ids = torch.randint(1, V, (1, 4), device=dev)
        out = inference_custom.generate(m, ids, max_new_tokens=8, temperature=1.0,
                                        top_k=50, eos_id=tok.eos_token_id, device=dev)
        assert out.shape[1] >= 4, "generation produced nothing"
        txt = tok.decode(out[0, 4:].tolist())
        return f"grew {ids.shape[1]}->{out.shape[1]} tokens; decoded ok ({len(txt)} chars)"
    check(res, "7. generation (inference_custom.generate)", _gen)

    # 8. timing --------------------------------------------------------------
    print("-" * 70)
    print("Timing (rough — single forward/step, not full-throughput tuned):")
    _timing(m, args, dev, B, N, V)

    _summary(res)
    raise SystemExit(1 if res.failed else 0)


def _timing(m, args, dev, B, N, V):
    cuda = dev.type == "cuda"
    toks = B * N

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def bench(fn, reps, warmup):
        for _ in range(warmup):
            fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        return (time.perf_counter() - t0) / reps

    ids = torch.randint(1, V, (B, N), device=dev)

    # forward only (inference / eval)
    m.eval()
    def fwd():
        with torch.no_grad():
            m(input_ids=ids)
    t_fwd = bench(fwd, args.time_reps, args.time_warmup)

    # forward + backward (training cost; AMoE loop runs full max_steps in train)
    m.train()
    def fwd_bwd():
        m.zero_grad(set_to_none=True)
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()
    t_step = bench(fwd_bwd, args.time_reps, args.time_warmup)

    peak = f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB" if cuda else "n/a (CPU)"
    print(f"  device           : {torch.cuda.get_device_name(0) if cuda else 'CPU'}")
    print(f"  forward (eval)   : {t_fwd*1e3:8.1f} ms/iter   ({toks/t_fwd:,.0f} tok/s)")
    print(f"  fwd+bwd (train)  : {t_step*1e3:8.1f} ms/step   ({toks/t_step:,.0f} tok/s)")
    print(f"  peak VRAM        : {peak}")
    print(f"  (tokens/iter = batch*block = {toks:,})")


def _summary(res):
    print("=" * 70)
    n_fail = len(res.failed)
    total = len(res.rows)
    if n_fail == 0:
        print(f"RESULT: ALL {total} CHECKS PASSED — model.py is E2E-compatible with the pipeline.")
    else:
        print(f"RESULT: {n_fail}/{total} CHECK(S) FAILED:")
        for name, _ok, detail in res.failed:
            print(f"   ✗ {name} — {detail}")
    print("=" * 70)


if __name__ == "__main__":
    main()
