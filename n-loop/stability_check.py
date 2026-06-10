"""
stability_check.py — train.bin 일부로 1 forward + 1 backward 후
layer/expert/router/grad 상태를 한 번에 진단.

Part 1: StabilityProbe — hook으로 모든 raw 정보 수집
Part 2: analyze()     — 수집한 정보를 8가지 섹션으로 출력

사용:
  python stability_check.py --train_bin_path train.bin
  python stability_check.py --train_bin_path train.bin \
    --dim 768 --depth 12 --heads 8 --dim_head 64 --mlp_dim 2048 \
    --block_size 768 --batch_size 4 --experts 4
"""
import argparse
import math
import os
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file

from model import LLM, MemmapDataset, TiktokenHFWrapper


# ============================================================================
# Part 1: 수집부 — hook으로 grad/router/activation 정보 모음
# ============================================================================
class StabilityProbe:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.act_norms = {}                       # {layer_idx: float}
        self.gate_selections = defaultdict(list)  # {layer_idx: [tensor[B*N]   per step]}

        # gradient checkpoint는 켠 채로 둔다(메모리). 단 backward 시 recompute로 MoE hook이
        # 한 번 더 fire하므로, 호출당 max_steps 만큼만 기록하고 나머지는 무시 → 데이터 중복 방지.
        self._register()

    def _register(self):
        for i, layer in enumerate(self.model.transformer.layers):
            _atten, amoe = layer
            cap = amoe.max_steps   # 첫 max_steps번만 기록, 그 이후(backward recompute)는 무시

            # AMoE는 layer당 1회만 호출되므로 dedupe 불필요
            def make_amoe_hook(idx):
                def hook(module, inputs, output):
                    state, _lbl = output
                    self.act_norms[idx] = state.detach().float().norm(dim=-1).mean().item()
                return hook
            self.handles.append(amoe.register_forward_hook(make_amoe_hook(i)))

            def make_gate_hook(idx, cap):
                def hook(module, inputs, output):
                    if len(self.gate_selections[idx]) >= cap:
                        return
                    selected = output.argmax(dim=-1).detach()
                    self.gate_selections[idx].append(selected)
                return hook
            self.handles.append(amoe.moe.gate.register_forward_hook(make_gate_hook(i, cap)))

    def run_step(self, input_ids):
        self.model.zero_grad(set_to_none=True)
        out = self.model(input_ids, labels=input_ids)
        out.loss.backward()
        return out.loss.item()

    def collect_grad_stats(self):
        stats = {}
        for name, p in self.model.named_parameters():
            wn = p.detach().float().norm().item()
            if p.grad is None:
                stats[name] = dict(
                    shape=tuple(p.shape), grad_norm=0.0, weight_norm=wn,
                    max_abs=0.0, mean_abs=0.0,
                    has_nan=False, has_inf=False, n_zero_frac=1.0,
                )
                continue
            g = p.grad.detach().float()
            stats[name] = dict(
                shape=tuple(p.shape),
                grad_norm=g.norm().item(),
                weight_norm=wn,
                max_abs=g.abs().max().item(),
                mean_abs=g.abs().mean().item(),
                has_nan=bool(torch.isnan(g).any().item()),
                has_inf=bool(torch.isinf(g).any().item()),
                n_zero_frac=(g == 0).float().mean().item(),
            )
        return stats

    def close(self):
        for h in self.handles:
            h.remove()


# ============================================================================
# Part 2: 분석부 — 8가지 진단 섹션 출력
# ============================================================================
def _bar(v, vmax, width=20):
    if vmax <= 0:
        return ""
    return "█" * max(1, int(width * v / vmax))


def analyze(*, grad_stats, gate_selections,
            act_norms, depth, n_experts,
            batch_size, block_size, dump_tokens, do_dump):

    print("\n" + "=" * 72)
    print("STABILITY ANALYSIS")
    print("=" * 72)

    # ---- [A] NaN/Inf scan ----
    print("\n[A] NaN/Inf scan")
    nan_params = [n for n, s in grad_stats.items() if s["has_nan"]]
    inf_params = [n for n, s in grad_stats.items() if s["has_inf"]]
    if nan_params:
        print(f"  ❌ NaN in {len(nan_params)} param(s): {nan_params[:5]}"
              + (" ..." if len(nan_params) > 5 else ""))
    if inf_params:
        print(f"  ❌ Inf in {len(inf_params)} param(s): {inf_params[:5]}"
              + (" ..." if len(inf_params) > 5 else ""))
    if not nan_params and not inf_params:
        print(f"  ✅ no NaN/Inf in any of {len(grad_stats)} params")

    # ---- [B] Dead params ----
    print("\n[B] Dead params (grad_norm == 0)")
    dead = [n for n, s in grad_stats.items() if s["grad_norm"] == 0.0]
    if dead:
        print(f"  ⚠️  {len(dead)} dead: {dead[:8]}"
              + (" ..." if len(dead) > 8 else ""))
    else:
        print(f"  ✅ all {len(grad_stats)} params receive grad")

    # ---- [C] Depth-wise grad norm ----
    print("\n[C] Depth-wise grad norm (per-layer aggregated)")
    layer_grad_norms = []
    for i in range(depth):
        prefix = f"transformer.layers.{i}."
        norms = [s["grad_norm"] for n, s in grad_stats.items() if n.startswith(prefix)]
        total = math.sqrt(sum(g * g for g in norms))
        layer_grad_norms.append(total)
    vmax = max(layer_grad_norms) if layer_grad_norms else 0.0
    for i, n in enumerate(layer_grad_norms):
        print(f"  L{i:2d}: {n:.4e}  {_bar(n, vmax)}")
    if layer_grad_norms and layer_grad_norms[0] > 0:
        ratio = layer_grad_norms[-1] / layer_grad_norms[0]
        if ratio < 0.01:
            verdict = f"❌ ratio={ratio:.2e} — vanishing"
        elif ratio > 100:
            verdict = f"❌ ratio={ratio:.2e} — exploding"
        else:
            verdict = f"✅ ratio={ratio:.2e} (within 0.01~100)"
        print(f"  last/first {verdict}")

    # ---- [D] Grad/Weight ratio ----
    print("\n[D] Grad/Weight ratio (top-5 high, bottom-5 low among non-zero)")
    rats = [(n, s["grad_norm"] / s["weight_norm"])
            for n, s in grad_stats.items()
            if s["weight_norm"] > 0 and s["grad_norm"] > 0]
    rats.sort(key=lambda x: x[1], reverse=True)
    for n, r in rats[:5]:
        print(f"  high: {r:.4e}  {n}")
    if len(rats) > 10:
        print("  ...")
    for n, r in rats[-5:]:
        print(f"  low : {r:.4e}  {n}")
    explosive = [(n, r) for n, r in rats if r > 1.0]
    if explosive:
        print(f"  ⚠️  {len(explosive)} param(s) with grad/weight > 1.0 — lr/init 부적절")
    else:
        print("  ✅ all grad/weight < 1.0")

    # ---- [E] Router expert utilization ----
    print(f"\n[E] Router expert utilization ({n_experts} experts, max_steps 누적)")
    for i in range(depth):
        all_sel = torch.cat(gate_selections[i])
        counts = torch.bincount(all_sel, minlength=n_experts).float()
        pct = (counts / counts.sum() * 100).tolist()
        line = "  ".join(f"E{e}:{p:5.1f}%" for e, p in enumerate(pct))
        max_pct = max(pct)
        if max_pct > 80:
            flag = "❌ collapse"
        elif max_pct > 50:
            flag = "⚠️ skew"
        else:
            flag = "✅"
        print(f"  L{i:2d}: {line}  [{flag}]")

    # ---- [G] Layer activation norm ----
    print("\n[G] Layer output activation norm")
    vmax = max(act_norms.values()) if act_norms else 0.0
    for i in range(depth):
        v = act_norms[i]
        print(f"  L{i:2d}: {v:.4e}  {_bar(v, vmax)}")

    # ---- [H] Per-token expert dump (재귀 step별 라우팅) ----
    if do_dump:
        n_show = min(dump_tokens, block_size)
        print(f"\n[H] Per-token expert (b=0, first {n_show} of {block_size} tokens)")
        for i in range(depth):
            for t, sel in enumerate(gate_selections[i]):
                sel_b0 = sel.view(batch_size, block_size)[0, :n_show].tolist()
                row = " ".join(f"E{e}" for e in sel_b0)
                print(f"  L{i:2d} step {t:2d}: {row}")
            print()


# ============================================================================
# main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_bin_path", default="train.bin")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_size", type=int, default=768)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=2048)
    p.add_argument("--rope_base", type=int, default=10000)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--ponder_beta", type=float, default=0.01)
    p.add_argument("--lambda_p", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=576)
    p.add_argument("--dump_tokens", type=int, default=32)
    p.add_argument("--no_dump", action="store_true")
    p.add_argument("--checkpoint_dir", default=None,
                   help="HF Trainer 저장 폴더. None이면 random init.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    tok = TiktokenHFWrapper("r50k_base")

    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tok.vocab_size, padding_idx=tok.pad_token_id,
        experts=args.experts, base=args.rope_base, dropout=args.dropout,
        ponder_beta=args.ponder_beta, lambda_p=args.lambda_p,
    ).to(device)

    if args.checkpoint_dir:
        st_path = os.path.join(args.checkpoint_dir, "model.safetensors")
        pt_path = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(st_path):
            state = load_file(st_path, device=device)
            src = st_path
        elif os.path.exists(pt_path):
            state = torch.load(pt_path, map_location=device, weights_only=True)
            src = pt_path
        else:
            raise FileNotFoundError(
                f"no model.safetensors / pytorch_model.bin in {args.checkpoint_dir}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ckpt] loaded {src}")
        if missing:
            print(f"[ckpt] missing keys ({len(missing)}): {missing[:5]}"
                  + (" ..." if len(missing) > 5 else ""))
        if unexpected:
            print(f"[ckpt] unexpected keys ({len(unexpected)}): {unexpected[:5]}"
                  + (" ..." if len(unexpected) > 5 else ""))

    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params / 1e6:.2f}M  dim={args.dim} depth={args.depth} "
          f"experts={args.experts}")

    ds = MemmapDataset(args.train_bin_path, args.block_size)
    assert len(ds) >= args.batch_size, \
        f"dataset has {len(ds)} blocks, need >= {args.batch_size}"
    ids = torch.stack([ds[i]["input_ids"] for i in range(args.batch_size)]).to(device)
    print(f"[batch] input_ids shape={tuple(ids.shape)}")

    probe = StabilityProbe(model)
    loss = probe.run_step(ids)
    print(f"[loss] {loss:.4f}")

    grad_stats = probe.collect_grad_stats()
    probe.close()

    analyze(
        grad_stats=grad_stats,
        gate_selections=probe.gate_selections,
        act_norms=probe.act_norms,
        depth=args.depth,
        n_experts=args.experts,
        batch_size=args.batch_size,
        block_size=args.block_size,
        dump_tokens=args.dump_tokens,
        do_dump=not args.no_dump,
    )


if __name__ == "__main__":
    main()
