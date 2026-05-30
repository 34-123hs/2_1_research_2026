"""
diagnostics.py

Pure metric helpers for training diagnostics (no I/O, no global state). Extracted
from the repeated per-gate computation inside
train_with_hooks.compute_aux_and_metrics so it can be unit-tested in isolation.
"""
import math

import torch
import torch.nn.functional as F


def switch_gate_stats(gate_logit):
    """
    Switch-Transformer load-balance + router-collapse stats for one captured
    gate-logit tensor (top-1 routing).

    input : gate_logit [N, E]   N tokens, E experts (raw router logits)
    output: balance  (scalar tensor, grad-alive) — E * sum_e f_e * P_e, where
            f = dispatch fraction per expert, P = mean gate prob per expert.
            max_pct  (scalar tensor, detached) — fraction routed to the busiest expert.
            ent_norm (scalar tensor, detached) — mean routing entropy / log(E), in [0, 1].
    """
    p = F.softmax(gate_logit.float(), dim=-1)                     # [N, E]
    E = p.size(-1)
    sel = p.argmax(dim=-1)                                        # [N]
    f = F.one_hot(sel, num_classes=E).to(p.dtype).mean(dim=0)     # [E] dispatch fraction
    P = p.mean(dim=0)                                             # [E] mean gate prob
    balance = E * (f * P).sum()                                  # scalar
    max_pct = f.max().detach()                                   # scalar
    ent_norm = (-(p * p.clamp_min(1e-12).log()).sum(-1).mean() / math.log(E)).detach()
    return balance, max_pct, ent_norm
