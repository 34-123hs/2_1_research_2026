"""
pondernet.py

Faithful canonical PonderNet (Banino et al. 2021, arXiv:2107.05407) adapted to an
autoregressive LM — the CONTROL/baseline for AMoE.

Difference from AMoE (for the record):
  - PonderNet ponders ONE weight-shared step-cell up to `max_steps` times over the
    input; each step n emits a prediction (logits_n) and a per-token halting prob
    λ_n. Halting distribution p_n = λ_n · ∏_{j<n}(1−λ_j). Loss is the EXPECTED
    per-step loss  Σ_n p_n · CE(logits_n, y)  +  β · KL(p ‖ geometric(λ_p)).
  - AMoE instead does per-layer halting-weighted *state averaging* + a single final
    prediction.

Self-contained: only copied/local modules (layers, muon via optim) — no AMOE imports.
Memory: the per-step full-vocab projection is the cost, so each ponder step is wrapped
in gradient checkpointing during training (same trick AMoE uses), keeping activation
memory ~1× instead of max_steps×.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutput

from layers import Attention


class FFN(nn.Module):
    """
    Dense feed-forward (the PonderNet counterpart of one AMoE expert): same shape
    as AMoE's expert MLP for a fair comparison — RMSNorm → Linear(d,4d) → GELU →
    Linear(4d,d), with dropout.
    """

    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """input: x [B, N, D] → output: [B, N, D]"""
        return self.net(x)


class PonderCore(nn.Module):
    """
    The weight-shared recurrent step-cell that is re-applied at every ponder step.
    A pre-norm transformer stack of `core_depth` (Attention + dense FFN) blocks.
    Faithful to PonderNet's s(x, h_{n-1}): the input embedding x is re-injected at
    the start of every ponder step.
    """

    def __init__(self, dim, core_depth, max_len, heads, dim_head, base=10000, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, max_len, heads, dim_head, base, dropout),
                FFN(dim, dropout),
            ]) for _ in range(core_depth)
        ])

    def forward(self, h, x):
        """
        input : h [B, N, D] previous ponder state, x [B, N, D] input embedding
        output:   [B, N, D] new ponder state
        """
        h = h + x                                  # re-inject input each ponder step
        for atten, ff in self.layers:
            h = atten(h) + h                       # [B, N, D] attention residual
            h = ff(h) + h                          # [B, N, D] FFN residual
        return h


class PonderLLM(nn.Module):
    """
    Decoder-only PonderNet LM. Same external contract as AMoE's LLM:
    forward(input_ids, labels) → CausalLMOutput(loss, logits), so it runs under the
    same training/inference machinery and optimizer split.

    train : run all `max_steps`; loss = expected per-step CE + ponder_beta·KL.
    infer : per-token early-exit once cumulative halting mass ≈ 1; returns expected logits.
    """

    def __init__(self, dim, max_len, mlp_dim, heads, dim_head,
                 vocab_size, padding_idx, core_depth=6,
                 base=10000, dropout=0., max_steps=10, eps=1e-2,
                 lambda_p=0.2, ponder_beta=0.01, use_checkpoint=True):
        super().__init__()
        self.padding_idx = padding_idx
        self.max_steps = max_steps
        self.eps = eps
        self.lambda_p = lambda_p
        self.ponder_beta = ponder_beta
        self.use_checkpoint = use_checkpoint
        # `embedding` / `mlp_head` names → routed to AdamW by optim.split_params (like AMoE)
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.core = PonderCore(dim, core_depth, max_len, heads, dim_head, base, dropout)
        self.norm = nn.RMSNorm(dim)
        self.mlp_head = nn.Linear(dim, vocab_size)
        self.halt_head = nn.Linear(dim, 1)         # per-token λ logit

    def _ponder_step(self, h, x, unhalted, force_halt, labels):
        """
        One ponder step. Returns (new_h, p, step_loss, logits_detached).

        input : h [B,N,D], x [B,N,D], unhalted [B,N,1] (=∏_{j<n}(1−λ_j)),
                force_halt (bool; True on last step → λ=1), labels [B,N] | None
        output: new_h    [B,N,D]
                p        [B,N,1]  unconditional halting prob p_n = unhalted·λ_n
                step_loss scalar  Σ_token p_n·CE(logits_n) (unnormalized; 0 if no labels)
                logits   [B,N,V]  detached (for the expected-logits output only)
        """
        h = self.core(h, x)                                    # [B, N, D]
        hn = self.norm(h)                                      # [B, N, D]
        logits = self.mlp_head(hn)                             # [B, N, V]
        if force_halt:
            lam = torch.ones_like(unhalted)                    # [B, N, 1] force halt
        else:
            lam = torch.sigmoid(self.halt_head(hn))            # [B, N, 1] λ_n
        p = unhalted * lam                                     # [B, N, 1] p_n

        if labels is not None:
            V = logits.size(-1)
            shift_logits = logits[:, :-1, :]                   # [B, N-1, V]
            shift_labels = labels[:, 1:]                       # [B, N-1]
            ce = F.cross_entropy(
                shift_logits.reshape(-1, V), shift_labels.reshape(-1),
                ignore_index=-100, reduction="none",
            ).view(shift_labels.shape)                         # [B, N-1] per-token CE
            step_loss = (p[:, :-1, 0] * ce).sum()              # scalar (weighted by p_n)
        else:
            step_loss = logits.sum() * 0.0                     # 0, keeps dtype/device

        return h, p, step_loss, logits.detach()

    def _ponder_loss(self, halting_probs):
        """
        KL(halting distribution ‖ geometric prior(λ_p)), mean over tokens.
        input : halting_probs [T, B, N]  (p_n per token)
        output: scalar
        """
        T = halting_probs.size(0)
        prior = torch.tensor(
            [self.lambda_p * (1 - self.lambda_p) ** t for t in range(T)],
            device=halting_probs.device, dtype=halting_probs.dtype,
        )
        prior = prior / prior.sum()                            # [T]
        kl = (halting_probs * (halting_probs.clamp_min(1e-8).log()
                               - prior.clamp_min(1e-8).log().view(T, 1, 1))).sum(dim=0)
        return kl.mean()

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        """
        input : input_ids [B, N], labels [B, N] | None
        output: CausalLMOutput(loss, logits[B, N, V])
                logits = expected logits Σ_n p_n·logits_n (for metrics/generation).
        """
        x = self.embedding(input_ids)                          # [B, N, D]
        x = self.dropout(x)
        B, N, D = x.shape

        state = x                                              # [B, N, D] h_0
        unhalted = torch.ones_like(state[..., :1])             # [B, N, 1] ∏(1−λ)
        exp_logits = None                                      # [B, N, V] Σ p_n·logits_n (no grad)
        task_loss = state.new_zeros(())                        # scalar accumulator (grad)
        halting_probs = []                                     # list of [B, N] p_n (grad, for KL)

        for t in range(self.max_steps):
            force = (t == self.max_steps - 1)
            if self.use_checkpoint and self.training:
                state, p, step_loss, logits_d = checkpoint(
                    self._ponder_step, state, x, unhalted, force, labels,
                    use_reentrant=False)
            else:
                state, p, step_loss, logits_d = self._ponder_step(
                    state, x, unhalted, force, labels)

            task_loss = task_loss + step_loss                  # accumulate weighted CE
            contrib = p.detach() * logits_d                    # [B, N, V] expected-logits term (no grad)
            exp_logits = contrib if exp_logits is None else exp_logits + contrib
            halting_probs.append(p.squeeze(-1))                # [B, N]
            unhalted = unhalted - p                            # ∏(1−λ) update (= unhalted·(1−λ))

            # inference: all tokens halted → remaining mass < eps, stop early
            if (not self.training) and bool((unhalted < self.eps).all()):
                break

        hp = torch.stack(halting_probs, dim=0)                 # [T, B, N]
        # diagnostic: per-token Σ_n p_n (≈1 in train; 1−remainder if inference early-exits)
        self._last_halt_sum = hp.sum(dim=0).detach()           # [B, N]

        loss = None
        if labels is not None:
            task_loss = task_loss / (B * (N - 1))              # mean over predicted tokens
            ponder_loss = self._ponder_loss(hp)
            loss = task_loss + self.ponder_beta * ponder_loss

        return CausalLMOutput(loss=loss, logits=exp_logits)
