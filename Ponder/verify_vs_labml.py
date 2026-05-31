"""
verify_vs_labml.py

Confirm that pondernet.py's loss math matches the labml_nn reference
(labml_nn.adaptive_computation.ponder_net), and quantify the KL difference.

- Reconstruction loss: should be NUMERICALLY IDENTICAL (same formula, ours just
  accumulates in a memory-efficient loop instead of stacking all per-step logits).
- Regularization (KL): ours and labml_nn differ — see the report.
"""
import torch
import torch.nn as nn
from labml_nn.adaptive_computation.ponder_net import ReconstructionLoss, RegularizationLoss

torch.manual_seed(0)
N, S, V = 6, 64, 200          # ponder steps, samples (=B*T flattened), vocab
lambda_p = 0.2

# ---- build a valid halting distribution p [N, S] (sums to 1 over N per sample) ----
lam = torch.rand(N, S)
lam[-1] = 1.0                                   # force halt at last step
p = torch.zeros(N, S)
unhalted = torch.ones(S)
for n in range(N):
    p[n] = unhalted * lam[n]
    unhalted = unhalted * (1 - lam[n])
assert torch.allclose(p.sum(0), torch.ones(S), atol=1e-5), "p must sum to 1 over steps"

logits = torch.randn(N, S, V)                   # per-step predictions ŷ_n
labels = torch.randint(0, V, (S,))

# ===== Reconstruction loss =====
ce = nn.CrossEntropyLoss(reduction="none")
# ours: Σ_n (p_n · CE_n).sum() / S   (exactly what PonderLLM.forward accumulates)
mine_recon = sum((p[n] * ce(logits[n], labels)).sum() for n in range(N)) / S
# labml_nn: Σ_n (p_n · CE_n).mean()
labml_recon = ReconstructionLoss(nn.CrossEntropyLoss(reduction="none"))(p, logits, labels)
recon_diff = (mine_recon - labml_recon).abs().item()

# ===== Regularization (KL) =====
# ours: forward KL(p ‖ prior), prior renormalized over the N used steps  (paper formula)
prior = torch.tensor([lambda_p * (1 - lambda_p) ** k for k in range(N)])
prior = prior / prior.sum()
mine_kl = (p * (p.clamp_min(1e-8).log()
                - prior.clamp_min(1e-8).log().view(N, 1))).sum(0).mean()
# labml_nn: KLDivLoss(p.log(), p_g) = reverse KL(prior ‖ p), prior NOT renormalized
labml_kl = RegularizationLoss(lambda_p, max_steps=N)(p)

print("=" * 64)
print("RECONSTRUCTION LOSS")
print(f"  mine  = {mine_recon.item():.8f}")
print(f"  labml = {labml_recon.item():.8f}")
print(f"  |diff|= {recon_diff:.2e}   -> {'IDENTICAL ✓' if recon_diff < 1e-5 else 'DIFFERS ✗'}")
print("=" * 64)
print("REGULARIZATION (KL)")
print(f"  mine  KL(p ‖ prior)   = {mine_kl.item():.6f}   [forward, prior renormalized — paper formula]")
print(f"  labml KL(prior ‖ p)   = {labml_kl.item():.6f}   [reverse, prior truncated/unnormalized]")
print("  -> directions differ (forward vs reverse) AND prior normalization differs.")
print("=" * 64)
