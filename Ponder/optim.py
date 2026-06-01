"""
optim.py

Optimizer construction for training (wrapper-side). The Muon optimizer updates
2D hidden weight matrices; everything else (embeddings, LM head, biases, norms)
is handled by the AdamW aux group. Deduplicated from the previously-identical
`create_muon_optimizer` in train_custom.py and train_with_hooks.py.
"""
from muon import SingleDeviceMuonWithAuxAdam as MuonWithAuxAdam


def split_params(model):
    """
    Partition trainable params into the Muon group and the AdamW aux group.
    Rule: a weight matrix of ndim>=2 that is NOT an embedding and NOT the LM head
    goes to Muon (this now includes the 3D grouped-expert tensors expert1/expert2 —
    muon.py's Newton-Schulz is batched over the last 2 dims, so each [D,H] expert
    matrix is orthogonalized); everything else (embeddings, `mlp_head`, biases,
    norms) goes to aux.

    input : model (nn.Module)
    output: (muon_params, aux_params) — two disjoint lists of nn.Parameter whose
            union is every `requires_grad` parameter.
    """
    muon_params, aux_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        use_muon = p.ndim >= 2 and "embedding" not in name and "mlp_head" not in name
        (muon_params if use_muon else aux_params).append(p)
    return muon_params, aux_params


def build_muon_optimizer(model, args):
    """
    Build the hybrid Muon + AdamW optimizer.

    input : model (nn.Module)
            args  — needs `.muon_lr`, `.muon_momentum`, `.lr`, `.weight_decay`
    output: SingleDeviceMuonWithAuxAdam optimizer (also prints group param counts).
    """
    muon_params, aux_params = split_params(model)
    n_muon = sum(p.numel() for p in muon_params)
    n_aux = sum(p.numel() for p in aux_params)
    print(f"[Optimizer] Muon params={n_muon:,}  Aux params={n_aux:,}")
    return MuonWithAuxAdam([
        dict(params=muon_params, lr=args.muon_lr, momentum=args.muon_momentum,
             weight_decay=args.weight_decay, use_muon=True),
        dict(params=aux_params, lr=args.lr,
             weight_decay=args.weight_decay, use_muon=False),
    ])
