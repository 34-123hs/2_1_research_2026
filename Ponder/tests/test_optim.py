"""
tests/test_optim.py

Verify the deduplicated optimizer split: the partition is disjoint + exhaustive,
follows the Muon-vs-aux rule, and the built optimizer covers every param.
"""
import torch

import model as M
from optim import split_params, build_muon_optimizer


def tiny():
    return M.LLM(dim=32, depth=2, max_len=64, mlp_dim=64, heads=2, dim_head=16,
                 vocab_size=128, padding_idx=0, experts=2, dropout=0.0)


class _Args:
    muon_lr = 0.02
    muon_momentum = 0.95
    lr = 3e-4
    weight_decay = 0.1


def test_partition_disjoint_and_exhaustive():
    m = tiny()
    muon, aux = split_params(m)
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    ids_muon = {id(p) for p in muon}
    ids_aux = {id(p) for p in aux}
    assert ids_muon.isdisjoint(ids_aux)
    assert ids_muon | ids_aux == trainable
    assert len(muon) + len(aux) == len(trainable)


def test_rule_embeddings_head_norms_to_aux_2d_hidden_to_muon():
    m = tiny()
    muon, aux = split_params(m)
    muon_ids = {id(p) for p in muon}
    aux_ids = {id(p) for p in aux}
    # embeddings, head (weight + bias) -> aux
    assert id(m.embedding.weight) in aux_ids
    assert id(m.mlp_head.weight) in aux_ids
    assert id(m.mlp_head.bias) in aux_ids
    # a 2D hidden weight matrix (attention qkv) -> muon
    assert id(m.transformer.layers[0][0].to_qkv.weight) in muon_ids
    # every muon param is a 2D matrix
    assert all(p.ndim == 2 for p in muon)


def test_build_optimizer_covers_all_params():
    m = tiny()
    opt = build_muon_optimizer(m, _Args())
    assert len(opt.param_groups) == 2
    n_in_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_in_opt == n_trainable
