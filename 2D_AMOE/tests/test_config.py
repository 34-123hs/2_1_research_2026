"""
tests/test_config.py

Verify the deduplicated arg parsing preserves defaults: shared base defaults are
identical, only --output_dir differs per entry, and train_with_hooks keeps its
extra flags.
"""
import sys

import train_custom
import train_with_hooks


def test_train_custom_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_custom.py"])
    a = train_custom.parse_args()
    assert a.output_dir == "custom-llm-out"
    assert a.dim == 512 and a.depth == 6 and a.experts == 4
    assert a.lr == 3e-4 and a.muon_lr == 0.02 and a.weight_decay == 0.1
    assert a.max_size == 50_000_000 and a.seed == 576


def test_hooks_defaults_and_extras(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_with_hooks.py"])
    a = train_with_hooks.parse_args()
    # base shared with same defaults, only output_dir differs
    assert a.output_dir == "hooks_outputs"
    assert a.dim == 512 and a.lr == 3e-4 and a.weight_decay == 0.1
    # extras preserved
    assert a.balance_beta == 0.01
    assert a.router_bias_init_mean == -0.05
    assert a.router_bias_init_std == 0.02
    assert a.max_grad_norm == 1.0
    assert a.log_per_layer is False
    assert a.print_console is False
    assert a.log_grad_detail is False
