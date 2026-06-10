"""
tests/test_diagnostics.py

Verify the pure switch_gate_stats helper and that train_with_hooks'
compute_aux_and_metrics still works end-to-end with a fake collector.
"""
import torch

from diagnostics import switch_gate_stats


def test_shapes_bounds_and_grad():
    torch.manual_seed(0)
    gl = torch.randn(500, 4)
    b, mx, ent = switch_gate_stats(gl)
    assert b.ndim == 0 and mx.ndim == 0 and ent.ndim == 0
    assert 0.0 <= float(ent) <= 1.0 + 1e-5
    assert 0.0 <= float(mx) <= 1.0
    assert float(b) > 0
    gl2 = torch.randn(500, 4, requires_grad=True)
    switch_gate_stats(gl2)[0].backward()
    assert gl2.grad is not None and torch.isfinite(gl2.grad).all()


def test_balanced_routing_max_pct():
    # one token routed to each of 4 experts -> dispatch fraction 1/4 each
    gl = torch.zeros(4, 4)
    for i in range(4):
        gl[i, i] = 10.0
    _, mx, _ = switch_gate_stats(gl)
    assert abs(float(mx) - 0.25) < 1e-5


def test_collapsed_routing():
    # all tokens to expert 0 -> max_pct = 1.0, low entropy
    gl = torch.zeros(8, 4)
    gl[:, 0] = 10.0
    _, mx, ent = switch_gate_stats(gl)
    assert abs(float(mx) - 1.0) < 1e-5
    assert float(ent) < 0.5


def test_compute_aux_and_metrics_integration():
    import train_with_hooks as twh

    class FakeCollector:
        def __init__(self, gate_logits):
            self.gate_logits = gate_logits

    depth = 2
    # one gate capture per layer (recurrence step 0): (layer, step, [S, E])
    gl = [(li, 0, torch.randn(50, 4)) for li in range(depth)]
    bal, metrics, dispatch_layer, dispatch_step, table = twh.compute_aux_and_metrics(
        FakeCollector(gl), depth, log_per_layer=True)
    assert bal.ndim == 0 and torch.isfinite(bal)
    assert "aux/balance_loss" in metrics
    assert "router/max_pct_global" in metrics
    assert dispatch_layer.shape == (depth, 4)
    assert len(table["balance"]) == depth
