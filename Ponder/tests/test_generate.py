"""
tests/test_generate.py

Verify the pure sampling step: top-k restricts the support, top_k<=0 allows all,
top_k=1 is deterministic (argmax) regardless of temperature, and batch shape holds.
"""
import torch

from generate import sample_next_token


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    samples = torch.cat([sample_next_token(logits, 1.0, 2) for _ in range(64)])
    assert set(samples.flatten().tolist()) <= {3, 4}  # only the top-2 ids


def test_top_k_zero_allows_all():
    out = sample_next_token(torch.zeros(1, 5), 1.0, 0)
    assert out.shape == (1, 1)
    assert 0 <= out.item() < 5


def test_top_k_one_is_argmax_any_temperature():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    for temp in (0.5, 1.0, 2.0):
        assert sample_next_token(logits, temp, 1).item() == 1


def test_batch_shape():
    out = sample_next_token(torch.randn(4, 10), 1.0, 5)
    assert out.shape == (4, 1)
