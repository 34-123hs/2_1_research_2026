"""
tests/test_smoke.py

CPU smoke tests that close the runtime-verification gap for the train/inference
model merge (model.py). Tiny dims, no GPU. Covers:
  - train forward+backward: finite loss, finite grads
  - inference logits: shape + finite, loss is None
  - AMoE returns (state, LBL): state shape + finite scalar LBL
  - fixed N-step recurrence actually runs (more steps -> different output)
  - MemmapDataset reads the synthetic uint16 .bin
  - the inference generate() loop decodes
"""
import numpy as np
import torch

import model as M
import inference_custom as inf


def tiny_llm(**over):
    cfg = dict(dim=64, depth=2, max_len=128, mlp_dim=128, heads=2, dim_head=32,
               vocab_size=256, padding_idx=0, experts=2, dropout=0.0)
    cfg.update(over)
    return M.LLM(**cfg)


def test_train_forward_backward_finite():
    torch.manual_seed(0)
    m = tiny_llm().train()
    ids = torch.randint(1, 256, (2, 16))
    out = m(input_ids=ids, labels=ids)
    assert out.loss is not None and torch.isfinite(out.loss), out.loss
    out.loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)


def test_inference_logits_shape_and_finite():
    torch.manual_seed(0)
    m = tiny_llm().eval()
    ids = torch.randint(1, 256, (2, 16))
    with torch.no_grad():
        out = m(input_ids=ids)
    assert out.loss is None
    assert out.logits.shape == (2, 16, 256)
    assert torch.isfinite(out.logits).all()


def test_amoe_output_shape_and_lbl():
    torch.manual_seed(0)
    amoe = M.AMoE(dim=32, hidden_dim=64, experts=2, max_steps=5).train()
    state, lbl = amoe(torch.randn(2, 8, 32))
    assert state.shape == (2, 8, 32)
    assert lbl.ndim == 0 and torch.isfinite(lbl)


def test_amoe_fixed_recurrence_runs():
    # same weights, only step count differs -> the N-step recurrence must change output
    torch.manual_seed(0)
    x = torch.randn(2, 8, 32)
    amoe1 = M.AMoE(dim=32, hidden_dim=64, experts=2, max_steps=1).eval()
    amoe3 = M.AMoE(dim=32, hidden_dim=64, experts=2, max_steps=3).eval()
    amoe3.load_state_dict(amoe1.state_dict())
    with torch.no_grad():
        out1, _ = amoe1(x)
        out3, _ = amoe3(x)
    assert not torch.allclose(out1, out3), "recurrence steps must change the output"


def test_memmap_dataset(tmp_path):
    p = tmp_path / "tiny.bin"
    np.random.default_rng(0).integers(0, 256, 1000, dtype=np.uint16).tofile(p)
    ds = M.MemmapDataset(str(p), block_size=64)
    assert len(ds) == 1000 // 64
    item = ds[0]
    assert item["input_ids"].shape == (64,)
    assert item["input_ids"].dtype == torch.int64


def test_generate_decodes():
    tok = M.TiktokenHFWrapper("r50k_base")
    torch.manual_seed(0)
    m = tiny_llm(vocab_size=tok.vocab_size, max_len=64).eval()
    ids = torch.randint(1, 1000, (1, 4))
    out = inf.generate(m, ids, max_new_tokens=5, temperature=1.0, top_k=10,
                       eos_id=tok.eos_token_id, device=torch.device("cpu"))
    assert 4 <= out.shape[1] <= 4 + 5  # may stop early on eos
