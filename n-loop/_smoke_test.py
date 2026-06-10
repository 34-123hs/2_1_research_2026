"""
Smoke test: critical 버그 두 개가 잡혔는지 확인.
- (A) inference LLM: 같은 입력에 대해 forward 2회 → identical (in-place 변형 없음)
- (B) inference MoE: 디버그 print가 더 이상 안 뜸 (호출 후 stderr 캡쳐로 검증은 생략, 코드 grep으로 이미 확인)
- (C) train LLM: forward + backward 정상
"""
import torch

print("=" * 50)
print("(A) inference forward 일관성 (residual 보존 확인)")
print("=" * 50)
from custom_model_inference import LLM as InfLLM, TiktokenHFWrapper as InfTok

tok = InfTok()
print(f"pad_token_id = {tok.pad_token_id}")  # 50256 이어야 함

torch.manual_seed(0)
m = InfLLM(dim=32, depth=2, max_len=16, mlp_dim=64, heads=2, dim_head=16,
           vocab_size=tok.vocab_size, padding_idx=tok.pad_token_id, experts=2)
m.eval()
ids = torch.randint(0, 1000, (1, 8))

with torch.no_grad():
    out1 = m(ids).logits
    out2 = m(ids).logits

diff = (out1 - out2).abs().max().item()
print(f"max abs diff between two forwards: {diff}")
assert diff == 0.0, f"FAIL: in-place mutation still present (diff={diff})"
print("PASS: forwards are identical (no in-place mutation)")

print()
print("=" * 50)
print("(C) train forward + backward")
print("=" * 50)
from custom_model_train import LLM as TrainLLM, TiktokenHFWrapper as TrainTok

t = TrainTok()
torch.manual_seed(0)
m2 = TrainLLM(dim=32, depth=2, max_len=16, mlp_dim=64, heads=2, dim_head=16,
              vocab_size=t.vocab_size, padding_idx=t.pad_token_id, experts=2)
out = m2(ids, labels=ids)
print(f"loss = {out.loss.item():.4f}")
print(f"logits.shape = {out.logits.shape}")
out.loss.backward()
n_grad = sum(1 for p in m2.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_total = sum(1 for _ in m2.parameters())
print(f"parameters with non-zero grad: {n_grad}/{n_total}")
print("PASS")
