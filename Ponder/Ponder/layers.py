"""layers.py — RoPE + Attention copied verbatim from AMOE/model.py for a fair comparison."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RoPE(nn.Module):
    """
    Rotary Positional Embedding.
    위치별 회전 행렬을 q, k에 곱해 상대 위치 정보를 주입한다.
    sin / cos 버퍼는 init에서 미리 만들어두고 forward에서 시퀀스 길이만큼 잘라 쓴다.
    """

    def __init__(self, max_len, dim_head, base):
        """
        input : max_len  (int) 최대 시퀀스 길이
                dim_head (int) 헤드 1개의 차원 (짝수)
                base     (int) 주파수 베이스 (보통 10000)
        output: 없음. sin / cos 버퍼 [max_len, dim_head] 등록.
        """
        super().__init__()
        t = torch.arange(max_len).float()                                          # [max_len]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2).float() / dim_head))  # [dim_head/2]
        freqs = torch.einsum('i,j->ij', t, inv_freq)                               # [max_len, dim_head/2]
        emb = torch.cat((freqs, freqs), dim=-1)                                    # [max_len, dim_head]
        self.register_buffer("sin", emb.sin())                                     # [max_len, dim_head]
        self.register_buffer("cos", emb.cos())                                     # [max_len, dim_head]

    def Rotate(self, x):
        """
        회전쌍 트릭: 뒤 절반의 부호를 뒤집어 앞 절반과 swap.
        input : x [..., dim_head]
        output:   [..., dim_head]
        """
        x1, x2 = x.chunk(2, dim=-1)              # 각 [..., dim_head/2]
        return torch.cat((-x2, x1), dim=-1)      # [..., dim_head]

    def forward(self, x):
        """
        input : x [B, H, N, dim_head]   (H=헤드 수, N=시퀀스 길이)
        output:   [B, H, N, dim_head]   회전이 적용된 텐서
        """
        seq_len = x.size(2)                                                        # N
        # cos[:N], sin[:N]: [N, dim_head] → [B, H, N, dim_head]로 브로드캐스트
        return x * self.cos[:seq_len].to(x.dtype) + self.Rotate(x) * self.sin[:seq_len].to(x.dtype)


class Attention(nn.Module):
    """
    RMSNorm pre-norm 멀티헤드 self-attention. RoPE + causal SDPA.
    """

    def __init__(self, dim, max_len, heads=8, dim_head=64, base=10000, dropout=0.):
        """
        input : dim (int) 모델 차원 D, max_len (int) 최대 길이,
                heads (int) 헤드 수 H, dim_head (int) 헤드 차원,
                base (int) RoPE 베이스, dropout (float)
        output: 없음.
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.dropout = dropout
        self.heads = heads
        self.norm = nn.RMSNorm(dim)
        self.rope = RoPE(max_len, dim_head, base)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)   # [D] → [3*H*dim_head]
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if not (heads == 1 and dim_head == dim) else nn.Identity()

    def forward(self, x):
        """
        input : x [B, N, D]
        output:   [B, N, D]   (residual 가산은 호출부 Transformer에서)
        """
        x = self.norm(x)                                          # [B, N, D]
        dropout_p = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)                     # 각 [B, N, H*dim_head]
        # [B, N, H*dim_head] → [B, H, N, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q_rope = self.rope(q)                                     # [B, H, N, dim_head]
        k_rope = self.rope(k)                                     # [B, H, N, dim_head]
        out = F.scaled_dot_product_attention(
            q_rope, k_rope, v, is_causal=True, dropout_p=dropout_p
        )                                                         # [B, H, N, dim_head]
        out = rearrange(out, "b h n d -> b n (h d)")             # [B, N, H*dim_head]
        return self.to_out(out)
