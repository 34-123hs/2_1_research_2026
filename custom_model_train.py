import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutput


class RoPE(nn.Module):
    def __init__(self, max_len, dim_head, base):
        super().__init__()
        t = torch.arange(max_len).float()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2).float() / dim_head))
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("sin", emb.sin())
        self.register_buffer("cos", emb.cos())

    def Rotate(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        seq_len = x.size(2)
        return x * self.cos[:seq_len].to(x.dtype) + self.Rotate(x) * self.sin[:seq_len].to(x.dtype)


class MoE(nn.Module):
    def __init__(self, dim, hidden_dim, experts, dropout=0.):
        super().__init__()
        self.gate = nn.Linear(dim, experts)
        self.how_much_certainty = nn.Linear(dim, 1)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.RMSNorm(dim),
                nn.Dropout(dropout),
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * dim, dim),
                nn.Dropout(dropout),
            ) for _ in range(experts)
        ])

    def forward(self, x):
        # x: [S, D]
        gate_probs = F.softmax(self.gate(x), dim=-1)            # [S, E]
        weights, selected = torch.topk(gate_probs, 1, dim=-1)   # [S, 1]
        weights  = weights.squeeze(-1)                          # [S]
        selected = selected.squeeze(-1)                         # [S]

        certainty = torch.sigmoid(self.how_much_certainty(x))   # [S, 1]

        results = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            s_idx, = torch.where(selected == i)
            if s_idx.numel() == 0:
                continue
            tokens = x[s_idx]                                   # [matched, D]
            out = expert(tokens)                                # [matched, D]
            results[s_idx] = weights[s_idx, None] * out

        return results, certainty


class AMoE(nn.Module):
    def __init__(self, dim, hidden_dim, experts, dropout=0.,
                 max_steps=10, eps=1e-2, use_checkpoint=True):
        super().__init__()
        self.moe = MoE(dim=dim, hidden_dim=hidden_dim, experts=experts, dropout=dropout)
        self.max_steps = max_steps
        self.eps = eps
        self.use_checkpoint = use_checkpoint

    def _moe_call(self, flat):
        """checkpointing 래퍼. 학습 중에만 checkpoint 적용."""
        if self.use_checkpoint and self.training:
            return checkpoint(self.moe, flat, use_reentrant=False)
        return self.moe(flat)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        state          = x
        sum_certainty  = torch.zeros_like(state[..., :1])  # [B, N, 1]
        sum_logit      = torch.zeros_like(state)           # [B, N, D]

        halting_probs = []  # 매 스텝의 step_cert를 모음 (정규화용)

        for t in range(self.max_steps):
            flat = state.reshape(B * N, D)
            new_flat, cert_flat = self._moe_call(flat)
            new_state = new_flat.reshape(B, N, D)
            cert      = cert_flat.reshape(B, N, 1)

            # active: 아직 halt 안 된 토큰만 기여
            active = (sum_certainty < 1 - self.eps).to(cert.dtype)  # [B, N, 1]

            if t == self.max_steps - 1:
                # 마지막 스텝: 남은 mass 전부 할당
                step_cert = (1 - sum_certainty) * active
            else:
                step_cert = torch.min(1 - sum_certainty, cert) * active

            sum_logit     = sum_logit + new_state * step_cert
            sum_certainty = sum_certainty + step_cert
            # state는 active 토큰만 갱신, halted는 freeze
            state = torch.where(active > 0.5, new_state, state)

            halting_probs.append(step_cert.squeeze(-1))  # [B, N]

        halting_probs = torch.stack(halting_probs, dim=0)  # [T, B, N]
        return sum_logit, halting_probs


class Attention(nn.Module):
    def __init__(self, dim, max_len, heads=8, dim_head=64, base=10000, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.dropout = dropout
        self.heads = heads
        self.norm = nn.RMSNorm(dim)
        self.rope = RoPE(max_len, dim_head, base)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if not (heads == 1 and dim_head == dim) else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        dropout_p = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q_rope = self.rope(q)
        k_rope = self.rope(k)
        out = F.scaled_dot_product_attention(
            q_rope, k_rope, v, is_causal=True, dropout_p=dropout_p
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, max_len, mlp_dim, heads, dim_head,
                 experts, base=10000, dropout=0.):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, max_len, heads, dim_head, base, dropout),
                AMoE(dim=dim, hidden_dim=mlp_dim, experts=experts, dropout=dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        all_halting_probs = []
        for atten, ff in self.layers:
            x = atten(x) + x
            ff_out, hp = ff(x)      # AMoE: (sum_logit, halting_probs [T,B,N])
            x = ff_out + x
            all_halting_probs.append(hp)
        return self.norm(x), all_halting_probs


class LLM(nn.Module):
    def __init__(self, dim, depth, max_len, mlp_dim, heads, dim_head,
                 vocab_size, padding_idx, experts,
                 base=10000, dropout=0., ponder_beta=0.01, lambda_p=0.2):
        super().__init__()
        self.padding_idx = padding_idx
        self.ponder_beta = ponder_beta
        self.lambda_p = lambda_p
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.transformer = Transformer(dim, depth, max_len, mlp_dim, heads,
                                       dim_head, experts, base, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp_head = nn.Linear(dim, vocab_size)

    def _ponder_loss(self, all_halting_probs):
        total = 0.0
        for hp in all_halting_probs:                # hp: [T, B, N]
            T = hp.size(0)
            prior = torch.tensor(
                [self.lambda_p * (1 - self.lambda_p) ** t for t in range(T)],
                device=hp.device, dtype=hp.dtype,
            )
            prior = prior / prior.sum()
            kl = (hp * (hp.clamp_min(1e-8).log()
                        - prior.view(T, 1, 1).log())).sum(dim=0)
            total = total + kl.mean()
        return total / len(all_halting_probs)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        x, all_halting_probs = self.transformer(x)
        logits = self.mlp_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            ponder_loss = self._ponder_loss(all_halting_probs)
            loss = task_loss + self.ponder_beta * ponder_loss
        return CausalLMOutput(loss=loss, logits=logits)


class TiktokenHFWrapper(PreTrainedTokenizer):
    vocab_files_names = {}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, encoding_name="r50k_base", **kwargs):
        self._enc = tiktoken.get_encoding(encoding_name)
        self._eot = self._enc.eot_token
        eot_str = "<|endoftext|>"
        kwargs.setdefault("eos_token", eot_str)
        kwargs.setdefault("bos_token", eot_str)
        kwargs.setdefault("unk_token", eot_str)
        kwargs.setdefault("pad_token", eot_str)
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self._enc.n_vocab

    def get_vocab(self):
        return {self._enc.decode([i]): i for i in range(self.vocab_size)}

    def _tokenize(self, text):
        return [str(i) for i in self._enc.encode(text, allowed_special={"<|endoftext|>"})]

    def _convert_token_to_id(self, token):
        return int(token)

    def _convert_id_to_token(self, index):
        return str(index)

    def convert_tokens_to_string(self, tokens):
        return self._enc.decode([int(t) for t in tokens])

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


class MemmapDataset(Dataset):
    def __init__(self, path, block_size, dtype=np.uint16, max_tokens=None):
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.block_size = block_size

        n_tokens = len(self.data)
        if max_tokens is not None:
            n_tokens = min(n_tokens, max_tokens)

        self.n_blocks = n_tokens // block_size

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.from_numpy(self.data[start:end].astype(np.int64))
        return {"input_ids": x}
