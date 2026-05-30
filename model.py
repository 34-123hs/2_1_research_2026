"""
model.py

train / inference 공용 모델 정의 (B 방식 통합 파일).
- 동작 차이는 nn.Module의 self.training 플래그로만 갈린다:
    * AMoE      : train은 gradient checkpointing 적용 + 수직 루프 max_steps 고정,
                  inference는 checkpointing 미적용 + 전 토큰 halt 시 조기 break.
    * LLM       : labels가 주어지면(=train) task_loss + ponder_loss를 계산,
                  없으면(=inference) logits만 반환.
- 수학은 통합 전 custom_model_train.py와 동일하다. 유일한 동작 추가는 AMoE의 추론 조기 종료.
"""

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


class MoE(nn.Module):
    """
    Top-1 라우팅 Mixture-of-Experts (1 스텝).
    토큰마다 게이트가 전문가 1명을 고르고, 동시에 이 스텝의 certainty(확신도)를 낸다.
    certainty는 상위 AMoE가 토큰별 halt 여부를 정하는 데 쓴다.
    """

    def __init__(self, dim, hidden_dim, experts, dropout=0.):
        """
        input : dim        (int) 모델 차원 D
                hidden_dim (int) (미사용 인자 — 전문가 내부는 4*dim 고정)
                experts    (int) 전문가 수 E
                dropout    (float)
        output: 없음.
        """
        super().__init__()
        self.gate = nn.Linear(dim, experts)              # [D] → [E] 라우팅 로짓
        self.how_much_certainty = nn.Linear(dim, 1)      # [D] → [1] 확신도 로짓
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
        """
        input : x [S, D]   (S = B*N, 토큰 평탄화)
        output: results   [S, D]   선택된 전문가 출력 * 게이트 가중치
                certainty [S, 1]   이 스텝의 토큰별 확신도 (sigmoid, 0~1)
        """
        gate_probs = F.softmax(self.gate(x), dim=-1)            # [S, E]
        weights, selected = torch.topk(gate_probs, 1, dim=-1)   # 각 [S, 1]
        weights  = weights.squeeze(-1)                          # [S]
        selected = selected.squeeze(-1)                         # [S]

        certainty = torch.sigmoid(self.how_much_certainty(x))   # [S, 1]

        results = torch.zeros_like(x)                           # [S, D]
        for i, expert in enumerate(self.experts):
            s_idx, = torch.where(selected == i)                 # 전문가 i가 맡은 토큰 인덱스
            if s_idx.numel() == 0:
                continue
            tokens = x[s_idx]                                   # [matched, D]
            out = expert(tokens)                                # [matched, D]
            results[s_idx] = weights[s_idx, None] * out         # 게이트 가중치 곱해 자리 채움

        return results, certainty


class AMoE(nn.Module):
    """
    Adaptive MoE: MoE를 토큰별로 적응적 횟수만큼 반복(수직축 반복).
    토큰별 누적 certainty가 1-eps를 넘으면 그 토큰은 halt(동결)되어 더는 갱신 안 됨.

    train     : 루프를 항상 max_steps번 돌린다(고정). → halting_probs가 항상 max_steps개라
                PonderNet 손실이 일관됨. gradient checkpointing 적용.
    inference : 모든 토큰이 halt되면 남은 스텝은 기여가 0이라 무의미 → 조기 break.
    """

    def __init__(self, dim, hidden_dim, experts, dropout=0.,
                 max_steps=10, eps=1e-2, use_checkpoint=True):
        """
        input : dim, hidden_dim, experts, dropout — MoE에 전달
                max_steps (int)   수직 루프 최대 반복 수
                eps       (float) halt 임계 여유 (누적 certainty >= 1-eps면 halt)
                use_checkpoint (bool) 학습 중 gradient checkpointing 사용 여부
        output: 없음.
        """
        super().__init__()
        self.moe = MoE(dim=dim, hidden_dim=hidden_dim, experts=experts, dropout=dropout)
        self.max_steps = max_steps
        self.eps = eps
        self.use_checkpoint = use_checkpoint

    def _moe_call(self, flat):
        """
        checkpointing 래퍼. 학습 중에만 checkpoint 적용(메모리 절약).
        input : flat [B*N, D]
        output: ([B*N, D], [B*N, 1])  MoE의 (results, certainty)
        """
        if self.use_checkpoint and self.training:
            return checkpoint(self.moe, flat, use_reentrant=False)
        return self.moe(flat)

    def forward(self, x):
        """
        토큰별 적응적 반복(Adaptive Computation): MoE를 최대 max_steps번 반복하되,
        토큰별 누적 certainty가 1-eps를 넘으면 그 토큰은 halt(동결)된다.

        input : x [B, N, D]   (B=배치, N=시퀀스 길이, D=모델 차원)
        output: sum_logit     [B, N, D]   step_cert로 가중합된 MoE 출력
                halting_probs  [T, B, N]   각 스텝의 step_cert (PonderNet 손실용), T<=max_steps
        """
        B, N, D = x.shape
        state          = x                                  # [B, N, D] 현재 토큰 상태
        sum_certainty  = torch.zeros_like(state[..., :1])   # [B, N, 1] 누적 확신도
        sum_logit      = torch.zeros_like(state)            # [B, N, D] 누적 출력

        halting_probs = []  # 매 스텝의 step_cert를 모음 (정규화/손실용)

        for t in range(self.max_steps):
            flat = state.reshape(B * N, D)                  # [B*N, D] MoE는 토큰 평탄화 입력
            new_flat, cert_flat = self._moe_call(flat)      # [B*N, D], [B*N, 1]
            new_state = new_flat.reshape(B, N, D)           # [B, N, D]
            cert      = cert_flat.reshape(B, N, 1)          # [B, N, 1] 이번 스텝 확신도

            # active: 아직 halt 안 된 토큰만 기여(1), halt된 토큰은 0
            active = (sum_certainty < 1 - self.eps).to(cert.dtype)  # [B, N, 1]

            if t == self.max_steps - 1:
                # 마지막 스텝: 남은 mass 전부 할당
                step_cert = (1 - sum_certainty) * active            # [B, N, 1]
            else:
                step_cert = torch.min(1 - sum_certainty, cert) * active  # [B, N, 1]

            sum_logit     = sum_logit + new_state * step_cert  # [B, N, D] step_cert 가중 누적
            sum_certainty = sum_certainty + step_cert          # [B, N, 1]
            # state는 active 토큰만 갱신, halted는 freeze
            state = torch.where(active > 0.5, new_state, state)  # [B, N, D]

            halting_probs.append(step_cert.squeeze(-1))     # [B, N]

            # inference: 모든 토큰이 halt되면 남은 스텝은 step_cert=0이라 출력에 무의미 → break
            # train: self.training=True 이므로 이 분기는 절대 안 탐 → 항상 max_steps 고정
            if not self.training and bool((sum_certainty >= 1 - self.eps).all()):
                break

        halting_probs = torch.stack(halting_probs, dim=0)   # [T, B, N]  (train이면 T=max_steps)
        return sum_logit, halting_probs


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
        return self.to_out(out)                                  # [B, N, D]


class Transformer(nn.Module):
    """
    (Attention, AMoE) 쌍을 depth개 쌓은 본체. 각 블록은 residual.
    AMoE가 내는 layer별 halting_probs를 모아 PonderNet 손실용으로 surface한다.
    """

    def __init__(self, dim, depth, max_len, mlp_dim, heads, dim_head,
                 experts, base=10000, dropout=0.):
        """
        input : dim, depth(레이어 수), max_len, mlp_dim, heads, dim_head, experts, base, dropout
        output: 없음.
        """
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, max_len, heads, dim_head, base, dropout),
                AMoE(dim=dim, hidden_dim=mlp_dim, experts=experts, dropout=dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        """
        input : x [B, N, D]
        output: x [B, N, D]                  최종 RMSNorm 출력
                all_halting_probs (list)     레이어별 [T, B, N] 리스트 (길이=depth)
        """
        all_halting_probs = []
        for atten, ff in self.layers:
            x = atten(x) + x                        # [B, N, D] attention residual
            ff_out, hp = ff(x)                      # AMoE: ([B, N, D], [T, B, N])
            x = ff_out + x                          # [B, N, D] AMoE residual
            all_halting_probs.append(hp)
        return self.norm(x), all_halting_probs


class LLM(nn.Module):
    """
    임베딩 → Transformer → 선형 head 의 디코더-온리 LM.
    labels가 주어지면(train) task_loss(CE) + ponder_beta * PonderNet KL을 손실로 반환.
    labels가 없으면(inference) loss=None, logits만 반환.
    """

    def __init__(self, dim, depth, max_len, mlp_dim, heads, dim_head,
                 vocab_size, padding_idx, experts,
                 base=10000, dropout=0., ponder_beta=0.01, lambda_p=0.2):
        """
        input : 모델 하이퍼파라미터들 + vocab_size, padding_idx,
                ponder_beta(PonderNet 손실 가중), lambda_p(기하분포 prior 파라미터)
        output: 없음.
        """
        super().__init__()
        self.padding_idx = padding_idx
        self.ponder_beta = ponder_beta
        self.lambda_p = lambda_p
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.transformer = Transformer(dim, depth, max_len, mlp_dim, heads,
                                       dim_head, experts, base, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp_head = nn.Linear(dim, vocab_size)              # [D] → [V]

    def _ponder_loss(self, all_halting_probs):
        """
        PonderNet 정규화: 레이어별 halting 분포가 기하분포 prior(lambda_p)를 따르도록 KL.
        input : all_halting_probs (list)  레이어별 [T, B, N]
        output: scalar tensor             레이어 평균 KL
        """
        total = 0.0
        for hp in all_halting_probs:                # hp: [T, B, N]
            T = hp.size(0)
            prior = torch.tensor(
                [self.lambda_p * (1 - self.lambda_p) ** t for t in range(T)],
                device=hp.device, dtype=hp.dtype,
            )                                       # [T]
            prior = prior / prior.sum()             # [T] 정규화
            kl = (hp * (hp.clamp_min(1e-8).log()
                        - prior.view(T, 1, 1).log())).sum(dim=0)   # [B, N]
            total = total + kl.mean()
        return total / len(all_halting_probs)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        """
        input : input_ids [B, N]   토큰 id
                labels    [B, N]   (선택) 다음 토큰 예측 라벨. 있으면 손실 계산(train).
                attention_mask     (미사용, HF 인터페이스 호환용)
        output: CausalLMOutput(loss, logits)
                logits [B, N, V]; loss는 labels 있을 때만 scalar, 없으면 None.
        """
        x = self.embedding(input_ids)                          # [B, N, D]
        x = self.dropout(x)                                    # [B, N, D]
        x, all_halting_probs = self.transformer(x)             # [B, N, D], list of [T, B, N]
        logits = self.mlp_head(x)                              # [B, N, V]

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()      # [B, N-1, V]
            shift_labels = labels[:, 1:].contiguous()          # [B, N-1]
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # [B*(N-1), V]
                shift_labels.view(-1),                         # [B*(N-1)]
                ignore_index=-100,
            )
            ponder_loss = self._ponder_loss(all_halting_probs)
            loss = task_loss + self.ponder_beta * ponder_loss
        return CausalLMOutput(loss=loss, logits=logits)


class TiktokenHFWrapper(PreTrainedTokenizer):
    """
    tiktoken r50k_base(GPT-2 BPE, ~50k vocab)를 HuggingFace PreTrainedTokenizer
    인터페이스로 감싸 DataCollatorForLanguageModeling 등과 호환되게 한다.
    """

    vocab_files_names = {}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, encoding_name="r50k_base", **kwargs):
        """
        input : encoding_name (str) tiktoken 인코딩 이름
        output: 없음. eos/bos/unk/pad 토큰을 모두 <|endoftext|>로 설정.
        """
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
        """output: (int) vocab 크기."""
        return self._enc.n_vocab

    def get_vocab(self):
        """output: dict {decoded_token_str: id} (전체 vocab)."""
        return {self._enc.decode([i]): i for i in range(self.vocab_size)}

    def _tokenize(self, text):
        """input: text (str) → output: list[str] (id를 문자열화한 토큰)."""
        return [str(i) for i in self._enc.encode(text, allowed_special={"<|endoftext|>"})]

    def _convert_token_to_id(self, token):
        """input: token (str) → output: (int) id."""
        return int(token)

    def _convert_id_to_token(self, index):
        """input: index (int) → output: (str) 토큰."""
        return str(index)

    def convert_tokens_to_string(self, tokens):
        """input: tokens (list[str]) → output: (str) 디코딩된 텍스트."""
        return self._enc.decode([int(t) for t in tokens])

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """tiktoken은 별도 vocab 파일이 없으므로 빈 튜플 반환."""
        return ()


class MemmapDataset(Dataset):
    """
    사전 토크나이즈된 uint16 바이너리(train.bin / val.bin)를 numpy.memmap으로 읽어
    block_size 윈도우로 슬라이스해 제공하는 데이터셋.
    """

    def __init__(self, path, block_size, dtype=np.uint16, max_tokens=None):
        """
        input : path (str) .bin 경로, block_size (int) 윈도우 길이,
                dtype 토큰 dtype, max_tokens (int|None) 사용할 토큰 상한
        output: 없음.
        """
        self.data = np.memmap(path, dtype=dtype, mode="r")     # [n_tokens]
        self.block_size = block_size

        n_tokens = len(self.data)
        if max_tokens is not None:
            n_tokens = min(n_tokens, max_tokens)

        self.n_blocks = n_tokens // block_size

    def __len__(self):
        """output: (int) 블록(샘플) 개수."""
        return self.n_blocks

    def __getitem__(self, idx):
        """
        input : idx (int)
        output: dict {"input_ids": LongTensor [block_size]}
        """
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.from_numpy(self.data[start:end].astype(np.int64))   # [block_size]
        return {"input_ids": x}
