"""
model.py

대조군(control) 모델 정의. 주력 AMoE(적응형 halting + PonderNet)에서 적응성과
ponder 손실을 제거하고, FFN 슬롯을 고정 max_steps회 MoE 재귀로 치환한 ablation이다.
- 동작 차이는 nn.Module의 self.training 플래그로만 갈린다:
    * AMoE      : 수직 루프는 항상 max_steps회 고정(halting/조기 break 없음).
                  train에서는 MoE 호출마다 gradient checkpointing 적용.
    * LLM       : labels가 주어지면(=train) task_loss + alpha*LBL을 계산,
                  없으면(=inference) logits만 반환.
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
    토큰마다 게이트가 전문가 1명을 고르고, 그 전문가 출력과 load-balance loss(LBL)를 낸다.
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
        self.dim = dim
        self.num_experts = experts
        self.gate = nn.Linear(dim, experts)              # [D] → [E] 각 전문가당 확신도 출력 -> 가장 확신도가 높은 전문가한테 라우팅
        self.norm = nn.RMSNorm(dim)

        self.expert1 = nn.Parameter(torch.empty(experts, dim, hidden_dim)) # [D, H] Linear 8개를 묶음
        self.expert2 = nn.Parameter(torch.empty(experts, hidden_dim, dim)) # [H, D] Linear 8개를 묶음

        nn.init.xavier_uniform_(self.expert1) #가중치 초기화
        nn.init.xavier_uniform_(self.expert2) #가중치 초기화

        
    def forward(self, x):
        """
        input : x [S, D]   (S = B*N, 토큰 평탄화)
        output: results [S, D]   선택된 전문가 출력 * 게이트 가중치
                LBL     scalar    load-balance loss
        """
        gate_probs = F.softmax(self.gate(x), dim=-1)            # [S, E]
        weights, selected = torch.topk(gate_probs, 1, dim=-1)   # 각 [S, 1]
        expert_mask = F.one_hot(selected.squeeze(-1), num_classes=self.num_experts).to(x.dtype) #[S, E]

        #Load Balance Loss 계산
        P_i = gate_probs.mean(dim=0)  # [E]
        f_i = expert_mask.mean(dim=0) # [E]
        LBL = self.num_experts * torch.sum(f_i * P_i)

        #X 정렬시키기
        sort_idx = torch.argsort(selected.squeeze(-1)) # [S]
        x_sorted = x[sort_idx] # 전문가 순서대로 정렬된 토큰들 [S, D]

        # 전문가마다 자르기 / [input1, input2, ...]
        tokens_per_expert = expert_mask.sum(dim=0, dtype=torch.long) # [E]
        expert_inputs = torch.split(x_sorted, tokens_per_expert.tolist(), dim=0)
        expert_outputs = []

        # 자른 거를 연산하기
        for i in range(self.num_experts):
            if expert_inputs[i].size(0) == 0:
                # 해당 전문가에게 할당된 토큰이 없으면 빈 텐서 추가
                expert_outputs.append(expert_inputs[i].new_empty(0, self.dim))
                continue
            
            # i번째 전문가 연산 진행 (ex1 곱하고 GELU 거쳐 ex2 곱하기)
            h = F.gelu(torch.matmul(self.norm(expert_inputs[i]), self.expert1[i]))
            out = torch.matmul(h, self.expert2[i])           # [S, D]
            expert_outputs.append(out)

        #연산된 자른거를 합치기
        combined_outputs = torch.cat(expert_outputs, dim=0) # [S, D] (정렬된 상태)

        results = torch.empty_like(combined_outputs)

        results[sort_idx] = combined_outputs # [S, D] (원래 순서 복원)

        results = results * weights
        
        return results, LBL


class AMoE(nn.Module):
    """
    MoE를 고정 max_steps회 반복하는 블록 (대조군).
    매 스텝 state를 MoE 출력으로 교체하며, halting/조기 break/certainty 없이
    항상 max_steps회 돈다. train에서는 MoE 호출마다 gradient checkpointing 적용.
    """

    def __init__(self, dim, hidden_dim, experts, dropout=0.,
                 max_steps=10, eps=1e-2, use_checkpoint=True):
        super().__init__()
        # 외부에서 정의된 MoE 클래스를 사용한다고 가정
        self.moe = MoE(dim=dim, hidden_dim=hidden_dim, experts=experts, dropout=dropout)
        self.max_steps = max_steps
        self.eps = eps
        self.use_checkpoint = use_checkpoint

    def _moe_call(self, flat):
        if self.use_checkpoint and self.training:
            return checkpoint(self.moe, flat, use_reentrant=False)
        return self.moe(flat)

    def forward(self, x):
        B, N, D = x.shape
        state          = x                                  # [B, N, D] 현재 토큰 상태

        total_LBL = 0.0

        for t in range(self.max_steps):
            # state를 평탄화 → MoE 1스텝 → [B,N,D]로 복구해 다음 스텝 입력으로 사용
            flat = state.view(B * N, D)

            resudial, LBL = self._moe_call(flat)
            total_LBL = total_LBL + LBL
            
            flat = flat + resudial
            # 다시 3차원으로 복구
            state = flat.view(B, N, D)
       
        
        return state, total_LBL


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
    각 AMoE가 내는 load-balance loss(LBL)를 합산해 surface한다.
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
        output: x         [B, N, D]   최종 RMSNorm 출력
                total_LBL scalar      레이어별 LBL 합
        """
        total_LBL = 0

        for atten, ff in self.layers:
            x = atten(x) + x                        # [B, N, D] attention residual
            ff_out, LBL = ff(x)                 # AMoE: ([B, N, D], LBL scalar)
            x = ff_out                              # [B, N, D] AMoE가 내부 residual을 하므로 그대로 통과 (이중 잔차 방지)
            total_LBL = total_LBL + LBL
        return self.norm(x), total_LBL


class LLM(nn.Module):
    """
    임베딩 → Transformer → 선형 head 의 디코더-온리 LM.
    labels가 주어지면(train) task_loss(CE) + alpha * LBL을 손실로 반환.
    labels가 없으면(inference) loss=None, logits만 반환.
    """

    def __init__(self, dim, depth, max_len, mlp_dim, heads, dim_head,
                 vocab_size, padding_idx, experts,
                 base=10000, dropout=0., ponder_beta=0.01, lambda_p=0.2, alpha=0.01):
        """
        input : 모델 하이퍼파라미터들 + vocab_size, padding_idx,
                alpha(LBL 손실 가중). ponder_beta/lambda_p는 호환용으로 받되 미사용.
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
        self.alpha = alpha

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
        x, LBL = self.transformer(x)        # [B, N, D], LBL scalar
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
            loss = task_loss + self.alpha*LBL
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

if __name__ == '__main__':
    pass