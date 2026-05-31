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
        self.dim = dim
        self.num_experts = experts
        self.gate = nn.Linear(dim, experts)              # [D] → [E] 각 전문가당 확신도 출력 -> 가장 확신도가 높은 전문가한테 라우팅
        self.norm = nn.RMSNorm(dim)

        self.expert1 = nn.Parameter(torch.empty(experts, dim, hidden_dim)) # [D, H] Linear 8개를 묶음
        self.expert2 = nn.Parameter(torch.empty(experts, hidden_dim, dim + 1)) # [H, D] Linear 8개를 묶음

        nn.init.xavier_uniform_(self.expert1) #가중치 초기화
        nn.init.xavier_uniform_(self.expert2) #가중치 초기화

        
    def forward(self, x):
        """
        input : x [S, D]   (S = B*N, 토큰 평탄화)
        output: results   [S, D]   선택된 전문가 출력 * 게이트 가중치
                certainty [S, 1]   이 스텝의 토큰별 확신도 (sigmoid, 0~1)
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
                expert_outputs.append(expert_inputs[i].new_empty(0, self.dim+1))
                continue
            
            # i번째 전문가 연산 진행 (ex1 곱하고 GELU 거쳐 ex2 곱하기)
            h = F.gelu(torch.matmul(self.norm(expert_inputs[i]), self.expert1[i]))
            out = torch.matmul(h, self.expert2[i])           # [S, D+1]
            expert_outputs.append(out)

        #연산된 자른거를 합치기
        combined_outputs = torch.cat(expert_outputs, dim=0) # [S, D+1] (정렬된 상태)

        results = torch.empty_like(combined_outputs)

        results[sort_idx] = combined_outputs # [S, D+1] (원래 순서 복원)

        results, certainty = results[:, :-1] * weights, F.sigmoid(results[:, -1:])
        
        return results, certainty, LBL


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
        sum_certainty  = torch.zeros_like(state[..., :1])   # [B, N, 1] 누적 확신도
        sum_logit      = torch.zeros_like(state)            # [B, N, D] 누적 출력

        halting_probs = []  
        total_LBL = 0.0

        for t in range(self.max_steps):
            # =====================================================================
            # [수정된 부분 시작: 동적 텐서 추출을 통한 True Sparsity (FLOPs 절약) 구현]
            # =====================================================================
            flat = state.view(B * N, D)
            sum_cert_flat = sum_certainty.view(B * N)

            # 1. 활성 토큰 마스크 생성 (이번 스텝에 연산이 필요한 토큰만 True)
            active_mask = sum_cert_flat < (1 - self.eps)

            # Inference 시 모든 토큰이 halt 상태면 즉시 조기 종료 (행렬 곱 원천 차단)
            if not self.training and not active_mask.any():
                break

            # 2. 활성 토큰만 메모리에서 추출 (텐서 크기가 [B*N, D]에서 [Num_Active, D]로 동적 축소됨)
            active_tokens = flat[active_mask]  # [S, D]
            
            # 결과를 담을 빈 텐서 (halt된 토큰은 0으로 남음)
            new_flat = torch.zeros_like(flat)
            cert_flat = torch.zeros_like(sum_cert_flat).unsqueeze(-1) # [B*N, 1]

            # 3. 추출된 토큰이 있을 때만 MoE 연산 수행 (핵심 연산량 감소 구간)
            if active_tokens.numel() > 0:
                new_active, cert_active, LBL = self._moe_call(active_tokens)
                total_LBL = total_LBL + LBL
                
                # 4. 연산된 결과를 원래 배치 위치로 복원 (Scatter)
                # autocast(fp16/bf16)에서 MoE 출력 dtype이 new_flat/cert_flat(fp32)과
                # 다를 수 있어 dst dtype에 맞춰 캐스팅 (누적은 fp32로 유지)
                new_flat[active_mask] = new_active.to(new_flat.dtype)
                cert_flat[active_mask] = cert_active.to(cert_flat.dtype)

            # 다시 3차원으로 복구
            new_state = new_flat.view(B, N, D)
            cert      = cert_flat.view(B, N, 1)
            # =====================================================================
            # [수정된 부분 끝]
            # =====================================================================

            # active: 아직 halt 안 된 토큰만 기여(1), halt된 토큰은 0
            active = (sum_certainty < 1 - self.eps).to(cert.dtype)  # [B, N, 1]

            if t == self.max_steps - 1:
                # 마지막 스텝: 남은 mass 전부 할당
                step_cert = (1 - sum_certainty) * active            
            else:
                step_cert = torch.min(1 - sum_certainty, cert) * active  

            sum_logit     = sum_logit + new_state * step_cert  
            sum_certainty = sum_certainty + step_cert          
            
            # state는 active 토큰만 갱신, halted는 freeze (위에서 0으로 복원되었으므로 기존 state 유지)
            state = torch.where(active > 0.5, new_state, state)  

            halting_probs.append(step_cert.squeeze(-1))     

        halting_probs = torch.stack(halting_probs, dim=0)   # [T, B, N]
        return sum_logit, halting_probs, total_LBL


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
        total_LBL = 0

        for atten, ff in self.layers:
            x = atten(x) + x                        # [B, N, D] attention residual
            ff_out, hp, LBL = ff(x)                 # AMoE: ([B, N, D], [T, B, N])
            x = ff_out + x                          # [B, N, D] AMoE residual
            all_halting_probs.append(hp)
            total_LBL = total_LBL + LBL
        return self.norm(x), all_halting_probs, total_LBL


class LLM(nn.Module):
    """
    임베딩 → Transformer → 선형 head 의 디코더-온리 LM.
    labels가 주어지면(train) task_loss(CE) + ponder_beta * PonderNet KL을 손실로 반환.
    labels가 없으면(inference) loss=None, logits만 반환.
    """

    def __init__(self, dim, depth, max_len, mlp_dim, heads, dim_head,
                 vocab_size, padding_idx, experts,
                 base=10000, dropout=0., ponder_beta=0.01, lambda_p=0.2, alpha=0.01):
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
        self.alpha = alpha

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
        x, all_halting_probs, LBL = self.transformer(x)        # [B, N, D], list of [T, B, N], scala
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
            loss = task_loss + self.ponder_beta * ponder_loss + self.alpha*LBL
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