"""
train.py — Custom LLM (RoPE + RMSNorm + SDPA) 학습 본체
train.bin, eval.bin 필요
"""

import os
import json
import math
import signal
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutput
import wandb
from muon import SingleDeviceMuonWithAuxAdam as MuonWithAuxAdam

WORLD_RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN = (WORLD_RANK == 0)


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


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.layers = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


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
    def __init__(self, dim, depth, max_len, mlp_dim, heads, dim_head, base=10000, dropout=0.):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, max_len, heads, dim_head, base, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for atten, ff in self.layers:
            x = atten(x) + x
            x = ff(x) + x
        return self.norm(x)


class LLM(nn.Module):
    accepts_loss_kwargs = False

    def __init__(self, dim, depth, max_len, mlp_dim, heads, dim_head,
                 vocab_size, base=10000, dropout=0.):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.transformer = Transformer(dim, depth, max_len, mlp_dim, heads,
                                       dim_head, base, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        x = self.transformer(x)
        logits = self.mlp_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)

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
    
def install_signal_handlers():
    def _handler(signum, frame):
        print(f"[rank {WORLD_RANK}] signal {signum} → cleanup", flush=True)
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=143, quiet=True)
        finally:
            os._exit(143)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=None)#
    p.add_argument("--run_name", default=None)#
    p.add_argument("--train_bin_path", default="train.bin")#
    p.add_argument("--val_bin_path", default="val.bin")#
    p.add_argument("--output_dir", default="custom-llm-out")#
    p.add_argument("--block_size", type=int, default=512)#
    p.add_argument("--batch_size", type=int, default=8)#
    p.add_argument("--grad_accum", type=int, default=4)#
    p.add_argument("--lr", type=float, default=3e-4)#
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=100)#
    p.add_argument("--eval_interval", type=int, default=50)#
    p.add_argument("--max_size", type=int, default=50_000_000)#
    p.add_argument("--max_val_size", type=int, default=500_000)#
    p.add_argument("--seed", type=int, default=576)#
    p.add_argument("--dim", type=int, default=512)#
    p.add_argument("--depth", type=int, default=6)#
    p.add_argument("--heads", type=int, default=8)#
    p.add_argument("--rope_base", type=int, default=10000)#
    p.add_argument("--dropout", type=float, default=0.0)#

    # Muon
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=0.1)
    return p.parse_args()

def init_wandb(args):
    if not IS_MAIN:
        return args
    wandb.init(project=args.project, name=args.run_name, config=vars(args),
               allow_val_change=True)
    for k, v in dict(wandb.config).items():
        if hasattr(args, k):
            setattr(args, k, v)
    print(f"[rank 0] args={vars(args)}")
    return args

def create_muon_optimizer(model, args):
    hidden_matrix_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        use_muon = (
            p.ndim == 2
            and "embedding" not in name
            and "mlp_head" not in name
        )

        if use_muon:
            hidden_matrix_params.append(p)
        else:
            other_params.append(p)

    if IS_MAIN:
        n_muon = sum(p.numel() for p in hidden_matrix_params)
        n_other = sum(p.numel() for p in other_params)
        print(f"[Optimizer] Muon params={n_muon:,}  Aux params={n_other:,}")

    param_groups = [
        dict(
            params=hidden_matrix_params,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=args.weight_decay,
            use_muon=True,
        ),
        dict(
            params=other_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_muon=False,
        ),
    ]
    return MuonWithAuxAdam(param_groups)

def run_training(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert os.path.exists(args.train_bin_path), f"파일 없음: {args.train_bin_path}"
    assert os.path.exists(args.val_bin_path), f"파일 없음: {args.val_bin_path}"

    enc = tiktoken.get_encoding("r50k_base")
    vocab_size = enc.n_vocab  # 50257

    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.dim*4, heads=args.heads, dim_head=args.dim//args.heads,
        vocab_size=vocab_size, base=args.rope_base,
        dropout=args.dropout,
    )

    if IS_MAIN:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] params={n_params/1e6:.2f}M  world_size={WORLD_SIZE}")
        wandb.run.summary["n_params_M"] = n_params / 1e6
    
    train_ds = MemmapDataset(args.train_bin_path, args.block_size)
    eval_ds = MemmapDataset(args.val_bin_path, args.block_size, max_tokens=args.max_val_size)

    def collate_fn(examples):
        input_ids = torch.stack([e["input_ids"] for e in examples])
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    tokens_per_step = args.batch_size * args.grad_accum * args.block_size * WORLD_SIZE
    max_steps = max(1, math.ceil(args.max_size / tokens_per_step))
    if IS_MAIN:
        print(f"[Budget] max_size={args.max_size:,} tokens → max_steps={max_steps:,} "
              f"(tokens/step={tokens_per_step:,})")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_interval,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="wandb",
        run_name=args.run_name,
        dataloader_pin_memory=True,
        seed=args.seed,
        ddp_find_unused_parameters=False,
        max_steps=max_steps,
    )

    optimizer = create_muon_optimizer(model, args)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        optimizers=(optimizer, None),
    )
    trainer.train()

    metrics = trainer.evaluate()
    if IS_MAIN:
        ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
        print(f"[Eval] loss={metrics['eval_loss']:.4f}  ppl={ppl:.2f}")
        wandb.log({"final/eval_loss": metrics["eval_loss"], "final/perplexity": ppl})
        trainer.save_model(args.output_dir)
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        wandb.finish()


def main():
    install_signal_handlers()
    args = init_wandb(parse_args())
    run_training(args)


if __name__ == "__main__":
    main()