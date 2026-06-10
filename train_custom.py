"""
train_custom.py
"""

import os
import math
import argparse
import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import wandb
from model import LLM, TiktokenHFWrapper, MemmapDataset
from optim import build_muon_optimizer
from config import add_base_args
from train_common import install_signal_handlers, init_wandb


def parse_args():
    p = argparse.ArgumentParser()
    add_base_args(p, output_dir_default="custom-llm-out")
    return p.parse_args()


def run_training(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert os.path.exists(args.train_bin_path), f"파일 없음: {args.train_bin_path}"
    assert os.path.exists(args.val_bin_path), f"파일 없음: {args.val_bin_path}"

    tokenizer = TiktokenHFWrapper("r50k_base")

    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id,
        experts=args.experts,
        base=args.rope_base, dropout=args.dropout,
        ponder_beta=args.ponder_beta, lambda_p=args.lambda_p, alpha=args.alpha,
    )

    # 고정 N회 재귀 대조군: ponder_steps → AMoE.max_steps, grad_checkpoint → use_checkpoint
    for _atten, _amoe in model.transformer.layers:
        _amoe.max_steps = args.ponder_steps
        _amoe.use_checkpoint = bool(args.grad_checkpoint)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] params={n_params/1e6:.2f}M")
    wandb.run.summary["n_params_M"] = n_params / 1e6

    train_ds = MemmapDataset(args.train_bin_path, args.block_size)
    eval_ds = MemmapDataset(args.val_bin_path, args.block_size, max_tokens=args.max_val_size)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    max_steps = max(1, math.ceil(args.max_size / tokens_per_step))
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
        save_strategy="no",          # 체크포인트 저장 안 함 (로그만 남김)
        bf16=torch.cuda.is_available(),
        report_to="wandb",
        run_name=args.run_name,
        dataloader_pin_memory=True,
        seed=args.seed,
        max_steps=max_steps,
    )

    optimizer = build_muon_optimizer(model, args)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        optimizers=(optimizer, None),
    )
    # HF Trainer >=4.46 GA loss bug fix:
    # LLM.forward has **kwargs → HF infers model_accepts_loss_kwargs=True → loss
    # is NOT divided by grad_accum_steps for reporting → train/loss is inflated
    # by grad_accum. Force False to restore correct mean-reduction scaling.
    trainer.model_accepts_loss_kwargs = False
    trainer.train()

    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
    print(f"[Eval] loss={metrics['eval_loss']:.4f}  ppl={ppl:.2f}")
    wandb.log({"final/eval_loss": metrics["eval_loss"], "final/perplexity": ppl})
    trainer.save_model(args.output_dir)
    wandb.finish()


def main():
    install_signal_handlers()
    args = init_wandb(parse_args())
    run_training(args)


if __name__ == "__main__":
    main()