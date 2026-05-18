"""
train_custom.py
"""

import os
import math
import signal
import argparse
import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import wandb
from muon import SingleDeviceMuonWithAuxAdam as MuonWithAuxAdam
from custom_model_train import LLM, TiktokenHFWrapper, MemmapDataset



def install_signal_handlers():
    def _handler(signum, frame):
        print(f"signal {signum} → cleanup", flush=True)
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=143, quiet=True)
        finally:
            os._exit(143)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--train_bin_path", default="train.bin")
    p.add_argument("--val_bin_path", default="val.bin")
    p.add_argument("--output_dir", default="custom-llm-out")
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--max_size", type=int, default=50_000_000)
    p.add_argument("--max_val_size", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=576)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=2048)
    p.add_argument("--rope_base", type=int, default=10000)
    p.add_argument("--dropout", type=float, default=0.0)

    # AMoE
    p.add_argument("--experts",      type=int,   default=4)
    p.add_argument("--ponder_beta",  type=float, default=0.01)
    p.add_argument("--lambda_p",     type=float, default=0.2)
    p.add_argument("--balance_beta", type=float, default=0.01)

    # Muon
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=0.1)
    return p.parse_args()


def init_wandb(args):
    wandb.init(project=args.project, name=args.run_name, config=vars(args),
               allow_val_change=True)
    for k, v in dict(wandb.config).items():
        if hasattr(args, k):
            setattr(args, k, v)
    print(f"args={vars(args)}")
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

    tokenizer = TiktokenHFWrapper("r50k_base")

    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id,
        experts=args.experts,
        base=args.rope_base, dropout=args.dropout,
        ponder_beta=args.ponder_beta, lambda_p=args.lambda_p,
        balance_beta=args.balance_beta,
    )

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
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        run_name=args.run_name,
        dataloader_pin_memory=True,
        seed=args.seed,
        max_steps=max_steps,
    )

    optimizer = create_muon_optimizer(model, args)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        optimizers=(optimizer, None),
    )
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