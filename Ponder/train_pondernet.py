"""
train_pondernet.py

Training entry for the PonderNet baseline. Mirrors AMOE/train_custom.py (same HF
Trainer + Muon optimizer + data + schedule) but builds PonderLLM, so the baseline
trains under identical conditions to AMoE for a fair comparison. Self-contained:
uses only the local copied modules.
"""
import os
import math
import argparse
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint
import wandb

from data import TiktokenHFWrapper, MemmapDataset
from optim import build_muon_optimizer
from config import add_base_args
from common import install_signal_handlers, init_wandb
from pondernet import PonderLLM


def parse_args():
    p = argparse.ArgumentParser()
    add_base_args(p, output_dir_default="pondernet-out")
    # PonderNet-specific (base args also supply --ponder_beta, --lambda_p, --dim, --mlp_dim, ...)
    p.add_argument("--ponder_steps", type=int, default=10,
                   help="max ponder steps N (PonderNet halting horizon)")
    p.add_argument("--core_depth", type=int, default=6,
                   help="depth of the weight-shared recurrent core")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the ponder hot path (~1.8x; needs torch>=2.12)")
    p.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=False,
                   help="gradient checkpointing per ponder step (less VRAM, ~1.4x slower)")
    return p.parse_args()


def run_training(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert os.path.exists(args.train_bin_path), f"파일 없음: {args.train_bin_path}"
    assert os.path.exists(args.val_bin_path), f"파일 없음: {args.val_bin_path}"

    tokenizer = TiktokenHFWrapper("r50k_base")

    model = PonderLLM(
        dim=args.dim, max_len=args.block_size, mlp_dim=args.mlp_dim,
        heads=args.heads, dim_head=args.dim_head,
        vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id,
        core_depth=args.core_depth, base=args.rope_base, dropout=args.dropout,
        max_steps=args.ponder_steps, lambda_p=args.lambda_p, ponder_beta=args.ponder_beta,
        use_checkpoint=args.grad_checkpoint,
    )
    if args.compile:
        # compile the hot path (core + head + per-step CE); avoids graph-breaks in the
        # outer ponder loop / halting-sum side effect. ~1.8x on H100. Needs torch>=2.12.
        model._ponder_step = torch.compile(model._ponder_step)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] PonderNet params={n_params/1e6:.2f}M  "
          f"(core_depth={args.core_depth}, ponder_steps={args.ponder_steps})")
    wandb.run.summary["n_params_M"] = n_params / 1e6

    train_ds = MemmapDataset(args.train_bin_path, args.block_size)
    # 중간 모니터링용은 슬라이스(--max_val_size)로 빠르게, 최종 보고값은 val 전체로.
    eval_ds = MemmapDataset(args.val_bin_path, args.block_size, max_tokens=args.max_val_size)
    eval_full_ds = MemmapDataset(args.val_bin_path, args.block_size)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    max_steps = max(1, math.ceil(args.max_size / tokens_per_step))
    print(f"[Budget] max_size={args.max_size:,} → max_steps={max_steps:,} "
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
    # HF Trainer >=4.46 GA loss fix (same as AMoE train_custom.py)
    trainer.model_accepts_loss_kwargs = False
    # 중간에 끊겨도 재실행하면 output_dir의 마지막 체크포인트에서 자동 재개(없으면 처음부터).
    last_ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last_ckpt:
        print(f"[Resume] 체크포인트에서 재개: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    metrics = trainer.evaluate(eval_dataset=eval_full_ds)  # val 전체(20M) 기준 최종값
    ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
    print(f"[Eval-full] loss={metrics['eval_loss']:.4f}  ppl={ppl:.2f}  (full val)")
    wandb.log({"final/eval_loss": metrics["eval_loss"], "final/perplexity": ppl})
    trainer.save_model(args.output_dir)
    wandb.finish()


def main():
    install_signal_handlers()
    args = init_wandb(parse_args())
    run_training(args)


if __name__ == "__main__":
    main()
