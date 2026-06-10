"""
config.py

Shared CLI argument definitions for the training entry points. The base args are
identical across train_custom.py and train_with_hooks.py except for --output_dir's
default, so they're defined here once; each entry passes its own output_dir default
and adds its own extra flags.
"""
import argparse  # noqa: F401  (re-exported convenience for entry points)


def add_base_args(parser, output_dir_default):
    """
    Add the training args common to train_custom.py and train_with_hooks.py.

    input : parser (argparse.ArgumentParser)
            output_dir_default (str)  per-entry default for --output_dir
    output: parser (mutated in place, returned for convenience)
    """
    # paths / wandb
    parser.add_argument("--project", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--train_bin_path", default="train.bin")
    parser.add_argument("--val_bin_path", default="val.bin")
    parser.add_argument("--output_dir", default=output_dir_default)

    # data + schedule
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_size", type=int, default=50_000_000)
    parser.add_argument("--max_val_size", type=int, default=500_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=576)

    # model
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--mlp_dim", type=int, default=2048,
                        help="현재 무효 — MoE 내부 4*dim 하드코딩")
    parser.add_argument("--rope_base", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.0)

    # MoE / 고정 N회 재귀 (대조군)
    parser.add_argument("--experts",     type=int,   default=4)
    parser.add_argument("--ponder_beta", type=float, default=0.01,
                        help="(레거시·미사용) PonderNet 손실 가중 — 대조군은 ponder 손실 없음")
    parser.add_argument("--lambda_p",    type=float, default=0.2,
                        help="(레거시·미사용) PonderNet prior — 대조군은 미사용")
    parser.add_argument("--alpha",       type=float, default=0.01,
                        help="load-balance loss (LBL) weight in LLM loss")
    parser.add_argument("--ponder_steps", type=int, default=10,
                        help="고정 수직 재귀 횟수 N (AMoE.max_steps로 세팅)")
    parser.add_argument("--grad_checkpoint", type=int, default=1,
                        help="1=AMoE MoE 호출에 gradient checkpointing 적용, 0=미적용")
    parser.add_argument("--compile", type=int, default=0,
                        help="1=torch.compile 적용 (HF Trainer torch_compile)")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="체크포인트 저장 간격(스텝)")
    parser.add_argument("--resume", type=int, default=0,
                        help="1=고정 output_dir의 마지막 체크포인트에서 재개 (메인 학습용)")

    # Muon / AdamW
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="AdamW (embedding/head/bias/norm) learning rate")
    parser.add_argument("--muon_lr", type=float, default=0.02,
                        help="Muon (2D hidden weight) learning rate")
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    return parser
