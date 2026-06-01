"""
train_common.py

Shared training-entry boilerplate (signal handling + W&B init), deduplicated
from the byte-identical helpers in train_custom.py and train_with_hooks.py.
"""
import os
import signal

import wandb


def install_signal_handlers():
    """
    Install SIGTERM/SIGINT handlers that finish the W&B run cleanly, then hard
    exit with code 143. No inputs/outputs; registers process-level handlers.
    """
    def _handler(signum, frame):
        print(f"signal {signum} → cleanup", flush=True)
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=143, quiet=True)
        finally:
            os._exit(143)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def init_wandb(args):
    """
    Start W&B and fold any sweep-provided config back onto `args`.

    input : args (Namespace) — needs `.project`, `.run_name`; all of `vars(args)`
            is logged as config.
    output: args (same object, mutated with sweep overrides).
    """
    wandb.init(project=args.project, name=args.run_name, config=vars(args),
               allow_val_change=True)
    # sweep override 반영
    for k, v in dict(wandb.config).items():
        if hasattr(args, k):
            setattr(args, k, v)
    print(f"args={vars(args)}")
    return args
