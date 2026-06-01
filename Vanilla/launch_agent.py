"""
launch_agent.py — wandb sweep agent 런처

역할:
  - wandb.agent를 띄워 큐에서 다음 trial을 받음
  - 받은 hyperparameter를 CLI 인자로 펼쳐 train.py를 subprocess로 실행
  - 단일 GPU면 그냥 python, 멀티 GPU면 torchrun으로 자동 전환
  - SIGTERM/SIGINT를 자식 process group 전체에 전파해 DDP 정리

사용:
  python launch_agent.py --sweep_id <entity>/<project>/<sweep_id>
                         [--count N] [--nproc K]
"""

import os
import sys
import signal
import argparse
import subprocess
import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True, help="<entity>/<project>/<sweep_id>")
    p.add_argument("--count", type=int, default=None, help="이 agent가 처리할 trial 수")
    p.add_argument("--nproc", type=int, default=1, help="trial당 GPU 수 (1=단일, 2+=DDP)")
    args = p.parse_args()

    parts = args.sweep_id.split("/")
    if len(parts) < 3:
        sys.exit("--sweep_id must be in <entity>/<project>/<sweep_id> form")
    project = parts[1]
    nproc = args.nproc

    def runner():
        wandb.init(project=project)
        config = dict(wandb.config)
        run_id = wandb.run.id

        if nproc <= 1:
            cmd = [sys.executable, "train.py"]
        else:
            cmd = ["torchrun", f"--nproc_per_node={nproc}", "--standalone", "train.py"]
        cmd += ["--project", project]
        for k, v in config.items():
            cmd += [f"--{k}", str(v)]
        print(f"[agent] launch: {' '.join(cmd)}", flush=True)

        # 자식 train.py가 같은 wandb run에 이어 쓰도록 부모 run을 닫고 환경 전달
        wandb.finish()
        env = os.environ.copy()
        env["WANDB_RUN_ID"] = run_id
        env["WANDB_RESUME"] = "allow"

        # 자식을 새 process group leader로 띄움 → 그룹째 신호 전파 가능
        proc = subprocess.Popen(cmd, preexec_fn=os.setsid, env=env)

        def _forward(signum, frame):
            print(f"[agent] forward signal {signum} → pgid {proc.pid}", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signum)
            except ProcessLookupError:
                pass
        try:
            signal.signal(signal.SIGTERM, _forward)
            signal.signal(signal.SIGINT, _forward)
        except ValueError:
            pass

        ret = proc.wait()
        print(f"[agent] child exited code={ret}", flush=True)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    print(f"[agent] CUDA_VISIBLE_DEVICES={visible}  nproc={nproc}  "
          f"sweep={args.sweep_id}  project={project}  "
          f"count={args.count or 'inf'}", flush=True)

    wandb.agent(args.sweep_id, function=runner, count=args.count)


if __name__ == "__main__":
    main()
