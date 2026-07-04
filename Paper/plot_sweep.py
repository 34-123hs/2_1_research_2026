"""Render sweep result graphs from the AMOE-SWEEP123 dump."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DUMP = "wandb_data_sweep/wandb_dump.json"
OUTDIR = "assets"

ALPHA_COLORS = {0.005: "#2ca02c", 0.01: "#1f77b4", 0.02: "#ff7f0e", 0.05: "#d62728"}


def finished_sweep_runs():
    d = json.load(open(DUMP, encoding="utf-8"))
    runs = d["projects"][0]["runs"]
    return [r for r in runs if "sweep" in r["name"]
            and r["state"] == "finished"
            and r["summary"].get("eval/loss") is not None]


def plot_loss_by_alpha(runs, fname):
    plt.figure(figsize=(7, 4.5))
    best = min(runs, key=lambda r: r["summary"]["eval/loss"])
    for r in runs:
        a = r["config"]["alpha"]
        loss = r["summary"]["eval/loss"]
        jitter = (hash(r["name"]) % 100 / 100 - 0.5) * 0.006
        is_best = r is best
        plt.scatter(a + jitter, loss, s=180 if is_best else 60,
                    color=ALPHA_COLORS.get(a, "#777"),
                    edgecolor="white", zorder=4 if is_best else 3,
                    marker="*" if is_best else "o")
    plt.scatter([], [], marker="*", s=180, color="#333", label="최적 run")
    plt.xscale("log")
    plt.xticks(sorted(ALPHA_COLORS), [str(a) for a in sorted(ALPHA_COLORS)])
    plt.xlabel("alpha (load-balancing 가중치, log scale)")
    plt.ylabel("eval/loss")
    plt.title("2D-AMoE Sweep: alpha에 따른 검증 손실")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print("wrote", path)


def plot_loss_vs_lr(runs, fname):
    plt.figure(figsize=(7, 4.5))
    for a in sorted(ALPHA_COLORS):
        pts = [(r["config"]["lr"], r["summary"]["eval/loss"])
               for r in runs if r["config"]["alpha"] == a]
        if pts:
            xs, ys = zip(*pts)
            plt.scatter(xs, ys, s=55, color=ALPHA_COLORS[a],
                        edgecolor="white", zorder=3, label=f"alpha={a}")
    plt.xscale("log")
    plt.xlabel("lr (log scale)")
    plt.ylabel("eval/loss")
    plt.title("2D-AMoE Sweep: 학습률 / 검증 손실 (alpha별)")
    plt.legend(title="alpha")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print("wrote", path)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    runs = finished_sweep_runs()
    plot_loss_by_alpha(runs, "sweep_loss_by_alpha.png")
    plot_loss_vs_lr(runs, "sweep_loss_vs_lr.png")


if __name__ == "__main__":
    main()
