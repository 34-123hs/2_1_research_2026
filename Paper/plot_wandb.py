"""Render PNG graphs for the paper from wandb_dump.json.

Reads the dump produced by fetch_wandb.py and writes PNGs into assets/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DUMP = "wandb_data/wandb_dump.json"
OUTDIR = "assets"
PROJECT = "nLoopMoE"

# fixed order n = 1,2,4,6,8
ORDER = ["full-n1", "full-n2", "full-n4", "full-n6", "full-n8"]
COLORS = {
    "full-n1": "#1f77b4",
    "full-n2": "#2ca02c",
    "full-n4": "#ff7f0e",
    "full-n6": "#d62728",
    "full-n8": "#9467bd",
}


def load_runs():
    data = json.load(open(DUMP, encoding="utf-8"))
    proj = next(p for p in data["projects"] if p["project"] == PROJECT)
    by_name = {r["name"]: r for r in proj["runs"]}
    return [by_name[n] for n in ORDER if n in by_name]


def label(run):
    return f"n={run['config_key'].get('ponder_steps', '?')}"


def plot_curve(runs, key, title, ylabel, fname):
    plt.figure(figsize=(7, 4.5))
    for run in runs:
        s = run["history"].get(key)
        if not s:
            continue
        plt.plot(s["step"], s["value"], label=label(run),
                 color=COLORS.get(run["name"]), linewidth=1.4)
    plt.xlabel("training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="loops")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print("wrote", path)


def plot_final_vs_n(runs, fname):
    ns = [run["config_key"]["ponder_steps"] for run in runs]
    loss = [run["summary"]["eval/loss"] for run in runs]
    ppl = [run["summary"]["final/perplexity"] for run in runs]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ns, loss, "o-", color="#1f77b4", linewidth=1.8, label="eval/loss")
    ax1.set_xlabel("수직 반복 횟수 n (ponder_steps)")
    ax1.set_ylabel("최종 eval/loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(ns)
    for x, y in zip(ns, loss):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(ns, ppl, "s--", color="#d62728", linewidth=1.2, label="perplexity")
    ax2.set_ylabel("perplexity", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("n-Loop MoE: 반복 횟수에 따른 최종 손실")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print("wrote", path)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    runs = load_runs()
    plot_curve(runs, "train/loss", "n-Loop MoE: 학습 손실 곡선",
               "train/loss", "nloop_train_loss.png")
    plot_curve(runs, "eval/loss", "n-Loop MoE: 검증 손실 곡선",
               "eval/loss", "nloop_eval_loss.png")
    plot_final_vs_n(runs, "nloop_final_vs_n.png")


if __name__ == "__main__":
    main()
