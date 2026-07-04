"""nLoopMoE sweep graphs, grouped by vertical loop count n (ponder_steps)."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DUMP = "wandb_data_nloop_all/wandb_dump.json"
OUTDIR = "assets"
N_ORDER = [1, 2, 4, 6, 8]
ALPHA_COLORS = {0.005: "#2ca02c", 0.01: "#1f77b4", 0.02: "#ff7f0e", 0.05: "#d62728"}


def finished_sweeps():
    d = json.load(open(DUMP, encoding="utf-8"))
    runs = d["projects"][0]["runs"]
    return [r for r in runs if "sweep" in r["name"]
            and r["state"] == "finished"
            and r["summary"].get("eval/loss") is not None]


def best_by_n(runs):
    by_n = defaultdict(list)
    for r in runs:
        by_n[r["config"].get("ponder_steps")].append(r)
    return {n: min(by_n[n], key=lambda r: r["summary"]["eval/loss"])
            for n in N_ORDER if n in by_n}


def plot_best_by_n(best, fname):
    ns = [n for n in N_ORDER if n in best]
    loss = [best[n]["summary"]["eval/loss"] for n in ns]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ns, loss, "o-", color="#2ca02c", linewidth=1.8)
    for x, y in zip(ns, loss):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)
    plt.xticks(ns)
    plt.xlabel("수직 반복 횟수 n (ponder_steps)")
    plt.ylabel("최적 eval/loss (n별 sweep 최저)")
    plt.title("nLoopMoE Sweep: n별 최적 검증 손실")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print("wrote", path)


def plot_all_by_n(runs, fname):
    plt.figure(figsize=(7.5, 4.5))
    seen = set()
    for r in runs:
        n = r["config"].get("ponder_steps")
        a = r["config"].get("alpha")
        if n not in N_ORDER:
            continue
        x = N_ORDER.index(n) + (hash(r["name"]) % 100 / 100 - 0.5) * 0.5
        lbl = f"alpha={a}" if a not in seen else None
        seen.add(a)
        plt.scatter(x, r["summary"]["eval/loss"], s=35,
                    color=ALPHA_COLORS.get(a, "#777"), edgecolor="white",
                    zorder=3, label=lbl, alpha=0.85)
    plt.xticks(range(len(N_ORDER)), [f"n={n}" for n in N_ORDER])
    plt.xlabel("수직 반복 횟수 n (ponder_steps)")
    plt.ylabel("eval/loss")
    plt.title("nLoopMoE Sweep: 전체 run (n별 그룹, alpha 색상)")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    plt.legend([handles[i] for i in order], [labels[i] for i in order],
               title="alpha", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print("wrote", path)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    runs = finished_sweeps()
    best = best_by_n(runs)
    plot_best_by_n(best, "nloop_sweep_best_by_n.png")
    plot_all_by_n(runs, "nloop_sweep_all_by_n.png")

    # emit HTML table rows for best-per-n
    rows = []
    for n in N_ORDER:
        if n not in best:
            continue
        r = best[n]; c = r["config"]; s = r["summary"]
        rows.append(
            f'    <tr><td>{n}</td><td>{r["name"]}</td>'
            f'<td>{c.get("alpha")}</td><td>{c.get("lr"):.4g}</td>'
            f'<td>{c.get("muon_lr"):.4g}</td>'
            f'<td>{s["eval/loss"]:.4f}</td>'
            f'<td>{s.get("final/perplexity"):.1f}</td></tr>')
    print("\n".join(rows))


if __name__ == "__main__":
    main()
