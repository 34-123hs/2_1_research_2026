"""Read-only W&B fetcher.

Pulls config + summary + (downsampled) history for every run under an entity
(optionally limited to one project) and writes an AI/human-friendly dump.

API key is read from the WANDB_API_KEY environment variable. It is never
written to disk. This script only READS from W&B.

Usage:
    set WANDB_API_KEY=...        # PowerShell: $env:WANDB_API_KEY="..."
    python fetch_wandb.py [--entity ENTITY] [--project PROJECT] [--outdir DIR]
"""
import argparse
import json
import math
import os
import sys

import wandb

# Metrics worth pulling as time series (only those present are used).
HISTORY_KEYS = [
    "train/loss",
    "eval/loss",
    "train/aux/balance_loss",
    "train/aux/base_loss",
    "train/router/max_pct_global",
    "train/router/entropy_norm_global",
    "train/grad_norm/global_l2",
    "train/learning_rate",
]

# Scalar summary metrics we care about for the paper.
SUMMARY_KEYS = [
    "eval/loss",
    "final/eval_loss",
    "final/perplexity",
    "n_params_M",
    "total_flos",
    "train_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    # present on eval-compute runs (kept if available):
    "bench/flops_per_token",
    "bench/tokens_per_s",
    "bench/avg_ponder_step",
]

# Hyperparameters that distinguish the runs / matter for the paper.
CONFIG_KEYS = [
    "ponder_steps", "experts", "depth", "dim", "mlp_dim", "heads",
    "lr", "muon_lr", "alpha", "ponder_beta", "lambda_p",
    "batch_size", "block_size", "max_steps",
]

MAX_HISTORY_POINTS = 500  # downsample longer series to keep the dump small


def clean(value):
    """Make a value JSON-serializable and drop NaN/inf."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)  # fall back to repr for odd types


def downsample(rows, n=MAX_HISTORY_POINTS):
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def collect_history(run):
    """Return downsampled history series, scanning each metric separately.

    scan_history(keys=[...]) only returns rows where *all* keys are non-null,
    so eval-vs-train metrics (logged at different cadences) must be pulled one
    key at a time against _step.
    """
    series = {}
    for k in HISTORY_KEYS:
        pts = []
        for row in run.scan_history(keys=["_step", k]):
            v = clean(row.get(k))
            if v is not None:
                pts.append((row.get("_step"), v))
        if pts:
            pts = downsample(pts)
            series[k] = {"step": [s for s, _ in pts], "value": [v for _, v in pts]}
    return series


def collect_run(run):
    summary = {k: clean(run.summary.get(k)) for k in SUMMARY_KEYS
               if k in run.summary.keys()}
    config_full = clean(dict(run.config))
    config_key = {k: config_full.get(k) for k in CONFIG_KEYS if k in config_full}
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "url": run.url,
        "config_key": config_key,
        "config": config_full,
        "summary": summary,
        "history": collect_history(run),
    }


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_markdown(data, path):
    lines = ["# W&B 데이터 요약", "",
             f"엔티티: `{data['entity']}`  ·  프로젝트 {len(data['projects'])}개", ""]
    for proj in data["projects"]:
        lines.append(f"## 프로젝트: `{proj['project']}`  ({len(proj['runs'])} runs)")
        lines.append("")
        if not proj["runs"]:
            lines.append("_run 없음_\n")
            continue
        # union of summary keys actually present
        keys = []
        for r in proj["runs"]:
            for k in r["summary"]:
                if k not in keys:
                    keys.append(k)
        header = ["run", "state"] + keys
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in proj["runs"]:
            row = [r["name"] or r["id"], r["state"]] + [fmt(r["summary"].get(k)) for k in keys]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        # key hyperparameters
        ckeys = []
        for r in proj["runs"]:
            for k in r["config_key"]:
                if k not in ckeys:
                    ckeys.append(k)
        if ckeys:
            lines.append("**주요 하이퍼파라미터**")
            lines.append("")
            chead = ["run"] + ckeys
            lines.append("| " + " | ".join(chead) + " |")
            lines.append("|" + "|".join(["---"] * len(chead)) + "|")
            for r in proj["runs"]:
                row = [r["name"] or r["id"]] + [fmt(r["config_key"].get(k)) for k in ckeys]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        # note which runs have history series
        for r in proj["runs"]:
            if r["history"]:
                series = ", ".join(f"{k}({len(v['step'])}pts)" for k, v in r["history"].items())
                lines.append(f"- `{r['name'] or r['id']}` history: {series}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="choijiwan1229-hansung-science-high-school")
    ap.add_argument("--project", default=None, help="limit to one project")
    ap.add_argument("--runs", default=None,
                    help="comma-separated run names to keep (exact match)")
    ap.add_argument("--outdir", default="wandb_data")
    args = ap.parse_args()
    wanted = [s.strip() for s in args.runs.split(",")] if args.runs else None

    if not os.environ.get("WANDB_API_KEY"):
        sys.exit("WANDB_API_KEY 환경변수가 설정되지 않았습니다. 키를 환경변수로 주입한 뒤 실행하세요.")

    api = wandb.Api(timeout=60)

    if args.project:
        project_names = [args.project]
    else:
        project_names = [p.name for p in api.projects(entity=args.entity)]

    data = {"entity": args.entity, "projects": []}
    for pname in project_names:
        print(f"[project] {pname}", flush=True)
        runs = list(api.runs(f"{args.entity}/{pname}"))
        if wanted is not None:
            found = {r.name for r in runs}
            missing = [w for w in wanted if w not in found]
            if missing:
                print(f"  [warn] 요청한 run을 못 찾음: {missing}", flush=True)
                print(f"  [info] 사용 가능한 run: {sorted(found)}", flush=True)
            runs = [r for r in runs if r.name in wanted]
        proj = {"project": pname, "runs": []}
        for run in runs:
            print(f"  - run {run.name} ({run.id}) state={run.state}", flush=True)
            proj["runs"].append(collect_run(run))
        data["projects"].append(proj)

    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, "wandb_dump.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    import pathlib
    write_markdown(data, pathlib.Path(args.outdir) / "wandb_summary.md")

    n_runs = sum(len(p["runs"]) for p in data["projects"])
    print(f"\n완료: {len(data['projects'])} projects, {n_runs} runs")
    print(f"  {json_path}")
    print(f"  {os.path.join(args.outdir, 'wandb_summary.md')}")


if __name__ == "__main__":
    main()
