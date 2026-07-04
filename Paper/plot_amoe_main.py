"""2D-AMoE main run train/eval loss curve from AMOE-SWEEP123 dump."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

d = json.load(open("wandb_data_sweep/wandb_dump.json", encoding="utf-8"))
main = next(r for r in d["projects"][0]["runs"] if r["name"] == "main")
h = main["history"]

plt.figure(figsize=(7, 4.5))
tr = h["train/loss"]
plt.plot(tr["step"], tr["value"], color="#1f77b4", linewidth=1.2, label="train/loss")
if "eval/loss" in h:
    ev = h["eval/loss"]
    plt.plot(ev["step"], ev["value"], color="#d62728", linewidth=1.6, label="eval/loss")
plt.xlabel("training step")
plt.ylabel("loss")
plt.title("2D-AMoE main: 학습 손실 곡선")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs("assets", exist_ok=True)
path = "assets/amoe_main_train_loss.png"
plt.savefig(path, dpi=150)
plt.close()
print("wrote", path)
