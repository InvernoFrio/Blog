"""Render the recorded CPU experiments; never invent or interpolate metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / "public" / "learning" / "llm-from-scratch"
results = json.loads((root / "experiment_results.json").read_text(encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
fig.patch.set_facecolor("#fbfaf8")
for ax, key, title in zip(axes, ("pretraining", "sft"), ("Pretraining from random weights", "SFT on a tiny answer dataset")):
    report = results[key]
    steps = [0] + [record["step"] for record in report["history"]]
    values = [report["initial_validation"]["loss"]] + [record["validation"]["loss"] for record in report["history"]]
    ax.set_facecolor("#fbfaf8")
    ax.plot(steps, values, "o-", color="#426f87", linewidth=2, markersize=4, label="Fixed validation mean")
    ax.plot(steps[1:], [record["train_batch_loss"] for record in report["history"]], "s--",
            color="#ab6944", linewidth=1.5, markersize=4, label="Current train microbatch")
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross entropy (nats / supervised token)")
    ax.grid(axis="y", color="#deded9", linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#a4a4a0")
    ax.spines["bottom"].set_color("#a4a4a0")
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("CPU learning lab | 120,768 parameters | seed 17 | byte tokens", fontsize=12)
fig.savefig(root / "validation-curves.svg", facecolor=fig.get_facecolor(), metadata={"Date": None})
fig.savefig(root / "validation-curves.png", dpi=170, facecolor=fig.get_facecolor())
plt.close(fig)
print("Rendered measured validation/training curves for both stages.")
