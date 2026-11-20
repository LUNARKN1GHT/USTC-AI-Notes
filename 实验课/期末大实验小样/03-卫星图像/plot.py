"""Render the EuroSAT region comparison from the pilot's saved results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS = {
    "logistic_color": "Color + logistic",
    "logistic_color_texture": "Color + texture + logistic",
    "cnn": "Small CNN",
}


def main():
    result = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    labels = np.arange(len(MODELS))
    fig, axis = plt.subplots(figsize=(7.3, 3.9), layout="constrained")
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.25)
    for offset, key, legend, color in (
        (-0.18, "in_region_test", "Within training region", "#367998"),
        (0.18, "spatial_test", "Held-out region", "#b05b38"),
    ):
        scores = [result["models"][model][key]["macro_f1"] for model in MODELS]
        bars = axis.bar(labels + offset, scores, width=0.34, label=legend, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    axis.set_title("EuroSAT pilot: geographic generalization", loc="left", pad=14, weight="bold")
    axis.set_ylabel("Macro-F1")
    axis.set_ylim(0, 0.72)
    axis.set_xticks(labels, MODELS.values())
    axis.legend(loc="upper right", frameon=False, ncols=2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(ROOT / "region_comparison.png", dpi=160)


if __name__ == "__main__":
    main()
