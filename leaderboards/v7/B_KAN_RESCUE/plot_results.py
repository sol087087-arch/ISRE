"""Plot append-only KAN rescue comparison charts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent


MODELS = [
    {
        "name": "raw KAN h16",
        "params": 4480,
        "val_loss": 0.5877,
        "greedy_opt": 84.5,
        "beam5_opt": 98.5,
        "beam5_overhead": 0.024,
        "color": "#8C6D31",
    },
    {
        "name": "KAN-AE h16",
        "params": 5840,
        "val_loss": 0.5924,
        "greedy_opt": 85.9,
        "beam5_opt": 98.6,
        "beam5_overhead": 0.021,
        "color": "#2F7E79",
    },
    {
        "name": "Encoder-KAN",
        "params": 5224,
        "val_loss": 0.4763,
        "greedy_opt": 89.5,
        "beam5_opt": 98.8,
        "beam5_overhead": 0.018,
        "color": "#C4512D",
    },
    {
        "name": "micro-MLP h8",
        "params": 2793,
        "val_loss": 0.4967,
        "greedy_opt": 91.1,
        "beam5_opt": 99.1,
        "beam5_overhead": 0.013,
        "color": "#3B5BA9",
    },
    {
        "name": "micro-MLP h16",
        "params": 10065,
        "val_loss": 0.3326,
        "greedy_opt": 93.7,
        "beam5_opt": 99.5,
        "beam5_overhead": 0.007,
        "color": "#6B4FA3",
    },
]


def annotate_bars(ax, values, fmt="{:.1f}", dy=0.01):
    ymax = max(values)
    for idx, value in enumerate(values):
        ax.text(
            idx,
            value + ymax * dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    names = [m["name"] for m in MODELS]
    colors = [m["color"] for m in MODELS]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        "KAN Rescue B-Campaign: representation fixes help, micro-MLP still leads rollout",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    vals = [m["beam5_opt"] for m in MODELS]
    ax.bar(names, vals, color=colors)
    ax.set_title("Beam-5 BFS-optimal rate")
    ax.set_ylabel("% optimal rollouts")
    ax.set_ylim(98.0, 99.7)
    ax.tick_params(axis="x", labelrotation=25)
    annotate_bars(ax, vals, "{:.1f}", dy=0.002)

    ax = axes[0, 1]
    vals = [m["greedy_opt"] for m in MODELS]
    ax.bar(names, vals, color=colors)
    ax.set_title("Greedy BFS-optimal rate")
    ax.set_ylabel("% optimal rollouts")
    ax.set_ylim(80, 96)
    ax.tick_params(axis="x", labelrotation=25)
    annotate_bars(ax, vals, "{:.1f}", dy=0.004)

    ax = axes[1, 0]
    vals = [m["val_loss"] for m in MODELS]
    ax.bar(names, vals, color=colors)
    ax.set_title("Best validation loss")
    ax.set_ylabel("cross-entropy loss, lower is better")
    ax.tick_params(axis="x", labelrotation=25)
    annotate_bars(ax, vals, "{:.4f}", dy=0.015)

    ax = axes[1, 1]
    vals = [m["beam5_overhead"] for m in MODELS]
    ax.bar(names, vals, color=colors)
    ax.set_title("Beam-5 mean overhead vs BFS")
    ax.set_ylabel("extra steps, lower is better")
    ax.tick_params(axis="x", labelrotation=25)
    annotate_bars(ax, vals, "{:.3f}", dy=0.02)

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / "kan_rescue_comparison.png"
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
