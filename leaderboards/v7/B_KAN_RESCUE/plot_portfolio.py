"""Generate portfolio-ready KAN rescue figures.

This is append-only: the original plot_results.py remains as the historical
early B-campaign chart. These figures include the later Encoder-KAN sweep.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent


SUMMARY_MODELS = [
    {"name": "raw KAN h16", "params": 4480, "val_loss": 0.5877,
     "greedy": 84.5, "beam5": 98.5, "beam5_overhead": 0.024,
     "color": "#8C6D31"},
    {"name": "KAN-AE h16", "params": 5840, "val_loss": 0.5924,
     "greedy": 85.9, "beam5": 98.6, "beam5_overhead": 0.021,
     "color": "#2F7E79"},
    {"name": "Encoder-KAN b16 h16", "params": 5224, "val_loss": 0.4763,
     "greedy": 89.5, "beam5": 98.8, "beam5_overhead": 0.018,
     "color": "#C4512D"},
    {"name": "Encoder-KAN b24 h16", "params": 6704, "val_loss": 0.4338,
     "greedy": 91.4, "beam5": 99.1, "beam5_overhead": 0.011,
     "color": "#D65F00"},
    {"name": "Encoder-KAN b28 h16", "params": 7444, "val_loss": 0.4464,
     "greedy": 92.2, "beam5": 99.2, "beam5_overhead": 0.010,
     "color": "#B22222"},
    {"name": "Encoder-KAN b24 h32", "params": 10704, "val_loss": 0.4697,
     "greedy": 90.8, "beam5": 99.3, "beam5_overhead": 0.009,
     "color": "#E69F00"},
    {"name": "micro-MLP h8", "params": 2793, "val_loss": 0.4967,
     "greedy": 91.1, "beam5": 99.1, "beam5_overhead": 0.013,
     "color": "#3B5BA9"},
    {"name": "micro-MLP h16", "params": 10065, "val_loss": 0.3326,
     "greedy": 93.7, "beam5": 99.5, "beam5_overhead": 0.007,
     "color": "#6B4FA3"},
    {"name": "MLP-128", "params": 474753, "val_loss": 0.4996,
     "greedy": 89.4, "beam5": 98.5, "beam5_overhead": 0.024,
     "color": "#666666"},
]

SWEEP_MODELS = [
    {"name": "b16 h16", "params": 5224, "val_loss": 0.4763,
     "greedy": 89.5, "beam5": 98.8, "beam5_overhead": 0.018},
    {"name": "b16 h32", "params": 7944, "val_loss": 0.4599,
     "greedy": 90.9, "beam5": 99.2, "beam5_overhead": 0.011},
    {"name": "b20 h24", "params": 7644, "val_loss": 0.4387,
     "greedy": 88.3, "beam5": 98.8, "beam5_overhead": 0.016},
    {"name": "b24 h12", "params": 5704, "val_loss": 0.4570,
     "greedy": 91.9, "beam5": 99.2, "beam5_overhead": 0.013},
    {"name": "b24 h16", "params": 6704, "val_loss": 0.4338,
     "greedy": 91.4, "beam5": 99.1, "beam5_overhead": 0.011},
    {"name": "b24 h24", "params": 8704, "val_loss": 0.4464,
     "greedy": 90.3, "beam5": 99.1, "beam5_overhead": 0.012},
    {"name": "b24 h32", "params": 10704, "val_loss": 0.4697,
     "greedy": 90.8, "beam5": 99.3, "beam5_overhead": 0.009},
    {"name": "b28 h16", "params": 7444, "val_loss": 0.4464,
     "greedy": 92.2, "beam5": 99.2, "beam5_overhead": 0.010},
    {"name": "b32 h32", "params": 13464, "val_loss": 0.4464,
     "greedy": 90.5, "beam5": 99.1, "beam5_overhead": 0.011},
]


def annotate_bars(ax, values, fmt="{:.1f}", dy=0.01):
    ymax = max(values)
    ymin = min(values)
    span = max(ymax - ymin, ymax * 0.05)
    for idx, value in enumerate(values):
        ax.text(
            idx,
            value + span * dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def polish(ax):
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_summary():
    models = SUMMARY_MODELS
    names = [m["name"] for m in models]
    colors = [m["color"] for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(19, 10.5))
    fig.suptitle(
        "ISRE v7 KAN Rescue: representation fixes turn raw KAN into a competitive policy",
        fontsize=17,
        fontweight="bold",
    )

    ax = axes[0, 0]
    vals = [m["greedy"] for m in models]
    ax.bar(names, vals, color=colors)
    ax.axhline(91.1, color="#3B5BA9", ls="--", lw=1.1, alpha=0.75)
    ax.text(len(names) - 2.4, 91.25, "micro-MLP h8 = 91.1%", fontsize=9)
    ax.set_title("Greedy rollout, no search")
    ax.set_ylabel("% BFS-optimal")
    ax.set_ylim(83, 95.5)
    ax.tick_params(axis="x", labelrotation=27)
    annotate_bars(ax, vals, "{:.1f}", dy=0.03)
    polish(ax)

    ax = axes[0, 1]
    vals = [m["beam5"] for m in models]
    ax.bar(names, vals, color=colors)
    ax.set_title("Beam-5 rollout")
    ax.set_ylabel("% BFS-optimal")
    ax.set_ylim(98.2, 99.65)
    ax.tick_params(axis="x", labelrotation=27)
    annotate_bars(ax, vals, "{:.1f}", dy=0.025)
    polish(ax)

    ax = axes[1, 0]
    vals = [m["val_loss"] for m in models]
    ax.bar(names, vals, color=colors)
    ax.set_title("Best validation loss")
    ax.set_ylabel("cross-entropy loss")
    ax.tick_params(axis="x", labelrotation=27)
    annotate_bars(ax, vals, "{:.4f}", dy=0.03)
    polish(ax)

    ax = axes[1, 1]
    params = [m["params"] for m in models]
    greedy = [m["greedy"] for m in models]
    ax.scatter(params, greedy, s=145, c=colors, edgecolor="black", linewidth=0.8)
    for m in models:
        ax.annotate(m["name"], (m["params"], m["greedy"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_title("Parameter efficiency")
    ax.set_xlabel("parameters, log scale")
    ax.set_ylabel("greedy BFS-optimal %")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / "kan_rescue_portfolio_summary.png"
    fig.savefig(out, dpi=180)
    print(out)


def plot_sweep():
    models = SWEEP_MODELS
    names = [m["name"] for m in models]
    colors = [
        "#B22222" if m["name"] == "b28 h16"
        else "#E69F00" if m["name"] == "b24 h32"
        else "#6C8EBF"
        for m in models
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    fig.suptitle(
        "Encoder-KAN local sweep: rollout optimum is architecture-sensitive",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0]
    vals = [m["greedy"] for m in models]
    ax.bar(names, vals, color=colors)
    ax.set_title("Greedy BFS-optimal")
    ax.set_ylabel("%")
    ax.set_ylim(87.5, 92.8)
    ax.tick_params(axis="x", labelrotation=35)
    annotate_bars(ax, vals, "{:.1f}", dy=0.03)
    polish(ax)

    ax = axes[1]
    vals = [m["beam5"] for m in models]
    ax.bar(names, vals, color=colors)
    ax.set_title("Beam-5 BFS-optimal")
    ax.set_ylabel("%")
    ax.set_ylim(98.6, 99.45)
    ax.tick_params(axis="x", labelrotation=35)
    annotate_bars(ax, vals, "{:.1f}", dy=0.03)
    polish(ax)

    ax = axes[2]
    vals = [m["val_loss"] for m in models]
    ax.bar(names, vals, color=colors)
    ax.set_title("Best validation loss")
    ax.set_ylabel("loss")
    ax.tick_params(axis="x", labelrotation=35)
    annotate_bars(ax, vals, "{:.4f}", dy=0.03)
    polish(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = OUT_DIR / "kan_rescue_sweep_map.png"
    fig.savefig(out, dpi=180)
    print(out)


def main():
    plot_summary()
    plot_sweep()


if __name__ == "__main__":
    main()
