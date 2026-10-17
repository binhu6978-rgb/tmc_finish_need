#!/usr/bin/env python
"""Plot client-population robustness using only recorded communication rounds."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["Decoupled", "HeteroFL", "ScaleFL", "AdaptiveFL", "FedLAGC", "LiquidFL"]
COLORS = {
    "Decoupled": "#D55E00",
    "HeteroFL": "#0072B2",
    "ScaleFL": "#CC79A7",
    "AdaptiveFL": "#E69F00",
    "FedLAGC": "#8C6D8A",
    "LiquidFL": "#009E73",
}


def moving_average(values: np.ndarray, window: int = 15) -> np.ndarray:
    if values.size < 2 or window <= 1:
        return values.copy()
    window = min(window, values.size)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def load_curves(path: Path) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            values = []
            for key, value in row.items():
                if key.startswith("round_") and value not in (None, ""):
                    values.append(float(value))
            curves[method] = np.asarray(values, dtype=float)
    return curves


def real_round_limits(report_path: Path) -> dict[tuple[int, float, str], int]:
    limits: dict[tuple[int, float, str], int] = {}
    with report_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            note = row.get("note", "") or ""
            match = re.search(r"first\s+(\d+)\s+real rounds", note)
            if match:
                limits[(int(row["N"]), float(row["fraction"]), row["method"])] = int(match.group(1))
    return limits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    limits = real_round_limits(args.data_dir / "extraction_report.csv")
    populations = [50, 100, 200, 500]
    titles = ["(a) 50 Clients", "(b) 100 Clients", "(c) 200 Clients", "(d) 500 Clients"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True, sharey=True)

    for ax, population, title in zip(axes.flat, populations, titles):
        curves = load_curves(args.data_dir / f"summary_num_users_{population}.csv")
        for method in METHODS:
            values = curves[method]
            keep = limits.get((population, 0.1, method), values.size)
            values = values[:keep]
            rounds = np.arange(values.size)
            smooth = moving_average(values, window=15)
            ax.plot(rounds, values, color=COLORS[method], lw=0.55, alpha=0.16)
            ax.plot(
                rounds,
                smooth,
                color=COLORS[method],
                lw=2.25 if method == "LiquidFL" else 1.7,
                label=method,
                zorder=4 if method == "LiquidFL" else 3,
            )
        ax.set_title(title, fontsize=16, pad=7)
        ax.grid(True, linestyle="--", linewidth=0.75, alpha=0.52)
        ax.set_xlim(-5, 1000)
        ax.set_ylim(8, 82)
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
        ax.tick_params(width=1.05, length=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Full-model Accuracy (%)", fontsize=14, fontweight="bold")
    for ax in axes[1, :]:
        ax.set_xlabel("Communication Round", fontsize=14, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=11.5,
        bbox_to_anchor=(0.5, -0.005),
        handlelength=2.5,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.96, bottom=0.11, wspace=0.12, hspace=0.26)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=350)
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight", dpi=350)
    plt.close(fig)


if __name__ == "__main__":
    main()
