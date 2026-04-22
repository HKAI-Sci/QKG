"""
Generate the Qwen-as-validator comparison figure from CSV.
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "qwen_validator_results.csv")

COLOR_BASE = "#9ecae1"
COLOR_QKG = "#238b45"
COLOR_NOPC = "#fd8d3c"
COLOR_IMP = "#41ab5d"
COLOR_REG = "#d73027"

ORDER = ["No validator", "KG w/o context", "QKG w/ context"]
COLOR_BY_SETTING = {
    "No validator": COLOR_BASE,
    "KG w/o context": COLOR_NOPC,
    "QKG w/ context": COLOR_QKG,
}


def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rank = {name: i for i, name in enumerate(ORDER)}
    rows.sort(key=lambda r: rank[r["setting"]])
    return rows


def main() -> None:
    rows = load_rows(CSV_PATH)

    labels = [r["setting"] for r in rows]
    final_acc = [float(r["final_acc"]) for r in rows]
    improved = [int(r["improved"]) for r in rows]
    regressed = [int(r["regressed"]) for r in rows]
    deltas = [float(r["delta_pp"]) for r in rows]
    colors = [COLOR_BY_SETTING[r["setting"]] for r in rows]

    x = np.arange(len(rows))
    bar_w = 0.56

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.6))
    fig.subplots_adjust(wspace=0.38)

    bars = ax1.bar(x, final_acc, bar_w, color=colors, edgecolor="white", linewidth=0.8)
    for bar, acc, delta in zip(bars, final_acc, deltas):
        xc = bar.get_x() + bar.get_width() / 2
        ax1.text(xc, acc + 0.18, f"{acc:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if delta > 0:
            ax1.text(xc, acc + 0.88, f"(+{delta:.1f} pp)", ha="center", va="bottom",
                     fontsize=8.2, color="#1f2937", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Final accuracy (%)", fontsize=10)
    ax1.set_title("(a) Final Accuracy", fontsize=11, fontweight="bold")
    ax1.set_ylim(74, 85.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    y = np.arange(len(rows))
    rev_h = bar_w / 2
    ax2.barh(y - bar_w / 4, improved, rev_h, color=COLOR_IMP, edgecolor="white",
             linewidth=0.7, label="Improved (W→C)")
    ax2.barh(y + bar_w / 4, regressed, rev_h, color=COLOR_REG, edgecolor="white",
             linewidth=0.7, label="Regressed (C→W)")

    for yi, (imp, reg) in enumerate(zip(improved, regressed)):
        if imp == 0 and reg == 0:
            ax2.text(2.5, yi, "no revisions", ha="left", va="center", fontsize=8.2, color="#4b5563")
            continue
        ax2.text(imp + 3, yi - bar_w / 4, str(imp), ha="left", va="center", fontsize=8.5)
        ax2.text(reg + 3, yi + bar_w / 4, str(reg), ha="left", va="center", fontsize=8.5)
        net = imp - reg
        ax2.text(max(imp * 0.78, 12), yi - rev_h / 2 + 0.18, f"net +{net}",
                 ha="left", va="top", fontsize=8.3, fontweight="bold", color="#1f2937")

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Number of answers revised", fontsize=10)
    ax2.set_title("(b) Validator Revisions", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.88)

    legend_handles = [
        mpatches.Patch(color=COLOR_BASE, label="No validator"),
        mpatches.Patch(color=COLOR_NOPC, label="KG without context"),
        mpatches.Patch(color=COLOR_QKG, label="QKG with context"),
    ]
    ax1.legend(handles=legend_handles, fontsize=7.8, loc="upper left", framealpha=0.88)

    plt.suptitle("Qwen-Validator Results and Context Ablation", fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        out_path = os.path.join(OUT_DIR, f"qwen_validator_results_plot.{ext}")
        plt.savefig(out_path, bbox_inches="tight", dpi=160)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
