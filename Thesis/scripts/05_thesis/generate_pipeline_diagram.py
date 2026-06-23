"""Generate the spatial-prior routing pipeline diagram for Chapter 3."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# ── Layout constants ──────────────────────────────────────────────
FIG_W, FIG_H = 10, 14
BOX_W = 7.0
BOX_H_STD = 1.35
BOX_H_TALL = 1.7
BOX_H_ENSEMBLE = 1.9
CORNER_R = 0.15
X_CENTER = FIG_W / 2
ARROW_SHRINK = 4

# colours
C_INPUT   = "#E8F4FD"
C_ENSEMBLE = "#D4E8F7"
C_PRIOR   = "#FFF3CD"
C_ROUTE   = "#D5F5E3"
C_GUARD   = "#FADBD8"
C_OUTPUT  = "#E8DAEF"
C_EDGE    = "#2C3E50"
C_SUBBOX  = "#FFFFFF"
C_TEXT    = "#1A1A1A"

FONT_TITLE = {"fontsize": 11, "fontweight": "bold", "color": C_TEXT, "fontfamily": "serif"}
FONT_BODY  = {"fontsize": 8.5, "color": C_TEXT, "fontfamily": "serif"}
FONT_SUB   = {"fontsize": 8,   "color": "#444444", "fontfamily": "serif"}
FONT_LABEL = {"fontsize": 9.5, "fontweight": "bold", "color": C_TEXT, "fontfamily": "serif"}


def add_box(ax, y_center, height, colour, title, body_lines, *, sub_boxes=None):
    x_left = X_CENTER - BOX_W / 2
    y_bottom = y_center - height / 2
    box = FancyBboxPatch(
        (x_left, y_bottom), BOX_W, height,
        boxstyle=f"round,pad={CORNER_R}",
        facecolor=colour, edgecolor=C_EDGE, linewidth=1.3,
    )
    ax.add_patch(box)
    # title
    title_y = y_bottom + height - 0.30
    ax.text(X_CENTER, title_y, title, ha="center", va="center", **FONT_TITLE)
    # body lines
    for i, line in enumerate(body_lines):
        ax.text(X_CENTER, title_y - 0.28 - i * 0.24, line,
                ha="center", va="center", **FONT_BODY)
    # optional sub-boxes (for the 4 streams)
    if sub_boxes:
        n = len(sub_boxes)
        sub_w = 1.25
        gap = 0.25
        total = n * sub_w + (n - 1) * gap
        sx = X_CENTER - total / 2
        sy = y_bottom + 0.22
        sub_h = 0.42
        for j, label in enumerate(sub_boxes):
            rx = sx + j * (sub_w + gap)
            sub = FancyBboxPatch(
                (rx, sy), sub_w, sub_h,
                boxstyle=f"round,pad=0.06",
                facecolor=C_SUBBOX, edgecolor=C_EDGE, linewidth=0.8,
            )
            ax.add_patch(sub)
            ax.text(rx + sub_w / 2, sy + sub_h / 2, label,
                    ha="center", va="center", **FONT_SUB)
    return y_center


def add_arrow(ax, y_from, y_to, *, label=None):
    ax.annotate(
        "", xy=(X_CENTER, y_to + 0.05), xytext=(X_CENTER, y_from - 0.05),
        arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=1.5,
                        shrinkA=ARROW_SHRINK, shrinkB=ARROW_SHRINK),
    )
    if label:
        mid_y = (y_from + y_to) / 2
        ax.text(X_CENTER + 0.15, mid_y, label, ha="left", va="center", **FONT_SUB)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # ── Stage positions (top to bottom) ──
    y_input    = 12.8
    y_ensemble = 11.0
    y_prior    = 8.9
    y_route    = 6.7
    y_guard    = 4.6
    y_output   = 2.8

    # ── 1. Input ──
    add_box(ax, y_input, BOX_H_STD, C_INPUT,
            "Input Features (66 variables)",
            ["Satellite AOD  |  TROPOMI trace gases  |  ERA5 meteorology",
             "RFSI spatial neighbours  |  Land-use & urban  |  Temporal encodings"])

    add_arrow(ax, y_input - BOX_H_STD / 2, y_ensemble + BOX_H_ENSEMBLE / 2)

    # ── 2. XGBoost Ensemble ──
    add_box(ax, y_ensemble, BOX_H_ENSEMBLE, C_ENSEMBLE,
            "XGBoost-DART Ensemble (4 target-free streams)",
            ["Trained without oracle tier labels — each stream uses a different",
             "target transform or loss to capture complementary signal"],
            sub_boxes=["Log-target", "Raw-target", "Blend", "Gated"])

    add_arrow(ax, y_ensemble - BOX_H_ENSEMBLE / 2, y_prior + BOX_H_TALL / 2)

    # ── 3. Spatial Prior ──
    add_box(ax, y_prior, BOX_H_TALL, C_PRIOR,
            "Spatial Prior Computation",
            ["prior = Σ wᵢ · ȳᵢ / Σ wᵢ      (distance-weighted neighbour mean)",
             "wᵢ = exp(−dᵢ² / s²),  s ≈ 60 km,  top-k neighbours",
             "LOO trust calibration downweights unreliable anchors"])

    add_arrow(ax, y_prior - BOX_H_TALL / 2, y_route + BOX_H_TALL / 2)

    # ── 4. Three-Regime Routing ──
    add_box(ax, y_route, BOX_H_TALL, C_ROUTE,
            "Three-Regime Routing",
            ["prior ≥ 35 µg/m³ :  Gated stream  (high-pollution regime)",
             "prior < 12 µg/m³ :  Log stream + shift-to-prior  (clean regime)",
             "12–35 µg/m³ :  Blend stream + shift-to-prior  (moderate regime)"])

    add_arrow(ax, y_route - BOX_H_TALL / 2, y_guard + BOX_H_TALL / 2)

    # ── 5. Reliability & Safety Guards ──
    add_box(ax, y_guard, BOX_H_TALL, C_GUARD,
            "Reliability Guard + MODIS Seasonal Correction",
            ["Hidden-high detection: lift predictions where prior is high",
             "Non-high cap: suppress false high spikes in clean areas",
             "MODIS seasonal AOD correction via compact HistGradientBoosting"])

    add_arrow(ax, y_guard - BOX_H_TALL / 2, y_output + BOX_H_STD / 2)

    # ── 6. Output ──
    add_box(ax, y_output, BOX_H_STD, C_OUTPUT,
            "Hourly PM2.5 Prediction",
            ["Deployable spatial estimate at unmonitored locations",
             "Per-station mean R² ≈ 0.20  (vs. oracle ceiling 0.20)"])

    # ── Side label: "Monitoring Network" bracket ──
    bx = X_CENTER + BOX_W / 2 + 0.35
    by_top = y_prior + BOX_H_TALL / 2
    by_bot = y_guard - BOX_H_TALL / 2
    ax.annotate("", xy=(bx + 0.15, by_top), xytext=(bx + 0.15, by_bot),
                arrowprops=dict(arrowstyle="-", color="#888888", lw=1.0))
    ax.plot([bx + 0.05, bx + 0.15], [by_top, by_top], color="#888888", lw=1.0)
    ax.plot([bx + 0.05, bx + 0.15], [by_bot, by_bot], color="#888888", lw=1.0)
    ax.text(bx + 0.35, (by_top + by_bot) / 2, "Requires\nmonitoring\nnetwork\nanchor data",
            ha="left", va="center", fontsize=7.5, color="#666666",
            fontfamily="serif", fontstyle="italic")

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.97)

    # ── Save ──
    out_paths = [
        Path(r"d:\Geo\vn-air-pollution-analysis-main\Air_quality\Thesis\latex\Hinh_ve\fig_3_pipeline.png"),
        Path(r"d:\Geo\vn-air-pollution-analysis-main\Air_quality\Thesis\figures\fig_3_pipeline.png"),
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {p}")

    plt.close(fig)


if __name__ == "__main__":
    main()
