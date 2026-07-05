"""Generate the final Red River Delta pipeline diagram.

The diagram matches the thesis scope: national diagnostics motivate a focused
Red River Delta regional XGBoost model with RFSI and wind-aware neighbor
features.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


C_INPUT = "#D6E8F7"
C_PROC = "#DCEFD6"
C_MODEL = "#FCEBC8"
C_OUT = "#ECDCEF"
C_EDGE = "#2C3E50"
C_TEXT = "#1A1A1A"

FIG_W, FIG_H = 12.0, 13.0
XC = FIG_W / 2


def box(ax, cx, cy, w, h, colour, lines, fs=10.5):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.12",
            facecolor=colour,
            edgecolor=C_EDGE,
            linewidth=1.3,
        )
    )
    step = 0.32
    y0 = cy + (len(lines) - 1) * step / 2
    for i, line in enumerate(lines):
        ax.text(cx, y0 - i * step, line, ha="center", va="center", fontsize=fs, color=C_TEXT)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=1.5, shrinkA=2, shrinkB=2),
    )


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    y_in = 11.5
    inputs = [
        (2.4, ["Vệ tinh", "AOD, khí vết TROPOMI,", "LST và nguồn phát thải"]),
        (6.0, ["Khí tượng", "ERA5, mưa GPM,", "PBLH, gió, độ ẩm"]),
        (9.6, ["Trạm mặt đất", "trạm KK PM2.5", "và LCS kiểm định"]),
    ]
    for cx, lines in inputs:
        box(ax, cx, y_in, 3.25, 1.55, C_INPUT, lines)

    y_qc, y_diag, y_region, y_feat, y_model, y_eval = 9.4, 7.7, 6.0, 4.45, 2.95, 1.25

    box(ax, XC, y_qc, 9.5, 1.15, C_PROC, ["Hợp nhất dữ liệu và lọc QC PM2.5", "bảng đặc trưng theo trạm-giờ"])
    box(
        ax,
        XC,
        y_diag,
        9.5,
        1.25,
        C_PROC,
        ["Chẩn đoán toàn quốc", "AOD thưa, CTM yếu, trần trong trạm,", "LOSO toàn quốc và kiểm tra tầng ô nhiễm"],
    )
    box(
        ax,
        XC,
        y_region,
        9.5,
        1.10,
        C_PROC,
        ["Chọn phạm vi vùng", "tập trung vào đồng bằng sông Hồng:", "vùng ô nhiễm cao, ba trạm neo gần hơn"],
    )
    box(
        ax,
        XC,
        y_feat,
        9.5,
        1.25,
        C_MODEL,
        ["Đặc trưng vùng", "vệ tinh + khí tượng + chu kỳ thời gian +", "RFSI và PM2.5 trạm neo theo gió"],
    )
    box(
        ax,
        XC,
        y_model,
        9.5,
        1.05,
        C_MODEL,
        ["Mô hình cuối", "XGBoost gbtree với base margin vùng"],
    )
    box(
        ax,
        3.2,
        y_eval,
        4.6,
        1.35,
        C_OUT,
        ["LOSO nội bộ", "12 trạm KK vùng ĐBSH", "chỉ số theo giờ và theo ngày"],
    )
    box(
        ax,
        8.8,
        y_eval,
        4.6,
        1.35,
        C_OUT,
        ["Kiểm định ngoài", "trạm LCS trong vùng", "và Đại sứ quán Hoa Kỳ Hà Nội"],
    )

    for cx, _ in inputs:
        arrow(ax, cx, y_in - 0.78, XC, y_qc + 0.58)
    arrow(ax, XC, y_qc - 0.58, XC, y_diag + 0.62)
    arrow(ax, XC, y_diag - 0.62, XC, y_region + 0.55)
    arrow(ax, XC, y_region - 0.55, XC, y_feat + 0.62)
    arrow(ax, XC, y_feat - 0.62, XC, y_model + 0.52)
    arrow(ax, XC, y_model - 0.52, 3.2, y_eval + 0.68)
    arrow(ax, XC, y_model - 0.52, 8.8, y_eval + 0.68)

    plt.tight_layout(pad=0.4)
    out_paths = [
        Path(__file__).resolve().parents[2] / "latex" / "Hinh_ve" / "fig_3_pipeline.png",
        Path(__file__).resolve().parents[2] / "figures" / "fig_3_pipeline.png",
    ]
    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print("Saved:", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
