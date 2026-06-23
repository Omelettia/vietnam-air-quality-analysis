"""Generate Figures 1, 2, 3 for the paper."""

import io, sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopandas as gpd
import seaborn as sns

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

os.makedirs("outputs/figures", exist_ok=True)

CM = 1 / 2.54
FIG_W = 17.5 * CM

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titlesize": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

REGION_COLORS = {"North": "#D62728", "Central": "#2CA02C", "South": "#1F77B4"}

STATIONS = pd.DataFrame([
    (1, "Hà Nội (ĐHBK)", "North", 21.005, 105.842),
    (2, "Hà Nội (NVC)", "North", 21.049, 105.883),
    (3, "Hà Nội (Nhân Chính)", "North", 21.003, 105.795),
    (4, "Hải Dương", "North", 20.938, 106.336),
    (5, "Hưng Yên", "North", 20.662, 106.059),
    (6, "Thái Bình (city)", "North", 20.458, 106.344),
    (7, "Thái Bình (Thái Thọ)", "North", 20.462, 106.517),
    (8, "Hà Nam", "North", 20.536, 105.916),
    (9, "Phú Thọ", "North", 21.339, 105.367),
    (10, "Quảng Ninh", "North", 21.009, 107.274),
    (11, "Quảng Nam", "Central", 15.562, 108.487),
    (12, "Quảng Ngãi", "Central", 15.121, 108.804),
    (13, "Đà Nẵng", "Central", 16.062, 108.159),
    (14, "Bình Định", "Central", 13.785, 109.220),
    (15, "Lâm Đồng", "Central", 11.953, 108.430),
    (16, "Ninh Thuận", "Central", 11.574, 108.992),
    (17, "Gia Lai (Pleiku)", "Central", 14.018, 108.035),
    (18, "Gia Lai (An Khê)", "Central", 13.954, 108.656),
    (19, "HCM (Bình Thạnh)", "South", 10.782, 106.683),
    (20, "HCM (Q. 2)", "South", 10.783, 106.753),
    (21, "Bình Dương", "South", 10.992, 106.658),
    (22, "Tây Ninh", "South", 11.030, 106.356),
    (23, "Long An", "South", 10.539, 106.405),
    (24, "Trà Vinh (city)", "South", 9.924, 106.340),
    (25, "Trà Vinh (Đông Hải)", "South", 9.576, 106.488),
    (26, "Vĩnh Long", "South", 10.251, 105.947),
    (27, "Sóc Trăng", "South", 9.614, 105.968),
], columns=["num", "label", "region", "lat", "lon"])


# =====================================================================
# FIGURE 1: Study area map
# =====================================================================
print("Figure 1: Study area map...")

vn0 = gpd.read_file("data/boundaries/gadm41_VNM_0.shp")
vn1 = gpd.read_file("data/boundaries/gadm41_VNM_1.shp")

fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 1.6))

vn1.plot(ax=ax, color="#F0F0F0", edgecolor="#CCCCCC", linewidth=0.3)
vn0.boundary.plot(ax=ax, color="#333333", linewidth=0.8)

for region, grp in STATIONS.groupby("region"):
    ax.scatter(grp["lon"], grp["lat"], c=REGION_COLORS[region],
               s=40, edgecolors="black", linewidths=0.5, zorder=5,
               label=f"{region} ({len(grp)})")

city_labels = {
    "Hanoi": (105.85, 21.02, 105.0, 22.2),
    "Da Nang": (108.16, 16.06, 106.8, 16.5),
    "Ho Chi Minh City": (106.70, 10.78, 105.0, 10.0),
}
for city, (cx, cy, tx, ty) in city_labels.items():
    ax.annotate(city, xy=(cx, cy), xytext=(tx, ty),
                fontsize=9, fontstyle="italic",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6),
                ha="center", va="center")

ax.set_xlim(102, 112)
ax.set_ylim(8, 24)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("(a) Study area and monitoring station locations", fontsize=11, pad=8)
ax.legend(loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")

ax.set_aspect("equal")
fig.savefig("outputs/figures/fig1_study_area.png", dpi=300, facecolor="white")
plt.close()
print("  Saved: outputs/figures/fig1_study_area.png")


# =====================================================================
# FIGURE 2: Filtering progression bar chart
# =====================================================================
print("Figure 2: Filtering progression...")

stages = [
    ("F0: raw AOT", 0.110, "r"),
    ("F1: + RF ≥ 0.5", 0.183, "r"),
    ("F2: + Unc. ≤ 0.5", 0.179, "r"),
    ("F0: physics-corrected", 0.162, "r"),
    ("F2: physics-corrected", 0.193, "r"),
    ("Daily OLS (R²)", 0.065, "R2"),
    ("Daily RANSAC (R²)", 0.293, "R2"),
]
labels = [s[0] for s in stages]
values = [s[1] for s in stages]
kinds = [s[2] for s in stages]

colors = []
for k in kinds:
    if k == "r":
        colors.append("#4C72B0")
    else:
        colors.append("#DD8452")

fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.55))
y_pos = np.arange(len(labels))[::-1]
bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.65)

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Mean per-station correlation (r or R²)")
ax.set_title("(b) Impact of filtering, physics correction, and regression on AOD–PM₂.₅ correlation",
             fontsize=11, pad=8)
ax.set_xlim(0, 0.38)

legend_elements = [
    Line2D([0], [0], color="#4C72B0", lw=8, label="Hourly Pearson r"),
    Line2D([0], [0], color="#DD8452", lw=8, label="Daily R²"),
]
ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig("outputs/figures/fig2_filtering.png", dpi=300, facecolor="white")
plt.close()
print("  Saved: outputs/figures/fig2_filtering.png")


# =====================================================================
# FIGURE 3: Hours-since-rain accumulation
# =====================================================================
print("Figure 3: Hours-since-rain accumulation...")

df = pd.read_csv("data/merged/unified_paper_v5.csv",
                  dtype={"stationId": str}, parse_dates=["ts"])

bins = [0, 6, 12, 24, 72, 168, 336, 720, np.inf]
bin_labels = ["0–6 h", "6–12 h", "12–24 h", "1–3 d", "3–7 d", "7–14 d", "14–30 d", ">30 d"]

sub = df.dropna(subset=["hrs_since_rain", "PM2.5"]).copy()
sub["rain_bin"] = pd.cut(sub["hrs_since_rain"], bins=bins, labels=bin_labels, right=False)

fig, ax = plt.subplots(figsize=(FIG_W * 1.18, FIG_W * 0.55))

for region in ["North", "Central", "South"]:
    grp = sub[sub["region"] == region]
    medians = grp.groupby("rain_bin", observed=False)["PM2.5"].median()
    ax.plot(range(len(bin_labels)), medians.values,
            marker="o", markersize=5, linewidth=1.8,
            color=REGION_COLORS[region], label=f"{region}")

ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels, rotation=30, ha="right")
ax.set_xlabel("Hours since last rainfall")
ax.set_ylabel("Median PM₂.₅ (µg/m³)")
ax.legend(framealpha=0.9, edgecolor="#CCCCCC")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

fig.savefig("outputs/figures/fig3_rain_accumulation.png", dpi=300, facecolor="white")
plt.close()
print("  Saved: outputs/figures/fig3_rain_accumulation.png")

from PIL import Image
im = Image.open("outputs/figures/fig3_rain_accumulation.png")
w_px, h_px = im.size
dpi_x, dpi_y = im.info.get("dpi", (300, 300))
print(f"  Size: {w_px}x{h_px} px, {w_px/dpi_x*2.54:.1f} x {h_px/dpi_y*2.54:.1f} cm, DPI={dpi_x:.0f}")

print("\nAll 3 figures generated.")
