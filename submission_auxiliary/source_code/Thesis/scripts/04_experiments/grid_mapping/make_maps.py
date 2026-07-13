# -*- coding: utf-8 -*-
"""Render the 4 PM2.5 grid maps (2x2 panel) for the defense deck.
Sequential YlOrRd ramp (domain standard for PM2.5), shared scale across panels,
anchor stations overlaid with their observed PM2.5 in the same ramp."""
import numpy as np
import os
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from pathlib import Path

MAP_DATA = os.environ.get("MAP_DATA", "D:/map_data")

ROOT = Path(__file__).resolve().parents[4].as_posix()
OUTPUTS = [
    Path(ROOT) / "defense_assets" / "fig_pm25_maps.png",
    Path(ROOT) / "Thesis" / "figures" / "fig_pm25_maps.png",
    Path(ROOT) / "Thesis" / "latex" / "Hinh_ve" / "fig_pm25_maps.png",
]
BOX = (20.3, 21.3, 105.5, 107.0)

pred = pd.read_csv(MAP_DATA + "/maps/grid_predictions.csv", parse_dates=["ts"])
obs = pd.read_csv(MAP_DATA + "/maps/anchor_obs.csv", parse_dates=["ts"])
vn = gpd.read_file(ROOT + "/GADM_Vietnam/gadm41_VNM_1.shp")

lats = np.sort(pred["lat"].unique()); lons = np.sort(pred["lon"].unique())
LON, LAT = np.meshgrid(lons, lats)

vmax = float(np.nanpercentile(pred["pm25_pred"], 99))
vmax = max(60, np.ceil(vmax / 10) * 10)
norm = Normalize(vmin=0, vmax=vmax)
CMAP = "YlOrRd"

panels = [
    ("2025-12-09 08:00", "(a) 09/12/2025 — 08:00 (mùa đông, sáng)"),
    ("2025-12-09 20:00", "(b) 09/12/2025 — 20:00 (mùa đông, tối)"),
    ("2025-07-30 08:00", "(c) 30/07/2025 — 08:00 (mùa hè, sáng)"),
    ("2025-07-30 20:00", "(d) 30/07/2025 — 20:00 (mùa hè, tối)"),
]

CITIES = [  # (name, lat, lon)
    ("Hà Nội", 21.028, 105.834),
    ("Bắc Ninh", 21.186, 106.076),
    ("Hải Dương", 20.940, 106.333),
    ("Hưng Yên", 20.646, 106.051),
    ("Hà Nam", 20.541, 105.913),
    ("Nam Định", 20.438, 106.162),
    ("Thái Bình", 20.446, 106.336),
    ("Hải Phòng", 20.865, 106.683),
]

fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6), sharex=True, sharey=True)
for ax, (ts_s, title) in zip(axes.ravel(), panels):
    ts = pd.Timestamp(ts_s)
    g = pred[pred["ts"] == ts]
    Z = g.pivot_table(index="lat", columns="lon", values="pm25_pred").values
    pc = ax.pcolormesh(LON, LAT, Z, cmap=CMAP, norm=norm, shading="nearest",
                       rasterized=True)
    vn.boundary.plot(ax=ax, color="#7a8694", linewidth=0.55, alpha=0.85)
    o = obs[obs["ts"] == ts]
    ax.scatter(o["lon"], o["lat"], c=o["pm25_obs"], cmap=CMAP, norm=norm,
               s=95, edgecolor="black", linewidth=1.1, zorder=6)
    for nm, la, lo in CITIES:
        ax.annotate(nm, xy=(lo, la), xytext=(lo, la + 0.022),
                    ha="center", fontsize=8, color="#1c2634", zorder=8,
                    path_effects=None,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.55))
    n_anchor = int(g["n_anchors"].iloc[0])
    ax.set_xlim(BOX[2], BOX[3]); ax.set_ylim(BOX[0], BOX[1])
    ax.set_title(f"{title} · {n_anchor} trạm neo", fontsize=11.5)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.18, linewidth=0.4)

fig.subplots_adjust(right=0.9, hspace=0.16, wspace=0.06,
                    left=0.055, top=0.94, bottom=0.06)
cax = fig.add_axes([0.915, 0.11, 0.02, 0.78])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
cb.set_label("PM2.5 ước tính (µg/m³)", fontsize=11)
cb.ax.tick_params(labelsize=9)

for out in OUTPUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=175, bbox_inches="tight", facecolor="white")
    print("saved", out, "| vmax =", vmax)
