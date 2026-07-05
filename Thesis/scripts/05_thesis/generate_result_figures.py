"""Generate thesis figures for the Red River Delta pivot.

The figures summarize locked result tables. They do not rerun models or change
metrics; they only make the evidence easier to inspect in the PDF.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
FIG_DIRS = [ROOT / "Thesis" / "figures", ROOT / "Thesis" / "latex" / "Hinh_ve"]
for d in FIG_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for d in FIG_DIRS:
        path = d / name
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print("Saved:", path)
    plt.close(fig)


def regional_context():
    labels = ["Toàn quốc\n40 trạm", "ĐBSH\n12 trạm"]
    nearest3 = [27.2, 21.6]
    pm25 = [19.3, 37.4]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    x = np.arange(len(labels))
    axes[0].bar(x, nearest3, width=0.56, color="#4C78A8")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("km")
    axes[0].set_title("(a) Khoảng cách TB tới 3 trạm neo")
    axes[0].grid(axis="y", alpha=0.25)
    for i, v in enumerate(nearest3):
        axes[0].text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=9)

    axes[1].bar(x, pm25, width=0.56, color="#E45756")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("µg/m³")
    axes[1].set_title("(b) PM2.5 trung vị theo trạm")
    axes[1].grid(axis="y", alpha=0.25)
    for i, v in enumerate(pm25):
        axes[1].text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=9)
    fig.suptitle("Bối cảnh chọn vùng đồng bằng sông Hồng", y=1.02)
    save(fig, "fig_4_regional_context.png")


def delta_configs():
    rows = [
        ("Nền vùng", 0.2899, 0.1806),
        ("+ RFSI", 0.4161, 0.2635),
        ("+ RFSI và gió", 0.4233, 0.2709),
        ("Oracle\n(chẩn đoán)", 0.5907, 0.4989),
    ]
    labels = [r[0] for r in rows]
    pooled = [r[1] for r in rows]
    mean_station = [r[2] for r in rows]
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - w / 2, pooled, width=w, color="#4C78A8", label="$R^2$ gộp")
    ax.bar(x + w / 2, mean_station, width=w, color="#F58518", label="$R^2$ TB trạm")
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("$R^2$")
    ax.set_title("LOSO vùng ĐBSH: RFSI tạo phần cải thiện chính")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    for i, v in enumerate(pooled):
        ax.text(i - w / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(mean_station):
        ax.text(i + w / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    save(fig, "fig_5_delta_config_metrics.png")


def delta_station_r2():
    path = ROOT / "Thesis" / "results" / "04_validation" / "red_river_delta_internal_station_metrics.csv"
    if not path.exists():
        path = ROOT / "analysis" / "thesis_experiments" / "delta_v5h_test.csv"
    df = pd.read_csv(path)
    configs = ["delta_bm", "delta_rfsi", "delta_rfsi_wind"]
    labels = {
        "delta_bm": "Nền vùng",
        "delta_rfsi": "+ RFSI",
        "delta_rfsi_wind": "+ RFSI và gió",
    }
    sub = df[df["config"].isin(configs)].copy()
    order = (
        sub[sub["config"].eq("delta_rfsi_wind")]
        .sort_values("pm25_mean")
        [["station_id", "station_name", "pm25_mean", "tier"]]
    )
    pivot = sub.pivot_table(index="station_id", columns="config", values="r2_hourly", aggfunc="first")

    # Short station labels: province abbreviation + the most distinctive token.
    # Full names are in Appendix A.1.
    prov_abbr = {
        "Hà Nội": "HN", "Bắc Ninh": "BN", "Hải Dương": "HD",
        "Hà Nam": "HNa", "Thái Bình": "TB", "Hưng Yên": "HY",
    }

    def short_name(name):
        name = str(name).replace(" (KK)", "").strip()
        for prov, ab in prov_abbr.items():
            if name.startswith(prov):
                rest = name[len(prov):].strip(" -")
                token = rest.split(" - ")[0].split(",")[0].strip()
                return f"{ab}: {token[:20]}"
        return name[:24]

    y = np.arange(len(order))
    h = 0.26
    fig, ax = plt.subplots(figsize=(9.4, 7.6))
    colors = ["#9E9E9E", "#4C78A8", "#F58518"]
    offsets = [-h, 0, h]
    for cfg, color, off in zip(configs, colors, offsets):
        vals = [pivot.loc[sid, cfg] for sid in order["station_id"]]
        ax.barh(y + off, vals, height=h, color=color, label=labels[cfg])
    ax.axvline(0, color="#333", linewidth=0.9)
    ax.set_yticks(y, [short_name(n) for n in order["station_name"]], fontsize=11)
    ax.set_xlabel("$R^2$ theo giờ", fontsize=12)
    ax.set_xlim(-1.15, 1.05)
    ax.set_title("Hiệu năng theo từng trạm trong LOSO vùng ĐBSH\n(mã trạm rút gọn; tên đầy đủ trong Phụ lục A)", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=11)
    for yi, pm, tier in zip(y, order["pm25_mean"], order["tier"]):
        tier_txt = str(tier)
        if tier_txt.isdigit():
            tier_txt = "t" + tier_txt
        ax.text(0.86, yi, f"{tier_txt}, {pm:.0f}", va="center", fontsize=9)
    ax.text(0.86, len(order) - 0.2, "tier, PM2.5 TB", va="bottom", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.20, right=0.94, bottom=0.14)
    save(fig, "fig_5_delta_station_r2.png")


def feature_gain():
    path = ROOT / "Thesis" / "results" / "04_validation" / "red_river_delta_feature_gain_by_group.csv"
    df = pd.read_csv(path)
    models = ["delta_bm", "delta_rfsi", "delta_rfsi_wind"]
    keep = [
        ("nearby_pm25_rfsi", "PM2.5 trạm neo / RFSI"),
        ("aod_satellite", "AOD vệ tinh"),
        ("tropomi_gases", "Khí vết TROPOMI"),
        ("meteorology", "Thời tiết"),
        ("rain_dryness", "Mưa / số ngày khô"),
        ("time_cycle", "Chu kỳ thời gian"),
        ("wind_dispersion", "Gió / phân tán"),
        ("aod_physics", "Tương tác AOD"),
        ("gas_physics", "Tương tác khí"),
    ]
    pivot = df.pivot_table(index="model", columns="feature_group", values="gain_share_percent", aggfunc="sum").fillna(0)

    model_labels = ["Nền vùng", "+ RFSI", "+ RFSI và gió"]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    left = np.zeros(len(models))
    colors = plt.cm.tab10(np.linspace(0, 1, len(keep)))
    yb = np.arange(len(models))
    for (group, label), color in zip(keep, colors):
        vals = np.array([pivot.loc[m, group] if group in pivot.columns else 0 for m in models])
        ax.barh(yb, vals, left=left, color=color, label=label, height=0.6)
        for yi, v, l in zip(yb, vals, left):
            if v >= 5:  # annotate only sizeable segments
                ax.text(l + v / 2, yi, f"{v:.0f}", va="center", ha="center",
                        fontsize=9, color="white", fontweight="bold")
        left += vals
    ax.set_yticks(yb, model_labels, fontsize=11)
    ax.set_xlabel("Tỷ lệ gain theo nhóm (%)", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_title("Nhóm đặc trưng trong mô hình vùng", fontsize=12)
    ax.legend(frameon=False, bbox_to_anchor=(0.5, -0.18), loc="upper center",
              ncol=3, fontsize=9.5)
    ax.grid(axis="x", alpha=0.2)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.30, top=0.90)
    save(fig, "fig_5_feature_gain_groups.png")


def external_lcs():
    path = ROOT / "Thesis" / "results" / "04_validation" / "red_river_delta_external_station_metrics.csv"
    df = pd.read_csv(path)
    lcs = df[df["type"].eq("LCS")].copy()
    canonical_lcs_median = float(lcs["r2"].median())
    embassy = df[df["type"].eq("Embassy")]
    canonical_embassy_r2 = float(embassy["r2"].iloc[0]) if len(embassy) else np.nan

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    clipped = lcs["r2"].clip(lower=-1, upper=1)
    ax.hist(clipped, bins=np.linspace(-1, 1, 17), color="#72B7B2", edgecolor="white")
    ax.axvline(canonical_lcs_median, color="#222", linestyle="--", linewidth=1.4, label=f"Trung vị LCS = {canonical_lcs_median:.3f}")
    ax.axvline(canonical_embassy_r2, color="#E45756", linewidth=1.8, label=f"Đại sứ quán Hoa Kỳ = {canonical_embassy_r2:.3f}")
    ax.axvline(0, color="#666", linewidth=0.8)
    ax.set_xlabel("$R^2$ theo giờ của từng trạm kiểm định (cắt trong [-1, 1])")
    ax.set_ylabel("Số trạm")
    ax.set_title("Kiểm định ngoài: phần lớn trạm dương nhưng không đồng đều")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_5_external_lcs_r2.png")


def external_failure_cases():
    path = ROOT / "Thesis" / "results" / "04_validation" / "red_river_delta_external_predictions.csv"
    if not path.exists():
        path = ROOT / "analysis" / "thesis_experiments" / "delta_v5h_lcs_validation_predictions.csv"
    if not path.exists():
        return

    df = pd.read_csv(path, parse_dates=["ts"])
    cases = [
        (
            "Hưng Yên Nhà văn hoá thế thôn Hồng Cầu (LCS)",
            "Hồng Cầu: điểm cao cục bộ",
        ),
        (
            "Hưng Yên Cổng bảo vệ  ủy ban xã Như Quỳnh (LCS)",
            "Như Quỳnh: gần Hồng Cầu nhưng thấp",
        ),
        (
            "Hải Phòng cột điện Phường Bạch Đằng (anh Vân - 098 9325417) (LCS)",
            "Bạch Đằng: điểm thấp cục bộ",
        ),
    ]

    fig, axes = plt.subplots(len(cases), 1, figsize=(9.4, 7.2), sharex=True)
    for ax, (station, title) in zip(axes, cases):
        sub = df[df["station"].eq(station)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        daily = (
            sub.set_index("ts")[["y_true", "y_pred"]]
            .resample("D")
            .mean()
            .rolling(7, min_periods=3)
            .mean()
        )
        ax.plot(daily.index, daily["y_true"], color="#222222", linewidth=1.6, label="Thực đo")
        ax.plot(daily.index, daily["y_pred"], color="#E45756", linewidth=1.5, label="Dự đoán")
        mean_y = sub["y_true"].mean()
        mean_p = sub["y_pred"].mean()
        r2 = 1.0 - np.sum((sub["y_true"] - sub["y_pred"]) ** 2) / np.sum(
            (sub["y_true"] - sub["y_true"].mean()) ** 2
        )
        ax.set_title(f"{title}: thực đo {mean_y:.1f}, dự đoán {mean_p:.1f}, R²={r2:.2f}", loc="left")
        ax.set_ylabel("µg/m³")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Thời gian")
    fig.suptitle("Các kiểu thất bại trong kiểm định LCS ngoài", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig_5_external_failure_cases.png")


def main():
    regional_context()
    delta_configs()
    delta_station_r2()
    feature_gain()
    external_lcs()
    external_failure_cases()


if __name__ == "__main__":
    main()
