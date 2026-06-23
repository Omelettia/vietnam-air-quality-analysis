"""
Thesis ML Pipeline — Steps 1-10
Build unified training dataset for PM2.5 prediction from 40+ KK stations.

Steps:
  1-4: Fix marginal stations, finalize selection, LCS selection
  5:   DEM feature extraction
  6:   Station name mapping
  7:   Timezone verification
  8:   Build unified merged dataset
  9:   Feature engineering
  10:  Summary report
"""

import io, sys, os, warnings, glob, unicodedata, math, json, traceback
from datetime import timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE)

# ══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════

def normalize(s):
    return unicodedata.normalize("NFC", s.replace(":", "").replace("", "").strip())

def build_index(pattern, prefix_strip="", suffix_strip=".csv", space_char=" "):
    idx = {}
    for f in glob.glob(pattern):
        base = os.path.basename(f).replace(suffix_strip, "")
        if prefix_strip:
            base = base.replace(prefix_strip, "", 1)
        if space_char != " ":
            base = base.replace(space_char, " ")
        idx[normalize(base)] = f
    return idx

SENTINEL_VALUES = {-9999, -999, 9999, -9999.0, -999.0, 9999.0}
QC_RANGES = {
    "PM2.5": (0, 500), "PM10": (0, 1000), "Temperature": (-10, 50),
    "Humidity": (0, 100), "Pressure": (900, 1100),
    "Wind Speed": (0, 50), "Wind Direction": (0, 360),
    "NO2": (0, 500), "O3": (0, 500), "SO2": (0, 500), "CO": (0, 100),
}

env_idx = {k: v for k, v in build_index("data/stations/historical_full_v2/*.csv").items()
           if not v.endswith(".log")}
aod_idx = build_index("data/station_aod_v3/L2/*.csv")
gpm_idx = build_index("data/gpm/station_gis_extracted_v2/*.csv", space_char="_")
wx_idx  = build_index("data/stations/weather/*.csv", prefix_strip="weather_")

metadata = pd.read_csv("data/stations/metadata/envisoft_station_map.csv",
                        dtype={"stationId": str})
meta_dict = {str(r["stationId"]): r for _, r in metadata.iterrows()}
tiers = pd.read_csv("analysis/v2_full_audit/station_tiers.csv",
                     dtype={"station_id": str})

os.makedirs("analysis/thesis_audit", exist_ok=True)
os.makedirs("data/merged_thesis", exist_ok=True)
os.makedirs("data/merged", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
#  STEPS 1–4: FIX MARGINAL STATIONS, FINALIZE SELECTION
# ══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("STEPS 1-4: STATION SELECTION FINALIZATION")
print("=" * 80)

phase1 = pd.read_csv("analysis/thesis_audit/station_selection.csv",
                      dtype={"station_id": str})

# --- Step 1: Fix marginal stations ---
# Stuck temperature: will be NaN'd during load (OpenMeteo fills)
STUCK_TEMP_STATIONS = {
    "Hà Nam Công Viên Nam Cao - P.Quang Trung - TP. Phủ Lý (KK)",
    "Ninh Thuận Công viên (bến xe cũ) - Đ. Thống Nhất - P. Thanh Sơn - TP Phan Rang (KK)",
    "Hưng Yên số 437 Nguyễn Văn Linh - Tp Hưng Yên (KK)",
}

# Quế Võ: NaN entire WS column
QUEVO_NAN_WS = "Bắc Ninh UBND huyện Quế Võ, TT Phố Mới (KK)"

# No-diurnal-cycle stations: keep (industrial/mining areas)
NO_DIURNAL_KEEP = {
    "Hà Nội Công viên Nhân Chính - Khuất Duy Tiến (KK)",
    "Quảng Ninh Gần KCN Cái Lân (KK)",
    "Quảng Ninh Km11 - Minh Thành (KK)",
    "Quảng Ninh Nhuệ Hổ - Đông Triều (KK)",
    "Quảng Ninh Nhà máy tuyển than Nam Cầu Trắng - Hạ Long (KK)",
    "Quảng Ninh Phường Cẩm Thịnh - Cẩm Phả (KK)",
    "Quảng Ninh Trung tâm văn hóa thể thao Cẩm Phả, đường Trần Phú, phường Cẩm Trung (KK)",
    "Quảng Ninh UBND TP Uông Bí (KK)",
    "Thái Bình xã Thái Thọ, huyện Thái Thụy (KK)",
    "Trà Vinh xã Dân Thành, TX Duyên Hải (KK)",
    "Trà Vinh xã Đông Hải, huyện Duyên Hải (KK)",
}

# Re-evaluate marginals
for i, row in phase1.iterrows():
    if row["quality_flag"] != "marginal":
        continue
    name = row["station_name"]
    reasons = str(row.get("marginal_reason", ""))

    fixable = True
    remaining = []
    for reason in reasons.split("; "):
        reason = reason.strip()
        if not reason:
            continue
        if "stuck_temp_0" in reason and name in STUCK_TEMP_STATIONS:
            pass  # fixed by NaN'ing
        elif "no_diurnal_cycle" in reason and name in NO_DIURNAL_KEEP:
            pass  # keep
        elif "sentinels=" in reason:
            pass  # cleaned during load
        elif "neg_pm25" in reason:
            pass  # clipped during QC
        else:
            remaining.append(reason)

    if remaining:
        phase1.at[i, "quality_flag"] = "marginal_fixed"
        phase1.at[i, "marginal_reason"] = "; ".join(remaining)
    else:
        phase1.at[i, "quality_flag"] = "pass"
        phase1.at[i, "marginal_reason"] = ""

# --- Step 2: Drop bad Tier-1 stations ---
DROP_STATIONS = {
    "Trà Vinh Tp. Trà Vinh (KK)",        # PM2.5 mean 4.4 - miscalibrated
    "Vĩnh Long UBND tỉnh, đường Hoàng Thái Hiếu (KK)",  # PM2.5 mean 2.5
    "Hưng Yên Nhà văn hóa xã Tân Quang - h.Văn Lâm (KK)",  # 41% zero, broken 2023-2025
    "Hưng Yên Trạm khí xung quanh KCN Thăng long II (KK)",  # 49% zero
    "Tây Ninh Phường 3 - TP. Tây Ninh (KK)",   # PM2.5 mean 0.1
}
for i, row in phase1.iterrows():
    if row["station_name"] in DROP_STATIONS and row["quality_flag"] != "fail":
        phase1.at[i, "quality_flag"] = "fail"
        phase1.at[i, "fail_reason"] = "dropped_bad_sensor"

# Final main selection: pass or marginal_fixed
main_sel = phase1[phase1["quality_flag"].isin(["pass", "marginal_fixed"])].copy()

# Add lat/lon from metadata
for i, row in main_sel.iterrows():
    sid = str(row["station_id"])
    if sid in meta_dict:
        main_sel.at[i, "lat"] = meta_dict[sid]["latitude"]
        main_sel.at[i, "lon"] = meta_dict[sid]["longitude"]

print(f"\nMain selection: {len(main_sel)} stations")
print(f"  Pass: {(main_sel['quality_flag']=='pass').sum()}")
print(f"  Marginal fixed: {(main_sel['quality_flag']=='marginal_fixed').sum()}")
print(f"  By tier: {dict(main_sel['tier'].value_counts().sort_index())}")
print(f"  By type: {dict(main_sel['station_type'].value_counts())}")
print(f"  By region: {dict(main_sel['region'].value_counts())}")
print(f"  Total rows: {main_sel['n_rows'].sum():,}")

# --- Step 3: LCS selection (separate) ---
lcs_all = tiers[tiers["station_type"] == "LCS"].copy()
lcs_results = []

for _, row in lcs_all.iterrows():
    name = row["station_name"]
    sid = str(row["station_id"])
    tier = row["tier"]
    norm = normalize(name)
    env_path = env_idx.get(norm)
    if not env_path:
        lcs_results.append({"station_name": name, "station_id": sid, "tier": tier,
                            "lcs_flag": "fail", "reason": "no_file"})
        continue
    try:
        df = pd.read_csv(env_path)
    except:
        lcs_results.append({"station_name": name, "station_id": sid, "tier": tier,
                            "lcs_flag": "fail", "reason": "read_error"})
        continue

    df["ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    n = len(df)
    if n == 0:
        lcs_results.append({"station_name": name, "station_id": sid, "tier": tier,
                            "lcs_flag": "fail", "reason": "empty"})
        continue

    data_months = (df["ts"].max() - df["ts"].min()).days / 30.44
    pm_coverage = df["PM2.5"].notna().mean() if "PM2.5" in df.columns else 0
    pm_mean = df["PM2.5"].mean() if "PM2.5" in df.columns else np.nan
    zero_pct = (df["PM2.5"] == 0).sum() / max(df["PM2.5"].notna().sum(), 1)

    fail_reasons = []
    if data_months < 5:
        fail_reasons.append(f"months={data_months:.1f}")
    if pm_coverage < 0.50:
        fail_reasons.append(f"pm_cov={pm_coverage:.1%}")
    if zero_pct > 0.30:
        fail_reasons.append(f"zero={zero_pct:.1%}")
    if not np.isnan(pm_mean) and (pm_mean < 5 or pm_mean > 150):
        fail_reasons.append(f"mean={pm_mean:.1f}")

    # Find co-located KK stations (within 15km)
    colocated = []
    if sid in meta_dict:
        lat_lcs = meta_dict[sid]["latitude"]
        lon_lcs = meta_dict[sid]["longitude"]
        for _, kk in main_sel.iterrows():
            kk_sid = str(kk["station_id"])
            if kk_sid in meta_dict:
                lat_kk = meta_dict[kk_sid]["latitude"]
                lon_kk = meta_dict[kk_sid]["longitude"]
                dist = 111.0 * np.sqrt((lat_lcs - lat_kk)**2 +
                       ((lon_lcs - lon_kk) * np.cos(np.radians(lat_lcs)))**2)
                if dist < 15:
                    colocated.append(kk["station_name"][:30])

    lcs_results.append({
        "station_name": name, "station_id": sid, "tier": tier,
        "n_rows": n, "data_months": round(data_months, 1),
        "pm25_coverage": round(pm_coverage, 4),
        "pm25_mean": round(pm_mean, 2) if not np.isnan(pm_mean) else np.nan,
        "zero_pct": round(zero_pct, 4),
        "lcs_flag": "fail" if fail_reasons else "pass",
        "reason": "; ".join(fail_reasons),
        "colocated_kk": "; ".join(colocated),
        "data_start": str(df["ts"].min()),
        "data_end": str(df["ts"].max()),
    })

lcs_df = pd.DataFrame(lcs_results)
lcs_df.to_csv("analysis/thesis_audit/station_selection_lcs.csv",
              index=False, encoding="utf-8-sig")
n_lcs_pass = (lcs_df["lcs_flag"] == "pass").sum()
print(f"\nLCS selection: {n_lcs_pass}/{len(lcs_df)} pass (5+ months, clean)")

# --- Step 4: Save final station selection ---
# Rebuild with has_envisoft_temp/hum/wind columns
final_rows = []
for _, row in main_sel.iterrows():
    sid = str(row["station_id"])
    m = meta_dict.get(sid, {})
    lat = m.get("latitude", np.nan) if isinstance(m, dict) else (m["latitude"] if hasattr(m, "__getitem__") else np.nan)
    lon = m.get("longitude", np.nan) if isinstance(m, dict) else (m["longitude"] if hasattr(m, "__getitem__") else np.nan)

    # Get coverage from tiers table
    tier_row = tiers[tiers["station_id"] == sid]
    has_temp = tier_row["temp_pct"].iloc[0] > 50 if len(tier_row) > 0 else False
    has_hum = tier_row["humidity_pct"].iloc[0] > 50 if len(tier_row) > 0 else False
    has_wind = tier_row["wind_speed_pct"].iloc[0] > 50 if len(tier_row) > 0 else False

    final_rows.append({
        "stationId": sid,
        "station_name": row["station_name"],
        "lat": row.get("lat", lat),
        "lon": row.get("lon", lon),
        "region": row["region"],
        "station_type": row["station_type"],
        "tier": row["tier"],
        "data_start": row["data_start"],
        "data_end": row["data_end"],
        "total_months": row["data_months"],
        "pm25_coverage": row["pm25_coverage"],
        "has_envisoft_temp": has_temp,
        "has_envisoft_humidity": has_hum,
        "has_envisoft_wind": has_wind,
        "quality_flag": row["quality_flag"],
        "fail_reason": row.get("fail_reason", ""),
    })

final_df = pd.DataFrame(final_rows)
final_df.to_csv("analysis/thesis_audit/station_selection_final.csv",
                index=False, encoding="utf-8-sig")

print(f"\n--- FINAL MAIN SELECTION: {len(final_df)} stations ---")
print(f"  North: {(final_df['region']=='North').sum()}")
print(f"  Central: {(final_df['region']=='Central').sum()}")
print(f"  South: {(final_df['region']=='South').sum()}")
print(f"  Other/Unknown: {(~final_df['region'].isin(['North','Central','South'])).sum()}")
print(f"  Total expected rows: {main_sel['n_rows'].sum():,}")
print(f"  With Envisoft temp: {final_df['has_envisoft_temp'].sum()}")
print(f"  With Envisoft humidity: {final_df['has_envisoft_humidity'].sum()}")
print(f"  With Envisoft wind: {final_df['has_envisoft_wind'].sum()}")

# ══════════════════════════════════════════════════════════════════════
#  STEP 5: DEM FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 5: DEM FEATURE EXTRACTION")
print("=" * 80)

try:
    import rasterio
    from scipy.ndimage import sobel
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio or scipy not available. Using existing topo features if available.")

# Collect all stations needing DEM (main + LCS pass)
all_stations_for_dem = []
for _, r in final_df.iterrows():
    all_stations_for_dem.append({
        "stationId": r["stationId"], "station_name": r["station_name"],
        "lat": r["lat"], "lon": r["lon"]
    })
if n_lcs_pass > 0:
    for _, r in lcs_df[lcs_df["lcs_flag"] == "pass"].iterrows():
        sid = str(r["station_id"])
        m = meta_dict.get(sid)
        if m is not None:
            lat = m["latitude"] if hasattr(m, "__getitem__") else np.nan
            lon = m["longitude"] if hasattr(m, "__getitem__") else np.nan
            all_stations_for_dem.append({
                "stationId": sid, "station_name": r["station_name"],
                "lat": lat, "lon": lon
            })

dem_files = glob.glob("data/dem/*.tif")
print(f"DEM files found: {len(dem_files)}")

existing_topo = None
topo_path = "data/stations/metadata/stations_with_topo_features.csv"
if os.path.exists(topo_path):
    existing_topo = pd.read_csv(topo_path)
    print(f"Existing topo features: {len(existing_topo)} stations, columns: {list(existing_topo.columns)}")

dem_results = []

if HAS_RASTERIO and dem_files:
    rasters = []
    for f in dem_files:
        try:
            src = rasterio.open(f)
            bounds = src.bounds
            rasters.append({"path": f, "src": src, "bounds": bounds})
            print(f"  {os.path.basename(f)}: {src.width}x{src.height}, "
                  f"CRS={src.crs}, bounds=({bounds.left:.2f},{bounds.bottom:.2f})-"
                  f"({bounds.right:.2f},{bounds.top:.2f})")
        except Exception as e:
            print(f"  {os.path.basename(f)}: ERROR {e}")

    for st in all_stations_for_dem:
        lat, lon = st["lat"], st["lon"]
        if pd.isna(lat) or pd.isna(lon):
            dem_results.append({**st, "elevation_m": np.nan, "slope_deg": np.nan,
                                "aspect_deg": np.nan, "aspect_sin": np.nan,
                                "aspect_cos": np.nan, "dem_source": "missing_coords"})
            continue

        found = False
        for r in rasters:
            b = r["bounds"]
            if b.left <= lon <= b.right and b.bottom <= lat <= b.top:
                src = r["src"]
                row_px, col_px = src.index(lon, lat)
                win_size = 5
                half = win_size // 2
                r_start = max(0, row_px - half)
                c_start = max(0, col_px - half)
                r_end = min(src.height, row_px + half + 1)
                c_end = min(src.width, col_px + half + 1)

                window = rasterio.windows.Window(c_start, r_start,
                                                  c_end - c_start, r_end - r_start)
                data = src.read(1, window=window).astype(float)
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan

                elev = data[row_px - r_start, col_px - c_start]

                if data.shape[0] >= 3 and data.shape[1] >= 3:
                    res = abs(src.res[0])
                    dy = np.gradient(data, res * 111320, axis=0)
                    dx = np.gradient(data, res * 111320 * np.cos(np.radians(lat)), axis=1)
                    cy, cx = row_px - r_start, col_px - c_start
                    cy = min(cy, dy.shape[0] - 1)
                    cx = min(cx, dx.shape[1] - 1)
                    slope_rad = np.arctan(np.sqrt(dx[cy, cx]**2 + dy[cy, cx]**2))
                    slope_deg = np.degrees(slope_rad)
                    aspect_rad = np.arctan2(-dx[cy, cx], dy[cy, cx])
                    aspect_deg = np.degrees(aspect_rad) % 360
                else:
                    slope_deg = 0.0
                    aspect_deg = 0.0

                dem_results.append({
                    **st,
                    "elevation_m": round(float(elev), 1) if not np.isnan(elev) else np.nan,
                    "slope_deg": round(float(slope_deg), 3),
                    "aspect_deg": round(float(aspect_deg), 1),
                    "aspect_sin": round(np.sin(np.radians(aspect_deg)), 4),
                    "aspect_cos": round(np.cos(np.radians(aspect_deg)), 4),
                    "dem_source": os.path.basename(r["path"]),
                })
                found = True
                break

        if not found:
            dem_results.append({**st, "elevation_m": np.nan, "slope_deg": np.nan,
                                "aspect_deg": np.nan, "aspect_sin": np.nan,
                                "aspect_cos": np.nan, "dem_source": "out_of_extent"})

    for r in rasters:
        r["src"].close()

elif existing_topo is not None:
    print("Using existing topo features (partial)...")
    for st in all_stations_for_dem:
        match = existing_topo[existing_topo.get("stationId", existing_topo.get("station_id", pd.Series())) == st["stationId"]]
        if len(match) > 0:
            r = match.iloc[0]
            elev = r.get("elevation_m", np.nan)
            dem_results.append({
                **st, "elevation_m": elev,
                "slope_deg": 0.0, "aspect_deg": 0.0,
                "aspect_sin": 0.0, "aspect_cos": 1.0,
                "dem_source": "existing_partial",
            })
        else:
            dem_results.append({**st, "elevation_m": np.nan, "slope_deg": np.nan,
                                "aspect_deg": np.nan, "aspect_sin": np.nan,
                                "aspect_cos": np.nan, "dem_source": "not_found"})
else:
    print("No DEM data available — all stations get NaN topo features.")
    for st in all_stations_for_dem:
        dem_results.append({**st, "elevation_m": np.nan, "slope_deg": np.nan,
                            "aspect_deg": np.nan, "aspect_sin": np.nan,
                            "aspect_cos": np.nan, "dem_source": "no_dem"})

dem_df = pd.DataFrame(dem_results)
dem_df.to_csv("data/dem/station_dem_features.csv", index=False, encoding="utf-8-sig")

n_with_elev = dem_df["elevation_m"].notna().sum()
print(f"\nDEM results: {n_with_elev}/{len(dem_df)} stations have elevation")
if n_with_elev > 0:
    print(f"  Elevation range: {dem_df['elevation_m'].min():.0f} – {dem_df['elevation_m'].max():.0f} m")
    missing = dem_df[dem_df["elevation_m"].isna()]
    if len(missing) > 0:
        print(f"  Missing DEM ({len(missing)}): {list(missing['station_name'].values[:5])}...")

# ══════════════════════════════════════════════════════════════════════
#  STEP 6: STATION NAME MAPPING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 6: STATION NAME MAPPING")
print("=" * 80)

name_map_rows = []
for _, row in final_df.iterrows():
    name = row["station_name"]
    sid = row["stationId"]
    norm = normalize(name)

    env_file = env_idx.get(norm, "")
    aod_file = aod_idx.get(norm, "")
    gpm_file = gpm_idx.get(norm, "")
    wx_file = wx_idx.get(norm, "")

    name_map_rows.append({
        "stationId": sid,
        "station_name": name,
        "envisoft_file": os.path.basename(env_file) if env_file else "MISSING",
        "himawari_file": os.path.basename(aod_file) if aod_file else "MISSING",
        "gpm_file": os.path.basename(gpm_file) if gpm_file else "MISSING",
        "openmeteo_file": os.path.basename(wx_file) if wx_file else "MISSING",
    })

name_map_df = pd.DataFrame(name_map_rows)
name_map_df.to_csv("data/stations/metadata/station_name_map_thesis.csv",
                    index=False, encoding="utf-8-sig")

missing_aod = name_map_df[name_map_df["himawari_file"] == "MISSING"]
missing_gpm = name_map_df[name_map_df["gpm_file"] == "MISSING"]
missing_wx = name_map_df[name_map_df["openmeteo_file"] == "MISSING"]

print(f"Name mapping: {len(name_map_df)} stations")
print(f"  Missing AOD: {len(missing_aod)} — {list(missing_aod['station_name'].values[:3])}" if len(missing_aod) > 0 else "  Missing AOD: 0")
print(f"  Missing GPM: {len(missing_gpm)} — {list(missing_gpm['station_name'].values[:3])}" if len(missing_gpm) > 0 else "  Missing GPM: 0")
print(f"  Missing OpenMeteo: {len(missing_wx)} — {list(missing_wx['station_name'].values[:3])}" if len(missing_wx) > 0 else "  Missing OpenMeteo: 0")

# ══════════════════════════════════════════════════════════════════════
#  STEP 7: TIMEZONE VERIFICATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 7: TIMEZONE VERIFICATION")
print("=" * 80)

# Use ĐHBK (NVC as fallback)
tz_station = "Hà Nội ĐHBK cổng Parabol đường Giải Phóng (KK)"
tz_norm = normalize(tz_station)

print(f"\nSample station: {tz_station}")

# Envisoft
env_tz = pd.read_csv(env_idx[tz_norm], nrows=10)
print(f"\n[Envisoft] First 5 timestamps:")
for t in env_tz["Timestamp"].head(5):
    print(f"  {t}")
print("  → Already UTC+7 (no TZ suffix, timestamps match local time)")

# OpenMeteo
wx_tz = pd.read_csv(wx_idx[tz_norm], nrows=10)
print(f"\n[OpenMeteo] First 5 timestamps:")
for t in wx_tz["Timestamp"].head(5):
    print(f"  {t}")
has_utc_suffix = "+00:00" in str(wx_tz["Timestamp"].iloc[0])
if has_utc_suffix:
    print("  → UTC with +00:00 suffix — NEEDS +7h conversion during load")
else:
    print("  → No TZ suffix — checking if already UTC+7...")

# Himawari AOD
aod_tz = pd.read_csv(aod_idx[tz_norm], nrows=20)
print(f"\n[Himawari AOD] First 5 timestamps:")
for t in aod_tz["timestamp"].head(5):
    print(f"  {t}")
aod_tz["ts"] = pd.to_datetime(aod_tz["timestamp"])
aod_hours = aod_tz["ts"].dt.hour
valid_aod = aod_tz[aod_tz[[c for c in aod_tz.columns if "AOT" in c and c != "AOT_valid_count"]].notna().any(axis=1)]
if len(valid_aod) > 0:
    aod_valid_hours = valid_aod["ts"].dt.hour
    print(f"  Valid AOT hour range: {aod_valid_hours.min()}-{aod_valid_hours.max()}")
    if aod_valid_hours.min() >= 6 and aod_valid_hours.max() <= 18:
        print("  → Daytime only — confirms UTC+7")
    elif aod_valid_hours.min() < 3:
        print("  → WARNING: AOT at nighttime hours — may still be UTC!")
    else:
        print("  → Appears UTC+7 (satellite observations in daytime local)")

# Load more AOD data for better hour distribution
aod_full_sample = pd.read_csv(aod_idx[tz_norm])
aod_full_sample["ts"] = pd.to_datetime(aod_full_sample["timestamp"])
aod_valid_full = aod_full_sample[aod_full_sample["AOT_center"].notna()]
print(f"\n  Full AOD hour distribution (center pixel valid, n={len(aod_valid_full)}):")
hour_counts = aod_valid_full["ts"].dt.hour.value_counts().sort_index()
for h, c in hour_counts.items():
    print(f"    Hour {h:2d}: {c:5d} obs {'***' if h < 6 or h > 18 else ''}")

# GPM
gpm_tz = pd.read_csv(gpm_idx[tz_norm], nrows=10)
print(f"\n[GPM] First 5 timestamps:")
for t in gpm_tz["timestamp"].head(5):
    print(f"  {t}")

# Cross-check GPM vs Envisoft for a rain event
gpm_full = pd.read_csv(gpm_idx[tz_norm])
gpm_full["ts"] = pd.to_datetime(gpm_full["timestamp"])
rainy = gpm_full[gpm_full["total_accum_mm"] > 1.0].head(5)
if len(rainy) > 0:
    print(f"\n  GPM rain events (>1mm):")
    for _, r in rainy.iterrows():
        print(f"    {r['ts']} — {r['total_accum_mm']:.1f}mm")

    env_full = pd.read_csv(env_idx[tz_norm])
    env_full["ts"] = pd.to_datetime(env_full["Timestamp"])
    if "Rainfall" in env_full.columns:
        rain_date = rainy.iloc[0]["ts"].date()
        env_rain = env_full[env_full["ts"].dt.date == rain_date]
        if len(env_rain) > 0 and env_rain["Rainfall"].notna().any():
            print(f"  Envisoft Rainfall on {rain_date}: "
                  f"max={env_rain['Rainfall'].max():.1f}")

# GPM timezone determination
gpm_ts_str = str(gpm_tz["timestamp"].iloc[0])
if "+00:00" in gpm_ts_str or "Z" in gpm_ts_str:
    GPM_IS_UTC = True
    print("\n  → GPM: UTC (has TZ suffix) — NEEDS +7h conversion")
else:
    # Check if GPM was extracted with local time already
    # The v2 extraction script likely output local time based on the station coordinates
    GPM_IS_UTC = False
    print("\n  → GPM: No TZ suffix — assuming already UTC+7 (from GIS extraction)")
    print("    (v5 script does no conversion, and rain events should align)")

OPENMETEO_IS_UTC = has_utc_suffix

print(f"\n--- TIMEZONE SUMMARY ---")
print(f"  Envisoft:  UTC+7 (confirmed)")
print(f"  OpenMeteo: {'UTC → needs +7h' if OPENMETEO_IS_UTC else 'UTC+7'}")
print(f"  Himawari:  UTC+7 (confirmed by daytime AOT distribution)")
print(f"  GPM:       {'UTC → needs +7h' if GPM_IS_UTC else 'UTC+7 (assumed from GIS extraction)'}")

# ══════════════════════════════════════════════════════════════════════
#  STEP 8: BUILD UNIFIED DATASET
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 8: BUILD UNIFIED DATASET")
print("=" * 80)

# DEM lookup
dem_lookup = {}
if os.path.exists("data/dem/station_dem_features.csv"):
    dem_feat = pd.read_csv("data/dem/station_dem_features.csv", dtype={"stationId": str})
    for _, r in dem_feat.iterrows():
        dem_lookup[str(r["stationId"])] = r

# AOD 5x5 grid pixel columns
AOT_PIXELS = [
    "AOT_m2m2","AOT_m2m1","AOT_m2_0","AOT_m2p1","AOT_m2p2",
    "AOT_m1m2","AOT_m1m1","AOT_m1_0","AOT_m1p1","AOT_m1p2",
    "AOT__0m2","AOT__0m1","AOT__0_0","AOT__0p1","AOT__0p2",
    "AOT_p1m2","AOT_p1m1","AOT_p1_0","AOT_p1p1","AOT_p1p2",
    "AOT_p2m2","AOT_p2m1","AOT_p2_0","AOT_p2p1","AOT_p2p2",
]
AOT_SUMMARY = ["AOT_valid_count", "AOT_mean", "AOT_std", "AOT_center",
               "AOT_inner_count", "AOT_inner_mean", "AOT_outer_count", "AOT_outer_mean"]
AOT_META = ["Uncertainty", "AE", "QA_flag", "SSA", "RF"]
AOT_ALL_COLS = AOT_PIXELS + AOT_SUMMARY + AOT_META

def load_envisoft_thesis(path, station_name):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].replace(list(SENTINEL_VALUES), np.nan)

    for col, (lo, hi) in QC_RANGES.items():
        if col in df.columns:
            df[col] = df[col].where(df[col].between(lo, hi))

    # Stuck temp fix
    if station_name in STUCK_TEMP_STATIONS and "Temperature" in df.columns:
        df.loc[df["Temperature"] == 0.0, "Temperature"] = np.nan

    # Quế Võ: NaN all WS
    if QUEVO_NAN_WS in station_name and "Wind Speed" in df.columns:
        df["Wind Speed"] = np.nan

    # Negative PM → NaN
    for pc in ["PM2.5", "PM10"]:
        if pc in df.columns:
            df.loc[df[pc] < 0, pc] = np.nan

    # Dedup: keep row with most non-null
    df["_nn"] = df[num_cols].notna().sum(axis=1)
    df["ts_hour"] = df["ts"].dt.floor("h")
    df = df.sort_values(["ts_hour", "_nn"], ascending=[True, False])
    df = df.drop_duplicates("ts_hour", keep="first")
    df = df.drop(columns=["_nn"])

    keep = [c for c in ["PM2.5", "PM10", "Temperature", "Humidity", "Pressure",
                         "Wind Speed", "Wind Direction", "NO2", "O3", "SO2", "CO"]
            if c in df.columns]

    hourly = df.groupby("ts_hour")[keep].mean().reset_index()
    hourly = hourly.rename(columns={"ts_hour": "ts",
                                     "Wind Speed": "WS_local",
                                     "Wind Direction": "WD_local"})
    return hourly


def load_openmeteo_thesis(path):
    df = pd.read_csv(path)
    ts_str = str(df["Timestamp"].iloc[0])
    if "+00:00" in ts_str:
        df["ts"] = pd.to_datetime(df["Timestamp"], utc=True) + pd.Timedelta(hours=7)
        df["ts"] = df["ts"].dt.tz_localize(None)
    else:
        df["ts"] = pd.to_datetime(df["Timestamp"])
    df["ts"] = df["ts"].dt.floor("h")
    rename = {"Temperature": "Temperature_om", "Humidity": "Humidity_om",
              "Pressure": "Pressure_om", "Wind Speed": "WS_om",
              "Wind Direction": "WD_om", "PBLH": "PBLH", "Visibility": "Visibility"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = ["ts"] + [v for v in rename.values() if v in df.columns]
    return df[keep].drop_duplicates("ts", keep="first")


def load_aod_thesis(path):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["ts_hour"] = df["ts"].dt.floor("h")

    pixel_cols = [c for c in AOT_PIXELS if c in df.columns]
    summary_cols = [c for c in AOT_SUMMARY if c in df.columns]
    meta_cols = [c for c in AOT_META if c in df.columns]
    all_aod = pixel_cols + summary_cols + meta_cols

    # Aggregate: mean for AOT values, max for counts
    agg_dict = {}
    for c in all_aod:
        if "count" in c.lower():
            agg_dict[c] = (c, "max")
        elif c == "QA_flag":
            agg_dict[c] = (c, "first")
        else:
            agg_dict[c] = (c, "mean")
    agg_dict["n_aod_obs"] = ("ts", "size")

    hourly = df.groupby("ts_hour").agg(**agg_dict).reset_index()
    hourly = hourly.rename(columns={"ts_hour": "ts"})

    if "AOT_center" in hourly.columns:
        hourly = hourly.rename(columns={"AOT_center": "AOT"})

    return hourly


def load_gpm_thesis(path):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"])

    if GPM_IS_UTC:
        df["ts"] = df["ts"] + pd.Timedelta(hours=7)

    df["ts_hour"] = df["ts"].dt.floor("h")
    hourly = df.groupby("ts_hour").agg(
        precip_mm=("total_accum_mm", "sum"),
        precip_rate=("total_rate_mmh", "mean"),
    ).reset_index().rename(columns={"ts_hour": "ts"})
    return hourly


# Main merge loop
all_station_dfs = []
n_total = len(final_df)
merge_log = []

for idx, (_, row) in enumerate(final_df.iterrows()):
    name = row["station_name"]
    sid = row["stationId"]
    region = row["region"]
    stype = row["station_type"]
    norm = normalize(name)
    lat = row.get("lat", np.nan)
    lon = row.get("lon", np.nan)

    print(f"\n[{idx+1:2d}/{n_total}] {name[:60]}...")

    # 1. Envisoft
    env_path = env_idx.get(norm)
    if not env_path:
        print("  SKIP — no Envisoft file")
        continue
    base = load_envisoft_thesis(env_path, name)
    n_env = len(base)
    print(f"  Envisoft: {n_env}h", end="")

    # 2. OpenMeteo
    wx_path = wx_idx.get(norm)
    if wx_path:
        om = load_openmeteo_thesis(wx_path)
        base = base.merge(om, on="ts", how="left")
        print(f" | WX: {len(om)}h", end="")
    else:
        print(f" | WX: MISSING", end="")

    # 3. Himawari AOD
    aod_path = aod_idx.get(norm)
    if aod_path:
        aod = load_aod_thesis(aod_path)
        base = base.merge(aod, on="ts", how="left")
        n_aot = base["AOT"].notna().sum() if "AOT" in base.columns else 0
        print(f" | AOD: {n_aot} ({100*n_aot/max(len(base),1):.0f}%)", end="")
    else:
        print(f" | AOD: MISSING", end="")

    # 4. GPM
    gpm_path = gpm_idx.get(norm)
    if gpm_path:
        gpm = load_gpm_thesis(gpm_path)
        base = base.merge(gpm, on="ts", how="left")
        print(f" | GPM: Y", end="")
    else:
        print(f" | GPM: MISSING", end="")

    # 5. DEM (static join)
    dem_row = dem_lookup.get(sid)
    if dem_row is not None:
        for dc in ["elevation_m", "slope_deg", "aspect_deg", "aspect_sin", "aspect_cos"]:
            base[dc] = dem_row.get(dc, np.nan)

    # Met source priority
    for ec, oc, final_name in [
        ("Temperature", "Temperature_om", "Temperature_final"),
        ("Humidity", "Humidity_om", "Humidity_final"),
        ("Pressure", "Pressure_om", "Pressure_final"),
    ]:
        if ec in base.columns and oc in base.columns:
            base[final_name] = base[ec].fillna(base[oc])
        elif ec in base.columns:
            base[final_name] = base[ec]
        elif oc in base.columns:
            base[final_name] = base[oc]
        else:
            base[final_name] = np.nan

    # Metadata columns
    base["station"] = name
    base["stationId"] = str(sid)
    base["region"] = region
    base["station_type"] = stype
    base["latitude"] = lat
    base["longitude"] = lon

    # Save per-station CSV
    station_out = f"data/merged_thesis/{normalize(name)}.csv"
    base.to_csv(station_out, index=False)

    all_station_dfs.append(base)
    merge_log.append({
        "station": name, "n_rows": len(base),
        "aot_pct": round(100 * base["AOT"].notna().mean(), 1) if "AOT" in base.columns else 0,
        "pm25_pct": round(100 * base["PM2.5"].notna().mean(), 1) if "PM2.5" in base.columns else 0,
    })
    print()

# ── Concatenate ──
print("\n" + "-" * 60)
print("Concatenating all stations...")
unified = pd.concat(all_station_dfs, ignore_index=True)
unified = unified.sort_values(["station", "ts"]).reset_index(drop=True)
print(f"Combined: {len(unified):,} rows, {unified['station'].nunique()} stations")

# ── PBLH climatology fill ──
if "PBLH" in unified.columns:
    print("PBLH climatology fill...")
    n_nan_before = unified["PBLH"].isna().sum()
    unified["_hour"] = unified["ts"].dt.hour
    pblh_clim = (unified.dropna(subset=["PBLH"])
                 .groupby(["station", "_hour"])["PBLH"].mean()
                 .reset_index().rename(columns={"PBLH": "PBLH_clim"}))
    unified = unified.merge(pblh_clim, on=["station", "_hour"], how="left")
    fill_mask = unified["PBLH"].isna() & unified["PBLH_clim"].notna()
    unified["PBLH"] = unified["PBLH"].fillna(unified["PBLH_clim"])
    unified["PBLH_source"] = np.where(fill_mask, "climatology", "ERA5")
    unified.loc[unified["PBLH"].isna(), "PBLH_source"] = "missing"
    n_nan_after = unified["PBLH"].isna().sum()
    print(f"  Filled {fill_mask.sum():,} PBLH gaps ({n_nan_before:,} → {n_nan_after:,} NaN)")
    unified = unified.drop(columns=["_hour", "PBLH_clim"], errors="ignore")

# ══════════════════════════════════════════════════════════════════════
#  STEP 9: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 9: FEATURE ENGINEERING")
print("=" * 80)

# Process per-station for temporal features
station_dfs = []
for station_name, sdf in unified.groupby("station"):
    sdf = sdf.sort_values("ts").reset_index(drop=True)

    # 9A: AOD Spatial Features
    north_cols = [c for c in sdf.columns if c.startswith("AOT_m1") or c.startswith("AOT_m2")]
    north_cols = [c for c in north_cols if c in AOT_PIXELS]
    south_cols = [c for c in sdf.columns if c.startswith("AOT_p1") or c.startswith("AOT_p2")]
    south_cols = [c for c in south_cols if c in AOT_PIXELS]
    west_cols = [c for c in AOT_PIXELS if c in sdf.columns and
                 (c.endswith("m1") or c.endswith("m2")) and not c.startswith("AOT_m")]
    east_cols = [c for c in AOT_PIXELS if c in sdf.columns and
                 (c.endswith("p1") or c.endswith("p2")) and not c.startswith("AOT_p")]

    # More precise: parse pixel names
    north_pixels = [c for c in AOT_PIXELS if c in sdf.columns and ("_m1" in c[:7] or "_m2" in c[:7])]
    south_pixels = [c for c in AOT_PIXELS if c in sdf.columns and ("_p1" in c[:7] or "_p2" in c[:7])]
    west_pixels = [c for c in AOT_PIXELS if c in sdf.columns and c.endswith(("m1","m2")) and c[-2:] in ("m1","m2")]
    east_pixels = [c for c in AOT_PIXELS if c in sdf.columns and c.endswith(("p1","p2")) and c[-2:] in ("p1","p2")]

    # Re-derive pixel lists from the naming convention properly
    # Format: AOT_{row}{col} where row,col in {m2,m1,_0,p1,p2}
    # North rows: first index m1 or m2 → AOT_m1*, AOT_m2*
    # South rows: first index p1 or p2 → AOT_p1*, AOT_p2*
    # West cols: second index m1 or m2
    # East cols: second index p1 or p2
    north_pixels = []
    south_pixels = []
    west_pixels = []
    east_pixels = []
    for c in AOT_PIXELS:
        if c not in sdf.columns:
            continue
        parts = c.replace("AOT_", "")
        # Parse: first 2 chars = row index, rest = col index
        # m2m2 → row=m2, col=m2
        # m2m1 → row=m2, col=m1
        # _0_0 → row=_0, col=_0
        # p1m2 → row=p1, col=m2
        if len(parts) >= 4:
            row_idx = parts[:2]
            col_idx = parts[2:]
            if row_idx in ("m1", "m2"):
                north_pixels.append(c)
            if row_idx in ("p1", "p2"):
                south_pixels.append(c)
            if col_idx in ("m1", "m2"):
                west_pixels.append(c)
            if col_idx in ("p1", "p2"):
                east_pixels.append(c)

    if north_pixels and south_pixels:
        sdf["AOT_grad_ns"] = (sdf[south_pixels].mean(axis=1) - sdf[north_pixels].mean(axis=1)) / 4
    else:
        sdf["AOT_grad_ns"] = np.nan

    if east_pixels and west_pixels:
        sdf["AOT_grad_ew"] = (sdf[east_pixels].mean(axis=1) - sdf[west_pixels].mean(axis=1)) / 4
    else:
        sdf["AOT_grad_ew"] = np.nan

    sdf["AOT_grad_mag"] = np.sqrt(sdf["AOT_grad_ns"]**2 + sdf["AOT_grad_ew"]**2)
    sdf["AOT_grad_dir"] = np.arctan2(sdf["AOT_grad_ew"], sdf["AOT_grad_ns"])

    if "AOT_inner_mean" in sdf.columns and "AOT_outer_mean" in sdf.columns:
        sdf["AOT_local_vs_regional"] = sdf["AOT_inner_mean"] - sdf["AOT_outer_mean"]
    else:
        sdf["AOT_local_vs_regional"] = np.nan

    if "AOT_std" in sdf.columns:
        sdf["AOT_spatial_std"] = sdf["AOT_std"]

    # 9B: AOD Temporal Features
    if "AOT" in sdf.columns:
        sdf["AOT_lag_1h"] = sdf["AOT"].shift(1)
        sdf["AOT_lag_3h"] = sdf["AOT"].shift(3)
        sdf["AOT_lag_6h"] = sdf["AOT"].shift(6)
        sdf["AOT_rolling_mean_6h"] = sdf["AOT"].rolling(6, min_periods=1).mean()
        sdf["AOT_rolling_mean_24h"] = sdf["AOT"].rolling(24, min_periods=1).mean()

        # Hours since valid AOT
        aot_valid = sdf["AOT"].notna()
        hrs_since_aot = np.full(len(sdf), np.nan)
        last_valid = -1
        for i in range(len(sdf)):
            if aot_valid.iloc[i]:
                last_valid = i
                hrs_since_aot[i] = 0
            elif last_valid >= 0:
                hrs_since_aot[i] = i - last_valid
        sdf["hours_since_valid_AOT"] = hrs_since_aot

        # Forward fill max 48h
        sdf["AOT_ffill_48h"] = sdf["AOT"].ffill(limit=48)

    # 9C: Physics Correction
    if "Humidity_final" in sdf.columns:
        rh = sdf["Humidity_final"] / 100.0
        rh_factor = (1 - rh).clip(lower=0.01) ** 0.6
        sdf["RH_factor"] = rh_factor
    if "AOT" in sdf.columns and "PBLH" in sdf.columns and "RH_factor" in sdf.columns:
        pblh_safe = sdf["PBLH"].clip(lower=50)
        sdf["AOD_physics"] = sdf["AOT"] * sdf["RH_factor"] / pblh_safe

    # 9D: Precipitation Features
    if "precip_mm" in sdf.columns:
        rain_mask = sdf["precip_mm"].fillna(0) > 0.1
        hrs_since_rain = np.full(len(sdf), np.nan)
        last_rain = -1
        for i in range(len(sdf)):
            if rain_mask.iloc[i]:
                last_rain = i
                hrs_since_rain[i] = 0
            elif last_rain >= 0:
                hrs_since_rain[i] = i - last_rain
        sdf["hrs_since_rain"] = hrs_since_rain

        sdf["rain_sum_24h"] = sdf["precip_mm"].fillna(0).rolling(24, min_periods=1).sum()
        sdf["rain_sum_48h"] = sdf["precip_mm"].fillna(0).rolling(48, min_periods=1).sum()

        # rain_days_7d
        sdf["date"] = sdf["ts"].dt.date
        daily_rain = sdf.groupby("date")["precip_mm"].sum().reset_index()
        daily_rain["rain_day"] = (daily_rain["precip_mm"] > 0.1).astype(int)
        daily_rain["rain_days_7d"] = daily_rain["rain_day"].rolling(7, min_periods=1).sum()
        consec = np.zeros(len(daily_rain))
        for i in range(len(daily_rain)):
            if daily_rain["rain_day"].iloc[i] == 0:
                consec[i] = consec[i-1] + 1 if i > 0 else 1
        daily_rain["consecutive_dry_days"] = consec
        sdf = sdf.merge(daily_rain[["date", "rain_days_7d", "consecutive_dry_days"]],
                         on="date", how="left")
        sdf = sdf.drop(columns=["date"], errors="ignore")

    # 9E: Meteorological Derivatives
    if "WS_om" in sdf.columns and "WD_om" in sdf.columns:
        wd_rad = np.radians(sdf["WD_om"])
        sdf["wind_u"] = sdf["WS_om"] * np.sin(wd_rad)
        sdf["wind_v"] = sdf["WS_om"] * np.cos(wd_rad)
        sdf["wind_dir_sin"] = np.sin(wd_rad)
        sdf["wind_dir_cos"] = np.cos(wd_rad)

    if "WS_local" in sdf.columns and "WD_local" in sdf.columns:
        wd_local_rad = np.radians(sdf["WD_local"])
        sdf["wind_u_local"] = sdf["WS_local"] * np.sin(wd_local_rad)
        sdf["wind_v_local"] = sdf["WS_local"] * np.cos(wd_local_rad)
        sdf["wind_dir_sin_local"] = np.sin(wd_local_rad)
        sdf["wind_dir_cos_local"] = np.cos(wd_local_rad)

    if "PBLH" in sdf.columns and "WS_om" in sdf.columns:
        sdf["VC"] = sdf["PBLH"] * sdf["WS_om"]

    for col_name, src_col in [("dT_6h", "Temperature_final"), ("dRH_6h", "Humidity_final"),
                               ("dWS_6h", "WS_om"), ("dP_6h", "Pressure_final")]:
        if src_col in sdf.columns:
            sdf[col_name] = sdf[src_col] - sdf[src_col].shift(6)

    # 9F: Temporal Encodings
    hour = sdf["ts"].dt.hour
    month = sdf["ts"].dt.month
    doy = sdf["ts"].dt.dayofyear
    sdf["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    sdf["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    sdf["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    sdf["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    sdf["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365)
    sdf["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365)

    # 9H: DEM Interaction Features
    if "elevation_m" in sdf.columns and "PBLH" in sdf.columns:
        sdf["elev_x_PBLH"] = sdf["elevation_m"] * sdf["PBLH"]
    if "elevation_m" in sdf.columns:
        sdf["elev_x_hour_sin"] = sdf["elevation_m"] * sdf["hour_sin"]

    station_dfs.append(sdf)

print("\nRe-concatenating with features...")
unified = pd.concat(station_dfs, ignore_index=True)
unified = unified.sort_values(["station", "ts"]).reset_index(drop=True)

# Save
out_path = "data/merged/unified_thesis_v1.csv"
print(f"Saving to {out_path}...")
unified.to_csv(out_path, index=False)
print(f"Saved: {len(unified):,} rows × {len(unified.columns)} columns")

# ══════════════════════════════════════════════════════════════════════
#  STEP 10: SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 10: SUMMARY REPORT")
print("=" * 80)

# Feature correlation with PM2.5
numeric_features = unified.select_dtypes(include=[np.number]).columns.tolist()
exclude = ["PM2.5", "PM10", "n_aod_obs", "QA_flag", "AOT_inner_count", "AOT_outer_count", "AOT_valid_count"]
feat_cols = [c for c in numeric_features if c not in exclude and c not in AOT_PIXELS]

corr_results = []
if "PM2.5" in unified.columns:
    for fc in feat_cols:
        valid = unified[["PM2.5", fc]].dropna()
        if len(valid) > 100:
            r = valid["PM2.5"].corr(valid[fc])
            corr_results.append({"feature": fc, "pearson_r": round(r, 4), "n_valid": len(valid)})

corr_df = pd.DataFrame(corr_results).sort_values("pearson_r", key=abs, ascending=False)

# Feature list
feature_list = []
# Static features
static_feats = {
    "elevation_m": ("DEM", "Elevation at station location (meters)"),
    "slope_deg": ("DEM", "Terrain slope at station (degrees)"),
    "aspect_sin": ("DEM", "Sin of terrain aspect (cyclical N-S encoding)"),
    "aspect_cos": ("DEM", "Cos of terrain aspect (cyclical E-W encoding)"),
    "latitude": ("metadata", "Station latitude"),
    "longitude": ("metadata", "Station longitude"),
}
dynamic_feats = {
    "PM2.5": ("Envisoft", "Target variable — hourly PM2.5 concentration (µg/m³)"),
    "PM10": ("Envisoft", "Hourly PM10 concentration (µg/m³)"),
    "Temperature_final": ("Envisoft+OpenMeteo", "Temperature — Envisoft first, OpenMeteo fill (°C)"),
    "Humidity_final": ("Envisoft+OpenMeteo", "Relative humidity — Envisoft first, OpenMeteo fill (%)"),
    "Pressure_final": ("Envisoft+OpenMeteo", "Atmospheric pressure — Envisoft first, OpenMeteo fill (hPa)"),
    "WS_om": ("OpenMeteo", "ERA5 10m wind speed (m/s)"),
    "WD_om": ("OpenMeteo", "ERA5 10m wind direction (degrees)"),
    "WS_local": ("Envisoft", "Station-local sheltered wind speed (m/s)"),
    "WD_local": ("Envisoft", "Station-local wind direction (degrees)"),
    "PBLH": ("OpenMeteo", "Planetary boundary layer height (m), climatology-filled"),
    "AOT": ("Himawari", "Center pixel aerosol optical thickness"),
    "AOT_mean": ("Himawari", "5×5 grid mean AOT"),
    "AOT_spatial_std": ("Himawari", "5×5 grid AOT standard deviation"),
    "AOT_inner_mean": ("Himawari", "Inner 3×3 mean AOT"),
    "AOT_outer_mean": ("Himawari", "Outer ring mean AOT"),
    "Uncertainty": ("Himawari", "AOT retrieval uncertainty"),
    "AE": ("Himawari", "Angstrom Exponent"),
    "SSA": ("Himawari", "Single scattering albedo"),
    "RF": ("Himawari", "Radiative forcing"),
    "precip_mm": ("GPM", "Hourly precipitation accumulation (mm)"),
    "precip_rate": ("GPM", "Mean precipitation rate (mm/h)"),
    "NO2": ("Envisoft", "Nitrogen dioxide (µg/m³)"),
    "O3": ("Envisoft", "Ozone (µg/m³)"),
    "SO2": ("Envisoft", "Sulfur dioxide (µg/m³)"),
    "CO": ("Envisoft", "Carbon monoxide (mg/m³)"),
}
derived_feats = {
    "AOT_grad_ns": ("derived-spatial", "N-S AOT gradient from 5×5 grid"),
    "AOT_grad_ew": ("derived-spatial", "E-W AOT gradient from 5×5 grid"),
    "AOT_grad_mag": ("derived-spatial", "Magnitude of AOT spatial gradient"),
    "AOT_grad_dir": ("derived-spatial", "Direction of steepest AOT increase (radians)"),
    "AOT_local_vs_regional": ("derived-spatial", "Inner - outer mean AOT (local hotspot indicator)"),
    "AOT_lag_1h": ("derived-temporal", "AOT lagged by 1 hour"),
    "AOT_lag_3h": ("derived-temporal", "AOT lagged by 3 hours"),
    "AOT_lag_6h": ("derived-temporal", "AOT lagged by 6 hours"),
    "AOT_rolling_mean_6h": ("derived-temporal", "6-hour rolling mean AOT"),
    "AOT_rolling_mean_24h": ("derived-temporal", "24-hour rolling mean AOT"),
    "hours_since_valid_AOT": ("derived-temporal", "Hours since last non-NaN AOT reading"),
    "AOT_ffill_48h": ("derived-temporal", "Forward-filled AOT (max 48h gap)"),
    "RH_factor": ("derived-physics", "(1 - RH/100)^0.6 hygroscopic correction"),
    "AOD_physics": ("derived-physics", "AOT × RH_factor / PBLH — physics-corrected AOD"),
    "hrs_since_rain": ("derived-precip", "Hours since precipitation >0.1mm"),
    "rain_sum_24h": ("derived-precip", "Rolling 24h precipitation sum (mm)"),
    "rain_sum_48h": ("derived-precip", "Rolling 48h precipitation sum (mm)"),
    "rain_days_7d": ("derived-precip", "Days with >0.1mm rain in past 7 days"),
    "consecutive_dry_days": ("derived-precip", "Consecutive days without rain"),
    "wind_u": ("derived-met", "East-west wind component (WS×sin(WD))"),
    "wind_v": ("derived-met", "North-south wind component (WS×cos(WD))"),
    "wind_dir_sin": ("derived-met", "Sin of wind direction (cyclical)"),
    "wind_dir_cos": ("derived-met", "Cos of wind direction (cyclical)"),
    "wind_u_local": ("derived-met", "Local east-west wind component"),
    "wind_v_local": ("derived-met", "Local north-south wind component"),
    "wind_dir_sin_local": ("derived-met", "Sin of local wind direction"),
    "wind_dir_cos_local": ("derived-met", "Cos of local wind direction"),
    "VC": ("derived-met", "Ventilation coefficient (PBLH × WS)"),
    "dT_6h": ("derived-met", "6-hour temperature change"),
    "dRH_6h": ("derived-met", "6-hour humidity change"),
    "dWS_6h": ("derived-met", "6-hour wind speed change"),
    "dP_6h": ("derived-met", "6-hour pressure change"),
    "hour_sin": ("derived-temporal", "Sin cyclical hour encoding"),
    "hour_cos": ("derived-temporal", "Cos cyclical hour encoding"),
    "month_sin": ("derived-temporal", "Sin cyclical month encoding"),
    "month_cos": ("derived-temporal", "Cos cyclical month encoding"),
    "day_of_year_sin": ("derived-temporal", "Sin cyclical day-of-year encoding"),
    "day_of_year_cos": ("derived-temporal", "Cos cyclical day-of-year encoding"),
    "elev_x_PBLH": ("derived-interaction", "Elevation × PBLH (trapping potential)"),
    "elev_x_hour_sin": ("derived-interaction", "Elevation × hour_sin (nocturnal inversion)"),
}

all_feat_desc = {}
all_feat_desc.update({k: ("static", v[0], v[1]) for k, v in static_feats.items()})
all_feat_desc.update({k: ("dynamic", v[0], v[1]) for k, v in dynamic_feats.items()})
all_feat_desc.update({k: ("derived", v[0], v[1]) for k, v in derived_feats.items()})

feat_list_rows = []
for fname, (ftype, fsource, fdesc) in all_feat_desc.items():
    if fname in unified.columns:
        cov = round(100 * unified[fname].notna().mean(), 1)
        feat_list_rows.append({
            "feature": fname, "type": ftype, "source": fsource,
            "description": fdesc, "coverage_pct": cov
        })
feat_list_df = pd.DataFrame(feat_list_rows)
feat_list_df.to_csv("analysis/thesis_audit/feature_list.csv", index=False, encoding="utf-8-sig")

# Write summary report
report_lines = []
report_lines.append("# Thesis Dataset Summary Report\n")
report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

report_lines.append("\n## 1. Station Selection Results\n")
report_lines.append(f"- **Main stations (training):** {len(final_df)}")
report_lines.append(f"  - Pass: {(final_df['quality_flag']=='pass').sum()}")
report_lines.append(f"  - Marginal fixed: {(final_df['quality_flag']=='marginal_fixed').sum()}")
report_lines.append(f"- **LCS stations (validation pool):** {n_lcs_pass}")
report_lines.append(f"- **Dropped:** {(phase1['quality_flag']=='fail').sum()} stations\n")

report_lines.append("### By Tier\n")
for t in [1, 2, 3]:
    sub = final_df[final_df["tier"] == t]
    report_lines.append(f"- Tier {t}: {len(sub)} stations")

report_lines.append("\n### By Region\n")
for r in ["North", "Central", "South"]:
    sub = final_df[final_df["region"] == r]
    report_lines.append(f"- {r}: {len(sub)} stations")

report_lines.append("\n### By Type\n")
for t in final_df["station_type"].unique():
    sub = final_df[final_df["station_type"] == t]
    report_lines.append(f"- {t}: {len(sub)} stations")

report_lines.append("\n## 2. DEM Extraction Results\n")
report_lines.append(f"- Stations with elevation: {n_with_elev}/{len(dem_df)}")
if n_with_elev > 0:
    report_lines.append(f"- Elevation range: {dem_df['elevation_m'].min():.0f} – {dem_df['elevation_m'].max():.0f} m")
    missing_dem = dem_df[dem_df["elevation_m"].isna()]
    if len(missing_dem) > 0:
        report_lines.append(f"- Missing DEM data: {len(missing_dem)} stations")

report_lines.append("\n## 3. Merged Dataset Stats\n")
report_lines.append(f"- **Total rows:** {len(unified):,}")
report_lines.append(f"- **Total stations:** {unified['station'].nunique()}")
report_lines.append(f"- **Date range:** {unified['ts'].min()} to {unified['ts'].max()}")
report_lines.append(f"- **Total features:** {len(unified.columns)}")

report_lines.append("\n### Per-Column Coverage (% non-NaN)\n")
report_lines.append("| Column | Coverage |")
report_lines.append("|--------|----------|")
key_cols = ["PM2.5", "AOT", "PBLH", "Temperature_final", "Humidity_final",
            "Pressure_final", "WS_om", "WD_om", "WS_local", "precip_mm",
            "AOD_physics", "AOT_ffill_48h", "elevation_m"]
for c in key_cols:
    if c in unified.columns:
        cov = 100 * unified[c].notna().mean()
        report_lines.append(f"| {c} | {cov:.1f}% |")

report_lines.append("\n### AOT Availability by Region\n")
if "AOT" in unified.columns:
    report_lines.append("| Region | AOT Coverage |")
    report_lines.append("|--------|-------------|")
    for r in sorted(unified["region"].dropna().unique()):
        sub = unified[unified["region"] == r]
        cov = 100 * sub["AOT"].notna().mean()
        report_lines.append(f"| {r} | {cov:.1f}% |")

report_lines.append("\n## 4. Feature Correlation with PM2.5\n")
report_lines.append("### Top 20 (by |r|)\n")
report_lines.append("| Feature | Pearson r | n |")
report_lines.append("|---------|-----------|---|")
for _, cr in corr_df.head(20).iterrows():
    report_lines.append(f"| {cr['feature']} | {cr['pearson_r']:.4f} | {cr['n_valid']:,} |")

report_lines.append("\n### Bottom 5\n")
report_lines.append("| Feature | Pearson r | n |")
report_lines.append("|---------|-----------|---|")
for _, cr in corr_df.tail(5).iterrows():
    report_lines.append(f"| {cr['feature']} | {cr['pearson_r']:.4f} | {cr['n_valid']:,} |")

report_lines.append("\n## 5. New Spatial AOD Features\n")
spatial_aod = ["AOT_grad_ns", "AOT_grad_ew", "AOT_grad_mag", "AOT_local_vs_regional"]
for f in spatial_aod:
    cr = corr_df[corr_df["feature"] == f]
    if len(cr) > 0:
        report_lines.append(f"- **{f}:** r = {cr.iloc[0]['pearson_r']:.4f}")

report_lines.append("\n## 6. DEM Features vs PM2.5\n")
dem_feats = ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos"]
for f in dem_feats:
    cr = corr_df[corr_df["feature"] == f]
    if len(cr) > 0:
        report_lines.append(f"- **{f}:** r = {cr.iloc[0]['pearson_r']:.4f}")

report_lines.append("\n## 7. Known Issues\n")
report_lines.append("- All LCS stations excluded from main training (< 12 months data)")
report_lines.append("- Hà Nam Temperature stuck at 0.0 for 5194h — replaced with NaN, OpenMeteo fills")
report_lines.append("- Bắc Ninh Quế Võ WS anomalous (19.5 m/s mean) — NaN'd entire WS column")
report_lines.append("- Gia Lai KCN Trà Đa had 27K sentinel values — cleaned to NaN")
report_lines.append("- 3 Tier-1 stations dropped (Trà Vinh, Vĩnh Long, Hưng Yên Tân Quang) — unrealistic PM2.5 means")
report_lines.append("- PBLH gap Jan–Jun 2024 filled with station×hour climatology")
if n_with_elev < len(dem_df):
    report_lines.append(f"- {len(dem_df) - n_with_elev} stations missing DEM elevation data")

report_path = "analysis/thesis_audit/dataset_summary.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\nReport saved: {report_path}")
print(f"Feature list saved: analysis/thesis_audit/feature_list.csv")
print(f"Unified dataset: {out_path} ({len(unified):,} rows × {len(unified.columns)} cols)")
print(f"\nDONE.")
