"""Rebuild the canonical unified_thesis.csv with ERA5-only meteorology.

Every meteorological feature becomes gridded (ERA5/GPM), so the model has NO
in-situ-only dependency; the only station-derived features remain RFSI (neighbour
PM2.5). This is the deployable feature policy for the thesis.

Reads the in-situ backup and writes the canonical unified_thesis.csv. The in-situ
original is preserved as the backup, and every pipeline script reads the unified
table normally.

Transforms (formulas match build_unified.py so the dataset stays self-consistent):
 - overwrite in-situ-preferred met with ERA5 (_om): Temperature/_final, Humidity/_final,
   Pressure/_final, WS_local, WD_local
 - recompute 6h deltas (dT_6h, dRH_6h, dP_6h) per station from the ERA5 sources
 - recompute RH_factor = ((1-RH).clip(0.01))**0.6 and AOD_physics = AOT*RH_factor/PBLH
 - recompute local wind components (now ERA5-based)
Already-ERA5 features (PBLH, VC=PBLH*WS_om, wind_u/v from WS_om, dWS_6h, GPM rain) untouched.
"""
import os, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "data", "merged", "unified_thesis_insitu.csv")
DST = os.path.join(ROOT, "data", "merged", "unified_thesis.csv")

print(f"loading {SRC} ...")
df = pd.read_csv(SRC, dtype={"stationId": str}, parse_dates=["ts"])
df = df.sort_values(["stationId", "ts"]).reset_index(drop=True)
print(f"  {len(df):,} rows, {df['stationId'].nunique()} stations")

# 1. overwrite in-situ / in-situ-preferred met columns with ERA5 (_om)
swaps = [("Temperature_final", "Temperature_om"), ("Humidity_final", "Humidity_om"),
         ("Pressure_final", "Pressure_om"), ("Temperature", "Temperature_om"),
         ("Humidity", "Humidity_om"), ("Pressure", "Pressure_om"),
         ("WS_local", "WS_om"), ("WD_local", "WD_om")]
done = []
for dst, src in swaps:
    if dst in df.columns and src in df.columns:
        df[dst] = df[src].values
        done.append(dst)
print("  overwrote with ERA5:", done)

# 2. recompute 6h deltas per station from the ERA5 sources
for col, src in [("dT_6h", "Temperature_final"), ("dRH_6h", "Humidity_final"), ("dP_6h", "Pressure_final")]:
    if col in df.columns and src in df.columns:
        df[col] = df.groupby("stationId")[src].diff(6)

# 3. RH_factor + AOD_physics from ERA5 humidity (dataset formula)
if "Humidity_final" in df.columns:
    df["RH_factor"] = ((1.0 - df["Humidity_final"] / 100.0).clip(lower=0.01)) ** 0.6
if all(c in df.columns for c in ["AOT", "PBLH", "RH_factor"]):
    df["AOD_physics"] = df["AOT"] * df["RH_factor"] / df["PBLH"].clip(lower=50)

# 4. local wind components (now ERA5-based since WS_local=WS_om, WD_local=WD_om)
if "WS_local" in df.columns and "WD_local" in df.columns:
    wr = np.radians(df["WD_local"])
    for c, val in [("wind_u_local", df["WS_local"] * np.sin(wr)), ("wind_v_local", df["WS_local"] * np.cos(wr)),
                   ("wind_dir_sin_local", np.sin(wr)), ("wind_dir_cos_local", np.cos(wr))]:
        if c in df.columns:
            df[c] = val

print(f"writing {DST} ...")
df.to_csv(DST, index=False)
print(f"done: {len(df):,} rows -> {DST}")
