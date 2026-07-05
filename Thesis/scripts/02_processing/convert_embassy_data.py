"""
convert_embassy_data.py

Converts US Embassy/Consulate PM2.5 data into the station_historical_full format
and writes the result to that folder so it can be used in all analyses alongside
the Vietnamese monitoring station data.

Input:  data/embassy/HaNoi/*.csv
        data/embassy/HCM/*.csv
Output: data/stations/historical_full/US Embassy Hanoi.csv
        data/stations/historical_full/US Consulate HCMC.csv
"""

from pathlib import Path
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parents[3]
EMBS_DIR  = BASE_DIR / 'data' / 'embassy'
OUT_DIR   = BASE_DIR / 'data' / 'stations' / 'historical_full'

STATIONS = [
    {
        'name':       'US Embassy Hanoi',
        'src_folder': EMBS_DIR / 'HaNoi',
        'out_file':   OUT_DIR / 'US Embassy Hanoi.csv',
    },
    {
        'name':       'US Consulate HCMC',
        'src_folder': EMBS_DIR / 'HCM',
        'out_file':   OUT_DIR / 'US Consulate HCMC.csv',
    },
]

# Target column order (matches data/stations/historical_full files)
OUTPUT_COLS = [
    'Timestamp', 'Record_ID', 'PM2.5', 'PM10', 'CO', 'NO2',
    'Temperature', 'Humidity', 'Wind Speed', 'Wind Direction',
    'Pressure', 'Radiation', 'Detailed_Status',
]


# ── Core conversion ───────────────────────────────────────────────────────────

def load_embassy_files(folder: Path) -> pd.DataFrame:
    """Read and concatenate all annual CSV files from an embassy folder."""
    files = sorted(folder.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No CSV files found in {folder}')

    chunks = []
    for f in files:
        df = pd.read_csv(f, dtype=str)
        chunks.append(df)
        print(f'  Loaded {f.name} ({len(df):,} rows)')

    return pd.concat(chunks, ignore_index=True)


def convert(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert embassy-format DataFrame to data/stations/historical_full format."""

    # Parse timestamp — format: '2024-01-01 01:00 AM'
    ts = pd.to_datetime(raw['Date (LT)'], format='%Y-%m-%d %I:%M %p', errors='coerce')
    out = pd.DataFrame()
    out['Timestamp'] = ts.dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Embassy data has no record ID
    out['Record_ID'] = ''

    # PM2.5: use Raw Conc., blank out Invalid rows and -999 sentinel
    raw_conc = pd.to_numeric(raw['Raw Conc.'], errors='coerce')
    is_invalid = (raw['QC Name'].str.strip().str.lower() != 'valid') | (raw_conc < 0)
    out['PM2.5'] = raw_conc.where(~is_invalid)

    # Columns that embassy data doesn't provide — leave empty
    for col in ['PM10', 'CO', 'NO2', 'Temperature', 'Humidity',
                'Wind Speed', 'Wind Direction', 'Pressure', 'Radiation']:
        out[col] = ''

    # Status: map Valid → Done (matches existing station files), else keep QC name
    out['Detailed_Status'] = raw['QC Name'].str.strip().apply(
        lambda q: 'Done' if q.lower() == 'valid' else q
    )

    return out[OUTPUT_COLS]


def process_station(cfg: dict):
    print(f"\n{'─' * 60}")
    print(f"Station: {cfg['name']}")
    print(f"Source:  {cfg['src_folder']}")

    raw = load_embassy_files(cfg['src_folder'])
    out = convert(raw)

    # Sort by time, drop exact duplicates (monthly files overlap yearly files)
    out = out.sort_values('Timestamp').drop_duplicates(subset='Timestamp').reset_index(drop=True)

    # Stats
    total   = len(out)
    valid   = out['PM2.5'].notna().sum()
    invalid = total - valid
    date_min = out['Timestamp'].min()
    date_max = out['Timestamp'].max()

    print(f"Rows:    {total:,}  (valid PM2.5: {valid:,} | invalid/missing: {invalid:,})")
    print(f"Period:  {date_min}  →  {date_max}")

    out.to_csv(cfg['out_file'], index=False)
    print(f"Saved → {cfg['out_file']}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for station_cfg in STATIONS:
        process_station(station_cfg)

    print(f"\n{'─' * 60}")
    print("Done. Both stations written to data/stations/historical_full/")
