import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Input: Raw telemetry from crawler.py
# Output: Standardized reference dataset for multivariate correlation analysis [cite: 381]
INPUT_FILE = "data/merged/air_quality_weather_data.csv"
OUTPUT_FILE = "data/merged/filtered_envisoft_air_quality_data.csv"

def filter_and_normalize():
    """
    Applies statistical integrity protocols to Envisoft data to remove sensor 
    artifacts and ensure range consistency before modeling[cite: 156, 157].
    """
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    # 1. Source Attribution [Section 2.1.3]
    # Assign 'gov' tag to maintain distinction from IQAir aggregator data during 
    # cross-source validation steps[cite: 70, 304].
    df['Source'] = 'gov'

    # 2. Schema Alignment
    # Normalizes columns to a common schema including meteorological catalysts 
    # required for analyzing the North-South dichotomy[cite: 57, 71, 784].
    schema = [
        "Source", "Id", "Timestamp", "Name", "Latitude", "Longitude", 
        "AQI", "PM2.5", "PM10", "CO", "NO2", "O3", "SO2",
        "Temperature", "Humidity", "Pressure", "Wind Speed"
    ]
    
    # Ensure all columns are present, padding missing telemetry with NaN [cite: 165]
    for col in schema:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[schema]

    # 3. Statistical Integrity Filtering [Section 3.2]
    # Protocol A: Range Consistency [cite: 168, 171]
    # Removes physically impossible negative concentrations to prevent model bias.
    numeric_cols = ["AQI", "PM2.5", "PM10", "CO", "NO2", "O3", "SO2"]
    for col in numeric_cols:
        df.loc[df[col] < 0, col] = np.nan

    # Protocol B: Artifact Removal (Flat-lining) [cite: 169]
    # Detects 'data logger freezing' where values remain mathematically constant 
    # for > 3 hours, a common failure in telemetry networks.
    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
    df = df.sort_values(['Name', 'Timestamp_dt'])

    def remove_flatlines(group):
        for col in numeric_cols:
            # Identity sequence detection
            is_same = group[col] == group[col].shift(1)
            # Flag sequences persisting beyond the 3-hour threshold [cite: 169]
            streak = is_same.groupby((is_same != is_same.shift()).cumsum()).cumsum()
            group.loc[streak >= 3, col] = np.nan
        return group

    # Apply per-station artifact filtering to preserve regional signals [cite: 41, 106]
    df = df.groupby('Name', group_keys=False).apply(remove_flatlines)
    
    # Remove helper columns before final persistence
    df = df.drop(columns=['Timestamp_dt'])

    # 4. Final Data Export
    # Saves clean, research-ready data for multivariate correlation checks[cite: 166, 517].
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"✅ Normalized and Filtered data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    filter_and_normalize()