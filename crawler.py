import requests
import pandas as pd
import time
import os
from datetime import datetime

CSV_FILE = "air_quality_weather_data.csv"

def get_all_stations():
    """Fetches every available station ID from Envisoft's map service."""
    url = "https://envisoft.gov.vn/eos/services/call/json/get_all_station_map_v2"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            stations = response.json().get('stations', [])
            return [s['station_id'] for s in stations if 'station_id' in s]
    except Exception as e:
        print(f"Error fetching station list: {e}")
    return []

def fetch_air_quality(station_id):
    url = f"https://envisoft.gov.vn/eos/services/call/json/qi_detail_for_eip?station_id={station_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return None
        data = response.json()
        res = data.get('res', {})
        return {
            "Timestamp": data.get('qi_time_2'),
            "Name": data.get('station_name'),
            "AQI": data.get('qi_value'),
            "PM2.5": res.get('PM-2-5', {}).get('current'),
            "PM10": res.get('PM-10', {}).get('current'),
            "CO": res.get('CO', {}).get('current'),
            "NO2": res.get('NO2', {}).get('current'),
            "O3": res.get('O3', {}).get('current'),
            "SO2": res.get('SO2', {}).get('current'),
        }
    except: return None

def crawl_all_data():
    # Load existing data or create new
    if os.path.exists(CSV_FILE):
        df_existing = pd.read_csv(CSV_FILE)
    else:
        df_existing = pd.DataFrame()

    ids = get_all_stations()
    print(f"Found {len(ids)} stations. Starting crawl...")
    
    new_records = []
    for sid in ids:
        data = fetch_air_quality(sid)
        if data and data["Name"] and data["Timestamp"]:
            # Basic duplicate check in current run
            new_records.append(data)
            print(f"Collected: {data['Name']}")
        time.sleep(0.5) # Gentle crawling

    if new_records:
        df_new = pd.DataFrame(new_records)
        df_final = pd.concat([df_existing, df_new]).drop_duplicates(subset=['Name', 'Timestamp'])
        df_final.to_csv(CSV_FILE, index=False, encoding="utf-8")
        print("CSV updated successfully.")

if __name__ == "__main__":
    crawl_all_data()