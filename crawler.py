import requests
import pandas as pd
import time
import os

CSV_FILE = "air_quality_weather_data.csv"

def get_all_stations():
    """Fetches all Envisoft station metadata including coordinates."""
    url = "https://envisoft.gov.vn/eos/services/call/json/get_all_station_map_v2"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.json().get('stations', [])
    except Exception as e:
        print(f"Error fetching station list: {e}")
    return []

def fetch_envisoft_data(station_id):
    """Fetches AQI and any available weather data directly from the station."""
    url = f"https://envisoft.gov.vn/eos/services/call/json/qi_detail_for_eip?station_id={station_id}"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200: return None
        data = response.json()
        res = data.get('res', {})
        
        # Mapping possible Envisoft weather keys
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
            # Attempt to grab internal station weather if it exists
            "Env_Wind_Speed": res.get('WS', {}).get('current'),
            "Env_Wind_Dir": res.get('WindDir', {}).get('current') or res.get('WD', {}).get('current')
        }
    except: return None

def fetch_open_meteo(lat, lon):
    """Fallback: Keyless weather fetch for Wind Speed and Direction."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m,surface_pressure"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return {}
        curr = response.json().get('current', {})
        return {
            "Temp": curr.get('temperature_2m'),
            "Humidity": curr.get('relative_humidity_2m'),
            "Pressure": curr.get('surface_pressure'),
            "Wind_Speed": curr.get('wind_speed_10m'),
            "Wind_Dir": curr.get('wind_direction_10m') # 0-360 degrees
        }
    except: return {}

def crawl_all_data():
    df_existing = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame()
    stations = get_all_stations()
    print(f"Found {len(stations)} stations. Starting Hybrid Crawl...")
    
    new_records = []
    for s in stations:
        sid = s.get('station_id')
        lat, lon = s.get('map_lat'), s.get('map_lng')
        
        aqi_data = fetch_envisoft_data(sid)
        if aqi_data and aqi_data["Name"] and aqi_data["Timestamp"]:
            # If Envisoft lacks weather, use Open-Meteo fallback
            if aqi_data.get("Env_Wind_Dir") is None:
                weather = fetch_open_meteo(lat, lon)
                aqi_data.update(weather)
            else:
                # Use Envisoft's own weather values
                aqi_data["Wind_Speed"] = aqi_data.get("Env_Wind_Speed")
                aqi_data["Wind_Dir"] = aqi_data.get("Env_Wind_Dir")
            
            aqi_data["Latitude"], aqi_data["Longitude"] = lat, lon
            new_records.append(aqi_data)
            print(f"✅ {aqi_data['Name']} | Wind: {aqi_data.get('Wind_Dir')}°")
            
        time.sleep(0.5) # Protect against IP blocking

    if new_records:
        df_new = pd.DataFrame(new_records)
        df_final = pd.concat([df_existing, df_new]).drop_duplicates(subset=['Name', 'Timestamp'])
        df_final.to_csv(CSV_FILE, index=False, encoding="utf-8")
        print(f"Done. Database now has {len(df_final)} total records.")

if __name__ == "__main__":
    crawl_all_data()