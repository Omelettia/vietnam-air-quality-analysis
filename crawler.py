import requests
import pandas as pd
import time
import os
import urllib3
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- Configuration & Environment ---
load_dotenv()
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")

CSV_FILE = "air_quality_weather_data.csv"
# Envisoft endpoints for station list and detailed pollutant data
URL_REGISTRY = "https://envisoft.gov.vn/eos/services/call/json/get_stations"
URL_DETAIL = "https://envisoft.gov.vn/eos/services/call/json/qi_detail_for_eip"

# Suppress insecure request warnings caused by government SSL certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_active_network():
    """
    Fetches the current list of active monitoring stations from Envisoft.
    Filters for '(KK)' to isolate Air Quality stations from water monitoring.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://cem.gov.vn",
        "Host": "envisoft.gov.vn"
    }
    # Parameters to trigger the AQI layer in the backend
    payload = {"is_qi": "true", "is_public": "true", "qi_type": "aqi"}
    try:
        res = requests.post(URL_REGISTRY, headers=headers, data=payload, timeout=20, verify=False)
        if res.status_code == 200:
            full_list = res.json().get('stations', [])
            return [s for s in full_list if "(KK)" in s.get('station_name', '')]
    except Exception as e:
        print(f"[!] Registry discovery error: {e}")
    return []

def fetch_openweather_live(lat, lon):
    """
    Retrieves real-time meteorological data for a specific location.
    Note: Only used when the station's AQI recording matches the current hour.
    """
    if not OPENWEATHER_KEY: return {}
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return {
                "Temperature": data['main']['temp'],
                "Humidity": data['main']['humidity'],
                "Pressure": data['main']['pressure'],
                "Wind Speed": data['wind']['speed'],
                "Wind Direction": data['wind']['deg']
            }
    except: pass
    return {}

def acquire_telemetry(station):
    """
    Gathers air quality and weather data for a single station.
    Strictly prevents 'time-drift' errors by checking data freshness 
    before calling the OpenWeather API.
    """
    sid = station.get('id')
    lat = station.get('latitude')
    lon = station.get('longitude')
    
    try:
        res = requests.get(f"{URL_DETAIL}?station_id={sid}", timeout=10, verify=False)
        if res.status_code != 200: return None
        
        raw = res.json()
        metrics = raw.get('res', {})
        time_str = raw.get('qi_time_2') # Recording timestamp (VN local time)
        
        if not time_str: return None

        # --- DATA INTEGRITY CHECK (FRESHNESS) ---
        # We compare the station's 'Recording Time' to the 'Current System Time'.
        # If the gap > 60 mins, current weather doesn't match the pollution state.
        data_time = datetime.strptime(time_str, "%d/%m/%Y %H:%M")
        is_fresh = (datetime.now() - data_time) < timedelta(minutes=60)

        # Base AQI payload
        payload = {
            "Timestamp": time_str,
            "Name": raw.get('station_name'),
            "Id": sid,
            "Latitude": lat,
            "Longitude": lon,
            "AQI": raw.get('qi_value'),
            "PM2.5": metrics.get('PM-2-5', {}).get('current'),
            "PM10": metrics.get('PM-10', {}).get('current'),
            "CO": metrics.get('CO', {}).get('current'),
            "NO2": metrics.get('NO2', {}).get('current'),
            "O3": metrics.get('O3', {}).get('current'),
            "SO2": metrics.get('SO2', {}).get('current'),
        }

        # --- METEOROLOGICAL SOURCE LOGIC ---
        # Priority 1: Physical sensor at the station (Ground Truth)
        # Priority 2: Real-time API fallback (Only if data is fresh)
        # Priority 3: Label for historical backfilling later
        internal_weather = {
            "Temperature": metrics.get('Temp', {}).get('current'),
            "Humidity": metrics.get('Hum', {}).get('current'),
            "Pressure": metrics.get('Press', {}).get('current'),
            "Wind Speed": metrics.get('WS', {}).get('current'),
            "Wind Direction": metrics.get('WD', {}).get('current') or metrics.get('WindDir', {}).get('current')
        }

        if any(v is not None for v in internal_weather.values()):
            payload.update(internal_weather)
            payload["Weather_Source"] = "Internal_Sensor"
        elif is_fresh:
            ow_data = fetch_openweather_live(lat, lon)
            payload.update(ow_data)
            payload["Weather_Source"] = "OpenWeather_Live"
        else:
            # Data is 'Lagging'. Keep empty to avoid inaccurate time-shifted weather.
            payload["Weather_Source"] = "TO_BACKFILL"
            for key in ["Temperature", "Humidity", "Pressure", "Wind Speed", "Wind Direction"]:
                payload[key] = None

        return payload
    except: return None

def update_dataset(new_records):
    """
    Saves new records to CSV. Uses 'Name' + 'Timestamp' as a composite key 
    to prevent duplicate entries during multiple runs.
    """
    if not new_records: return
    schema = [
        "Timestamp", "Name", "Id", "Latitude", "Longitude", 
        "AQI", "PM2.5", "PM10", "CO", "NO2", "O3", "SO2",
        "Temperature", "Humidity", "Pressure", "Wind Speed", "Wind Direction", "Weather_Source"
    ]

    if os.path.exists(CSV_FILE):
        df_master = pd.read_csv(CSV_FILE)
    else:
        df_master = pd.DataFrame(columns=schema)

    df_new = pd.DataFrame(new_records)
    
    # Concatenate and remove duplicates (keeps the most recently pulled version)
    df_final = pd.concat([df_master, df_new], ignore_index=True)
    df_final.drop_duplicates(subset=["Name", "Timestamp"], keep="last", inplace=True)
    
    df_final[schema].to_csv(CSV_FILE, index=False, encoding="utf-8")
    print(f"[*] Batch committed: {len(new_records)} records updated in {CSV_FILE}.")

def main():
    """ Orchestrates the topology discovery and data crawl sequence. """
    print(f"[*] Starting Envisoft crawl at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    network = get_active_network()
    if not network: return
    
    batch_buffer = []
    for node in network:
        data = acquire_telemetry(node)
        if data:
            batch_buffer.append(data)
            print(f"[+] Processed: {data['Name']} | {data['Weather_Source']}")
            # Delay to respect the server's rate limits
            time.sleep(0.3)

    update_dataset(batch_buffer)

if __name__ == "__main__":
    main()