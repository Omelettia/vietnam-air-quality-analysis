import requests
import pandas as pd

def fetch_station_identifiers():
    """
    Retrieves official station names and IDs from the TEDP API
    for multi-source data integration.
    """
    # Endpoint identified in image_2a3f89.png
    url = "https://tedp.vn/api/public-data/search/findPublicDataWithValidParentIn?stationType=4&size=5000"
    
    try:
        print("📡 Connecting to TEDP API...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        # Extract embedded records as shown in image_2a3f89.png
        raw_stations = data.get('_embedded', {}).get('public-data', [])
        
        station_list = []
        for item in raw_stations:
            station_list.append({
                'stationId': item.get('stationId'),
                'stationName': item.get('stationName'),
                'latitude': item.get('latitude'),
                'longitude': item.get('longtitude')
            })
            
        df = pd.DataFrame(station_list)
        
        # Save to your feature branch project directory
        df.to_csv("data/stations/metadata/envisoft_station_map.csv", index=False)
        print(f"✅ Successfully mapped {len(df)} stations to 'data/stations/metadata/envisoft_station_map.csv'")
        return df

    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

if __name__ == "__main__":
    fetch_station_identifiers()