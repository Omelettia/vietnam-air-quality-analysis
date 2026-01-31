import os
import sys
import time
from ftplib import FTP
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# --- Configuration for import paths ---
# Dynamically add the parent directory to sys.path to access local config files
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_script_dir, os.pardir)
sys.path.append(parent_dir)

from config.config import aod_config

# --- FTP and Directory Configuration ---
# JAXA P-Tree System is the primary source for Himawari-8 Aerosol Property data [cite: 80]
FTP_HOST = "ftp.ptree.jaxa.jp"
FTP_USER = aod_config.FTP_USER
FTP_PASS = aod_config.FTP_PASS

# BASE_DIRS covers both research perspectives:
# L2: High-frequency snapshots for diurnal variations (every 10 minutes) [cite: 82]
# L3: Spatiotemporally aggregated data for high-precision regional analysis [cite: 83]
BASE_DIRS = {
    "L2": "/pub/himawari/L2/ARP/031",
    "L3": "/pub/himawari/L3/ARP/031"
}

# LOCAL_BASE: Directory where raw .nc files are stored before Geospatial Trimming [cite: 104]
LOCAL_BASE = "/home/slow_data/Air_Quality/AOD/full_aod"
# PROCESS_SCRIPT: Executes nc-to-geotiff conversion and Vietnam boundary clipping [cite: 102, 105]
PROCESS_SCRIPT = "/home/work1/projects/Air_Quality/AOD data/process_aod_data.py"
# MISSING_LOG_FILE: Critical for identifying temporal gaps in the 2025-2026 data range [cite: 84, 199]
MISSING_LOG_FILE = "/home/work1/projects/Air_Quality/AOD data/missing_data.log"

# Historical Window: Aligns with the synchronized transition period analyzed in the report [cite: 47]
start_time_holder = datetime(2025, 10, 11, 15, 0)
# Gap handling: If data is missing for 24 hours, the script assumes a JAXA server lag or outage [cite: 160]
MAX_CONSECUTIVE_MISSING = 24 

# --- Missing Data Log Management ---
def load_missing_data_log():
    """Loads existing log of missing timestamps to attempt data recovery during historical runs [cite: 160, 165]"""
    missing_data = {}
    if not os.path.exists(MISSING_LOG_FILE):
        return missing_data
    try:
        with open(MISSING_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    parts = line.split(" | ")
                    if len(parts) >= 3:
                        # Extract data timestamp to prioritize recovery of specific gaps
                        data_timestamp = datetime.strptime(parts[1], '%Y-%m-%d %H:%M')
                        missing_data[data_timestamp] = line
                except: continue
        if missing_data:
            print(f"📋 Loaded {len(missing_data)} missing data entries from log")
    except Exception as e:
        print(f"⚠️ Error loading missing data log: {e}")
    return missing_data

def remove_from_missing_log(timestamp):
    """Removes a timestamp from the missing log once a successful recovery download is verified [cite: 161]"""
    if not os.path.exists(MISSING_LOG_FILE): return
    try:
        with open(MISSING_LOG_FILE, "r") as f:
            lines = f.readlines()
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M')
        filtered_lines = [line for line in lines if f" | {timestamp_str} | " not in line]
        with open(MISSING_LOG_FILE, "w") as f:
            f.writelines(filtered_lines)
    except: pass

def log_missing_data(timestamp, remote_path, reason="Directory not found"):
    """Documents connection failures or directory gaps to support Section 3.1: Handling Missing Data [cite: 159, 160]"""
    try:
        with open(MISSING_LOG_FILE, "a") as f:
            log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {timestamp.strftime('%Y-%m-%d %H:%M')} | {remote_path} | {reason}\n"
            f.write(log_entry)
    except: pass

def get_local_files(local_path):
    """Retrieves a list of files already processed into GeoTIFFs to avoid redundant API calls and downloads [cite: 65, 102]"""
    if not os.path.exists(local_path): return set()
    try:
        # Prefix 'aod_vietnam_' indicates the file has successfully passed Geospatial Trimming [cite: 105, 110]
        return set([f.removeprefix("aod_vietnam_").removesuffix(".tif") for f in os.listdir(local_path)])
    except: return set()

def fetch_file(file, local_path, ftp):
    """Downloads a raw NetCDF file and immediately triggers the boundary clipping subprocess [cite: 102, 105]"""
    local_file = os.path.join(local_path, file)
    try:
        with open(local_file, "wb") as f:
            # Implements retry logic to handle intermittent network failures common in high-volume satellite data transfers [cite: 160]
            for attempt in range(3):
                try:
                    ftp.retrbinary(f"RETR {file}", f.write)
                    break
                except:
                    if attempt < 2: time.sleep(5)
                    else: raise
        print(f"✅ Downloaded: {file}")
        # Pass file to process_aod_data.py for masking and GeoTIFF conversion [cite: 102, 105]
        subprocess.run(["python", PROCESS_SCRIPT, local_file], check=True, timeout=300)
    except Exception as e:
        print(f"❌ Error with file {file}: {e}")
        if os.path.exists(local_file): os.remove(local_file)

# --- Core FTP and Processing Logic ---
def download_and_process(ftp, remote_path, local_path, timestamp, missing_data_log):
    """Manages the directory-level crawl, filtering for unique files and handling server synchronization."""
    os.makedirs(local_path, exist_ok=True)
    try:
        ftp.cwd(remote_path)
        remote_nc_files = [f for f in ftp.nlst() if f.endswith('.nc')]
        local_files = get_local_files(local_path)
        # Strategic selection: Only fetch files not already existing in the local 'Vietnam-clipped' archive [cite: 100]
        files_to_download = [f for f in remote_nc_files if f.removesuffix(".nc") not in local_files]

        if not files_to_download:
            # If files exist, verify if we can resolve a previously logged 'missing data' status [cite: 161]
            if timestamp in missing_data_log and remote_nc_files:
                remove_from_missing_log(timestamp)
            return True

        for file in files_to_download:
            fetch_file(file, local_path, ftp)
        
        if timestamp in missing_data_log:
            remove_from_missing_log(timestamp)
        return True
    except Exception as e:
        if timestamp not in missing_data_log:
            log_missing_data(timestamp, remote_path, str(e))
        return False

def historical_mode(missing_data_log):
    """Executes the bulk download phase to cover the April-December 2025 research window [cite: 47, 84]"""
    global start_time_holder
    consecutive_missing_count = 0
    
    while True:
        current_time = start_time_holder
        # Stop historical phase when within 2 hours of real-time to avoid racing JAXA upload times
        if current_time >= datetime.now() - timedelta(hours=2):
            return 
        
        found_any_level = False
        try:
            with FTP(FTP_HOST, timeout=30) as ftp:
                ftp.login(FTP_USER, FTP_PASS)
                
                # Iterates through L2 and L3 to ensure synchronized datasets for the North-South dichotomy analysis
                for level, base_dir in BASE_DIRS.items():
                    ymd, dd, hh = current_time.strftime("%Y%m"), current_time.strftime("%d"), current_time.strftime("%H")
                    remote_path = f"{base_dir}/{ymd}/{dd}/{hh}/"
                    # Maintain separate directory structures for L2 (6 bands) and L3 (15 bands) [cite: 85, 88]
                    local_path = os.path.join(LOCAL_BASE, level, ymd, dd, hh)
                    
                    if download_and_process(ftp, remote_path, local_path, current_time, missing_data_log):
                        found_any_level = True
            
            if found_any_level:
                consecutive_missing_count = 0
                start_time_holder += timedelta(hours=1)
            else:
                consecutive_missing_count += 1
                if consecutive_missing_count >= MAX_CONSECUTIVE_MISSING:
                    return
                start_time_holder += timedelta(hours=1)
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ FTP Connection error: {e}")
            time.sleep(30)

def realtime_mode():
    """Continuously monitors JAXA for new hourly updates to maintain an operational monitoring stream [cite: 69]"""
    while True:
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        # Check a 3-hour trailing window to account for JAXA upload latency and processing time [cite: 118]
        check_times = [current_hour - timedelta(hours=i) for i in range(1, 4)]
        
        try:
            with FTP(FTP_HOST, timeout=30) as ftp:
                ftp.login(FTP_USER, FTP_PASS)
                for check_time in check_times:
                    for level, base_dir in BASE_DIRS.items():
                        ymd, dd, hh = check_time.strftime("%Y%m"), check_time.strftime("%d"), check_time.strftime("%H")
                        remote_path = f"{base_dir}/{ymd}/{dd}/{hh}/"
                        local_path = os.path.join(LOCAL_BASE, level, ymd, dd, hh)
                        download_and_process(ftp, remote_path, local_path, check_time, {})
            # Wait 10 minutes between checks to respect server rate limits [cite: 69]
            time.sleep(600)
        except:
            time.sleep(30)

def main():
    """Initializes the multi-source integration pipeline, transitioning from historical recovery to real-time sync [cite: 18, 36]"""
    print("📖 Initializing Dual L2/L3 Himawari Downloader...")
    missing_data_log = load_missing_data_log()
    if start_time_holder < datetime.now() - timedelta(hours=6):
        print("📚 Executing Historical Recovery Phase...")
        historical_mode(missing_data_log)
    print("🔄 Activating Real-time Synchronization Loop...")
    realtime_mode()

if __name__ == "__main__":
    main()