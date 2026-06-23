"""
Download Himawari-8 L2/L3 Aerosol Property (ARP) data from JAXA P-Tree FTP.

Downloads raw NetCDF files, then triggers process_himawari.py for each file.
Operates in two phases: historical bulk download, then real-time monitoring.

Configuration:
  Set JAXA_FTP_USER and JAXA_FTP_PASS environment variables before running.
  Adjust LOCAL_BASE to the directory where raw .nc files should be stored.

Original location: data/himawari/raw_scripts/download_himawari.py
"""
import os
import sys
import time
from ftplib import FTP
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

FTP_HOST = "ftp.ptree.jaxa.jp"
FTP_USER = os.environ.get("JAXA_FTP_USER", "")
FTP_PASS = os.environ.get("JAXA_FTP_PASS", "")

if not FTP_USER or not FTP_PASS:
    print("ERROR: Set JAXA_FTP_USER and JAXA_FTP_PASS environment variables.")
    sys.exit(1)

BASE_DIRS = {
    "L2": "/pub/himawari/L2/ARP/031",
    "L3": "/pub/himawari/L3/ARP/031"
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_BASE = PROJECT_ROOT / "data" / "himawari" / "raw"
PROCESS_SCRIPT = Path(__file__).resolve().parent / "process_himawari.py"
MISSING_LOG_FILE = PROJECT_ROOT / "data" / "himawari" / "missing_data.log"

start_time_holder = datetime(2025, 10, 11, 15, 0)
MAX_CONSECUTIVE_MISSING = 24


def load_missing_data_log():
    missing_data = {}
    if not MISSING_LOG_FILE.exists():
        return missing_data
    try:
        with open(MISSING_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split(" | ")
                    if len(parts) >= 3:
                        data_timestamp = datetime.strptime(parts[1], '%Y-%m-%d %H:%M')
                        missing_data[data_timestamp] = line
                except:
                    continue
        if missing_data:
            print(f"Loaded {len(missing_data)} missing data entries from log")
    except Exception as e:
        print(f"Error loading missing data log: {e}")
    return missing_data


def remove_from_missing_log(timestamp):
    if not MISSING_LOG_FILE.exists():
        return
    try:
        with open(MISSING_LOG_FILE, "r") as f:
            lines = f.readlines()
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M')
        filtered_lines = [line for line in lines if f" | {timestamp_str} | " not in line]
        with open(MISSING_LOG_FILE, "w") as f:
            f.writelines(filtered_lines)
    except:
        pass


def log_missing_data(timestamp, remote_path, reason="Directory not found"):
    try:
        MISSING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MISSING_LOG_FILE, "a") as f:
            log_entry = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                         f"{timestamp.strftime('%Y-%m-%d %H:%M')} | "
                         f"{remote_path} | {reason}\n")
            f.write(log_entry)
    except:
        pass


def get_local_files(local_path):
    if not os.path.exists(local_path):
        return set()
    try:
        return set([f.removeprefix("aod_vietnam_").removesuffix(".tif")
                    for f in os.listdir(local_path)])
    except:
        return set()


def fetch_file(file, local_path, ftp):
    local_file = os.path.join(local_path, file)
    try:
        with open(local_file, "wb") as f:
            for attempt in range(3):
                try:
                    ftp.retrbinary(f"RETR {file}", f.write)
                    break
                except:
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        raise
        print(f"Downloaded: {file}")
        subprocess.run(
            [sys.executable, str(PROCESS_SCRIPT), local_file],
            check=True, timeout=300
        )
    except Exception as e:
        print(f"Error with file {file}: {e}")
        if os.path.exists(local_file):
            os.remove(local_file)


def download_and_process(ftp, remote_path, local_path, timestamp, missing_data_log):
    os.makedirs(local_path, exist_ok=True)
    try:
        ftp.cwd(remote_path)
        remote_nc_files = [f for f in ftp.nlst() if f.endswith('.nc')]
        local_files = get_local_files(local_path)
        files_to_download = [f for f in remote_nc_files
                             if f.removesuffix(".nc") not in local_files]

        if not files_to_download:
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
    global start_time_holder
    consecutive_missing_count = 0

    while True:
        current_time = start_time_holder
        if current_time >= datetime.now() - timedelta(hours=2):
            return

        found_any_level = False
        try:
            with FTP(FTP_HOST, timeout=30) as ftp:
                ftp.login(FTP_USER, FTP_PASS)

                for level, base_dir in BASE_DIRS.items():
                    ymd = current_time.strftime("%Y%m")
                    dd = current_time.strftime("%d")
                    hh = current_time.strftime("%H")
                    remote_path = f"{base_dir}/{ymd}/{dd}/{hh}/"
                    local_path = str(LOCAL_BASE / level / ymd / dd / hh)

                    if download_and_process(ftp, remote_path, local_path,
                                            current_time, missing_data_log):
                        found_any_level = True

            if found_any_level:
                consecutive_missing_count = 0
            else:
                consecutive_missing_count += 1
                if consecutive_missing_count >= MAX_CONSECUTIVE_MISSING:
                    return
            start_time_holder += timedelta(hours=1)
            time.sleep(1)
        except Exception as e:
            print(f"FTP Connection error: {e}")
            time.sleep(30)


def realtime_mode():
    while True:
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        check_times = [current_hour - timedelta(hours=i) for i in range(1, 4)]

        try:
            with FTP(FTP_HOST, timeout=30) as ftp:
                ftp.login(FTP_USER, FTP_PASS)
                for check_time in check_times:
                    for level, base_dir in BASE_DIRS.items():
                        ymd = check_time.strftime("%Y%m")
                        dd = check_time.strftime("%d")
                        hh = check_time.strftime("%H")
                        remote_path = f"{base_dir}/{ymd}/{dd}/{hh}/"
                        local_path = str(LOCAL_BASE / level / ymd / dd / hh)
                        download_and_process(ftp, remote_path, local_path,
                                             check_time, {})
            time.sleep(600)
        except:
            time.sleep(30)


def main():
    print("Initializing Himawari L2/L3 downloader...")
    missing_data_log = load_missing_data_log()
    if start_time_holder < datetime.now() - timedelta(hours=6):
        print("Running historical recovery phase...")
        historical_mode(missing_data_log)
    print("Switching to real-time monitoring...")
    realtime_mode()


if __name__ == "__main__":
    main()
