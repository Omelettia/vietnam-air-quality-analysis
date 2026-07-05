"""
Download GPM IMERG half-hourly precipitation data from NASA FTPS.

Downloads GIS-format zip files (30-min accumulation GeoTIFFs) from:
  - Final run: arthurhou.pps.eosdis.nasa.gov  /gpmdata/YYYY/MM/DD/gis/
  - NRT fallback: jsimpsonhttps.pps.eosdis.nasa.gov  /data/imerg/gis/YYYY/MM/

Requires NASA PMM registration (free): https://registration.pps.eosdis.nasa.gov/

Configuration:
  Set environment variables before running:
    GPM_FTP_USER     — NASA PMM username (email)
    GPM_FTP_PASSWORD  — NASA PMM password

  Or create Thesis/scripts/01_collection/config_local.py with:
    GPM_FTP_USER = "your_email@example.com"
    GPM_FTP_PASSWORD = "your_password"

Usage:
  python download_gpm.py --start 2023-01-01 --end 2026-04-09 --output data/gpm/raw

Adapted from: github.com/sfatew/Air_Quality/blob/main/GIS/get_file_list.py
"""
import os
import sys
import re
import ssl
import time
import argparse
from datetime import datetime, timedelta
from ftplib import FTP_TLS, error_perm

# Config
FINAL_SERVER = "arthurhou.pps.eosdis.nasa.gov"
NRT_SERVER = "jsimpsonhttps.pps.eosdis.nasa.gov"
RECONNECT_AFTER_FILES = 50
MAX_RETRIES = 3

FTP_TLS.ssl_version = ssl.PROTOCOL_TLSv1_2


def get_credentials():
    user = os.environ.get("GPM_FTP_USER")
    pwd = os.environ.get("GPM_FTP_PASSWORD")
    if user and pwd:
        return user, pwd
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        from config_local import GPM_FTP_USER, GPM_FTP_PASSWORD
        return GPM_FTP_USER, GPM_FTP_PASSWORD
    except ImportError:
        pass
    print("ERROR: Set GPM_FTP_USER and GPM_FTP_PASSWORD env vars, "
          "or create config_local.py")
    sys.exit(1)


def connect_ftps(server, user, password):
    try:
        ftps = FTP_TLS(timeout=60)
        ftps.connect(server, 21)
        ftps.login(user, password)
        ftps.prot_p()
        ftps.voidcmd("NOOP")
        print(f"  Connected to {server}")
        return ftps
    except Exception as e:
        print(f"  Connection failed ({server}): {e}")
        return None


def is_alive(ftps):
    try:
        ftps.voidcmd("NOOP")
        return True
    except Exception:
        return False


def get_file_list(ftps, remote_path):
    for attempt in range(MAX_RETRIES):
        try:
            ftps.cwd("/")
            ftps.cwd(remote_path)
            raw = ftps.nlst()
            return [f for f in raw if re.search(r"HHR.*\.zip", f, re.I)]
        except error_perm as e:
            if "550" in str(e):
                return []
            if attempt == MAX_RETRIES - 1:
                return []
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return []
            time.sleep(2 ** attempt)
    return []


def download_file(ftps, remote_path, filename, local_path):
    output = os.path.join(local_path, filename)
    if os.path.exists(output):
        return True
    for attempt in range(MAX_RETRIES):
        try:
            ftps.cwd("/")
            ftps.cwd(remote_path)
            with open(output, "wb") as f:
                ftps.retrbinary(f"RETR {filename}", f.write)
            return True
        except Exception as e:
            if os.path.exists(output):
                os.remove(output)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed: {filename} ({e})")
                return False
    return False


def download_date(ftps_final, ftps_nrt, date_obj, output_dir):
    y, m, d = date_obj.strftime("%Y"), date_obj.strftime("%m"), date_obj.strftime("%d")
    local_path = os.path.join(output_dir, y, m, d)

    # Try Final server
    remote = f"/gpmdata/{y}/{m}/{d}/gis/"
    files = get_file_list(ftps_final, remote)
    if files:
        os.makedirs(local_path, exist_ok=True)
        count = sum(1 for f in files if download_file(ftps_final, remote, f, local_path))
        return count

    # Fallback: NRT
    if ftps_nrt is None:
        return 0
    nrt_remote = f"/data/imerg/gis/{y}/{m}/"
    date_str = date_obj.strftime("%Y%m%d")
    try:
        ftps_nrt.cwd("/")
        ftps_nrt.cwd(nrt_remote)
        raw = ftps_nrt.nlst()
        nrt_files = [f for f in raw if re.search(r"HHR.*\.30min\.zip", f, re.I) and date_str in f]
    except Exception:
        nrt_files = []

    if nrt_files:
        os.makedirs(local_path, exist_ok=True)
        count = sum(1 for f in nrt_files if download_file(ftps_nrt, nrt_remote, f, local_path))
        print(f"    NRT fallback: {count} files")
        return count
    return 0


def main():
    parser = argparse.ArgumentParser(description="Download GPM IMERG precipitation data")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="data/gpm/raw", help="Output directory")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    user, pwd = get_credentials()

    print(f"GPM IMERG download: {args.start} to {args.end}")
    print(f"Output: {args.output}")

    ftps = connect_ftps(FINAL_SERVER, user, pwd)
    if not ftps:
        sys.exit(1)
    ftps_nrt = connect_ftps(NRT_SERVER, user, pwd)

    current = start
    files_total = 0
    files_since_reconnect = 0

    while current <= end:
        if files_since_reconnect >= RECONNECT_AFTER_FILES or not is_alive(ftps):
            try:
                ftps.quit()
            except Exception:
                pass
            ftps = connect_ftps(FINAL_SERVER, user, pwd)
            if not ftps:
                time.sleep(60)
                continue
            files_since_reconnect = 0

        try:
            n = download_date(ftps, ftps_nrt, current, args.output)
            files_total += n
            files_since_reconnect += n
            if n > 0:
                print(f"  {current.date()}: {n} files")
            else:
                print(f"  {current.date()}: no data")
        except Exception as e:
            print(f"  {current.date()}: error ({e}), reconnecting...")
            try:
                ftps.quit()
            except Exception:
                pass
            ftps = connect_ftps(FINAL_SERVER, user, pwd)
            if not ftps:
                time.sleep(60)
            continue

        current += timedelta(days=1)

    for conn in (ftps, ftps_nrt):
        if conn:
            try:
                conn.quit()
            except Exception:
                pass

    print(f"\nDone. Total files: {files_total}")


if __name__ == "__main__":
    main()
