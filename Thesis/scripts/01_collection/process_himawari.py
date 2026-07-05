"""
Convert Himawari-8 NetCDF to GeoTIFF and clip to Vietnam boundary.

Usage: python process_himawari.py <input.nc>

Reads a raw Himawari ARP NetCDF file, writes a 6-band GeoTIFF (AOT,
AOT_uncertainty, AE, QA_flag, SSA, RF), clips to Vietnam using GADM
shapefile, then passes the clipped file to extract_himawari_stations.py.

Original location: data/himawari/raw_scripts/process_aod_data.py
"""
import os
import sys
import subprocess
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHAPEFILE = PROJECT_ROOT / "data" / "boundaries" / "gadm41_VNM_0.shp"
EXTRACT_SCRIPT = Path(__file__).resolve().parent / "extract_himawari_stations.py"


def nc_to_geotiff(nc_file, output_path):
    ds = xr.open_dataset(nc_file, decode_timedelta=True)
    aot = ds['AOT'].values
    aot_uncertainty = ds['AOT_uncertainty'].values
    ae = ds['AE'].values
    qa_flag = ds['QA_flag'].values
    ssa = ds['SSA'].values
    rf = ds['RF'].values
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    ds.close()

    pixel_size = 0.05
    lon_start = np.floor(lon.min() / pixel_size) * pixel_size
    lat_start = np.ceil(lat.max() / pixel_size) * pixel_size

    transform = rasterio.transform.from_origin(
        lon_start, lat_start, pixel_size, pixel_size
    )

    profile = {
        'driver': 'GTiff',
        'height': aot.shape[0],
        'width': aot.shape[1],
        'count': 6,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform,
    }

    band_names = ['AOT', 'AOT_uncertainty', 'AE', 'QA_flag', 'SSA', 'RF']
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(aot.astype('float32'), 1)
        dst.write(aot_uncertainty.astype('float32'), 2)
        dst.write(ae.astype('float32'), 3)
        dst.write(qa_flag.astype('float32'), 4)
        dst.write(ssa.astype('float32'), 5)
        dst.write(rf.astype('float32'), 6)
        for i, name in enumerate(band_names):
            dst.set_band_description(i + 1, name)


def crop_to_vietnam(input_tif, output_tif):
    with rasterio.open(input_tif) as src:
        shape = gpd.read_file(SHAPEFILE)
        shape = shape.to_crs(src.crs)
        out_image, out_transform = mask(src, shape.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    with rasterio.open(output_tif, "w", **out_meta) as dest:
        dest.write(out_image)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_himawari.py <input.nc>")
        sys.exit(1)

    nc_path = sys.argv[1]
    base_dir = os.path.dirname(nc_path)
    filename = os.path.basename(nc_path).replace(".nc", "")
    aod_full_path = os.path.join(base_dir, f"aod_full_{filename}.tif")
    aod_vietnam_path = os.path.join(base_dir, f"aod_vietnam_{filename}.tif")

    nc_to_geotiff(nc_path, aod_full_path)
    crop_to_vietnam(aod_full_path, aod_vietnam_path)

    os.remove(nc_path)
    os.remove(aod_full_path)

    subprocess.run(
        [sys.executable, str(EXTRACT_SCRIPT), aod_vietnam_path],
        check=True, timeout=300
    )
