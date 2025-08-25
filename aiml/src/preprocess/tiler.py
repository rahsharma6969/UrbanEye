import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

def generate_windows(width, height, tile_size):
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            if w == tile_size and h == tile_size:
                yield Window(x, y, tile_size, tile_size)

def reproject_to(dst_crs, src_ds):
    # For brevity, assume inputs are already in same CRS in this starter.
    return src_ds

def read_bands_as_array(path, band_indices=None):
    with rasterio.open(path) as ds:
        if band_indices is None:
            arr = ds.read()
            transform = ds.transform
            crs = ds.crs
        else:
            arr = ds.read(band_indices)
            transform = ds.transform
            crs = ds.crs
    return arr, transform, crs
