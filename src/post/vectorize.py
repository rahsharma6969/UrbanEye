# src/post/vectorize.py
import numpy as np, geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes

def mask_to_polygons(mask, transform, crs, min_area_px=25):
    """
    Convert a binary mask to polygons using a real geotransform + CRS.

    mask: 2D numpy array of 0/1
    transform: rasterio Affine for the chip (meters if UTM)
    crs: e.g. 'EPSG:32643'
    """
    geoms = []
    vals  = []
    for geom, val in shapes(mask.astype('uint8'), mask=mask.astype(bool), transform=transform):
        if val != 1:
            continue
        shp = shape(geom)
        if shp.area < min_area_px:  # filter tiny blobs (in pixel units)
            continue
        geoms.append(shp); vals.append(val)

    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame({"value": vals}, geometry=geoms, crs=crs)
    # Convert to WGS84 for GeoJSON/web maps
    return gdf.to_crs("EPSG:4326")
