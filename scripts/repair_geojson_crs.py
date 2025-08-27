import argparse, os, glob, geopandas as gpd, rasterio

def main(args):
    # 1) detect the true CRS from any *_change.tif
    tif_list = glob.glob(os.path.join(args.preds_dir, "*_change.tif"))
    if not tif_list:
        raise SystemExit(f"No TIFFs found in {args.preds_dir}")
    with rasterio.open(tif_list[0]) as src:
        true_crs = src.crs
    print("Detected raster CRS:", true_crs)

    # 2) read the broken GeoJSON (assumed labeled as EPSG:4326)
    gdf = gpd.read_file(args.in_geojson)
    print("GeoJSON reported CRS:", gdf.crs)

    # 3) Reassign (don't reproject) to the TRUE CRS
    #    Because coordinates are actually in UTM meters, but file says 4326.
    gdf = gdf.set_crs(true_crs, allow_override=True)

    # 4) Compute area in meters in this CRS (UTM is metric)
    gdf["area_m2"] = gdf.area

    # 5) Save a clean metric version (same CRS as rasters)
    os.makedirs(args.out_dir, exist_ok=True)
    fixed_metric = os.path.join(args.out_dir, "changes_metric.geojson")
    gdf.to_file(fixed_metric, driver="GeoJSON")
    print("Wrote (metric CRS) ->", fixed_metric, "| CRS:", gdf.crs)

    # 6) Also save a WGS84 version for web maps
    gdf_wgs = gdf.to_crs("EPSG:4326")
    fixed_wgs = os.path.join(args.out_dir, "changes_wgs84.geojson")
    gdf_wgs.to_file(fixed_wgs, driver="GeoJSON")
    print("Wrote (WGS84) ->", fixed_wgs, "| CRS:", gdf_wgs.crs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_geojson", default="outputs/polys/changes.geojson")
    ap.add_argument("--preds_dir", default="outputs/preds")
    ap.add_argument("--out_dir", default="outputs/polys_fixed")
    args = ap.parse_args()
    main(args)
