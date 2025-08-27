# src/infer/polygonize_changes.py
import os, argparse, json, numpy as np, geopandas as gpd, rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

def morph(mask, close_iters=1, open_iters=0):
    from numpy.lib.stride_tricks import sliding_window_view
    def dilate(x):
        w = sliding_window_view(x.astype(np.uint8), (3,3))
        return (w.max(axis=(-1,-2)) > 0).astype(np.uint8)
    def erode(x):
        w = sliding_window_view(x.astype(np.uint8), (3,3))
        return (w.min(axis=(-1,-2)) > 0).astype(np.uint8)
    y = mask.astype(np.uint8)
    for _ in range(close_iters): y = dilate(y); y = erode(y)
    for _ in range(open_iters):  y = erode(y);  y = dilate(y)
    return y.astype(bool)

def polygonize_one(tif_path, prob_thresh, min_area_m2, blur=False, close_iters=1, open_iters=0):
    with rasterio.open(tif_path) as src:
        prob = src.read(1).astype("float32")
        prob = np.nan_to_num(prob, nan=0.0)
        transform = src.transform
        crs = src.crs

    if blur:
        from numpy.lib.stride_tricks import sliding_window_view
        p = np.pad(prob, 1, mode="edge")
        win = sliding_window_view(p, (3,3))
        prob = win.mean(axis=(-1,-2))

    mask = (prob >= prob_thresh)
    mask = morph(mask, close_iters=close_iters, open_iters=open_iters)

    geoms = []
    for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if val != 1: continue
        poly = shape(geom)
        if not poly.is_valid: poly = poly.buffer(0)
        geoms.append(poly)

    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    # dissolve within tile to reduce speckle
    dissolved = unary_union(geoms)
    if dissolved.geom_type == "Polygon":
        polys = [dissolved]
    else:
        polys = list(dissolved.geoms)

    gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)
    # area calculation in metric CRS:
    if gdf.crs and not str(gdf.crs).startswith("EPSG:4326"):
        gdf["area_m2"] = gdf.area
    else:
        # fallback: project to local UTM for area
        gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
        gdf["area_m2"] = gdf_m.area
    gdf = gdf[gdf["area_m2"] >= min_area_m2].copy()
    return gdf

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    tifs = [f for f in os.listdir(args.preds_dir) if f.endswith("_change.tif")]
    all_gdfs = []
    for fname in tifs:
        gdf = polygonize_one(
            os.path.join(args.preds_dir, fname),
            prob_thresh=args.prob_thresh,
            min_area_m2=args.min_area_m2,
            blur=args.blur,
            close_iters=args.close_iters,
            open_iters=args.open_iters
        )
        if not gdf.empty:
            gdf["chip_id"] = fname.replace("_change.tif","")
            all_gdfs.append(gdf)

    if not all_gdfs:
        print("No polygons found with current thresholds.")
        return

    # concatenate in native raster CRS (UTM per tile)
    gdf_all = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)

    # dissolve across tiles to remove overlaps (reproject to a common CRS first)
    gdf_eq = gdf_all.to_crs(gdf_all.estimate_utm_crs())  # single local UTM for dissolution
    dissolved = unary_union(list(gdf_eq.geometry))
    polys = [dissolved] if dissolved.geom_type == "Polygon" else list(dissolved.geoms)
    gdf_diss = gpd.GeoDataFrame(geometry=polys, crs=gdf_eq.crs)
    gdf_diss["area_m2"] = gdf_diss.area

    # save WGS84 for the app + CSV summary
    gdf_wgs = gdf_diss.to_crs("EPSG:4326")
    out_geojson = os.path.join(args.out_dir, "changes.geojson")
    gdf_wgs.to_file(out_geojson, driver="GeoJSON")
    gdf_wgs.drop(columns="geometry").to_csv(os.path.join(args.out_dir, "changes.csv"), index=False)

    print(f"Wrote {len(gdf_wgs)} polygons → {out_geojson}")

if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", default="outputs/preds")
    ap.add_argument("--out_dir", default="outputs/polys")
    ap.add_argument("--prob_thresh", type=float, default=0.3)
    ap.add_argument("--min_area_m2", type=float, default=100.0)
    ap.add_argument("--blur", action="store_true")
    ap.add_argument("--close_iters", type=int, default=1)
    ap.add_argument("--open_iters", type=int, default=0)
    args = ap.parse_args()
    main(args)
