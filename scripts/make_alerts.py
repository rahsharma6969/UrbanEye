import argparse, os, glob
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box

def _tif_index(preds_dir):
    rows = []
    for p in glob.glob(os.path.join(preds_dir, "*_change.tif")):
        with rasterio.open(p) as src:
            b = src.bounds
            crs = src.crs
            gsrc = gpd.GeoSeries([box(b.left, b.bottom, b.right, b.top)], crs=crs)
            gwgs = gsrc.to_crs("EPSG:4326")
            rows.append(dict(path=p, crs=str(crs), wgs_geom=gwgs.iloc[0]))
    if not rows:
        raise SystemExit(f"No change TIFFs found in {preds_dir}")
    return gpd.GeoDataFrame(rows, geometry="wgs_geom", crs="EPSG:4326")

def _mean_prob_for_polygon(poly_wgs, tif_gdf):
    hits = tif_gdf[tif_gdf.intersects(poly_wgs)]
    if hits.empty:
        return np.nan, 0
    vals = []
    for _, r in hits.iterrows():
        with rasterio.open(r["path"]) as src:
            poly_src = gpd.GeoSeries([poly_wgs], crs="EPSG:4326").to_crs(src.crs).iloc[0]
            try:
                arr, _ = rio_mask(src, [poly_src.__geo_interface__], crop=True, nodata=np.nan)
            except Exception:
                continue
            a = arr[0].astype("float32")
            m = a[np.isfinite(a)]
            if m.size > 0:
                vals.append(m.mean())
    if not vals:
        return np.nan, len(hits)
    return float(np.mean(vals)), len(vals)

def _write_empty(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(
        os.path.join(out_dir, "alerts.geojson"), driver="GeoJSON"
    )
    pd.DataFrame().to_csv(os.path.join(out_dir, "alerts.csv"), index=False)
    print("Alerts: 0")
    print("GeoJSON:", os.path.join(out_dir, "alerts.geojson"))
    print("CSV    :", os.path.join(out_dir, "alerts.csv"))

def main(a):
    os.makedirs(a.out_dir, exist_ok=True)

    # 1) Load polygons
    changes = gpd.read_file(a.changes_geojson)
    if changes.empty:
        print("No polygons in changes_geojson.")
        return _write_empty(a.out_dir)
    changes = changes.to_crs("EPSG:4326")

    # Ensure area_m2 exists
    if "area_m2" not in changes.columns:
        ch_m = changes.to_crs(changes.estimate_utm_crs())
        changes["area_m2"] = ch_m.area

    total0 = len(changes)
    print(f"Loaded {total0} change polygons")

    # 2) Restricted zones (optional)
    changes["in_restricted_zone"] = False
    rz_used = False
    if a.restricted_geojson:
        if os.path.exists(a.restricted_geojson):
            rz = gpd.read_file(a.restricted_geojson)
            if not rz.empty:
                rz = rz.to_crs("EPSG:4326")
                j = gpd.sjoin(changes, rz[["geometry"]], predicate="intersects", how="left")
                changes["in_restricted_zone"] = j["index_right"].notna().values
                print("Polygons in restricted zone:", int(changes["in_restricted_zone"].sum()))
                rz_used = True
            else:
                print("Restricted zones file is EMPTY:", a.restricted_geojson)
        else:
            print("Restricted zones file NOT FOUND:", a.restricted_geojson)
    if a.only_restricted and not rz_used:
        print("only_restricted=True but no valid restricted file -> will yield 0 alerts.")

    # 3) Confidence from TIFFs
    tif_gdf = _tif_index(a.preds_dir)
    mean_probs, hits = [], []
    for poly in changes.geometry:
        mp, n = _mean_prob_for_polygon(poly, tif_gdf)
        mean_probs.append(mp); hits.append(n)
    changes["mean_prob"] = mean_probs
    changes["tif_hits"] = hits

    mp_series = pd.Series([v for v in mean_probs if pd.notna(v)])
    print("Mean prob (non-NaN) stats:" if not mp_series.empty else "Mean prob: no values")
    if not mp_series.empty:
        print(mp_series.describe())

    # 4) Apply filters with guards
    q = changes.copy()
    before = len(q)

    if a.area_min_m2 is not None:
        q = q[q["area_m2"] >= float(a.area_min_m2)]
    print(f"After area ≥ {a.area_min_m2} m²: {len(q)} (dropped {before-len(q)})"); before = len(q)
    if len(q) == 0:
        return _write_empty(a.out_dir)

    if a.min_mean_prob is not None:
        q = q[q["mean_prob"].fillna(0) >= float(a.min_mean_prob)]
    print(f"After mean_prob ≥ {a.min_mean_prob}: {len(q)} (dropped {before-len(q)})"); before = len(q)
    if len(q) == 0:
        return _write_empty(a.out_dir)

    if a.only_restricted:
        q = q[q["in_restricted_zone"]]
    print(f"After only_restricted={a.only_restricted}: {len(q)} (dropped {before-len(q)})")
    if len(q) == 0:
        return _write_empty(a.out_dir)

    # 5) Safe centroids: compute in projected CRS only if non-empty
    q_m = q.to_crs(q.estimate_utm_crs())
    cents_m = q_m.centroid
    cents = gpd.GeoSeries(cents_m, crs=q_m.crs).to_crs("EPSG:4326")
    q["centroid_lon"] = cents.x
    q["centroid_lat"] = cents.y

    # 6) Save
    out_geo = os.path.join(a.out_dir, "alerts.geojson")
    out_csv = os.path.join(a.out_dir, "alerts.csv")
    q.to_file(out_geo, driver="GeoJSON")
    q.drop(columns="geometry").to_csv(out_csv, index=False)

    print(f"Alerts: {len(q)}")
    print("GeoJSON:", out_geo)
    print("CSV    :", out_csv)

if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--changes_geojson", default="outputs/polys/changes_with_areas.geojson")
    ap.add_argument("--preds_dir", default="outputs/preds")
    ap.add_argument("--restricted_geojson", default="")
    ap.add_argument("--area_min_m2", type=float, default=400)
    ap.add_argument("--min_mean_prob", type=float, default=0.25)
    ap.add_argument("--only_restricted", action="store_true")
    ap.add_argument("--out_dir", default="outputs/alerts")
    args = ap.parse_args()
    main(args)
