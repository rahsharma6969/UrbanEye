# scripts/make_chips_s2_pairs.py
import time
import os
import argparse, json, yaml, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import rasterio
import planetary_computer as pc
import pystac

from src.preprocess.s2_prep import cloud_mask_from_scl, apply_mask


# ---------------- utils ----------------
def load_cfg(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def item_time(it) -> datetime:
    """Use the inner STAC item timestamp."""
    dt_str = it["item"]["properties"]["datetime"].replace("Z", "")
    return datetime.fromisoformat(dt_str)

def _choose_asset_href(stac_item: pystac.Item, key: str) -> str:
    """
    Pick the correct asset for a band key, strictly preferring COG TIFFs.
    Falls back to any asset with matching key (case-insensitive).
    """
    candidates = [key, key.upper(), key.lower()]
    # 1) exact key & .tif/.tiff
    for k in candidates:
        if k in stac_item.assets:
            href = stac_item.assets[k].href
            if href.lower().endswith((".tif", ".tiff")):
                return href
    # 2) exact key, any extension
    for k in candidates:
        if k in stac_item.assets:
            return stac_item.assets[k].href
    # 3) scan by lowercased key
    lk = key.lower()
    for k, a in stac_item.assets.items():
        if k.lower() == lk:
            return a.href
    raise KeyError(f"Asset '{key}' not found on item {stac_item.id}")

def sign_assets(item_dict: dict) -> dict:
    """
    Return dict of lowercased keys -> href for S2 assets with fresh SAS.
    """
    it = pystac.Item.from_dict(item_dict)
    it = pc.sign(it)  # fresh SAS (anonymous is fine; key optional)
    needed = ["B02", "B03", "B04", "B08", "SCL"]
    out = {}
    for k in needed:
        out[k.lower()] = _choose_asset_href(it, k)
    return out

def open_da_with_resign(item_dict: dict, asset_key: str, retries=5, sleep=1.0):
    """
    Open an asset; on each attempt, re-sign to avoid 403 due to expired SAS.
    asset_key lower-case: 'b02','b03','b04','b08','scl'.
    """
    last = None
    for _ in range(retries):
        try:
            href = sign_assets(item_dict)[asset_key]   # FRESH SAS every attempt
            da = rxr.open_rasterio(href, masked=True, cache=False).squeeze()
            # Touch a tiny slice to fail fast if SAS/network is bad
            _ = da.isel(x=slice(0,1), y=slice(0,1)).values
            return da
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise last

def aoi_union(aoi_dir: Path) -> gpd.GeoDataFrame:
    files = list(aoi_dir.glob("*.geojson"))
    if not files:
        sys.exit(f"No AOI files in {aoi_dir}")
    gdfs = []
    for f in files:
        g = gpd.read_file(f)
        g = g[g.geometry.notna()]
        if not g.empty:
            gdfs.append(g.to_crs("EPSG:4326"))
    if not gdfs:
        sys.exit("AOI files contain no valid geometries.")
    merged = pd.concat(gdfs, ignore_index=True).explode(index_parts=False, ignore_index=True)
    try:
        u = merged.geometry.union_all()
    except AttributeError:
        u = merged.geometry.unary_union
    return gpd.GeoDataFrame(geometry=[u], crs="EPSG:4326")

def pair_nearest(t0_items, t1_items):
    if not t0_items or not t1_items:
        return []
    t1t = [(item_time(i1), i1) for i1 in t1_items]
    out = []
    for i0 in t0_items:
        t0t = item_time(i0)
        best = min(t1t, key=lambda x: abs(x[0] - t0t))[1]
        out.append((i0, best))
    return out


# -------------- main logic --------------
def main(config, items_json, t0_start, t0_end, t1_start, t1_end, out_index, tile):
    cfg = load_cfg(config)

    # Robust GDAL defaults for HTTP COGs
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.jp2,.json")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "20000000")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")

    items = json.loads(Path(items_json).read_text())
    if not items:
        sys.exit("No STAC items; run search first.")

    # keep S2 only
    s2 = [it for it in items if it.get("collection") == "sentinel-2-l2a"]

    # filter by actual item datetime
    t0s = datetime.fromisoformat(t0_start)
    t0e = datetime.fromisoformat(t0_end)
    t1s = datetime.fromisoformat(t1_start)
    t1e = datetime.fromisoformat(t1_end)
    t0 = [it for it in s2 if t0s <= item_time(it) <= t0e]
    t1 = [it for it in s2 if t1s <= item_time(it) <= t1e]

    pairs = pair_nearest(t0, t1)
    print(f"Paired {len(pairs)} S2 t0/t1 scenes")

    aoi = aoi_union(Path(cfg["paths"]["aoi_dir"]))
    chips_dir = Path(cfg["paths"]["chips_dir"]); chips_dir.mkdir(parents=True, exist_ok=True)
    rows = []; made = 0
    res = float(cfg["preprocess"].get("resolution", 10))
    ts = int(tile)

    # Extra safety: also use explicit Env (even though env vars are set)
    env_kwargs = dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.jp2,.json",
        CPL_VSIL_CURL_USE_HEAD="NO",
        CPL_VSIL_CURL_CACHE_SIZE="20000000",
        GDAL_HTTP_MAX_RETRY="4",
        GDAL_HTTP_RETRY_DELAY="1",
    )

    with rasterio.Env(**env_kwargs):
        for i, (i0, i1) in enumerate(pairs):
            try:
                # ---- READ t1 (reference grid) with per-attempt re-sign ----
                b02_1 = open_da_with_resign(i1["item"], "b02")
                b03_1 = open_da_with_resign(i1["item"], "b03")
                b04_1 = open_da_with_resign(i1["item"], "b04")
                b08_1 = open_da_with_resign(i1["item"], "b08")
                scl1  = open_da_with_resign(i1["item"], "scl")

                # reproject t1 to desired res (keep native CRS)
                b02_1 = b02_1.rio.reproject(scl1.rio.crs, resolution=res)
                b03_1 = b03_1.rio.reproject(scl1.rio.crs, resolution=res)
                b04_1 = b04_1.rio.reproject(scl1.rio.crs, resolution=res)
                b08_1 = b08_1.rio.reproject(scl1.rio.crs, resolution=res)
                scl1  = scl1 .rio.reproject(scl1.rio.crs,  resolution=res)

                # ---- READ t0 and match t1 grid ----
                b02_0 = open_da_with_resign(i0["item"], "b02").rio.reproject_match(b02_1)
                b03_0 = open_da_with_resign(i0["item"], "b03").rio.reproject_match(b03_1)
                b04_0 = open_da_with_resign(i0["item"], "b04").rio.reproject_match(b04_1)
                b08_0 = open_da_with_resign(i0["item"], "b08").rio.reproject_match(b08_1)
                scl0  = open_da_with_resign(i0["item"], "scl").rio.reproject_match(scl1)

                # ---- CLIP both to AOI ----
                aoi_ref = aoi.to_crs(scl1.rio.crs)
                def clip_all(b02, b03, b04, b08, scl):
                    b02 = b02.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    b03 = b03.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    b04 = b04.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    b08 = b08.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    scl = scl.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    return b02, b03, b04, b08, scl
                b02_1, b03_1, b04_1, b08_1, scl1 = clip_all(b02_1, b03_1, b04_1, b08_1, scl1)
                b02_0, b03_0, b04_0, b08_0, scl0 = clip_all(b02_0, b03_0, b04_0, b08_0, scl0)

                # ---- STACKS + CLOUD MASKS ----
                t1_stack = np.stack([b02_1.values, b03_1.values, b04_1.values, b08_1.values])
                t0_stack = np.stack([b02_0.values, b03_0.values, b04_0.values, b08_0.values])

                keep1 = cloud_mask_from_scl(scl1); keep1[np.isnan(scl1.values)] = True
                keep0 = cloud_mask_from_scl(scl0); keep0[np.isnan(scl0.values)] = True
                t1_stack = apply_mask(t1_stack, keep1)
                t0_stack = apply_mask(t0_stack, keep0)

                H, W = t1_stack.shape[1:]
                if H < ts or W < ts:
                    print(f"Pair {i} too small after clip; skipping")
                    continue
                print(f"- pair {i}: dims {H}x{W}")

                transform = b02_1.rio.transform()
                chip_id = 0
                for y in range(0, H - ts + 1, ts):
                    for x in range(0, W - ts + 1, ts):
                        c1 = t1_stack[:, y:y + ts, x:x + ts]
                        c0 = t0_stack[:, y:y + ts, x:x + ts]
                        # Lenient during monsoon; adjust later if needed
                        if np.isfinite(c1).mean() < 0.10 or np.isfinite(c0).mean() < 0.10:
                            continue

                        out0 = chips_dir / f"s2_t0_{i}_{chip_id}.npy"
                        out1 = chips_dir / f"s2_t1_{i}_{chip_id}.npy"
                        np.save(out0, c0.astype("float32"))
                        np.save(out1, c1.astype("float32"))

                        # chip bounds (xmin, ymin, xmax, ymax)
                        x0_map, y0_map = transform * (x, y)
                        x1_map, y1_map = transform * (x + ts, y + ts)
                        xmin, xmax = (x0_map, x1_map) if x0_map <= x1_map else (x1_map, x0_map)
                        ymin, ymax = (y1_map, y0_map) if y1_map <= y0_map else (y0_map, y1_map)

                        rows.append(dict(
                            chip_id=f"s2_{i}_{chip_id}",
                            split="train",
                            t0_npy=str(out0), t1_npy=str(out1),
                            xmin=float(xmin), ymin=float(ymin),
                            xmax=float(xmax), ymax=float(ymax),
                            width=ts, height=ts, res=res, crs=str(b02_1.rio.crs),
                            mask_npy="data/labels/placeholder.npy"
                        ))
                        chip_id += 1
                        made += 1

            except Exception as e:
                print(f"Skipping pair {i}: processing error: {e}")
                continue

    df = pd.DataFrame(rows)
    Path(out_index).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_index)
    print(f"Wrote {made} paired S2 chips -> {out_index}")


# -------------- CLI --------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--t0", nargs=2, required=True)  # YYYY-MM-DD YYYY-MM-DD
    ap.add_argument("--t1", nargs=2, required=True)
    ap.add_argument("--out_index", required=True)
    ap.add_argument("--tile_size", type=int, default=256)
    a = ap.parse_args()
    main(a.config, a.items, a.t0[0], a.t0[1], a.t1[0], a.t1[1], a.out_index, a.tile_size)
