# scripts/make_chips_s2_pairs.py
import os
import argparse, json, yaml, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import rasterio
import pystac

from src.preprocess.s2_prep import cloud_mask_from_scl, apply_mask


# ---------------- utils ----------------
def load_cfg(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def item_time(it) -> datetime:
    """Use the inner STAC item timestamp (ISO8601)."""
    dt_str = it["item"]["properties"]["datetime"].replace("Z", "")
    return datetime.fromisoformat(dt_str)

# ESA code -> common-name aliases seen in many S2 L2A catalogs
BAND_ALIASES = {
    "B02": ["B02", "blue"],
    "B03": ["B03", "green"],
    "B04": ["B04", "red"],
    "B08": ["B08", "nir"],
    "B8A": ["B8A", "nir08", "nir_08", "nir08"],  # optional if you use 8A
    "SCL": ["SCL", "scl"],
}

# Preferred native resolution per band
PREFERRED_RES_PER_BAND = {
    "B02": ("10m", "20m", "60m"),
    "B03": ("10m", "20m", "60m"),
    "B04": ("10m", "20m", "60m"),
    "B08": ("10m", "20m", "60m"),
    "B8A": ("20m", "10m", "60m"),
    "SCL": ("20m", "10m", "60m"),
}

def _choose_asset_href(stac_item: pystac.Item, key: str) -> str:
    """
    Resolve an asset by ESA band key (e.g., 'B02','B03','B04','B08','SCL'),
    trying common-name aliases like 'blue','green','red','nir','scl'.
    Prefer COGs (.tif/.tiff); accept JP2 if needed.
    Also tries resolution-suffixed names like 'blue_10m', 'scl_20m'.
    """
    def ext_ok(h: str) -> bool:
        h = h.lower()
        return h.endswith((".tif", ".tiff", ".jp2"))

    key_up = key.upper()
    aliases = BAND_ALIASES.get(key_up, [key, key_up, key.lower()])

    # Build a prioritized probe list
    prefer_res = PREFERRED_RES_PER_BAND.get(key_up, ("10m", "20m", "60m"))

    # 1) Exact alias keys first (already include possible "-jp2")
    for k in aliases:
        if k in stac_item.assets:
            href = stac_item.assets[k].href
            if ext_ok(href):
                return href

    # 2) Try resolution-suffixed variants (alias_res)
    for res in prefer_res:
        for base in aliases:
            if base.endswith(("_10m", "_20m", "_60m")):
                continue
            candidate = f"{base}_{res}"
            if candidate in stac_item.assets:
                href = stac_item.assets[candidate].href
                if ext_ok(href):
                    return href

    # 3) Try ESA code with resolution suffix (B02_10m etc.)
    for res in prefer_res:
        candidate = f"{key_up}_{res}"
        if candidate in stac_item.assets:
            href = stac_item.assets[candidate].href
            if ext_ok(href):
                return href

    # 4) Accept any extension for candidates above (as last resort)
    for res in prefer_res:
        for base in aliases + [key_up]:
            if base.endswith(("_10m", "_20m", "_60m")):
                if base in stac_item.assets:
                    return stac_item.assets[base].href
                continue
            cand = f"{base}_{res}"
            if cand in stac_item.assets:
                return stac_item.assets[cand].href
    for base in aliases:
        if base in stac_item.assets:
            return stac_item.assets[base].href

    # 5) Case-insensitive match on asset keys
    alias_lowers = set(a.lower() for a in aliases)
    for k, a in stac_item.assets.items():
        if k.lower() in alias_lowers and ext_ok(a.href):
            return a.href

    # 6) Search by eo:bands / raster:bands metadata (name matches an alias)
    for k, a in stac_item.assets.items():
        meta = a.extra_fields or {}
        for b in (meta.get("eo:bands") or []):
            name = (b.get("name") or "").lower()
            if name in alias_lowers and ext_ok(a.href):
                return a.href
        for b in (meta.get("raster:bands") or []):
            name = (b.get("name") or "").lower()
            if name in alias_lowers and ext_ok(a.href):
                return a.href

    # 7) Not found → show available keys
    avail = ", ".join(stac_item.assets.keys())
    raise KeyError(f"Asset '{key}' not found on item {stac_item.id}. Available assets: {avail}")

def open_da_pick(item_dict: dict, asset_key_upper: str):
    """
    Open a STAC asset by ESA band key using the alias-aware resolver.
    asset_key_upper: 'B02','B03','B04','B08','B8A','SCL'
    """
    it = pystac.Item.from_dict(item_dict)
    href = _choose_asset_href(it, asset_key_upper)
    da = rxr.open_rasterio(href, masked=True, cache=False).squeeze()
    # touch a tiny slice to fail fast if the href is invalid
    _ = da.isel(x=slice(0, 1), y=slice(0, 1)).values
    return da

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

    # Robust GDAL defaults
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.jp2,.json")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "20000000")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")

    items = json.loads(Path(items_json).read_text())
    if not items:
        sys.exit("No STAC items; run search first.")

    s2 = [it for it in items if it.get("collection") == "sentinel-2-l2a"]

    # filter by datetime
    t0s, t0e = datetime.fromisoformat(t0_start), datetime.fromisoformat(t0_end)
    t1s, t1e = datetime.fromisoformat(t1_start), datetime.fromisoformat(t1_end)
    t0 = [it for it in s2 if t0s <= item_time(it) <= t0e]
    t1 = [it for it in s2 if t1s <= item_time(it) <= t1e]

    pairs = pair_nearest(t0, t1)
    print(f"Paired {len(pairs)} S2 t0/t1 scenes")

    aoi = aoi_union(Path(cfg["paths"]["aoi_dir"]))
    chips_dir = Path(cfg["paths"]["chips_dir"]); chips_dir.mkdir(parents=True, exist_ok=True)
    rows = []; made = 0
    res = float(cfg["preprocess"].get("resolution", 10))
    ts = int(tile)

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
            chips_this_pair = 0
            try:
                # ---- READ t1 ---- (alias-aware)
                b02_1 = open_da_pick(i1["item"], "B02")
                b03_1 = open_da_pick(i1["item"], "B03")
                b04_1 = open_da_pick(i1["item"], "B04")
                b08_1 = open_da_pick(i1["item"], "B08")
                scl1  = open_da_pick(i1["item"], "SCL")
                
                scl1 = scl1.rio.reproject_match(b02_1, resampling=rasterio.enums.Resampling.nearest)

                # ---- READ t0 (match grid) ----
                b02_0 = open_da_pick(i0["item"], "B02").rio.reproject_match(b02_1)
                b03_0 = open_da_pick(i0["item"], "B03").rio.reproject_match(b03_1)
                b04_0 = open_da_pick(i0["item"], "B04").rio.reproject_match(b04_1)
                b08_0 = open_da_pick(i0["item"], "B08").rio.reproject_match(b08_1)
                scl0  = open_da_pick(i0["item"], "SCL").rio.reproject_match(b02_1, resampling=rasterio.enums.Resampling.nearest)

                # ---- CLIP ----
                aoi_ref = aoi.to_crs(scl1.rio.crs)

                def clip_all(b02,b03,b04,b08,scl):
                    b02 = b02.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    b03 = b03.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    b04 = b04.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    b08 = b08.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    scl = scl.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    return b02,b03,b04,b08,scl

                b02_1,b03_1,b04_1,b08_1,scl1 = clip_all(b02_1,b03_1,b04_1,b08_1,scl1)
                b02_0,b03_0,b04_0,b08_0,scl0 = clip_all(b02_0,b03_0,b04_0,b08_0,scl0)

                # ---- STACK + MASK ----
                t1_stack = np.stack([b02_1.values,b03_1.values,b04_1.values,b08_1.values])
                t0_stack = np.stack([b02_0.values,b03_0.values,b04_0.values,b08_0.values])

                keep1 = cloud_mask_from_scl(scl1); keep1[np.isnan(scl1.values)] = True
                keep0 = cloud_mask_from_scl(scl0); keep0[np.isnan(scl0.values)] = True
                t1_stack = apply_mask(t1_stack, keep1)
                t0_stack = apply_mask(t0_stack, keep0)

                H,W = t1_stack.shape[1:]
                if H < ts or W < ts:
                    print(f"Pair {i} too small; skipping")
                    continue
                print(f"- pair {i}: dims {H}x{W}")

                transform = b02_1.rio.transform()
                chip_id = 0
                for y in range(0, H - ts + 1, ts):
                    for x in range(0, W - ts + 1, ts):
                        c1 = t1_stack[:, y:y+ts, x:x+ts]
                        c0 = t0_stack[:, y:y+ts, x:x+ts]
                        if np.isfinite(c1).mean() < 0.05 or np.isfinite(c0).mean() < 0.05:
                            continue

                        out0 = chips_dir / f"s2_t0_{i}_{chip_id}.npy"
                        out1 = chips_dir / f"s2_t1_{i}_{chip_id}.npy"
                        np.save(out0, c0.astype("float32"))
                        np.save(out1, c1.astype("float32"))

                        x0_map,y0_map = transform*(x,y)
                        x1_map,y1_map = transform*(x+ts,y+ts)
                        xmin,xmax = (x0_map,x1_map) if x0_map<=x1_map else (x1_map,x0_map)
                        ymin,ymax = (y1_map,y0_map) if y1_map<=y0_map else (y0_map,y1_map)

                        rows.append(dict(
                            chip_id=f"s2_{i}_{chip_id}",
                            split="train",
                            t0_npy=str(out0), t1_npy=str(out1),
                            xmin=float(xmin), ymin=float(ymin),
                            xmax=float(xmax), ymax=float(ymax),
                            width=ts, height=ts, res=res, crs=str(b02_1.rio.crs),
                            mask_npy="data/labels/placeholder.npy"
                        ))
                        chip_id += 1; made += 1; chips_this_pair += 1

                print(f"  wrote {chips_this_pair} chips for pair {i}")

            except Exception as e:
                # Helpful context to debug asset issues quickly
                try:
                    akeys0 = list(pystac.Item.from_dict(i0["item"]).assets.keys())
                except Exception:
                    akeys0 = ["<unavailable>"]
                try:
                    akeys1 = list(pystac.Item.from_dict(i1["item"]).assets.keys())
                except Exception:
                    akeys1 = ["<unavailable>"]
                print(f"Skipping pair {i}: processing error: {e}")
                print(f"  t0 assets: {akeys0}")
                print(f"  t1 assets: {akeys1}")
                continue

    df = pd.DataFrame(rows)
    Path(out_index).parent.mkdir(parents=True, exist_ok=True)
    if len(df) == 0:
        print("No chips were written; check AOI overlap, cloud mask, or assets naming.")
    df.to_parquet(out_index)
    print(f"Wrote {made} paired S2 chips -> {out_index}")


# -------------- CLI --------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--t0", nargs=2, required=True)
    ap.add_argument("--t1", nargs=2, required=True)
    ap.add_argument("--out_index", required=True)
    ap.add_argument("--tile_size", type=int, default=256)
    a = ap.parse_args()
    main(a.config, a.items, a.t0[0], a.t0[1], a.t1[0], a.t1[1], a.out_index, a.tile_size)
