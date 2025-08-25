import time
import rasterio
from rasterio.errors import RasterioIOError
import planetary_computer as pc
import pystac

import argparse, json, yaml, sys
import numpy as np, pandas as pd
from pathlib import Path
import geopandas as gpd
import rioxarray as rxr
from src.preprocess.s2_prep import cloud_mask_from_scl, apply_mask

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _sign_item_assets(item_dict):
    """Return lowercased asset hrefs with fresh SAS tokens."""
    item = pystac.Item.from_dict(item_dict)
    item = pc.sign(item)
    return {k.lower(): v.href for k, v in item.assets.items()}

def _open_band(href, retries=3, sleep=1.0):
    """Open a remote COG band with retries; returns a rioxarray DataArray (squeezed)."""
    last_err = None
    for _ in range(retries):
        try:
            da = rxr.open_rasterio(href, masked=True).squeeze()
            # Touch a tiny window to fail fast if token/band is bad
            _ = da.isel(x=slice(0,1), y=slice(0,1)).values
            return da
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise last_err

def _load_aoi_union(aoi_dir: Path) -> gpd.GeoDataFrame:
    files = list(aoi_dir.glob("*.geojson"))
    if not files:
        print(f"No AOI files in {aoi_dir}. Add at least one .geojson.", file=sys.stderr)
        sys.exit(1)
    gdfs = []
    for f in files:
        g = gpd.read_file(f)
        g = g[g.geometry.notna()]
        if not g.empty:
            gdfs.append(g.to_crs("EPSG:4326"))
    if not gdfs:
        print("AOI files contain no valid geometries.", file=sys.stderr)
        sys.exit(1)
    merged = pd.concat(gdfs, ignore_index=True).explode(index_parts=False, ignore_index=True)
    try:
        union_geom = merged.geometry.union_all()
    except AttributeError:
        union_geom = merged.geometry.unary_union
    return gpd.GeoDataFrame(geometry=[union_geom], crs="EPSG:4326")

def main(config, items_json, out_index, tile_size_override=None):
    cfg = load_cfg(config)
    items = json.loads(Path(items_json).read_text())
    if len(items) == 0:
        print("No STAC items found in outputs/stac_items.json — run search first or loosen cloud filter.", file=sys.stderr)
        Path(out_index).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([]).to_parquet(out_index)
        return

    aoi_dir = Path(cfg["paths"]["aoi_dir"])
    aoi_union = _load_aoi_union(aoi_dir)

    chips_dir = Path(cfg["paths"]["chips_dir"]); chips_dir.mkdir(parents=True, exist_ok=True)
    rows = []; made = 0
    tile_size = int(tile_size_override or cfg["preprocess"].get("tile_size", 256))
    res = float(cfg["preprocess"].get("resolution", 10))

    print(f"Processing {len(items)} items; tile_size={tile_size}, res={res}")

    for it in items:
        # 1) Sign assets for THIS item
        try:
            signed = _sign_item_assets(it["item"])
            href_b02 = signed["b02"]; href_b03 = signed["b03"]
            href_b04 = signed["b04"]; href_b08 = signed["b08"]
            href_scl = signed["scl"]
        except Exception as e:
            print(f"Skipping item: signing failed: {e}")
            continue

        # 2) Robust remote reads (GDAL/VSICURL) for THIS item
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="YES",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
            CPL_VSIL_CURL_USE_HEAD="NO",
            CPL_VSIL_CURL_CACHE_SIZE="20000000",
            GDAL_HTTP_MAX_RETRY="3",
            GDAL_HTTP_RETRY_DELAY="1"
        ):
            try:
                b02 = _open_band(href_b02); b03 = _open_band(href_b03)
                b04 = _open_band(href_b04); b08 = _open_band(href_b08)
                scl = _open_band(href_scl)
            except Exception as e:
                print(f"Skipping item: read error: {e}")
                continue

            # 3) Reproject to common CRS/res (use SCL’s CRS)
            target_crs = scl.rio.crs
            try:
                b02 = b02.rio.reproject(target_crs, resolution=res)
                b03 = b03.rio.reproject(target_crs, resolution=res)
                b04 = b04.rio.reproject(target_crs, resolution=res)
                b08 = b08.rio.reproject(target_crs, resolution=res)
                scl = scl.rio.reproject(target_crs, resolution=res)
            except Exception as e:
                print(f"Skipping item: reproject error: {e}")
                continue

            # 4) Clip to AOI
            try:
                aoi_proj = aoi_union.to_crs(target_crs)
                b02 = b02.rio.clip(aoi_proj.geometry, aoi_proj.crs, drop=True)
                b03 = b03.rio.clip(aoi_proj.geometry, aoi_proj.crs, drop=True)
                b04 = b04.rio.clip(aoi_proj.geometry, aoi_proj.crs, drop=True)
                b08 = b08.rio.clip(aoi_proj.geometry, aoi_proj.crs, drop=True)
                scl = scl.rio.clip(aoi_proj.geometry, aoi_proj.crs, drop=True)
            except Exception as e:
                print(f"Skipping item: clip error: {e}")
                continue

            # 5) Stack & cloud mask (NaN‑safe)
            stack = np.stack([b02.values, b03.values, b04.values, b08.values])  # [4,H,W]
            keep = cloud_mask_from_scl(scl)      # True = keep
            keep[np.isnan(scl.values)] = True    # don’t kill tiles due to missing SCL
            stack = apply_mask(stack, keep)

            H, W = stack.shape[1], stack.shape[2]
            if H < tile_size or W < tile_size:
                print(f"Item too small after clip: H={H}, W={W} (tile={tile_size}). Skipping.")
                continue

            valid_ratio = np.isfinite(stack).mean()
            print(f"- item dims: {H}x{W}, valid={valid_ratio:.2%}")

            # 6) Tile this item
            transform = b02.rio.transform()
            chip_id = 0
            for y in range(0, H - tile_size + 1, tile_size):
                for x in range(0, W - tile_size + 1, tile_size):
                    chip = stack[:, y:y+tile_size, x:x+tile_size]
                    # need some real signal
                    if np.isfinite(chip).mean() < 0.3:
                        continue

                    out_path = chips_dir / f"{it['start']}_{it['end']}_{chip_id}.npy"
                    np.save(out_path, chip.astype(np.float32))

                    # pixel -> map coords using Affine
                    x0, y0 = transform * (x, y)
                    x1, y1 = transform * (x + tile_size, y + tile_size)
                    xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
                    ymin, ymax = (y1, y0) if y1 <= y0 else (y0, y1)

                    rows.append(dict(
                        chip_id=f"{it['start']}_{it['end']}_{chip_id}",
                        split="train",   # TODO: stratified split later
                        # TODO: pair true t0/t1 in next iteration
                        t0_npy=str(out_path),
                        t1_npy=str(out_path),
                        xmin=float(xmin), ymin=float(ymin), xmax=float(xmax), ymax=float(ymax),
                        width=tile_size, height=tile_size,
                        res=res,
                        crs=str(target_crs),
                        mask_npy="data/labels/placeholder.npy"
                    ))
                    chip_id += 1
                    made += 1

    df = pd.DataFrame(rows)
    Path(out_index).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_index)
    print(f"Wrote {made} chips -> {out_index}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out_index", required=True)
    ap.add_argument("--tile_size", type=int, default=None)
    args = ap.parse_args()
    main(args.config, args.items, args.out_index, args.tile_size)
