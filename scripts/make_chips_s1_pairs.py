import argparse, json, yaml, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rioxarray as rxr, rasterio
import planetary_computer as pc, pystac
from src.preprocess.s1_prep import denoise_sar

def load_cfg(p): return yaml.safe_load(open(p,"r",encoding="utf-8"))
def dt(item): return datetime.fromisoformat(item["properties"]["datetime"].replace("Z",""))
def sign(item_dict):
    it=pystac.Item.from_dict(item_dict); it=pc.sign(it)
    return {k.lower(): v.href for k,v in it.assets.items()}
def open_da(h): 
    da=rxr.open_rasterio(h, masked=True).squeeze()
    _=da.isel(x=slice(0,1), y=slice(0,1)).values
    return da

def aoi_union(aoi_dir: Path):
    files=list(aoi_dir.glob("*.geojson")); 
    if not files: sys.exit(f"No AOIs in {aoi_dir}")
    gdfs=[]
    for f in files:
        g=gpd.read_file(f); g=g[g.geometry.notna()]
        if not g.empty: gdfs.append(g.to_crs("EPSG:4326"))
    m=pd.concat(gdfs, ignore_index=True).explode(index_parts=False, ignore_index=True)
    try: u=m.geometry.union_all()
    except AttributeError: u=m.geometry.unary_union
    return gpd.GeoDataFrame(geometry=[u], crs="EPSG:4326")

def pair_nearest(t0, t1):
    if not t0 or not t1: return []
    t1t=[(dt(i), i) for i in t1]
    return [(i0, min(t1t, key=lambda x: abs(x[0]-dt(i0)))[1]) for i0 in t0]

def main(config, items_json, t0_start, t0_end, t1_start, t1_end, out_index, tile):
    cfg=load_cfg(config)
    items=json.loads(Path(items_json).read_text())
    if not items: sys.exit("No STAC items")
    aoi=aoi_union(Path(cfg["paths"]["aoi_dir"]))
    s1=[it for it in items if it["collection"]=="sentinel-1-grd"]
    t0=[it for it in s1 if t0_start<=it["start"]<=t0_end]
    t1=[it for it in s1 if t1_start<=it["start"]<=t1_end]
    pairs=pair_nearest(t0,t1); print(f"Paired {len(pairs)} S1 scenes")

    chips_dir=Path(cfg["paths"]["chips_dir"]); chips_dir.mkdir(parents=True, exist_ok=True)
    rows=[]; made=0; res=float(cfg["preprocess"].get("resolution",10)); ts=int(tile)

    for i,(i0,i1) in enumerate(pairs):
        a0=sign(i0["item"]); a1=sign(i1["item"])
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="YES",
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                          CPL_VSIL_CURL_USE_HEAD="NO",
                          CPL_VSIL_CURL_CACHE_SIZE="20000000",
                          GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1"):
            vv1=open_da(a1["vv"]).rio.reproject(aoi.crs, resolution=res)
            vh1=open_da(a1["vh"]).rio.reproject(aoi.crs, resolution=res)
            vv0=open_da(a0["vv"]).rio.reproject_match(vv1)
            vh0=open_da(a0["vh"]).rio.reproject_match(vv1)

            # clip to AOI
            aoi1=aoi.to_crs(vv1.rio.crs)
            vv1=vv1.rio.clip(aoi1.geometry,aoi1.crs,drop=True); vh1=vh1.rio.clip(aoi1.geometry,aoi1.crs,drop=True)
            vv0=vv0.rio.clip(aoi1.geometry,aoi1.crs,drop=True); vh0=vh0.rio.clip(aoi1.geometry,aoi1.crs,drop=True)

            # denoise (simple median)
            vv1.values=denoise_sar(vv1.values[None,...], size=3)[0]
            vh1.values=denoise_sar(vh1.values[None,...], size=3)[0]
            vv0.values=denoise_sar(vv0.values[None,...], size=3)[0]
            vh0.values=denoise_sar(vh0.values[None,...], size=3)[0]

            H,W=vv1.shape; 
            if H<ts or W<ts: continue
            transform=vv1.rio.transform()

            chip_id=0
            for y in range(0, H-ts+1, ts):
                for x in range(0, W-ts+1, ts):
                    c1=np.stack([vv1.values[y:y+ts,x:x+ts], vh1.values[y:y+ts,x:x+ts]])
                    c0=np.stack([vv0.values[y:y+ts,x:x+ts], vh0.values[y:y+ts,x:x+ts]])
                    if np.isfinite(c1).mean()<0.3 or np.isfinite(c0).mean()<0.3: continue
                    out0=chips_dir/f"s1_t0_{i}_{chip_id}.npy"
                    out1=chips_dir/f"s1_t1_{i}_{chip_id}.npy"
                    np.save(out0, c0.astype("float32")); np.save(out1, c1.astype("float32"))
                    x0,y0=transform*(x,y); x1,y1=transform*(x+ts,y+ts)
                    xmin,xmax=min(x0,x1),max(x0,x1); ymin,ymax=min(y0,y1),max(y0,y1)
                    rows.append(dict(
                        chip_id=f"s1_{i}_{chip_id}", split="train",
                        t0_npy=str(out0), t1_npy=str(out1),
                        xmin=float(xmin), ymin=float(ymin), xmax=float(xmax), ymax=float(ymax),
                        width=ts, height=ts, res=res, crs=str(vv1.rio.crs),
                        mask_npy="data/labels/placeholder.npy"
                    ))
                    chip_id+=1; made+=1

    df=pd.DataFrame(rows); Path(out_index).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_index); print(f"Wrote {made} paired S1 chips -> {out_index}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--t0", nargs=2, required=True)
    ap.add_argument("--t1", nargs=2, required=True)
    ap.add_argument("--out_index", required=True)
    ap.add_argument("--tile_size", type=int, default=256)
    a=ap.parse_args()
    main(a.config, a.items, a.t0[0], a.t0[1], a.t1[0], a.t1[1], a.out_index, a.tile_size)
