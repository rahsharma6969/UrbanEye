import argparse, yaml, geopandas as gpd, rasterio, numpy as np, pandas as pd
from shapely.geometry import box
from rasterio.features import rasterize
from pathlib import Path

def load_cfg(p):
    import yaml
    with open(p,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def main(config, index_path):
    cfg = load_cfg(config)
    idx = pd.read_parquet(index_path)
    labels_dir = Path(cfg["paths"]["labels_dir"])
    for _, row in idx.iterrows():
        chip_bbox = box(row.xmin, row.ymin, row.xmax, row.ymax)
        label_files = list(labels_dir.glob("*.geojson"))
        if not label_files: continue
        gdf = gpd.read_file(label_files[0])  # starter: single label file
        gdf = gdf.to_crs(row.crs)
        inter = gdf[gdf.geometry.intersects(chip_bbox)]
        if inter.empty:
            mask = np.zeros((row.height, row.width), dtype=np.uint8)
        else:
            shapes = [(geom, 1) for geom in inter.intersection(chip_bbox).geometry]
            transform = rasterio.Affine(row.res, 0, row.xmin, 0, -row.res, row.ymax)
            mask = rasterize(shapes, out_shape=(row.height, row.width), transform=transform, fill=0, dtype="uint8")
        out_path = Path(row.mask_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", driver="GTiff", height=row.height, width=row.width, count=1, dtype="uint8",
                           transform=rasterio.Affine(row.res,0,row.xmin,0,-row.res,row.ymax), crs=row.crs) as dst:
            dst.write(mask, 1)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True)
    args = ap.parse_args()
    main(args.config, args.index)
