import argparse, yaml, pandas as pd, numpy as np, torch
from pathlib import Path
from rasterio.transform import Affine
from src.models.unet import UNet
from src.post.vectorize import mask_to_polygons

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(config, aoi, t0_start, t0_end, t1_start, t1_end, out, chip_id=None):
    cfg = load_cfg(config)

    # Pick device safely (fallback to CPU if CUDA build isn't present)
    device_cfg = cfg["model"]["device"]
    device = "cuda" if (device_cfg == "cuda" and torch.cuda.is_available()) else "cpu"
    if device_cfg == "cuda" and device == "cpu":
        print("CUDA not available; falling back to CPU.")

    # Load a chips index produced by your tiler
    index_path = Path("outputs/chips_index.parquet")
    if not index_path.exists():
        raise SystemExit("chips index not found: outputs/chips_index.parquet (run make_chips_s2.py first)")
    df = pd.read_parquet(index_path)

    # Expect columns like: t0_npy, t1_npy, xmin, ymin, xmax, ymax, res, crs, width, height
    required_cols = {"t0_npy","t1_npy","xmin","ymin","xmax","ymax","res","crs","width","height"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(f"chips_index.parquet missing required columns. Found: {list(df.columns)}")

    # Choose a chip: by id if provided, else first row
    if chip_id is not None:
        if "chip_id" not in df.columns:
            raise SystemExit("chip_id not present in chips index. Rebuild index with chip_id or omit --chip_id.")
        rows = df[df["chip_id"] == chip_id]
        if rows.empty:
            raise SystemExit(f"chip_id '{chip_id}' not found in chips index.")
        r = rows.iloc[0]
    else:
        r = df.iloc[0]

    # Load paired t0/t1 chips
    t0 = np.load(r.t0_npy).astype("float32")   # shape [C,H,W]
    t1 = np.load(r.t1_npy).astype("float32")   # shape [C,H,W]
    x  = np.concatenate([t0, t1], axis=0)      # [2C,H,W]
    x  = torch.from_numpy(x[None, ...]).to(device)  # add batch dim

    # Build model
    model = UNet(in_ch=cfg["model"]["in_channels"], base=cfg["model"]["base_channels"]).to(device)
    weights = Path("outputs/model_best.pt")
    if not weights.exists():
        raise SystemExit("weights not found: outputs/model_best.pt (train first)")
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state); model.eval()

    # Inference
    with torch.no_grad():
        logits = model(x)              # [1,1,H,W]
        y = torch.sigmoid(logits)[0,0].cpu().numpy()  # [H,W]

    # Threshold -> binary mask
    mask = (y > 0.5).astype("uint8")

    # Real geotransform from chip metadata (top-left origin, pixel size = res)
    # NOTE: This assumes r.xmin/xmax/ymin/ymax are in the chip's CRS (e.g., UTM).
    transform = Affine(r["res"], 0, r["xmin"], 0, -r["res"], r["ymax"])
    crs = r["crs"]  # e.g., 'EPSG:32643'

    # Vectorize to GeoDataFrame in WGS84 and write GeoJSON
    gdf = mask_to_polygons(mask, transform, crs, min_area_px=50)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GeoJSON")
    print("Wrote", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--t0", nargs=2, required=True)        # unused in this MVP, kept for API parity
    ap.add_argument("--t1", nargs=2, required=True)        # unused in this MVP, kept for API parity
    ap.add_argument("--out", required=True)
    ap.add_argument("--chip_id", default=None)             # optional: target a specific chip
    args = ap.parse_args()
    main(args.config, args.aoi, args.t0[0], args.t0[1], args.t1[0], args.t1[1], args.out, args.chip_id)
