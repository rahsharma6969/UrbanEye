# src/infer/infer_change.py
import os, argparse, numpy as np, pandas as pd, torch, rasterio
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader
from PIL import Image

# import using the package path (works with: python -m src.infer.infer_change)
from src.train.model import ChangeUNet
from src.train.dataset import ChipDataset

def load_flexible_state_dict(weights_path, device):
    """
    Loads either:
      - a raw state_dict (mapping of param_name -> tensor), or
      - a checkpoint dict containing 'model_state_dict'.
    """
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    # if it's already a raw state_dict, return as-is
    return ckpt

def save_png(arr01, out_png):
    arr01 = np.clip(arr01, 0, 1)
    img = (arr01 * 255).astype(np.uint8)
    Image.fromarray(img).save(out_png)

def save_geotiff(prob, meta_row, out_tif):
    H, W = prob.shape
    xmin, ymin, xmax, ymax = meta_row["xmin"], meta_row["ymin"], meta_row["xmax"], meta_row["ymax"]
    crs = meta_row["crs"]
    transform = from_bounds(xmin, ymin, xmax, ymax, W, H)
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    with rasterio.open(
        out_tif, "w",
        driver="GTiff",
        height=H, width=W, count=1, dtype="float32",
        crs=crs, transform=transform,
        compress="LZW"
    ) as dst:
        dst.write(prob.astype("float32"), 1)

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChangeUNet(in_ch=8, out_ch=1).to(device)

    state = load_flexible_state_dict(args.weights, device)
    model.load_state_dict(state)
    model.eval()

    df = pd.read_parquet(args.parquet)

    # dataset that also returns the row index
    class _DS(ChipDataset):
        def __getitem__(self, idx):
            x, valid, chip_id = super().__getitem__(idx)
            return x, valid, idx, chip_id

    ds = _DS(args.parquet, split=args.split) if args.split else _DS(args.parquet)
    if args.max_chips > 0:
        ds.df = ds.df.head(args.max_chips).reset_index(drop=True)
        df = df.head(args.max_chips).reset_index(drop=True)

    dl = DataLoader(ds, batch_size=1, shuffle=False)
    os.makedirs(args.out_dir, exist_ok=True)

    for (x, valid, idx_tensor, chip_id) in dl:
        idx = int(idx_tensor.item())
        meta_row = df.iloc[idx]

        x = x.to(device).float()                    # [1,8,H,W]
        y_pred = model(x).cpu().numpy()[0, 0]       # (H,W) in [0,1]
        vm = valid.numpy()[0]                       # (H,W) {0,1}
        y_pred = np.where(vm > 0.5, y_pred, 0.0)    # mask invalid

        out_png = os.path.join(args.out_dir, f"{meta_row['chip_id']}_change.png")
        out_tif = os.path.join(args.out_dir, f"{meta_row['chip_id']}_change.tif")
        save_png(y_pred, out_png)
        save_geotiff(y_pred, meta_row, out_tif)

    print(f"Saved predictions to {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet")
    ap.add_argument("--weights", required=True)  # point to your best checkpoint
    ap.add_argument("--split", default="val")    # or "train" / "" for all
    ap.add_argument("--max_chips", type=int, default=12)
    ap.add_argument("--out_dir", default="outputs/preds")
    args = ap.parse_args()
    main(args)
