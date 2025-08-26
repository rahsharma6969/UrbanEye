# src/train/baseline_change_map.py
import os, numpy as np, pandas as pd
from PIL import Image

def to_rgb(chip):  # chip: (4,H,W) = [B02,B03,B04,B08]
    R, G, B = chip[2], chip[1], chip[0]

    def stretch(x):
        if np.all(np.isnan(x)):
            return np.zeros_like(x, dtype=np.float32)
        lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
        y = (x - lo) / (hi - lo + 1e-6)
        return np.clip(y, 0, 1)

    rgb = np.stack([stretch(R), stretch(G), stretch(B)], -1)
    if np.all(np.isnan(rgb)):
        rgb = np.zeros_like(rgb, dtype=np.float32)
    return (rgb * 255).astype(np.uint8)

def change_heat(t0, t1):
    d = np.abs(t1 - t0)

    if np.all(np.isnan(d)):
        return np.zeros(t0.shape[1:], dtype=np.uint8)

    d = d / (np.nanstd(d, axis=(1, 2), keepdims=True) + 1e-6)
    h = np.nanmean(d, axis=0)   # (H,W)

    if np.all(np.isnan(h)):
        return np.zeros_like(h, dtype=np.uint8)

    lo, hi = np.nanpercentile(h, 2), np.nanpercentile(h, 98)
    h = (h - lo) / (hi - lo + 1e-6)
    h = np.clip(h, 0, 1)
    return (h * 255).astype(np.uint8)

df = pd.read_parquet("outputs/chips_index_s2.parquet")
os.makedirs("outputs/quick_changes", exist_ok=True)

for k in range(min(12, len(df))):
    r = df.iloc[k]
    t0 = np.load(r.t0_npy).astype("float32")
    t1 = np.load(r.t1_npy).astype("float32")

    rgb0 = to_rgb(t0)
    rgb1 = to_rgb(t1)
    heat = change_heat(t0, t1)

    Image.fromarray(rgb0).save(f"outputs/quick_changes/{r.chip_id}_t0.png")
    Image.fromarray(rgb1).save(f"outputs/quick_changes/{r.chip_id}_t1.png")
    Image.fromarray(heat).save(f"outputs/quick_changes/{r.chip_id}_heat.png")

print("Wrote previews to outputs/quick_changes (NaNs handled safely)")
