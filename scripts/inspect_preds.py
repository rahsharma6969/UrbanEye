# scripts/inspect_preds.py
import os, numpy as np, rasterio, glob

preds = glob.glob("outputs/preds/*_change.tif")
print("pred files:", len(preds))
for p in preds[:10]:
    with rasterio.open(p) as src:
        a = src.read(1).astype("float32")
        vm = np.isfinite(a)
        m = a[vm]
        if m.size == 0:
            print(os.path.basename(p), "ALL NaN")
            continue
        print(os.path.basename(p),
              "min=", float(np.nanmin(m)),
              "max=", float(np.nanmax(m)),
              "mean=", float(np.nanmean(m)),
              "#>0.2:", int((m>0.2).sum()),
              "#>0.3:", int((m>0.3).sum()),
              "#>0.5:", int((m>0.5).sum()))
