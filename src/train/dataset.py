import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ChipDataset(Dataset):
    def __init__(self, parquet_path, split=None):
        self.df = pd.read_parquet(parquet_path)
        if split:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

    def _percentile_stretch(self, x):
        # x: (C,H,W) with possible NaNs; stretch each channel using finite pixels only
        y = x.copy()
        for c in range(y.shape[0]):
            ch = y[c]
            finite = np.isfinite(ch)
            if finite.sum() < 10:  # too few pixels → skip
                continue
            lo = np.nanpercentile(ch[finite], 2)
            hi = np.nanpercentile(ch[finite], 98)
            if hi - lo <= 1e-6:
                continue
            ch = (ch - lo) / (hi - lo + 1e-6)
            y[c] = np.clip(ch, 0, 1)
        return y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        t0 = np.load(r.t0_npy).astype("float32")  # (4,H,W)
        t1 = np.load(r.t1_npy).astype("float32")

        # valid where ALL 4 bands are finite in both times
        valid0 = np.isfinite(t0).all(axis=0)
        valid1 = np.isfinite(t1).all(axis=0)
        valid = (valid0 & valid1).astype("float32")  # (H,W)

        # normalize per-chip using only finite pixels
        t0 = self._percentile_stretch(t0)
        t1 = self._percentile_stretch(t1)

        # fill remaining NaNs with 0 to be safe
        t0 = np.nan_to_num(t0, nan=0.0, posinf=0.0, neginf=0.0)
        t1 = np.nan_to_num(t1, nan=0.0, posinf=0.0, neginf=0.0)

        x = np.concatenate([t0, t1], axis=0)  # (8,H,W)

        return (
            torch.from_numpy(x),                  # float tensor
            torch.from_numpy(valid),              # (H,W) float mask in {0,1}
            r.chip_id
        )
