import torch, pandas as pd, numpy as np

class ChipDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_path, split=None):
        self.df = pd.read_parquet(parquet_path)
        if split:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        t0 = np.load(r.t0_npy).astype("float32")  # (4,H,W)
        t1 = np.load(r.t1_npy).astype("float32")

        # simple per-chip percentile stretch (0..1)
        def norm(x):
            x = x.copy()
            for c in range(x.shape[0]):
                lo = np.nanpercentile(x[c], 2)
                hi = np.nanpercentile(x[c], 98)
                x[c] = (x[c]-lo)/(hi-lo+1e-6)
            return np.clip(x, 0, 1)

        t0 = norm(t0); t1 = norm(t1)
        return torch.from_numpy(t0), torch.from_numpy(t1), r.chip_id
