import numpy as np, torch
from torch.utils.data import Dataset

class ChangeDataset(Dataset):
    def __init__(self, rows, aug=None):
        self.rows = rows
        self.aug = aug
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        t0 = np.load(r.t0_npy).astype(np.float32)  # [C,H,W]
        t1 = np.load(r.t1_npy).astype(np.float32)
        x = np.concatenate([t0, t1], axis=0)      # [2C,H,W]
        y = np.load(r.mask_npy).astype(np.float32) # [H,W]
        if self.aug:
            # TODO: basic flips/rotations
            pass
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).unsqueeze(0)
        return x, y
