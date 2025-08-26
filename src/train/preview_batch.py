import torch
from torch.utils.data import DataLoader
from dataset import ChipDataset

ds = ChipDataset("outputs/chips_index_s2.parquet")
print("chips:", len(ds))
dl = DataLoader(ds, batch_size=4, shuffle=True)
t0, t1, ids = next(iter(dl))
print("batch shapes:", t0.shape, t1.shape)  # expect [B,4,256,256]
print("sample ids:", list(ids))
