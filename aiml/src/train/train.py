import argparse, yaml, pandas as pd, torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from src.dataset.change_dataset import ChangeDataset
from src.models.unet import UNet
from tqdm import tqdm

def load_cfg(p):
    with open(p,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2*(pred*target).sum()+eps
    den = pred.sum()+target.sum()+eps
    return 1 - (num/den)

def main(config, index):
    cfg = load_cfg(config)
    device = cfg["model"]["device"]
    df = pd.read_parquet(index)
    # Expect df to have columns: t0_npy, t1_npy, mask_npy, split
    tr = df[df.split=="train"].reset_index(drop=True)
    va = df[df.split=="val"].reset_index(drop=True)
    train_ds = ChangeDataset(tr)
    val_ds   = ChangeDataset(va)
    train_dl = DataLoader(train_ds, batch_size=cfg["model"]["batch_size"], shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=cfg["model"]["batch_size"], shuffle=False, num_workers=2)

    model = UNet(in_ch=cfg["model"]["in_channels"], base=cfg["model"]["base_channels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["model"]["lr"])
    bce = nn.BCEWithLogitsLoss()

    best_val = 1e9
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)

    for epoch in range(cfg["model"]["epochs"]):
        model.train()
        tr_loss = 0.0
        for x,y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg['model']['epochs']}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = 0.5*bce(logits,y) + 0.5*dice_loss(logits,y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()*x.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss = 0.5*bce(logits,y) + 0.5*dice_loss(logits,y)
                va_loss += loss.item()*x.size(0)
        va_loss /= len(val_ds)

        print(f"Epoch {epoch+1}: train={tr_loss:.4f} val={va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), out_dir/"model_best.pt")
            print("Saved best -> outputs/model_best.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True)
    args = ap.parse_args()
    main(args.config, args.index)
