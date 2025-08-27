
# # src/train/train_model.py
# import os, argparse
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader
# from torch import nn, optim

# from .dataset import ChipDataset
# from .model import ChangeUNet

# def ensure_split(parquet_path: str, val_frac: float = 0.2, seed: int = 42):
#     df = pd.read_parquet(parquet_path)
#     need_split = ("split" not in df.columns) or (df["split"].nunique() == 1) or (df[df["split"] == "val"].empty)
#     if need_split:
#         rng = np.random.default_rng(seed)
#         idx = np.arange(len(df))
#         rng.shuffle(idx)
#         n_val = max(1, int(round(val_frac * len(df))))
#         val_idx = set(idx[:n_val])
#         split = np.array(["train"] * len(df), dtype=object)
#         for i in range(len(df)):
#             if i in val_idx:
#                 split[i] = "val"
#         df["split"] = split
#         df.to_parquet(parquet_path)
#         print(f"[split] Created/updated split with val_frac={val_frac:.2f}:",
#               df["split"].value_counts().to_dict())
#     else:
#         print(f"[split] Using existing split:", df["split"].value_counts().to_dict())
#     return df

# def masked_mse(pred, target, mask):
#     # pred/target: [B,1,H,W]; mask: [B,1,H,W] in {0,1}
#     diff2 = (pred - target) ** 2
#     diff2 = diff2 * mask
#     denom = mask.sum().clamp_min(1.0)
#     return diff2.sum() / denom

# def train_one_run(args):
#     parquet_path = args.parquet
#     df = ensure_split(parquet_path, val_frac=args.val_frac, seed=args.seed)
#     print(f"Total dataset size: {len(df)}")
#     print("Split distribution:", df["split"].value_counts().to_dict())

#     train_ds = ChipDataset(parquet_path, split="train")
#     val_ds   = ChipDataset(parquet_path, split="val")
#     if len(val_ds) == 0:
#         raise ValueError("No validation data after split creation — please check your parquet.")

#     train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
#     val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = ChangeUNet(in_ch=8, out_ch=1).to(device)
#     opt = optim.Adam(model.parameters(), lr=args.lr)

#     for epoch in range(1, args.epochs + 1):
#         # -------- Train --------
#         model.train()
#         tr_loss = 0.0
#         for x, valid, _ in train_dl:
#             x = x.to(device).float()                 # [B,8,H,W]
#             valid = valid.to(device).unsqueeze(1)    # [B,1,H,W]

#             # pseudo-target: mean abs-diff over 4 bands
#             diff = torch.abs(x[:, :4] - x[:, 4:])    # [B,4,H,W]
#             y = diff.mean(dim=1, keepdim=True)       # [B,1,H,W]

#             # forward
#             y_pred = model(x)
#             y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)
#             y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

#             loss = masked_mse(y_pred, y, valid)
#             opt.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()

#             tr_loss += loss.item() * x.size(0)
#         tr_loss /= max(1, len(train_ds))

#         # -------- Validate --------
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for x, valid, _ in val_dl:
#                 x = x.to(device).float()
#                 valid = valid.to(device).unsqueeze(1)
#                 diff = torch.abs(x[:, :4] - x[:, 4:])
#                 y = diff.mean(dim=1, keepdim=True)
#                 y_pred = model(x)
#                 y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)
#                 y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
#                 val_loss += masked_mse(y_pred, y, valid).item() * x.size(0)
#         val_loss /= max(1, len(val_ds))

#         print(f"Epoch {epoch:02d}/{args.epochs}  train_loss={tr_loss:.6f}  val_loss={val_loss:.6f}")

#     os.makedirs("outputs/models", exist_ok=True)
#     out_path = "outputs/models/change_unet.pth"
#     torch.save(model.state_dict(), out_path)
#     print(f"Model saved to {out_path}")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet")
#     ap.add_argument("--epochs", type=int, default=5)
#     ap.add_argument("--batch_size", type=int, default=4)
#     ap.add_argument("--lr", type=float, default=1e-3)
#     ap.add_argument("--val_frac", type=float, default=0.2)
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()
#     train_one_run(args)

# src/train/train_model.py

# src/train/train_model.py
import os, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from datetime import datetime
import json

from .dataset import ChipDataset
from .model import ChangeUNet

def ensure_split(parquet_path: str, val_frac: float = 0.2, seed: int = 42):
    df = pd.read_parquet(parquet_path)
    need_split = ("split" not in df.columns) or (df["split"].nunique() == 1) or (df[df["split"] == "val"].empty)
    if need_split:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(df))))
        val_idx = set(idx[:n_val])
        split = np.array(["train"] * len(df), dtype=object)
        for i in range(len(df)):
            if i in val_idx:
                split[i] = "val"
        df["split"] = split
        df.to_parquet(parquet_path)
        print(f"[split] Created/updated split with val_frac={val_frac:.2f}:",
              df["split"].value_counts().to_dict())
    else:
        print(f"[split] Using existing split:", df["split"].value_counts().to_dict())
    return df

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in change detection"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.mse_loss(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def masked_loss(pred, target, mask, loss_fn):
    """Apply loss function only to valid pixels"""
    # pred/target: [B,1,H,W]; mask: [B,1,H,W] in {0,1}
    valid_pixels = mask.sum().clamp_min(1.0)
    
    if loss_fn.__class__.__name__ == 'MSELoss':
        diff2 = (pred - target) ** 2 * mask
        return diff2.sum() / valid_pixels
    elif loss_fn.__class__.__name__ == 'L1Loss':
        diff = torch.abs(pred - target) * mask
        return diff.sum() / valid_pixels
    else:
        # For other losses, compute on flattened valid pixels
        pred_valid = pred[mask > 0]
        target_valid = target[mask > 0]
        return loss_fn(pred_valid, target_valid)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def compute_metrics(pred, target, mask, threshold=0.5):
    """Compute change detection metrics"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Only compute on valid pixels
    pred_valid = pred_binary[mask > 0]
    target_valid = target_binary[mask > 0]
    
    if len(pred_valid) == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
    
    tp = (pred_valid * target_valid).sum().item()
    fp = (pred_valid * (1 - target_valid)).sum().item()
    fn = ((1 - pred_valid) * target_valid).sum().item()
    tn = ((1 - pred_valid) * (1 - target_valid)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

def visualize_predictions(model, val_dl, device, epoch, save_dir="outputs/visualizations"):
    """Save sample predictions for visual inspection"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (x, valid, _) in enumerate(val_dl):
            if i >= 3:  # Only save first 3 batches
                break
                
            x = x.to(device).float()
            valid = valid.to(device).unsqueeze(1)
            
            # Generate target and prediction
            diff = torch.abs(x[:, :4] - x[:, 4:])
            target = diff.mean(dim=1, keepdim=True)
            pred = model(x)
            
            # Save first sample in batch
            sample_idx = 0
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Time 1 (RGB composite)
            t1_rgb = x[sample_idx, [2, 1, 0]].cpu().numpy().transpose(1, 2, 0)
            t1_rgb = np.clip(t1_rgb * 3, 0, 1)  # Enhance for visualization
            axes[0, 0].imshow(t1_rgb)
            axes[0, 0].set_title('Time 1 (RGB)')
            axes[0, 0].axis('off')
            
            # Time 2 (RGB composite)
            t2_rgb = x[sample_idx, [6, 5, 4]].cpu().numpy().transpose(1, 2, 0)
            t2_rgb = np.clip(t2_rgb * 3, 0, 1)  # Enhance for visualization
            axes[0, 1].imshow(t2_rgb)
            axes[0, 1].set_title('Time 2 (RGB)')
            axes[0, 1].axis('off')
            
            # Target change
            target_img = target[sample_idx, 0].cpu().numpy()
            im1 = axes[0, 2].imshow(target_img, cmap='hot', vmin=0, vmax=1)
            axes[0, 2].set_title('Target Change')
            axes[0, 2].axis('off')
            plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # Predicted change
            pred_img = pred[sample_idx, 0].cpu().numpy()
            im2 = axes[1, 0].imshow(pred_img, cmap='hot', vmin=0, vmax=1)
            axes[1, 0].set_title('Predicted Change')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # Difference map
            diff_img = np.abs(pred_img - target_img)
            im3 = axes[1, 1].imshow(diff_img, cmap='coolwarm')
            axes[1, 1].set_title('Prediction Error')
            axes[1, 1].axis('off')
            plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            # Valid mask
            mask_img = valid[sample_idx, 0].cpu().numpy()
            axes[1, 2].imshow(mask_img, cmap='gray')
            axes[1, 2].set_title('Valid Pixels')
            axes[1, 2].axis('off')
            
            plt.suptitle(f'Epoch {epoch} - Batch {i}')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/epoch_{epoch:03d}_batch_{i}.png", dpi=150, bbox_inches='tight')
            plt.close()

def train_one_run(args):
    # Setup directories
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"outputs/logs/training_log_{timestamp}.json"
    
    parquet_path = args.parquet
    df = ensure_split(parquet_path, val_frac=args.val_frac, seed=args.seed)
    print(f"Total dataset size: {len(df)}")
    print("Split distribution:", df["split"].value_counts().to_dict())

    train_ds = ChipDataset(parquet_path, split="train")
    val_ds   = ChipDataset(parquet_path, split="val")
    if len(val_ds) == 0:
        raise ValueError("No validation data after split creation — please check your parquet.")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                         num_workers=min(args.num_workers, 4), pin_memory=torch.cuda.is_available())
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                         num_workers=min(args.num_workers, 4), pin_memory=torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = ChangeUNet(in_ch=8, out_ch=1).to(device)
    
    # Enhanced optimizer and scheduler
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Loss function selection
    if args.loss == 'mse':
        criterion = nn.MSELoss(reduction='none')
    elif args.loss == 'l1':
        criterion = nn.L1Loss(reduction='none')
    elif args.loss == 'focal':
        criterion = FocalLoss()
    else:
        criterion = nn.MSELoss(reduction='none')
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-6)
    
    # Training history
    history = defaultdict(list)
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # -------- Train --------
        model.train()
        tr_loss = 0.0
        tr_metrics = defaultdict(float)
        
        for batch_idx, (x, valid, _) in enumerate(train_dl):
            # Use non_blocking only if CUDA is available
            if torch.cuda.is_available():
                x = x.to(device, non_blocking=True).float()
                valid = valid.to(device, non_blocking=True).unsqueeze(1)
            else:
                x = x.to(device).float()
                valid = valid.to(device).unsqueeze(1)

            # Enhanced pseudo-target generation
            diff = torch.abs(x[:, :4] - x[:, 4:])
            if args.target_mode == 'mean':
                y = diff.mean(dim=1, keepdim=True)
            elif args.target_mode == 'max':
                y = diff.max(dim=1, keepdim=True)[0]
            elif args.target_mode == 'weighted':
                # Weight bands differently (e.g., NIR band more important)
                weights = torch.tensor([0.2, 0.3, 0.3, 0.2]).view(1, 4, 1, 1).to(device)
                y = (diff * weights).sum(dim=1, keepdim=True)
            else:
                y = diff.mean(dim=1, keepdim=True)

            # Forward pass
            y_pred = model(x)
            y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)

            # Compute loss
            if args.loss in ['mse', 'l1']:
                loss = masked_loss(y_pred, y, valid, criterion)
            else:
                loss = criterion(y_pred, y)

            # Backward pass
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            tr_loss += loss.item() * x.size(0)
            
            # Compute metrics
            if batch_idx % args.metrics_freq == 0:
                metrics = compute_metrics(y_pred, y, valid)
                for k, v in metrics.items():
                    tr_metrics[k] += v

        tr_loss /= max(1, len(train_ds))
        for k in tr_metrics:
            tr_metrics[k] /= max(1, len(train_dl) // args.metrics_freq)

        # -------- Validate --------
        model.eval()
        val_loss = 0.0
        val_metrics = defaultdict(float)
        
        with torch.no_grad():
            for batch_idx, (x, valid, _) in enumerate(val_dl):
                # Use non_blocking only if CUDA is available
                if torch.cuda.is_available():
                    x = x.to(device, non_blocking=True).float()
                    valid = valid.to(device, non_blocking=True).unsqueeze(1)
                else:
                    x = x.to(device).float()
                    valid = valid.to(device).unsqueeze(1)
                
                diff = torch.abs(x[:, :4] - x[:, 4:])
                if args.target_mode == 'mean':
                    y = diff.mean(dim=1, keepdim=True)
                elif args.target_mode == 'max':
                    y = diff.max(dim=1, keepdim=True)[0]
                elif args.target_mode == 'weighted':
                    weights = torch.tensor([0.2, 0.3, 0.3, 0.2]).view(1, 4, 1, 1).to(device)
                    y = (diff * weights).sum(dim=1, keepdim=True)
                else:
                    y = diff.mean(dim=1, keepdim=True)
                
                y_pred = model(x)
                y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)
                y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
                
                if args.loss in ['mse', 'l1']:
                    loss = masked_loss(y_pred, y, valid, criterion)
                else:
                    loss = criterion(y_pred, y)
                    
                val_loss += loss.item() * x.size(0)
                
                # Compute metrics
                metrics = compute_metrics(y_pred, y, valid)
                for k, v in metrics.items():
                    val_metrics[k] += v
                    
        val_loss /= max(1, len(val_ds))
        for k in val_metrics:
            val_metrics[k] /= max(1, len(val_dl))

        # Update learning rate
        old_lr = opt.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]['lr']
        
        # Print LR change if it happened
        if current_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.2e} to {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"outputs/models/change_unet_best_{timestamp}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, best_model_path)
        
        # Log progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{args.epochs} ({epoch_time:.1f}s) | "
              f"LR: {current_lr:.2e} | "
              f"Train: {tr_loss:.6f} | Val: {val_loss:.6f} | "
              f"F1: {val_metrics.get('f1', 0):.3f} | IoU: {val_metrics.get('iou', 0):.3f}")
        
        # Store history
        history['epoch'].append(epoch)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        for k, v in val_metrics.items():
            history[f'val_{k}'].append(v)
        
        # Generate visualizations periodically
        if epoch % args.viz_freq == 0:
            visualize_predictions(model, val_dl, device, epoch)
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    # Save final model and history
    final_model_path = f"outputs/models/change_unet_final_{timestamp}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'history': dict(history),
        'args': vars(args)
    }, final_model_path)
    
    # Save training log
    with open(log_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'history': dict(history),
            'best_val_loss': best_val_loss,
            'total_time': total_time,
            'final_epoch': epoch
        }, f, indent=2)
    
    # Plot training curves
    if len(history['epoch']) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train')
        axes[0, 0].plot(history['epoch'], history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(history['epoch'], history['lr'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # F1 Score
        if 'val_f1' in history:
            axes[1, 0].plot(history['epoch'], history['val_f1'])
            axes[1, 0].set_title('F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1')
            axes[1, 0].grid(True)
        
        # IoU Score
        if 'val_iou' in history:
            axes[1, 1].plot(history['epoch'], history['val_iou'])
            axes[1, 1].set_title('IoU Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('IoU')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"outputs/logs/training_curves_{timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved: {best_model_path}")
    print(f"Training log: {log_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", choices=['mse', 'l1', 'focal'], default='mse')
    ap.add_argument("--target_mode", choices=['mean', 'max', 'weighted'], default='mean')
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--metrics_freq", type=int, default=10, help="Compute metrics every N batches")
    ap.add_argument("--viz_freq", type=int, default=5, help="Generate visualizations every N epochs")
    args = ap.parse_args()
    train_one_run(args)