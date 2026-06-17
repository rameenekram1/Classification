"""
Canola Cotyledon (Coty) vs Non-Coty Binary Classifier  -  v4
============================================================
Improvements over v3:
  * AdamW + weight_decay
  * Label smoothing
  * LR warmup -> CosineAnnealing
  * Mixed precision (AMP)
  * Stronger augmentation: RandAugment, RandomErasing, RandomAffine, stronger ColorJitter
  * MixUp (optional, default on)
  * EMA of model weights, eval on EMA
  * Class-weighted loss for imbalanced classes
  * Checkpoint by val F1-macro (not raw accuracy)
  * Val transform aligned with production: Resize((imgsz, imgsz))  (force-stretch)
  * drop_last=True on train loader, seeded for reproducibility

USAGE:
    python train_coty_classifier_v4.py \
        --model efficientnet_b0 \
        --data  /home/rameen/final_dataset_v7_1to1_final \
        --name  coty_effnetb0_v4 \
        --epochs 60 --batch 64
"""

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as tv_models
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score,
)


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Model ────────────────────────────────────────────────────────────────────

_WEIGHTS_MAP = {
    "efficientnet_b0":    tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    "efficientnet_b1":    tv_models.EfficientNet_B1_Weights.IMAGENET1K_V1,
    "efficientnet_b2":    tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1,
    "efficientnet_b3":    tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1,
    "efficientnet_v2_s":  tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    "resnet18":           tv_models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet50":           tv_models.ResNet50_Weights.IMAGENET1K_V2,
    "mobilenet_v3_small": tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    "mobilenet_v3_large": tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
}


def build_torchvision_model(model_name: str, num_classes: int = 2) -> nn.Module:
    if model_name not in _WEIGHTS_MAP:
        raise ValueError(f"Unsupported model '{model_name}'. Pick from {list(_WEIGHTS_MAP)}")
    constructor = getattr(tv_models, model_name)
    model = constructor(weights=_WEIGHTS_MAP[model_name])

    if hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Cannot replace head for model: {model_name}")
    return model


# ─── EMA ──────────────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ─── MixUp ────────────────────────────────────────────────────────────────────

def mixup_batch(x, y, alpha: float = 0.2):
    """Standard MixUp; returns (mixed_x, y_a, y_b, lam)."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], float(lam)


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_data_loaders(data_dir: str, imgsz: int, batch_size: int, workers: int, seed: int):
    """Train aug stronger than v3; val transform matches production (force-stretch)."""
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=20, translate=(0.20, 0.20), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0),
    ])

    # MATCH PRODUCTION INFERENCE: force-stretch resize (Resize((H, W))) not center-crop
    val_tfm = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfm)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tfm)

    print(f"\nClasses: {train_ds.classes}")
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    # Per-class counts (for class-weighted loss)
    counts = np.bincount([y for _, y in train_ds.samples], minlength=len(train_ds.classes))
    print(f"Train per-class: {dict(zip(train_ds.classes, counts.tolist()))}\n")

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True,
                              drop_last=True, generator=g, persistent_workers=workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True,
                              persistent_workers=workers > 0)
    return train_loader, val_loader, train_ds.classes, counts


# ─── Validate ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    losses = 0.0
    n      = 0
    all_y, all_p, all_prob1 = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(imgs)
        losses += criterion(out, labels).item() * imgs.size(0)
        n      += imgs.size(0)
        prob = torch.softmax(out, dim=1)
        all_y.append(labels.cpu().numpy())
        all_p.append(out.argmax(1).cpu().numpy())
        all_prob1.append(prob[:, 1].cpu().numpy())
    y      = np.concatenate(all_y)
    p      = np.concatenate(all_p)
    prob1  = np.concatenate(all_prob1)
    f1     = f1_score(y, p, average="macro")
    acc    = (y == p).mean()
    try:    auroc = roc_auc_score(y, prob1)
    except ValueError:    auroc = float("nan")
    try:    ap    = average_precision_score(y, prob1)
    except ValueError:    ap    = float("nan")
    return losses / n, float(acc), float(f1), float(auroc), float(ap), y, p


# ─── Train ────────────────────────────────────────────────────────────────────

def train_pytorch(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir    = os.path.join("runs", "classify", args.name)
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    train_loader, val_loader, classes, train_counts = get_data_loaders(
        args.data, args.imgsz, args.batch, args.workers, args.seed
    )

    model = build_torchvision_model(args.model, num_classes=len(classes)).to(device)

    # Class-weighted CE + label smoothing
    if args.class_weights == "balanced":
        inv = 1.0 / np.clip(train_counts, 1, None)
        w   = inv / inv.sum() * len(classes)
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"Class weights (balanced): {class_weight.cpu().tolist()}")
    else:
        class_weight = None
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)

    # LR schedule: linear warmup -> cosine annealing
    warmup_epochs = max(1, args.warmup_epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=args.lr0 * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type == "cuda"))
    ema    = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    best_metric = -1.0
    no_improve  = 0
    best_path   = os.path.join(weights_dir, "best.pt")
    last_path   = os.path.join(weights_dir, "last.pt")
    log_path    = os.path.join(save_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_f1,val_auroc,val_ap,lr,time_s\n")

    print(f"{'Epoch':>5} {'Tr_Loss':>8} {'Tr_Acc':>7} {'Va_Loss':>8} {'Va_Acc':>7} {'Va_F1':>7} "
          f"{'AUROC':>6} {'AP':>6} {'LR':>9}  {'t':>5}")
    print("-" * 84)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            do_mix = (args.mixup_alpha > 0) and (np.random.rand() < args.mixup_prob)
            if do_mix:
                imgs, y_a, y_b, lam = mixup_batch(imgs, labels, alpha=args.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(args.amp and device.type == "cuda")):
                outputs = model(imgs)
                if do_mix:
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            with torch.no_grad():
                tr_loss    += loss.item() * imgs.size(0)
                tr_correct += (outputs.argmax(1) == labels).sum().item()
                tr_total   += imgs.size(0)

        scheduler.step()

        eval_model = ema.ema if ema is not None else model
        v_loss, v_acc, v_f1, v_auroc, v_ap, _, _ = validate(eval_model, val_loader, device, criterion)
        t_loss = tr_loss / tr_total
        t_acc  = tr_correct / tr_total
        lr     = optimizer.param_groups[0]["lr"]
        dt     = time.time() - t0

        print(f"{epoch:>5} {t_loss:>8.4f} {t_acc:>7.4f} {v_loss:>8.4f} {v_acc:>7.4f} {v_f1:>7.4f} "
              f"{v_auroc:>6.4f} {v_ap:>6.4f} {lr:>9.6f}  {dt:>4.1f}s")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{t_loss:.6f},{t_acc:.6f},{v_loss:.6f},{v_acc:.6f},{v_f1:.6f},"
                    f"{v_auroc:.6f},{v_ap:.6f},{lr:.8f},{dt:.2f}\n")

        # Save last (training model, not EMA — for resuming)
        torch.save(model.state_dict(), last_path)

        # Best is the EMA model when EMA is on, else the training model. Selected by val F1.
        metric = v_f1
        if metric > best_metric:
            best_metric = metric
            no_improve  = 0
            torch.save(eval_model.state_dict(), best_path)
            print(f"      ✓ new best val_F1={best_metric:.4f}  (val_acc={v_acc:.4f})  → {best_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no F1 improvement for {args.patience} epochs).")
                break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Best val F1-macro : {best_metric:.4f}")
    print(f"  Best weights      : {best_path}")
    print(f"  Train log         : {log_path}")
    print("=" * 70)

    # Final report on best checkpoint
    print("\nFinal val report on best checkpoint:")
    model_eval = build_torchvision_model(args.model, num_classes=len(classes)).to(device)
    model_eval.load_state_dict(torch.load(best_path, map_location=device))
    model_eval.eval()
    _, acc, f1, auroc, ap, y, p = validate(model_eval, val_loader, device, criterion)
    cm   = confusion_matrix(y, p)
    prec, rec, f1c, _ = precision_recall_fscore_support(y, p, average=None, zero_division=0)
    print(f"  accuracy  = {acc:.4f}")
    print(f"  f1_macro  = {f1:.4f}")
    print(f"  AUROC     = {auroc:.4f}")
    print(f"  AP        = {ap:.4f}")
    print(f"  per-class:")
    for i, cls in enumerate(classes):
        print(f"    {cls:<12}  P={prec[i]:.4f}  R={rec[i]:.4f}  F1={f1c[i]:.4f}")
    print(f"  confusion matrix (rows=true, cols=pred):  classes={classes}")
    print(cm)

    return best_path, save_dir


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="efficientnet_b0",
                   help=f"One of: {list(_WEIGHTS_MAP)}")
    p.add_argument("--data",  type=str, required=True,
                   help="Dataset root with train/<class>/ and val/<class>/")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz",  type=int, default=224)
    p.add_argument("--batch",  type=int, default=64)
    p.add_argument("--lr0",    type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_epochs",   type=int,   default=3)
    p.add_argument("--class_weights",   type=str,   default="balanced",
                   choices=["balanced", "none"])
    p.add_argument("--mixup_alpha", type=float, default=0.2,
                   help="MixUp beta param (0 disables MixUp)")
    p.add_argument("--mixup_prob",  type=float, default=0.5,
                   help="Probability of applying MixUp on a batch")
    p.add_argument("--amp",  action="store_true", default=True,
                   help="Use mixed precision (AMP) for training")
    p.add_argument("--ema",  action="store_true", default=True,
                   help="Maintain EMA of model weights and evaluate on EMA")
    p.add_argument("--ema_decay", type=float, default=0.9998)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--device",  type=str, default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name",    type=str, default="coty_effnetb0_v4")
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()

    train_pytorch(args)


if __name__ == "__main__":
    main()

# python3 train_coty_classifier_v4.py --model efficientnet_b0 --data  /home/rameen/final_dataset_v7_1to1_split --name  coty_effnetb0_v8 --epochs 100 --batch 64
# python3 train_coty_classifier_v4.py --model efficientnet_b2 --data /home/rameen/final_dataset_v7_1to1_split --imgsz 224 --batch 32 --epochs 80 --patience 18 --lr0 0.0008 --ema_decay 0.9995 --name coty_effnetb2_v10 --device 0
# python3 train_coty_classifier_v4.py --model efficientnet_b2 --data /home/rameen/final_dataset_v7_1to1_final --imgsz 224 --batch 32 --epochs 100 --patience 28 --lr0 0.0005 --warmup_epochs 5 --ema_decay 0.9999 --label_smoothing 0.05 --mixup_alpha 0 --weight_decay 2e-4 --name coty_effnetb2_v12 --device 0