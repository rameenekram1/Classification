"""
Canola Cotyledon (Coty) vs Non-Coty Binary Classifier  -  v5
============================================================
Improvements over v4  (tuned for drone-captured field imagery):
  * Toned-down spatial augmentation — prevents cropping out small cotyledons
  * Reduced ColorJitter — preserves the green-hue signal that separates coty from grass
  * Lower RandAugment magnitude (7 vs 9)
  * MixUp off by default (blending coty+non-coty is harmful when non-coty contains green vegetation)
  * CutMix added as alternative (preserves local regions)
  * Staged unfreezing: freeze backbone for N epochs, then unfreeze with lower backbone LR
  * Higher classifier dropout (0.4) for EfficientNet to reduce overfitting on 12k images
  * Gradient accumulation support for effective larger batch sizes
  * TTA (test-time augmentation) option for final evaluation
  * Focal loss option for hard-example mining
  * Per-class metrics logged every epoch

USAGE:
    python train_coty_classifier_v5.py \
        --model efficientnet_b2 \
        --data  /home/rameen/final_dataset_v7_1to1_split \
        --name  coty_effnetb2_v5 \
        --epochs 80 --batch 32 --patience 25
"""

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def build_torchvision_model(model_name: str, num_classes: int = 2,
                             head_dropout: float = 0.3) -> nn.Module:
    if model_name not in _WEIGHTS_MAP:
        raise ValueError(f"Unsupported model '{model_name}'. Pick from {list(_WEIGHTS_MAP)}")
    constructor = getattr(tv_models, model_name)
    model = constructor(weights=_WEIGHTS_MAP[model_name])

    if hasattr(model, "classifier"):
        # EfficientNet family: classifier = [Dropout, Linear]
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Cannot replace head for model: {model_name}")
    return model


def freeze_backbone(model):
    """Freeze everything except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def get_param_groups(model, backbone_lr: float, head_lr: float):
    """Separate backbone and head into different param groups with different LRs."""
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


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


# ─── CutMix ──────────────────────────────────────────────────────────────────

def cutmix_batch(x, y, alpha: float = 1.0):
    """CutMix: cut and paste patches between images. Better than MixUp for
    tasks where local spatial information matters (small objects in noisy bg)."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.shape
    idx = torch.randperm(B, device=x.device)

    # Sample bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

    # Adjust lambda to actual area ratio
    lam = 1.0 - float((y2 - y1) * (x2 - x1)) / (H * W)
    return x_mixed, y, y[idx], lam


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalCrossEntropyLoss(nn.Module):
    """Focal loss for hard-example mining. Reduces loss for well-classified
    examples, focusing training on ambiguous/misclassified samples."""

    def __init__(self, weight=None, gamma: float = 2.0,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer_weight = weight  # stored manually
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)  # probability of correct class
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_data_loaders(data_dir: str, imgsz: int, batch_size: int,
                     workers: int, seed: int, aug_strength: str = "medium"):
    """
    Augmentation tuned for drone-captured cotyledon imagery with HIGH intra-class
    variance: coty ranges from tiny 2-leaf seedlings to large leafy plants filling
    the frame.  Non-coty contains green grass, straw, soil — visually similar.

    Design principles:
    - Wider scale range to handle both tiny seedlings and frame-filling plants
    - Moderate color jitter — enough to generalize lighting but not destroy
      the subtle shape/texture cues that separate coty leaves from grass
    - Strong occlusion simulation (RandomErasing) — straw/grass cross over leaves
    - Moderate spatial transforms — coty can be anywhere in the bbox crop
    """

    if aug_strength == "light":
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(imgsz, scale=(0.70, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.12, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
        ])
    elif aug_strength == "medium":
        # DEFAULT: balanced for high intra-class variance (tiny seedlings + large plants)
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(imgsz, scale=(0.60, 1.0), ratio=(0.82, 1.22)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=18, translate=(0.12, 0.12), scale=(0.88, 1.12)),
            transforms.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.20, hue=0.03),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.18), ratio=(0.3, 3.3), value=0),
        ])
    else:  # "strong" — use if medium still overfits
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(imgsz, scale=(0.50, 1.0), ratio=(0.78, 1.28)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=25, translate=(0.15, 0.15), scale=(0.82, 1.18)),
            transforms.ColorJitter(brightness=0.38, contrast=0.38, saturation=0.28, hue=0.04),
            transforms.RandAugment(num_ops=3, magnitude=8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.30, scale=(0.02, 0.22), ratio=(0.3, 3.3), value=0),
        ])

    # MATCH PRODUCTION INFERENCE: force-stretch resize (not center-crop)
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


# ─── TTA (Test-Time Augmentation) ────────────────────────────────────────────

@torch.no_grad()
def predict_with_tta(model, images, imgsz):
    """Simple TTA: original + hflip + vflip. Average logits."""
    logits = model(images)
    logits += model(torch.flip(images, dims=[3]))  # hflip
    logits += model(torch.flip(images, dims=[2]))  # vflip
    return logits / 3.0


# ─── Validate ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, criterion, use_tta: bool = False, imgsz: int = 224):
    model.eval()
    losses = 0.0
    n      = 0
    all_y, all_p, all_prob1 = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if use_tta:
            out = predict_with_tta(model, imgs, imgsz)
        else:
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
    return losses / n, float(acc), float(f1), float(auroc), float(ap), y, p, prob1


# ─── Threshold Optimization ───────────────────────────────────────────────────

def find_optimal_threshold(y_true, prob1, classes, metric="f1_macro",
                           num_steps=200):
    """
    Sweep thresholds on prob(non_coty) to find the one that maximizes the
    chosen metric.  prob1 = P(class 1) from softmax; class 0 = cn_coty,
    class 1 = non_coty (alphabetical ImageFolder order).

    Returns (best_threshold, best_metric_value, results_at_best).
    """
    thresholds = np.linspace(0.05, 0.95, num_steps)
    best_t, best_val = 0.5, -1.0
    best_info = {}

    for t in thresholds:
        preds = (prob1 >= t).astype(int)  # 1 = non_coty if prob1 >= t
        f1_mac = f1_score(y_true, preds, average="macro", zero_division=0)
        f1_per = f1_score(y_true, preds, average=None, zero_division=0)
        acc_val = (y_true == preds).mean()
        cm = confusion_matrix(y_true, preds)

        if metric == "f1_macro":
            score = f1_mac
        elif metric == "f1_min":
            score = f1_per.min()  # maximize worst-class F1
        elif metric == "balanced_acc":
            # average per-class recall
            per_class_recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
            score = per_class_recall.mean()
        else:
            score = f1_mac

        if score > best_val:
            best_val = score
            best_t   = t
            best_info = {
                "threshold": float(t),
                "accuracy": float(acc_val),
                "f1_macro": float(f1_mac),
                "confusion_matrix": cm,
            }

    # Print sweep summary
    print(f"\n{'='*60}")
    print(f"THRESHOLD OPTIMIZATION  (metric={metric})")
    print(f"{'='*60}")
    print(f"  Default threshold (0.50):")
    default_preds = (prob1 >= 0.5).astype(int)
    default_cm = confusion_matrix(y_true, default_preds)
    default_f1 = f1_score(y_true, default_preds, average="macro", zero_division=0)
    default_prec, default_rec, default_f1c, _ = precision_recall_fscore_support(
        y_true, default_preds, average=None, zero_division=0)
    print(f"    F1-macro={default_f1:.4f}")
    for i, cls in enumerate(classes):
        print(f"    {cls:<12}  P={default_prec[i]:.4f}  R={default_rec[i]:.4f}  F1={default_f1c[i]:.4f}")
    print(f"    CM:\n{default_cm}")

    print(f"\n  Optimal threshold = {best_t:.4f}:")
    opt_preds = (prob1 >= best_t).astype(int)
    opt_prec, opt_rec, opt_f1c, _ = precision_recall_fscore_support(
        y_true, opt_preds, average=None, zero_division=0)
    print(f"    F1-macro={best_info['f1_macro']:.4f}  accuracy={best_info['accuracy']:.4f}")
    for i, cls in enumerate(classes):
        print(f"    {cls:<12}  P={opt_prec[i]:.4f}  R={opt_rec[i]:.4f}  F1={opt_f1c[i]:.4f}")
    print(f"    CM:\n{best_info['confusion_matrix']}")

    # Also show threshold optimized for balanced per-class recall
    print(f"\n  --- Also showing threshold for balanced recall ---")
    best_bal_t, best_bal = 0.5, -1.0
    for t in thresholds:
        preds = (prob1 >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        per_class_recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        bal = per_class_recall.min()  # maximize worst-class recall
        if bal > best_bal:
            best_bal = bal
            best_bal_t = t
    bal_preds = (prob1 >= best_bal_t).astype(int)
    bal_cm = confusion_matrix(y_true, bal_preds)
    bal_prec, bal_rec, bal_f1c, _ = precision_recall_fscore_support(
        y_true, bal_preds, average=None, zero_division=0)
    bal_f1 = f1_score(y_true, bal_preds, average="macro", zero_division=0)
    print(f"  Balanced-recall threshold = {best_bal_t:.4f}:")
    print(f"    F1-macro={bal_f1:.4f}")
    for i, cls in enumerate(classes):
        print(f"    {cls:<12}  P={bal_prec[i]:.4f}  R={bal_rec[i]:.4f}  F1={bal_f1c[i]:.4f}")
    print(f"    CM:\n{bal_cm}")
    print(f"{'='*60}\n")

    return best_t, best_val, best_info


# ─── Train ────────────────────────────────────────────────────────────────────

def train_pytorch(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir    = os.path.join("runs", "classify", args.name)
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    train_loader, val_loader, classes, train_counts = get_data_loaders(
        args.data, args.imgsz, args.batch, args.workers, args.seed, args.aug_strength
    )

    model = build_torchvision_model(args.model, num_classes=len(classes),
                                     head_dropout=args.head_dropout).to(device)

    # ── Class-weighted loss ──
    if args.class_weights == "balanced":
        inv = 1.0 / np.clip(train_counts, 1, None)
        w   = inv / inv.sum() * len(classes)
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"Class weights (balanced): {class_weight.cpu().tolist()}")
    else:
        class_weight = None

    # ── Loss function ──
    if args.focal_gamma > 0:
        criterion = FocalCrossEntropyLoss(
            weight=class_weight, gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing)
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight,
                                         label_smoothing=args.label_smoothing)

    # ── Staged unfreezing ──
    if args.freeze_epochs > 0:
        print(f"\n>>> Freezing backbone for first {args.freeze_epochs} epochs (head-only training)")
        freeze_backbone(model)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr0, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr0,
                                weight_decay=args.weight_decay)

    # LR schedule: linear warmup -> cosine annealing
    warmup_epochs = max(1, args.warmup_epochs)
    total_cosine = max(1, args.epochs - warmup_epochs)
    warmup  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine  = CosineAnnealingLR(optimizer, T_max=total_cosine, eta_min=args.lr0 * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type == "cuda"))
    ema    = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    best_metric = -1.0
    no_improve  = 0
    best_path   = os.path.join(weights_dir, "best.pt")
    last_path   = os.path.join(weights_dir, "last.pt")
    log_path    = os.path.join(save_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_f1,val_auroc,val_ap,"
                "val_p_coty,val_r_coty,val_p_noncoty,val_r_noncoty,lr,time_s\n")

    print(f"\n{'Epoch':>5} {'Tr_Loss':>8} {'Tr_Acc':>7} {'Va_Loss':>8} {'Va_Acc':>7} "
          f"{'Va_F1':>7} {'AUROC':>6} {'AP':>6} {'LR':>9}  {'t':>5}")
    print("-" * 88)

    accum_steps = args.grad_accum

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Staged unfreezing: unfreeze backbone after freeze_epochs ──
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            print(f"\n>>> Unfreezing backbone at epoch {epoch} with backbone_lr={args.lr0 * args.backbone_lr_mult:.6f}")
            unfreeze_backbone(model)
            param_groups = get_param_groups(model, args.lr0 * args.backbone_lr_mult, args.lr0)
            optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
            # Reset scheduler for remaining epochs
            remaining = args.epochs - epoch + 1
            warmup2 = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=2)
            cosine2 = CosineAnnealingLR(optimizer, T_max=max(1, remaining - 2),
                                         eta_min=args.lr0 * 0.01)
            scheduler = SequentialLR(optimizer, schedulers=[warmup2, cosine2], milestones=[2])
            scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type == "cuda"))

        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # CutMix (preferred over MixUp for this task)
            do_cutmix = (args.cutmix_alpha > 0) and (np.random.rand() < args.cutmix_prob)
            do_mix    = (not do_cutmix) and (args.mixup_alpha > 0) and (np.random.rand() < args.mixup_prob)

            if do_cutmix:
                imgs, y_a, y_b, lam = cutmix_batch(imgs, labels, alpha=args.cutmix_alpha)
            elif do_mix:
                # Standard MixUp fallback
                lam_val = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                idx = torch.randperm(imgs.size(0), device=device)
                imgs = lam_val * imgs + (1 - lam_val) * imgs[idx]
                y_a, y_b, lam = labels, labels[idx], float(lam_val)
            else:
                y_a = y_b = labels
                lam = 1.0

            with torch.amp.autocast('cuda', enabled=(args.amp and device.type == "cuda")):
                outputs = model(imgs)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                loss = loss / accum_steps  # scale for accumulation

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

            with torch.no_grad():
                tr_loss    += loss.item() * accum_steps * imgs.size(0)
                tr_correct += (outputs.argmax(1) == labels).sum().item()
                tr_total   += imgs.size(0)

        scheduler.step()

        eval_model = ema.ema if ema is not None else model
        v_loss, v_acc, v_f1, v_auroc, v_ap, y_val, p_val, _ = validate(
            eval_model, val_loader, device, criterion, use_tta=False, imgsz=args.imgsz
        )
        t_loss = tr_loss / tr_total
        t_acc  = tr_correct / tr_total
        lr     = optimizer.param_groups[-1]["lr"]  # head LR
        dt     = time.time() - t0

        # Per-class precision/recall
        prec_c, rec_c, _, _ = precision_recall_fscore_support(
            y_val, p_val, average=None, zero_division=0)

        print(f"{epoch:>5} {t_loss:>8.4f} {t_acc:>7.4f} {v_loss:>8.4f} {v_acc:>7.4f} "
              f"{v_f1:>7.4f} {v_auroc:>6.4f} {v_ap:>6.4f} {lr:>9.6f}  {dt:>4.1f}s")
        with open(log_path, "a") as f:
            p0, r0 = (prec_c[0], rec_c[0]) if len(prec_c) > 0 else (0, 0)
            p1, r1 = (prec_c[1], rec_c[1]) if len(prec_c) > 1 else (0, 0)
            f.write(f"{epoch},{t_loss:.6f},{t_acc:.6f},{v_loss:.6f},{v_acc:.6f},{v_f1:.6f},"
                    f"{v_auroc:.6f},{v_ap:.6f},{p0:.4f},{r0:.4f},{p1:.4f},{r1:.4f},"
                    f"{lr:.8f},{dt:.2f}\n")

        # Save last
        torch.save(model.state_dict(), last_path)

        # Best by val F1
        metric = v_f1
        if metric > best_metric:
            best_metric = metric
            no_improve  = 0
            torch.save(eval_model.state_dict(), best_path)
            print(f"      >>> new best val_F1={best_metric:.4f}  (acc={v_acc:.4f}, "
                  f"AUROC={v_auroc:.4f})  -> {best_path}")
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

    # ── Final report on best checkpoint ──
    print("\nFinal val report on best checkpoint:")
    model_eval = build_torchvision_model(args.model, num_classes=len(classes),
                                          head_dropout=args.head_dropout).to(device)
    model_eval.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model_eval.eval()
    _, acc, f1, auroc, ap, y, p, prob1 = validate(model_eval, val_loader, device, criterion)
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

    # ── Threshold optimization on standard predictions ──
    print("\n--- Threshold optimization (standard) ---")
    opt_t, opt_val, _ = find_optimal_threshold(y, prob1, classes, metric="f1_macro")
    print(f"  >>> USE threshold={opt_t:.4f} at inference for best F1-macro")

    # ── TTA evaluation ──
    if args.tta:
        print("\n--- TTA (hflip + vflip) evaluation on best checkpoint ---")
        _, tta_acc, tta_f1, tta_auroc, tta_ap, tta_y, tta_p, tta_prob1 = validate(
            model_eval, val_loader, device, criterion, use_tta=True, imgsz=args.imgsz
        )
        tta_cm = confusion_matrix(tta_y, tta_p)
        tta_prec, tta_rec, tta_f1c, _ = precision_recall_fscore_support(
            tta_y, tta_p, average=None, zero_division=0)
        print(f"  TTA accuracy  = {tta_acc:.4f}")
        print(f"  TTA f1_macro  = {tta_f1:.4f}")
        print(f"  TTA AUROC     = {tta_auroc:.4f}")
        print(f"  TTA AP        = {tta_ap:.4f}")
        print(f"  TTA per-class:")
        for i, cls in enumerate(classes):
            print(f"    {cls:<12}  P={tta_prec[i]:.4f}  R={tta_rec[i]:.4f}  F1={tta_f1c[i]:.4f}")
        print(f"  TTA confusion matrix:  classes={classes}")
        print(tta_cm)

        # ── Threshold optimization on TTA predictions ──
        print("\n--- Threshold optimization (TTA) ---")
        tta_opt_t, _, _ = find_optimal_threshold(tta_y, tta_prob1, classes, metric="f1_macro")
        print(f"  >>> USE threshold={tta_opt_t:.4f} at inference with TTA for best F1-macro")

    # ── Save optimal threshold to file ──
    thresh_path = os.path.join(save_dir, "optimal_threshold.txt")
    with open(thresh_path, "w") as f:
        f.write(f"standard_threshold={opt_t:.4f}\n")
        if args.tta:
            f.write(f"tta_threshold={tta_opt_t:.4f}\n")
    print(f"\n  Optimal thresholds saved to: {thresh_path}")

    return best_path, save_dir


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="efficientnet_b2",
                   help=f"One of: {list(_WEIGHTS_MAP)}")
    p.add_argument("--data",  type=str, required=True,
                   help="Dataset root with train/<class>/ and val/<class>/")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz",  type=int, default=224)
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--lr0",    type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.08)
    p.add_argument("--warmup_epochs",   type=int,   default=3)
    p.add_argument("--class_weights",   type=str,   default="balanced",
                   choices=["balanced", "none"])

    # Augmentation
    p.add_argument("--aug_strength", type=str, default="medium",
                   choices=["light", "medium", "strong"],
                   help="Augmentation intensity (medium recommended for ~12k images)")
    p.add_argument("--head_dropout", type=float, default=0.4,
                   help="Dropout before classifier head")

    # MixUp / CutMix
    p.add_argument("--mixup_alpha", type=float, default=0.0,
                   help="MixUp beta param (0 disables — recommended for this task)")
    p.add_argument("--mixup_prob",  type=float, default=0.0)
    p.add_argument("--cutmix_alpha", type=float, default=0.0,
                   help="CutMix beta param (0 disables; try 1.0 if needed)")
    p.add_argument("--cutmix_prob",  type=float, default=0.0)

    # Focal loss
    p.add_argument("--focal_gamma", type=float, default=0.0,
                   help="Focal loss gamma (0 = standard CE; try 1.5-2.0)")

    # Staged unfreezing
    p.add_argument("--freeze_epochs", type=int, default=5,
                   help="Epochs to freeze backbone (0 = train everything from start)")
    p.add_argument("--backbone_lr_mult", type=float, default=0.1,
                   help="Backbone LR = lr0 * this (lower = more stable fine-tuning)")

    # EMA / AMP
    p.add_argument("--amp",  action="store_true", default=True)
    p.add_argument("--ema",  action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.9995)

    # Training control
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch * grad_accum)")
    p.add_argument("--tta", action="store_true", default=True,
                   help="Run TTA evaluation at end")

    p.add_argument("--device",  type=str, default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name",    type=str, default="coty_effnetb2_v5")
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()

    train_pytorch(args)


if __name__ == "__main__":
    main()

# ─── RECOMMENDED COMMANDS ────────────────────────────────────────────────────
#
# RUN 1: Recommended first run (effective batch=64 via grad_accum=2)
#   python3 train_coty_classifier_v5.py \
#     --model efficientnet_b2 \
#     --data /home/rameen/final_dataset_v7_1to1_split \
#     --imgsz 224 --batch 32 --grad_accum 2 --epochs 80 \
#     --patience 25 --lr0 0.001 \
#     --freeze_epochs 5 --backbone_lr_mult 0.1 \
#     --head_dropout 0.4 --ema_decay 0.9995 \
#     --aug_strength medium \
#     --label_smoothing 0.08 \
#     --name coty_effnetb2_v5_run1 --device 0
#
# RUN 2: + CutMix (if Run 1 overfits or val_acc plateaus early)
#   python3 train_coty_classifier_v5.py \
#     --model efficientnet_b2 \
#     --data /home/rameen/final_dataset_v7_1to1_split \
#     --imgsz 224 --batch 32 --grad_accum 2 --epochs 80 \
#     --patience 25 --lr0 0.001 \
#     --freeze_epochs 5 --backbone_lr_mult 0.1 \
#     --head_dropout 0.4 --ema_decay 0.9995 \
#     --aug_strength medium \
#     --cutmix_alpha 1.0 --cutmix_prob 0.3 \
#     --label_smoothing 0.08 \
#     --name coty_effnetb2_v5_run2_cutmix --device 0
#
# RUN 3: + Focal Loss (if confusion matrix shows one class dominating errors)
#   python3 train_coty_classifier_v5.py \
#     --model efficientnet_b2 \
#     --data /home/rameen/final_dataset_v7_1to1_split \
#     --imgsz 224 --batch 32 --grad_accum 2 --epochs 80 \
#     --patience 25 --lr0 0.001 \
#     --freeze_epochs 5 --backbone_lr_mult 0.1 \
#     --head_dropout 0.4 --ema_decay 0.9995 \
#     --aug_strength medium \
#     --focal_gamma 1.5 \
#     --label_smoothing 0.05 \
#     --name coty_effnetb2_v5_run3_focal --device 0
#
# RUN 4: Strong aug (if all above overfit — train_acc >> val_acc by 5%+)
#   python3 train_coty_classifier_v5.py \
#     --model efficientnet_b2 \
#     --data /home/rameen/final_dataset_v7_1to1_split \
#     --imgsz 224 --batch 32 --grad_accum 2 --epochs 100 \
#     --patience 30 --lr0 0.001 \
#     --freeze_epochs 5 --backbone_lr_mult 0.1 \
#     --head_dropout 0.45 --ema_decay 0.9995 \
#     --aug_strength strong \
#     --cutmix_alpha 1.0 --cutmix_prob 0.25 \
#     --label_smoothing 0.10 \
#     --name coty_effnetb2_v5_run4_strong --device 0