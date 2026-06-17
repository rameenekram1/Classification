"""
Clip Classification Script — EfficientNet
==========================================
Reads clipped images from a flat folder, classifies them using the trained
EfficientNet (or any torchvision) model from train_coty_classifier_v4.py,
and sorts them into coty / non_coty subfolders. Also copies companion JSON
files alongside the images.

Input:  Flat folder of clipped images (.jpg/.png) + optional .json per image
Output: Images + JSONs copied into predicted-class subfolders, CSV report, summary

Usage:
    python classify_clips_effinet.py
"""

import csv
import logging
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image

# ============== LOCAL CONFIGURATION ==============
MODEL_PATH   = r"/home/rameen/Desktop/computer_vision/OD_codes/prescout_effinet/effi-50/runs/classify/coty_effnetb2_v12/weights/best.pt"          # Path to best.pt from training
MODEL_NAME   = "efficientnet_b2"                  # Must match what you trained with
INPUT_FOLDER = r"/home/rameen/Desktop/Prescout_pipeline_output/pipeline_output/2026_Pre-Scout_5829_-_NE_18-11-6/object_detection/Tile_images"    # Flat folder of images to classify
OUTPUT_FOLDER= r"/home/rameen/Desktop/Prescout_pipeline_output/pipeline_output/2026_Pre-Scout_5829_-_NE_18-11-6/object_detection/output_v12"

# ============== INFERENCE SETTINGS ==============
BATCH_SIZE            = 32
PREDICTION_IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD  = 0.0    # 0 = keep all predictions; e.g. 0.7 = only keep high-confidence
CLASS_NAMES = ["cn_coty", "non_coty"]  # Must match your training folder names (alphabetical)
# =================================================

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', 'JPG')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Val transform (same as training) ────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize((PREDICTION_IMAGE_SIZE, PREDICTION_IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ─── Model loader ─────────────────────────────────────────────────────────────
def load_model(model_path=None, model_name=None, num_classes=2):
    """Load trained torchvision model from a state_dict .pt file."""
    model_path = model_path or MODEL_PATH
    model_name = model_name or MODEL_NAME

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Rebuild architecture
    weights_map = {
        "efficientnet_b0":    tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b1":    tv_models.EfficientNet_B1_Weights.IMAGENET1K_V1,
        "efficientnet_b2":    tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1,
        "efficientnet_b3":    tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        "efficientnet_b4":    tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        "efficientnet_v2_s":  tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        "efficientnet_v2_m":  tv_models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
        "resnet18":           tv_models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34":           tv_models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50":           tv_models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet101":          tv_models.ResNet101_Weights.IMAGENET1K_V2,
        "mobilenet_v3_small": tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "mobilenet_v3_large": tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "convnext_tiny":      tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "convnext_small":     tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
    }
    model = getattr(tv_models, model_name)(weights=weights_map[model_name])

    # Replace head (same logic as training script)
    if hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    # Load your trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {model_name}  |  Classes: {CLASS_NAMES}")
    return model, device


# ─── Main classify-and-sort ───────────────────────────────────────────────────
def classify_and_sort(input_folder=None, output_folder=None):
    input_folder  = input_folder  or INPUT_FOLDER
    output_folder = output_folder or OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    # File logging
    log_path = os.path.join(output_folder, f"classify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    model, device = load_model(num_classes=len(CLASS_NAMES))

    # Create one subfolder per class
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(output_folder, cls), exist_ok=True)
    # Extra folder for low-confidence predictions (only used if CONFIDENCE_THRESHOLD > 0)
    if CONFIDENCE_THRESHOLD > 0:
        os.makedirs(os.path.join(output_folder, "low_confidence"), exist_ok=True)

    # Collect images
    images = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(IMG_EXTS)
    ])
    logger.info(f"Found {len(images)} images in {input_folder}")

    if not images:
        logger.error("No images found. Check INPUT_FOLDER path.")
        return

    results_list = []

    for i in range(0, len(images), BATCH_SIZE):
        batch_names = images[i:i + BATCH_SIZE]
        batch_tensors = []
        valid_names   = []

        for img_name in batch_names:
            img_path = os.path.join(input_folder, img_name)
            try:
                # PIL handles more formats cleanly than cv2 for RGB
                img = Image.open(img_path).convert("RGB")
                batch_tensors.append(TRANSFORM(img))
                valid_names.append(img_name)
            except Exception as e:
                logger.warning(f"Could not read {img_name}: {e}")

        if not batch_tensors:
            continue

        try:
            batch = torch.stack(batch_tensors).to(device)   # (N, 3, H, W)
            with torch.no_grad():
                logits = model(batch)                        # (N, num_classes)
                probs  = torch.softmax(logits, dim=1)        # (N, num_classes)
                confs, preds = probs.max(dim=1)              # top-1 conf + index

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            traceback.print_exc()
            continue

        for img_name, conf, pred_idx in zip(valid_names,
                                             confs.cpu().numpy(),
                                             preds.cpu().numpy()):
            conf       = float(conf)
            pred_label = CLASS_NAMES[int(pred_idx)]

            # Route low-confidence images to their own folder
            if CONFIDENCE_THRESHOLD > 0 and conf < CONFIDENCE_THRESHOLD:
                dest_folder = os.path.join(output_folder, "low_confidence")
            else:
                dest_folder = os.path.join(output_folder, pred_label)

            # Copy image
            shutil.copy2(
                os.path.join(input_folder, img_name),
                os.path.join(dest_folder, img_name)
            )

            # Copy companion JSON if present
            json_name = os.path.splitext(img_name)[0] + ".json"
            src_json  = os.path.join(input_folder, json_name)
            if os.path.exists(src_json):
                shutil.copy2(src_json, os.path.join(dest_folder, json_name))

            results_list.append({
                "image":           img_name,
                "predicted_label": pred_label,
                "confidence":      round(conf, 4),
            })

            logger.info(f"  {img_name:55s} -> {pred_label}  ({conf:.3f})")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_folder, "classification_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(results_list)
    logger.info(f"CSV saved -> {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(results_list)
    pct   = lambda n: f"{n / total * 100:.1f}%" if total > 0 else "N/A"

    lines = [
        "=" * 50,
        "  CLASSIFICATION SUMMARY",
        "=" * 50,
        f"  Total images classified : {total}",
        "-" * 50,
    ]
    for cls in CLASS_NAMES:
        n = sum(1 for r in results_list if r["predicted_label"] == cls)
        lines.append(f"  {cls:25s} : {n:>5}  ({pct(n)})")
    lines.append("=" * 50)

    for line in lines:
        logger.info(line)

    report_path = os.path.join(output_folder, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary -> {report_path}")

    return results_list


if __name__ == "__main__":
    classify_and_sort()
