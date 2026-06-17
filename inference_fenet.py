import os
import csv
import shutil
import torch
import logging
from PIL import Image
from torchvision import models, transforms

# ============== LOCAL CONFIGURATION ==============

MODEL_PATH      = r"/home/rameen/Desktop/computer_vision/OD_codes/prescout_effinet/effi-50/runs/classify/coty_effnetb2_v12/weights/best.pt"
MODEL_NAME      = "efficientnet_b2"     # must match training
PIPELINE_OUTPUT = r"/home/rameen/Desktop/Prescout_pipeline_output/pipeline_output"
OUTPUT_FOLDER   = "output_v12"          # saved inside each field's Tile_images/

# Class names — must match training order (alphabetical)
CLASS_NAMES = ["cn_coty", "non_coty"]

# ============== INFERENCE SETTINGS ==============
BATCH_SIZE            = 32
PREDICTION_IMAGE_SIZE = 224
COPY_MODE             = True    # True = copy files, False = move files

# =================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

TRANSFORM = transforms.Compose([
    transforms.Resize((PREDICTION_IMAGE_SIZE, PREDICTION_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    constructor = getattr(models, MODEL_NAME)
    model = constructor(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    logger.info(f"Model loaded: {MODEL_NAME} | classes: {CLASS_NAMES}")
    return model, device


def predict_batch(model, device, pil_images):
    tensors = torch.stack([TRANSFORM(img.convert("RGB")) for img in pil_images]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensors), dim=1)
    top_confs, top_idxs = probs.max(dim=1)
    return [(CLASS_NAMES[i.item()], c.item()) for i, c in zip(top_idxs, top_confs)]


def run(data_path, output_path, model=None, device=None):
    # Create output class folders
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(output_path, cls), exist_ok=True)

    # Collect all images from the flat input folder
    image_names = sorted(f for f in os.listdir(data_path) if f.lower().endswith(IMAGE_EXTENSIONS))
    logger.info(f"Found {len(image_names)} images in {data_path}")

    if model is None or device is None:
        model, device = load_model()

    results = []
    counts  = {cls: 0 for cls in CLASS_NAMES}

    for i in range(0, len(image_names), BATCH_SIZE):
        batch_names  = image_names[i:i + BATCH_SIZE]
        batch_images = []
        valid_names  = []

        for name in batch_names:
            try:
                batch_images.append(Image.open(os.path.join(data_path, name)))
                valid_names.append(name)
            except Exception:
                logger.warning(f"Could not read: {name}")

        if not batch_images:
            continue

        for name, (label, conf) in zip(valid_names, predict_batch(model, device, batch_images)):
            src = os.path.join(data_path, name)
            dst = os.path.join(output_path, label, name)
            shutil.copy2(src, dst) if COPY_MODE else shutil.move(src, dst)

            counts[label] += 1
            results.append({'image': name, 'predicted_label': label, 'confidence': round(conf, 4)})
            logger.info(f"{name} -> {label} ({conf:.2%})")

    # Save summary CSV
    csv_path = os.path.join(output_path, "classification_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'predicted_label', 'confidence'])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"\n{'='*50}")
    for cls, n in counts.items():
        logger.info(f"  {cls}: {n} images")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    from pathlib import Path

    pipeline_root = Path(PIPELINE_OUTPUT)
    field_dirs    = sorted(d for d in pipeline_root.iterdir() if d.is_dir())

    logger.info(f"Model      : {MODEL_PATH}")
    logger.info(f"Model name : {MODEL_NAME}")
    logger.info(f"Fields     : {len(field_dirs)}")
    logger.info(f"Output dir : {OUTPUT_FOLDER}\n")

    # Load model once — reuse across all fields
    model, device = load_model()

    grand_total = {cls: 0 for cls in CLASS_NAMES}

    for field_dir in field_dirs:
        tile_dir   = field_dir / "object_detection" / "Tile_images"
        output_dir = tile_dir / OUTPUT_FOLDER

        if not tile_dir.exists():
            logger.warning(f"[skip] no Tile_images: {field_dir.name}")
            continue

        # Skip if already processed (CSV exists)
        if (output_dir / "classification_results.csv").exists():
            logger.info(f"[skip] already done: {field_dir.name}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Field: {field_dir.name}")
        logger.info(f"  Input : {tile_dir}")
        logger.info(f"  Output: {output_dir}")

        run(data_path=str(tile_dir), output_path=str(output_dir),
            model=model, device=device)

        # Accumulate totals from this field's CSV
        csv_path = output_dir / "classification_results.csv"
        if csv_path.exists():
            import csv as _csv
            with open(csv_path) as f:
                for row in _csv.DictReader(f):
                    grand_total[row['predicted_label']] = \
                        grand_total.get(row['predicted_label'], 0) + 1

    logger.info(f"\n{'='*60}")
    logger.info("ALL FIELDS DONE")
    for cls, n in grand_total.items():
        logger.info(f"  {cls}: {n} total detections")
    logger.info(f"{'='*60}")
