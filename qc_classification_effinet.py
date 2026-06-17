import os
import csv
import cv2
import torch
import logging
import numpy as np
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# ============== LOCAL CONFIGURATION ==============
MODEL_PATH   = r"/home/rameen/Desktop/computer_vision/OD_codes/prescout_effinet/effi-50/runs/classify/coty_effnetb2_v12/weights/best.pt"
MODEL_NAME   = "efficientnet_b2"       # Must match what was used for training
DATA_PATH    = r"/home/rameen/final_dataset_v7_1to1_final/val"
OUTPUT_PATH  = r"/home/rameen/Desktop/computer_vision/OD_codes/prescout_effinet/effi-50/runs/classify/coty_effnetb2_v12/output_v12"

# Class folders inside DATA_PATH (alphabetical order, same as training)
CLASS_FOLDERS = ["cn_coty", "non_coty"]

# ============== INFERENCE SETTINGS ==============
BATCH_SIZE            = 32
PREDICTION_IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD  = 0.0   # 0.0 = capture all predictions

# ============== QC SETTINGS ==============
MAX_PLOT_IMAGES = None   # None / 0 / -1 = export ALL misclassified images
SCALE_PERCENT   = 50

# =================================================

# ImageNet normalisation — must match training transform
_TRANSFORM = T.Compose([
    T.Resize((PREDICTION_IMAGE_SIZE, PREDICTION_IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def setup_logging(output_folder):
    """Setup logging to both console and file."""
    log_filename = f"qc_effinet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_folder, log_filename)
    os.makedirs(output_folder, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return log_path


class QCClassificationEffiNet:
    """
    QC pipeline for EfficientNet (torchvision) classification models.

    Folder structure:
        DATA_PATH/
        ├── cn_coty/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── non_coty/
            ├── image3.jpg
            └── image4.jpg

    Ground truth = folder name.
    """

    def __init__(
        self,
        model_path,
        model_name,
        data_path,
        output_path,
        class_folders=None,
        batch_size=32,
        image_size=224,
        confidence_threshold=0.0,
    ):
        self.model_path          = model_path
        self.model_name          = model_name
        self.data_path           = data_path
        self.output_path         = output_path
        self.batch_size          = batch_size
        self.image_size          = image_size
        self.confidence_threshold = confidence_threshold
        self.logger              = logging.getLogger(__name__)
        self.model               = None
        self.device              = None
        self.results             = []

        if class_folders and len(class_folders) > 0:
            self.class_folders = class_folders
        else:
            self.class_folders = self._detect_class_folders()

        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        os.makedirs(self.output_path, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _detect_class_folders(self):
        """Auto-detect subfolders as class names (alphabetical)."""
        folders = sorted([
            item for item in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, item))
        ])
        self.logger.info(f"Auto-detected class folders: {folders}")
        return folders

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def load_model(self):
        """Load the trained EfficientNet state-dict from best.pt."""
        self.logger.info(f"Loading model '{self.model_name}' from: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        num_classes = len(self.class_folders)

        # Supported architectures — extend as needed
        weights_map = {
            "efficientnet_b0":   tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            "efficientnet_b1":   tv_models.EfficientNet_B1_Weights.IMAGENET1K_V1,
            "efficientnet_b2":   tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1,
            "efficientnet_b3":   tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1,
            "efficientnet_b4":   tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1,
            "efficientnet_v2_s": tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
            "efficientnet_v2_m": tv_models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
        }

        if self.model_name not in weights_map:
            raise ValueError(
                f"Unsupported model_name '{self.model_name}'. "
                f"Choose from: {list(weights_map.keys())}"
            )

        model = getattr(tv_models, self.model_name)(weights=weights_map[self.model_name])

        # Replace final classifier head to match training
        if hasattr(model, "classifier"):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        self.logger.info(
            f"Model loaded — {self.model_name} | classes: {self.class_folders}"
        )
        return self.model

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def run_inference(self):
        """Run batch inference on all images in class folders."""
        if self.model is None:
            self.load_model()

        self.results = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

        for class_name in self.class_folders:
            folder_path = os.path.join(self.data_path, class_name)

            if not os.path.exists(folder_path):
                self.logger.warning(f"Folder not found, skipping: {folder_path}")
                continue

            images = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(image_extensions)
            ]
            self.logger.info(
                f"Processing {len(images)} images from '{class_name}' folder"
            )

            for i in range(0, len(images), self.batch_size):
                batch_names  = images[i:i + self.batch_size]
                batch_tensors = []
                batch_paths   = []

                for img_name in batch_names:
                    img_path = os.path.join(folder_path, img_name)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        batch_tensors.append(self.transform(img))
                        batch_paths.append(img_path)
                    except Exception as e:
                        self.logger.warning(f"Could not read {img_path}: {e}")

                if not batch_tensors:
                    continue

                batch = torch.stack(batch_tensors).to(self.device)

                with torch.no_grad():
                    logits = self.model(batch)                # (N, C)
                    probs  = torch.softmax(logits, dim=1)    # (N, C)
                    confs, pred_idxs = probs.max(dim=1)      # top-1

                for img_path, conf, pred_idx in zip(
                    batch_paths, confs.cpu().numpy(), pred_idxs.cpu().numpy()
                ):
                    conf        = float(conf)
                    pred_label  = self.class_folders[int(pred_idx)]

                    # Apply confidence filter (keep everything when threshold = 0)
                    if self.confidence_threshold > 0 and conf < self.confidence_threshold:
                        continue

                    self.results.append({
                        'image_path':   img_path,
                        'image_name':   os.path.basename(img_path),
                        'ground_truth': class_name,
                        'predicted':    pred_label,
                        'confidence':   conf,
                        'correct':      pred_label.lower() == class_name.lower(),
                    })

        self.logger.info(f"Total images processed: {len(self.results)}")
        return self.results

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def calculate_metrics(self):
        """Calculate TP, FP, FN per class and overall accuracy."""
        if not self.results:
            return {}

        all_classes = set()
        for r in self.results:
            all_classes.add(r['ground_truth'].lower())
            all_classes.add(r['predicted'].lower())
        all_classes = sorted(all_classes)

        confusion = {c1: {c2: 0 for c2 in all_classes} for c1 in all_classes}
        for r in self.results:
            confusion[r['ground_truth'].lower()][r['predicted'].lower()] += 1

        metrics = {'per_class': {}, 'confusion_matrix': confusion}

        for cls in all_classes:
            tp = confusion[cls][cls]
            fp = sum(confusion[other][cls] for other in all_classes if other != cls)
            fn = sum(confusion[cls][other] for other in all_classes if other != cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            metrics['per_class'][cls] = {
                'TP': tp, 'FP': fp, 'FN': fn,
                'precision': precision, 'recall': recall, 'f1': f1,
                'support': tp + fn,
            }

        total_correct = sum(1 for r in self.results if r['correct'])
        total         = len(self.results)
        metrics['overall'] = {
            'accuracy':     total_correct / total if total > 0 else 0.0,
            'total_images': total,
            'correct':      total_correct,
            'incorrect':    total - total_correct,
        }

        self.metrics = metrics
        return metrics

    # ------------------------------------------------------------------ #
    # Exports
    # ------------------------------------------------------------------ #

    def export_csv(self):
        """Export per-image prediction results to CSV."""
        csv_path = os.path.join(self.output_path, "classification_results.csv")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'image_name', 'image_path', 'ground_truth',
                'predicted', 'confidence', 'correct', 'error_type'
            ])
            for r in self.results:
                error_type = (
                    f"FN_{r['ground_truth']}_FP_{r['predicted']}"
                    if not r['correct'] else ''
                )
                writer.writerow([
                    r['image_name'], r['image_path'], r['ground_truth'],
                    r['predicted'], f"{r['confidence']:.4f}",
                    r['correct'], error_type,
                ])

        self.logger.info(f"Results exported to: {csv_path}")
        return csv_path

    def export_metrics_csv(self):
        """Export per-class and overall metrics to CSV."""
        if not hasattr(self, 'metrics'):
            self.calculate_metrics()

        csv_path = os.path.join(self.output_path, "metrics_summary.csv")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'class', 'TP', 'FP', 'FN',
                'precision', 'recall', 'f1', 'support'
            ])
            for cls, m in self.metrics['per_class'].items():
                writer.writerow([
                    cls, m['TP'], m['FP'], m['FN'],
                    f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}", m['support'],
                ])

            writer.writerow([])
            overall = self.metrics['overall']
            writer.writerow([
                'OVERALL', '', '', '',
                f"Accuracy: {overall['accuracy']:.4f}", '',
                f"Total: {overall['total_images']}",
                f"Correct: {overall['correct']}",
            ])

        self.logger.info(f"Metrics exported to: {csv_path}")
        return csv_path

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #

    def plot_confusion_matrix(self):
        """Plot and save confusion matrix."""
        if not hasattr(self, 'metrics'):
            self.calculate_metrics()

        confusion = self.metrics['confusion_matrix']
        classes   = sorted(confusion.keys())
        matrix    = np.array([
            [confusion[gt][pred] for pred in classes]
            for gt in classes
        ])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            title='Confusion Matrix',
            ylabel='Ground Truth',
            xlabel='Predicted',
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        thresh = matrix.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j, i, format(matrix[i, j], 'd'),
                    ha="center", va="center", fontsize=14,
                    color="white" if matrix[i, j] > thresh else "black",
                )

        fig.tight_layout()
        plot_path = os.path.join(self.output_path, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Confusion matrix saved to: {plot_path}")
        return plot_path

    def plot_metrics_dashboard(self):
        """Plot a 2×2 summary dashboard."""
        if not hasattr(self, 'metrics'):
            self.calculate_metrics()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        classes = list(self.metrics['per_class'].keys())
        x       = np.arange(len(classes))
        width   = 0.25

        # ── 1. TP / FP / FN bar chart ──────────────────────────────── #
        ax1 = axes[0, 0]
        tp_vals = [self.metrics['per_class'][c]['TP'] for c in classes]
        fp_vals = [self.metrics['per_class'][c]['FP'] for c in classes]
        fn_vals = [self.metrics['per_class'][c]['FN'] for c in classes]

        ax1.bar(x - width, tp_vals, width, label='TP', color='green',  alpha=0.7)
        ax1.bar(x,         fp_vals, width, label='FP', color='red',    alpha=0.7)
        ax1.bar(x + width, fn_vals, width, label='FN', color='orange', alpha=0.7)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('TP / FP / FN per Class')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes)
        ax1.legend()

        for i, (tp, fp, fn) in enumerate(zip(tp_vals, fp_vals, fn_vals)):
            ax1.text(i - width, tp + 1, str(tp), ha='center', fontsize=9)
            ax1.text(i,         fp + 1, str(fp), ha='center', fontsize=9)
            ax1.text(i + width, fn + 1, str(fn), ha='center', fontsize=9)

        # ── 2. Precision / Recall / F1 ─────────────────────────────── #
        ax2 = axes[0, 1]
        prec_vals = [self.metrics['per_class'][c]['precision'] for c in classes]
        rec_vals  = [self.metrics['per_class'][c]['recall']    for c in classes]
        f1_vals   = [self.metrics['per_class'][c]['f1']        for c in classes]

        ax2.bar(x - width, prec_vals, width, label='Precision', color='blue',   alpha=0.7)
        ax2.bar(x,         rec_vals,  width, label='Recall',    color='purple', alpha=0.7)
        ax2.bar(x + width, f1_vals,   width, label='F1',        color='teal',   alpha=0.7)
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision / Recall / F1 per Class')
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes)
        ax2.set_ylim(0, 1.1)
        ax2.legend()

        # ── 3. Confidence distribution ─────────────────────────────── #
        ax3 = axes[1, 0]
        correct_confs   = [r['confidence'] for r in self.results if     r['correct']]
        incorrect_confs = [r['confidence'] for r in self.results if not r['correct']]

        ax3.hist(correct_confs,   bins=20, alpha=0.7,
                 label=f'Correct (n={len(correct_confs)})',   color='green')
        ax3.hist(incorrect_confs, bins=20, alpha=0.7,
                 label=f'Incorrect (n={len(incorrect_confs)})', color='red')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Count')
        ax3.set_title('Confidence Distribution')
        ax3.legend()

        # ── 4. Summary text ────────────────────────────────────────── #
        ax4 = axes[1, 1]
        ax4.axis('off')

        overall = self.metrics['overall']
        summary_text = f"""
OVERALL SUMMARY
{'='*40}

Total Images : {overall['total_images']}
Correct      : {overall['correct']}
Incorrect    : {overall['incorrect']}

Accuracy     : {overall['accuracy']:.2%}

PER-CLASS METRICS:
"""
        for cls in classes:
            m = self.metrics['per_class'][cls]
            summary_text += f"""
{cls.upper()}:
  TP: {m['TP']}  FP: {m['FP']}  FN: {m['FN']}
  Precision : {m['precision']:.2%}
  Recall    : {m['recall']:.2%}
  F1        : {m['f1']:.2%}
"""
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plot_path = os.path.join(self.output_path, "metrics_dashboard.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Metrics dashboard saved to: {plot_path}")
        return plot_path

    def plot_misclassified_images(self, max_images=None, scale_percent=50):
        """
        Save annotated misclassified images into per-class subfolders.
        Each image shows GT (FN) and Prediction (FP) in the header.
        High-confidence errors are shown first.

        max_images: None / 0 / -1 / >= len(misclassified) -> export ALL.
        """
        misclassified = {}
        for cls in self.class_folders:
            cls_lower = cls.lower()
            misclassified[cls] = sorted(
                [r for r in self.results
                 if not r['correct'] and r['ground_truth'].lower() == cls_lower],
                key=lambda x: x['confidence'],
                reverse=True,
            )

        for cls, images in misclassified.items():
            if not images:
                self.logger.info(f"No misclassified images for class: {cls}")
                continue

            output_folder = os.path.join(self.output_path, f"misclassified_{cls}")
            os.makedirs(output_folder, exist_ok=True)
            # If max_images is falsy/non-positive, plot all available misclassifications.
            limit = len(images) if not max_images or max_images <= 0 else min(max_images, len(images))
            images_to_plot = images[:limit]
            self.logger.info(
                f"Plotting {len(images_to_plot)}/{len(images)} misclassified images for: {cls}"
            )

            for idx, r in enumerate(images_to_plot, 1):
                try:
                    img = cv2.imread(r['image_path'])
                    if img is None:
                        continue

                    # Rescale
                    scale      = scale_percent / 100
                    new_w      = int(img.shape[1] * scale)
                    new_h      = int(img.shape[0] * scale)
                    img        = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Add header bar
                    header_h   = 80
                    padded     = np.full((img.shape[0] + header_h, img.shape[1], 3),
                                        255, dtype=np.uint8)
                    padded[header_h:] = img

                    font       = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness  = 2

                    cv2.putText(padded, f"GT (FN): {r['ground_truth']}",
                                (10, 25), font, font_scale, (0, 150, 0), thickness)
                    cv2.putText(padded,
                                f"Pred (FP): {r['predicted']}  ({r['confidence']:.2%})",
                                (10, 50), font, font_scale, (0, 0, 200), thickness)

                    # Confidence bar
                    bar_w = int((img.shape[1] - 20) * r['confidence'])
                    cv2.rectangle(padded, (10, 60), (10 + bar_w, 75), (0, 0, 200), -1)
                    cv2.rectangle(padded, (10, 60), (img.shape[1] - 10, 75),
                                  (100, 100, 100), 1)

                    out_name = (
                        f"{idx:03d}_GT_{r['ground_truth']}"
                        f"_PRED_{r['predicted']}_{r['image_name']}"
                    )
                    cv2.imwrite(os.path.join(output_folder, out_name), padded)

                except Exception as e:
                    self.logger.error(f"Error plotting {r['image_name']}: {e}")

            self.logger.info(f"Saved misclassified images to: {output_folder}")

    # ------------------------------------------------------------------ #
    # Console summary
    # ------------------------------------------------------------------ #

    def print_summary(self):
        """Print per-class and overall metrics to the log."""
        if not hasattr(self, 'metrics'):
            self.calculate_metrics()

        overall = self.metrics['overall']

        self.logger.info("=" * 60)
        self.logger.info("EFFICIENTNET CLASSIFICATION QC SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Model      : {self.model_name}")
        self.logger.info(f"Total      : {overall['total_images']}")
        self.logger.info(f"Correct    : {overall['correct']}")
        self.logger.info(f"Incorrect  : {overall['incorrect']}")
        self.logger.info(f"Accuracy   : {overall['accuracy']:.2%}")
        self.logger.info("-" * 60)

        for cls in sorted(self.metrics['per_class'].keys()):
            m = self.metrics['per_class'][cls]
            self.logger.info(f"\n{cls.upper()}:")
            self.logger.info(f"  TP: {m['TP']:4d}  |  FP: {m['FP']:4d}  |  FN: {m['FN']:4d}")
            self.logger.info(f"  Precision : {m['precision']:.4f}")
            self.logger.info(f"  Recall    : {m['recall']:.4f}")
            self.logger.info(f"  F1 Score  : {m['f1']:.4f}")

        self.logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #

    def run(self, plot=True, max_plot_images=50, scale_percent=50):
        """Run the full QC pipeline: inference → metrics → exports → plots."""
        self.logger.info("Starting EfficientNet Classification QC...")
        self.logger.info(f"  Model      : {self.model_path}")
        self.logger.info(f"  Model name : {self.model_name}")
        self.logger.info(f"  Data       : {self.data_path}")
        self.logger.info(f"  Classes    : {self.class_folders}")
        self.logger.info(f"  Output     : {self.output_path}")

        self.load_model()
        self.run_inference()
        self.calculate_metrics()
        self.export_csv()
        self.export_metrics_csv()
        self.print_summary()

        if plot:
            self.plot_confusion_matrix()
            self.plot_metrics_dashboard()
            self.plot_misclassified_images(max_plot_images, scale_percent)

        self.logger.info("EfficientNet Classification QC completed!")
        return self.metrics


# ============== MAIN ==============
if __name__ == "__main__":
    setup_logging(OUTPUT_PATH)

    qc = QCClassificationEffiNet(
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        class_folders=CLASS_FOLDERS,
        batch_size=BATCH_SIZE,
        image_size=PREDICTION_IMAGE_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    qc.run(
        plot=True,
        max_plot_images=MAX_PLOT_IMAGES,
        scale_percent=SCALE_PERCENT,
    )
