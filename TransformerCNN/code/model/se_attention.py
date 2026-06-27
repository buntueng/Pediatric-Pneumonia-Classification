import os
import csv
import json
import time
import random
import warnings
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
RAW_PATH = "/home/eecommu06/Desktop/Bee/Pediatric_Pneumonia/dataset"
FASTER_GLCM_PATH = "/home/eecommu06/Desktop/Bee/Pediatric_Pneumonia/glcm_dataset"
OUTPUT_PATH = "/home/eecommu06/Desktop/Bee/Pediatric_Pneumonia/output/SE"

MODEL_NAME = "fusion_se_effnetb0_resnet18_glcm"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
EPOCHS = 30
EARLY_STOPPING_PATIENCE = None  
IMG_SIZE = (224, 224)
N_FOLDS = 5
NUM_WORKERS = 2
PIN_MEMORY = True
GENERATE_XAI = True
XAI_MAX_PER_CASE_TYPE = 1       
XAI_INTEGRATED_GRADIENT_STEPS = 20
DELETION_INSERTION_STEPS = 10
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

COMPARISON_PREDICTION_FILES = []

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
CHECKPOINT_DIR = os.path.join(OUTPUT_PATH, "checkpoints")
XAI_DIR = os.path.join(OUTPUT_PATH, "xai_outputs")
PLOTS_DIR = os.path.join(OUTPUT_PATH, "plots")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(XAI_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
for sub in [
    "gradcam_raw", "gradcam_glcm", "saliency_raw", "saliency_glcm",
    "integrated_gradients_raw", "integrated_gradients_glcm",
    "deletion_curves_raw", "insertion_curves_raw",
    "deletion_curves_glcm", "insertion_curves_glcm",
]:
    os.makedirs(os.path.join(XAI_DIR, sub), exist_ok=True)

MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_fold{{}}.pth")

# Required reviewer output paths
EXPERIMENT_CONFIG_PATH = os.path.join(OUTPUT_PATH, "experiment_config.json")
GLCM_CONFIG_PATH = os.path.join(OUTPUT_PATH, "glcm_config.json")
MODEL_CONFIG_PATH = os.path.join(OUTPUT_PATH, "model_config.json")
SPLIT_SUMMARY_PATH = os.path.join(OUTPUT_PATH, "split_summary.csv")
TRAINING_HISTORY_PATH = os.path.join(OUTPUT_PATH, "training_history.csv")
BEST_CHECKPOINT_SUMMARY_PATH = os.path.join(OUTPUT_PATH, "best_checkpoint_summary.json")
TEST_PREDICTIONS_PATH = os.path.join(OUTPUT_PATH, "test_predictions.csv")
TEST_METRICS_POINT_PATH = os.path.join(OUTPUT_PATH, "test_metrics_point.csv")
TEST_METRICS_BOOTSTRAP_CI_PATH = os.path.join(OUTPUT_PATH, "test_metrics_bootstrap_ci.csv")
CONFUSION_MATRIX_CSV_PATH = os.path.join(OUTPUT_PATH, "confusion_matrix.csv")
CONFUSION_MATRIX_PNG_PATH = os.path.join(OUTPUT_PATH, "confusion_matrix.png")
STATISTICAL_TESTS_PATH = os.path.join(OUTPUT_PATH, "statistical_tests.csv")
CALIBRATION_BINS_PATH = os.path.join(OUTPUT_PATH, "calibration_bins.csv")
RELIABILITY_CURVE_PATH = os.path.join(OUTPUT_PATH, "reliability_curve.png")
MODEL_COMPLEXITY_PATH = os.path.join(OUTPUT_PATH, "model_complexity.csv")
FUSION_WEIGHTS_PATH = os.path.join(OUTPUT_PATH, "fusion_weights_test.csv")
BRANCH_COMPARISON_PATH = os.path.join(OUTPUT_PATH, "branch_prediction_comparison.csv")
ERROR_ANALYSIS_PATH = os.path.join(OUTPUT_PATH, "error_analysis_cases.csv")
DELETION_INSERTION_PATH = os.path.join(OUTPUT_PATH, "deletion_insertion_results.csv")

# Original output names preserved for compatibility with your previous code
OLD_HISTORY_CSV_PATH = os.path.join(OUTPUT_PATH, "training_history_SE.csv")
OLD_RESULTS_CSV_PATH = os.path.join(OUTPUT_PATH, "SE_results.csv")

print(f"Running {MODEL_NAME} on: {DEVICE}")
print(f"Output folder: {OUTPUT_PATH}")


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


def get_gpu_name() -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "cuda_available_unknown_gpu"
    return "cpu"


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =============================================================================
# 3. DATASET
# =============================================================================
class DualInputDataset(Dataset):
    def __init__(self, raw_dir=None, faster_glcm_dir=None, transform=None, samples=None, classes=None):
        self.raw_dir = raw_dir
        self.faster_glcm_dir = faster_glcm_dir
        self.transform = transform

        if samples is not None and classes is not None:
            self.samples = list(samples)
            self.classes = list(classes)
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            return

        if raw_dir is None or faster_glcm_dir is None:
            raise ValueError("raw_dir and faster_glcm_dir are required when samples/classes are not provided.")

        self.samples = []
        valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        class_names = set()
        found_images = []

        print(f"Scanning raw dataset recursively at: {raw_dir}")
        for root, _, files in os.walk(raw_dir):
            for file in files:
                if file.lower().endswith(valid_ext):
                    raw_img_path = os.path.join(root, file)
                    class_name = os.path.basename(root)
                    rel_path = os.path.relpath(raw_img_path, raw_dir)
                    found_images.append((raw_img_path, rel_path, class_name))
                    class_names.add(class_name)

        self.classes = sorted(list(class_names))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Detected classes: {self.classes}")

        missing_glcm = 0
        for raw_path, rel_path, class_name in found_images:
            glcm_target = os.path.join(faster_glcm_dir, rel_path)
            final_glcm_path = None

            if os.path.exists(glcm_target):
                final_glcm_path = glcm_target
            else:
                base_no_ext = os.path.splitext(glcm_target)[0]
                for ext in valid_ext:
                    temp_path = base_no_ext + ext
                    if os.path.exists(temp_path):
                        final_glcm_path = temp_path
                        break

            if final_glcm_path:
                self.samples.append((raw_path, final_glcm_path, self.class_to_idx[class_name]))
            else:
                missing_glcm += 1

        print(f"Total matched raw-GLCM pairs found: {len(self.samples)}")
        print(f"Raw images without matched GLCM image: {missing_glcm}")
        if len(self.samples) == 0:
            raise RuntimeError("No matched raw-GLCM pairs found. Check RAW_PATH and FASTER_GLCM_PATH folder structures.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_path, glcm_path, label = self.samples[idx]
        try:
            raw_img = Image.open(raw_path).convert("RGB")
            glcm_img = Image.open(glcm_path).convert("RGB")
        except Exception:
            # If one file is corrupted, skip to the next valid sample.
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            raw_img = self.transform(raw_img)
            glcm_img = self.transform(glcm_img)

        return raw_img, glcm_img, label, raw_path, glcm_path


# Keep the original preprocessing behavior: resize + ImageNet normalization.
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

base_dataset = DualInputDataset(RAW_PATH, FASTER_GLCM_PATH, transform=None)
train_dataset = DualInputDataset(samples=base_dataset.samples, classes=base_dataset.classes, transform=train_transform)
val_dataset = DualInputDataset(samples=base_dataset.samples, classes=base_dataset.classes, transform=val_transform)

if len(base_dataset.classes) != 2:
    raise ValueError(f"This script expects binary classification, but found classes: {base_dataset.classes}")

# Prefer the class that contains "pneumonia" as the positive class.
positive_label = 1
for i, cls_name in enumerate(base_dataset.classes):
    if "pneumonia" in cls_name.lower():
        positive_label = i
        break
negative_label = 1 - positive_label

print(f"Positive class index: {positive_label} ({base_dataset.classes[positive_label]})")


def safe_class_col_name(name: str) -> str:
    return "prob_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


# =============================================================================
# 4. MODEL
# =============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for vector features.

    Input shape: [batch, channels].
    Output: recalibrated features and, optionally, channel-wise attention weights.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, return_attention=False):
        weights = self.fc(x)
        out = x * weights
        if return_attention:
            return out, weights
        return out


class FusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Branch 1: Raw image branch, EfficientNet-B0.
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.raw_branch = nn.Sequential(
            effnet.features,
            effnet.avgpool,
            nn.Flatten(),
        )
        self.raw_dim = 1280

        # Branch 2: precomputed GLCM texture-map branch, ResNet-18.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.faster_glcm_branch = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten(),
        )
        self.texture_dim = 512

        # Concatenate features, then apply SE channel recalibration.
        self.fusion_dim = self.raw_dim + self.texture_dim
        self.se_block = SEBlock(self.fusion_dim, reduction=16)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, raw, glcm, return_features=False):
        raw_vec = self.raw_branch(raw)                 # [B, 1280]
        texture_vec = self.faster_glcm_branch(glcm)    # [B, 512]
        fused_input = torch.cat((raw_vec, texture_vec), dim=1)
        fused_vec, se_attention = self.se_block(fused_input, return_attention=True)
        logits = self.classifier(fused_vec)

        if return_features:
            raw_attn = se_attention[:, :self.raw_dim]
            texture_attn = se_attention[:, self.raw_dim:]
            se_raw_channel_mean = raw_attn.mean(dim=1)
            se_texture_channel_mean = texture_attn.mean(dim=1)
            denom = se_raw_channel_mean + se_texture_channel_mean + 1e-12
            se_raw_weight = se_raw_channel_mean / denom
            se_texture_weight = se_texture_channel_mean / denom
            return (
                logits,
                raw_vec,
                texture_vec,
                fused_vec,
                se_raw_weight,
                se_texture_weight,
                se_raw_channel_mean,
                se_texture_channel_mean,
            )
        return logits


# =============================================================================
# 5. CONFIG FILES REQUESTED BY REVIEWERS
# =============================================================================
experiment_config = {
    "model_name": MODEL_NAME,
    "run_id": RUN_ID,
    "raw_dataset_path": RAW_PATH,
    "glcm_dataset_path": FASTER_GLCM_PATH,
    "output_path": OUTPUT_PATH,
    "seed": SEED,
    "device": str(DEVICE),
    "gpu_name": get_gpu_name(),
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "epochs": EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "n_folds": N_FOLDS,
    "num_workers": NUM_WORKERS,
    "pin_memory": PIN_MEMORY,
    "optimizer": "Adam",
    "scheduler": "CosineAnnealingLR(T_max=EPOCHS)",
    "loss": "CrossEntropyLoss",
    "augmentation": {
        "used": False,
        "details": "Original Se.py used resize + tensor conversion + ImageNet normalization only.",
        "rotation_degrees": 0,
        "translation": 0,
        "brightness_jitter": 0,
        "contrast_jitter": 0,
        "horizontal_flip": False,
    },
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "source": "ImageNet",
        "applied_to": ["raw_image", "precomputed_glcm_map_image"],
    },
    "created_at": RUN_ID,
}

# This training script consumes precomputed GLCM images from FASTER_GLCM_PATH.
# Fill the unknown fields with real values from your GLCM generation script before manuscript submission.
glcm_config = {
    "used_in_this_model": True,
    "model_name": MODEL_NAME,
    "glcm_input_path": FASTER_GLCM_PATH,
    "note": "This script uses precomputed GLCM map images. Generation parameters must match the separate GLCM data-preparation script.",
    "image_size_used_by_training": list(IMG_SIZE),
    "texture_maps_expected_from_manuscript": ["contrast", "dissimilarity", "entropy"],
    "compute_glcm_after_resize": "not_specified_in_original_Se_py",
    "gray_levels": "not_specified_in_original_Se_py",
    "window_size": "not_specified_in_original_Se_py",
    "distances": "not_specified_in_original_Se_py",
    "angles_degrees": "not_specified_in_original_Se_py",
    "offsets": "not_specified_in_original_Se_py",
    "direction_aggregation": "not_specified_in_original_Se_py",
    "entropy_formula": "not_specified_in_original_Se_py",
    "border_handling": "not_specified_in_original_Se_py",
    "map_channel_order": "read_as_RGB_from_precomputed_GLCM_image",
    "normalization": "ImageNet mean/std after reading the precomputed GLCM image as RGB",
}

model_config = {
    "model_name": MODEL_NAME,
    "input_type": "raw_image_plus_precomputed_glcm_texture_image",
    "num_classes": len(base_dataset.classes),
    "classes": base_dataset.classes,
    "positive_label": int(positive_label),
    "raw_branch": {
        "backbone": "EfficientNet-B0",
        "pretrained_weights": "torchvision.models.EfficientNet_B0_Weights.DEFAULT",
        "output_dim": 1280,
    },
    "texture_branch": {
        "backbone": "ResNet-18",
        "pretrained_weights": "torchvision.models.ResNet18_Weights.DEFAULT",
        "input": "precomputed_GLCM_texture_image_read_as_RGB",
        "output_dim": 512,
    },
    "fusion": {
        "type": "feature_concatenation_followed_by_SE_channel_recalibration",
        "attention_or_gate_used": True,
        "fusion_vector_dim": 1792,
        "se_block_position": "after concatenating EfficientNet-B0 raw feature vector and ResNet-18 GLCM feature vector, before classifier",
        "se_block": "Linear(1792,112) -> ReLU -> Linear(112,1792) -> Sigmoid",
        "gate_network": "not_a_two_scalar_gate; SE produces channel-wise attention over the concatenated feature vector",
        "fusion_weight_type": "SE channel-attention summary; w_raw and w_texture are branch-wise averages normalized to sum to 1 for auditing",
        "note": "w_raw and w_texture saved in fusion_weights_test.csv are derived from SE channel attention averages over raw and texture channel groups; they are not softmax gate outputs.",
    },
    "classifier": "Linear(1792,512) -> ReLU -> Dropout(0.3) -> Linear(512,num_classes)",
}

save_json(experiment_config, EXPERIMENT_CONFIG_PATH)
save_json(glcm_config, GLCM_CONFIG_PATH)
save_json(model_config, MODEL_CONFIG_PATH)


# =============================================================================
# 6. METRIC HELPERS
# =============================================================================
def binary_predictions_from_probs(probs_positive, threshold=0.5):
    preds_positive = (np.asarray(probs_positive) >= threshold).astype(int)
    if positive_label == 1:
        return preds_positive
    return 1 - preds_positive


def extract_positive_probs(prob_matrix):
    prob_matrix = np.asarray(prob_matrix)
    return prob_matrix[:, positive_label]


def compute_calibration_bins(y_true_binary, probs_positive, n_bins=10):
    y_true_binary = np.asarray(y_true_binary).astype(int)
    probs_positive = np.asarray(probs_positive).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(y_true_binary)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs_positive >= lo) & (probs_positive <= hi)
        else:
            mask = (probs_positive >= lo) & (probs_positive < hi)
        count = int(mask.sum())
        if count > 0:
            mean_conf = float(probs_positive[mask].mean())
            frac_pos = float(y_true_binary[mask].mean())
            abs_gap = abs(mean_conf - frac_pos)
            weighted_gap = (count / n) * abs_gap
        else:
            mean_conf = np.nan
            frac_pos = np.nan
            abs_gap = np.nan
            weighted_gap = 0.0
        ece += weighted_gap
        rows.append({
            "bin_id": i + 1,
            "bin_lower": lo,
            "bin_upper": hi,
            "count": count,
            "mean_confidence": mean_conf,
            "fraction_positive": frac_pos,
            "abs_gap": abs_gap,
            "weighted_gap": weighted_gap,
        })
    return float(ece), pd.DataFrame(rows)


def compute_binary_metrics(y_true, probs_positive, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probs_positive = np.asarray(probs_positive).astype(float)
    y_pred = binary_predictions_from_probs(probs_positive, threshold=threshold)

    labels_order = [negative_label, positive_label]
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    tn, fp, fn, tp = cm.ravel()

    y_true_binary = (y_true == positive_label).astype(int)
    y_pred_binary = (y_pred == positive_label).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "sensitivity": recall_score(y_true_binary, y_pred_binary, zero_division=0),
        "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    try:
        metrics["auroc"] = roc_auc_score(y_true_binary, probs_positive)
    except Exception:
        metrics["auroc"] = np.nan
    try:
        metrics["auprc"] = average_precision_score(y_true_binary, probs_positive)
    except Exception:
        metrics["auprc"] = np.nan
    try:
        metrics["brier_score"] = brier_score_loss(y_true_binary, probs_positive)
    except Exception:
        metrics["brier_score"] = np.nan

    ece, _ = compute_calibration_bins(y_true_binary, probs_positive, n_bins=10)
    metrics["ece"] = ece

    return metrics, y_pred


def bootstrap_metric_ci(y_true, probs_positive, threshold=0.5, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    probs_positive = np.asarray(probs_positive).astype(float)
    n = len(y_true)
    metric_values = {}

    metric_names = [
        "accuracy", "f1", "f1_weighted", "sensitivity", "precision", "specificity",
        "mcc", "auroc", "auprc", "brier_score", "ece"
    ]
    for name in metric_names:
        metric_values[name] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m, _ = compute_binary_metrics(y_true[idx], probs_positive[idx], threshold)
        for name in metric_names:
            metric_values[name].append(m.get(name, np.nan))

    rows = []
    point, _ = compute_binary_metrics(y_true, probs_positive, threshold)
    for name, values in metric_values.items():
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            rows.append({
                "model_name": MODEL_NAME,
                "metric": name,
                "point_estimate": point.get(name, np.nan),
                "bootstrap_mean": np.nan,
                "lower_95ci": np.nan,
                "upper_95ci": np.nan,
                "n_bootstrap_valid": 0,
                "n_bootstrap_requested": n_bootstrap,
                "seed": seed,
            })
        else:
            rows.append({
                "model_name": MODEL_NAME,
                "metric": name,
                "point_estimate": point.get(name, np.nan),
                "bootstrap_mean": float(np.mean(values)),
                "lower_95ci": float(np.percentile(values, 2.5)),
                "upper_95ci": float(np.percentile(values, 97.5)),
                "n_bootstrap_valid": int(len(values)),
                "n_bootstrap_requested": n_bootstrap,
                "seed": seed,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 7. TRAINING AND EVALUATION HELPERS
# =============================================================================
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for raw, glcm, labels, _raw_paths, _glcm_paths in loader:
            raw = raw.to(DEVICE)
            glcm = glcm.to(DEVICE)
            labels = labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(raw, glcm)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            total_loss += loss.item()
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / max(len(loader), 1)
    probs_positive = extract_positive_probs(np.asarray(all_probs))
    metrics, _ = compute_binary_metrics(np.asarray(all_labels), probs_positive, threshold=0.5)
    metrics["loss"] = avg_loss
    return metrics


def evaluate_with_predictions(model, loader, fold, threshold=0.5):
    model.eval()
    rows = []

    with torch.no_grad():
        for raw, glcm, labels, raw_paths, glcm_paths in loader:
            raw = raw.to(DEVICE)
            glcm = glcm.to(DEVICE)
            outputs, raw_vec, texture_vec, fused_vec, se_raw_w, se_texture_w, se_raw_channel_mean, se_texture_channel_mean = model(raw, glcm, return_features=True)
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            logits = outputs.detach().cpu().numpy()
            probs_positive = extract_positive_probs(probs)
            preds = binary_predictions_from_probs(probs_positive, threshold=threshold)

            raw_norm = torch.linalg.vector_norm(raw_vec, ord=2, dim=1).detach().cpu().numpy()
            texture_norm = torch.linalg.vector_norm(texture_vec, ord=2, dim=1).detach().cpu().numpy()
            fused_norm = torch.linalg.vector_norm(fused_vec, ord=2, dim=1).detach().cpu().numpy()
            denom = raw_norm + texture_norm + 1e-12
            raw_norm_ratio = raw_norm / denom
            texture_norm_ratio = texture_norm / denom
            se_raw_w_np = se_raw_w.detach().cpu().numpy()
            se_texture_w_np = se_texture_w.detach().cpu().numpy()
            se_raw_channel_mean_np = se_raw_channel_mean.detach().cpu().numpy()
            se_texture_channel_mean_np = se_texture_channel_mean.detach().cpu().numpy()

            for i in range(len(labels)):
                row = {
                    "model_name": MODEL_NAME,
                    "run_id": RUN_ID,
                    "seed": SEED,
                    "fold": fold,
                    "split": "fold_validation_out_of_fold",
                    "image_path": raw_paths[i],
                    "raw_image_path": raw_paths[i],
                    "glcm_image_path": glcm_paths[i],
                    "true_label": int(labels[i]),
                    "true_class": base_dataset.classes[int(labels[i])],
                    "pred_label": int(preds[i]),
                    "pred_class": base_dataset.classes[int(preds[i])],
                    "prob_positive": float(probs_positive[i]),
                    "prob_negative": float(1.0 - probs_positive[i]),
                    "prob_normal": float(probs[i][negative_label]),
                    "prob_pneumonia": float(probs[i][positive_label]),
                    "logit_negative": float(logits[i][negative_label]),
                    "logit_positive": float(logits[i][positive_label]),
                    "logit_normal": float(logits[i][negative_label]),
                    "logit_pneumonia": float(logits[i][positive_label]),
                    "correct": int(preds[i] == int(labels[i])),
                    "threshold": threshold,
                    "raw_feature_l2_norm": float(raw_norm[i]),
                    "texture_feature_l2_norm": float(texture_norm[i]),
                    "fused_feature_l2_norm": float(fused_norm[i]),
                    "raw_feature_norm_ratio": float(raw_norm_ratio[i]),
                    "texture_feature_norm_ratio": float(texture_norm_ratio[i]),
                    "se_raw_weight": float(se_raw_w_np[i]),
                    "se_texture_weight": float(se_texture_w_np[i]),
                    "se_raw_channel_mean": float(se_raw_channel_mean_np[i]),
                    "se_texture_channel_mean": float(se_texture_channel_mean_np[i]),
                }
                for c_idx, c_name in enumerate(base_dataset.classes):
                    row[safe_class_col_name(c_name)] = float(probs[i][c_idx])
                rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 8. SPLIT SUMMARY
# =============================================================================
def write_split_summary(kfold_obj):
    rows = []
    indices = np.arange(len(base_dataset))
    labels = np.array([label for _, _, label in base_dataset.samples])

    for fold, (train_idx, val_idx) in enumerate(kfold_obj.split(indices), start=1):
        for split_name, idxs in [("train", train_idx), ("fold_validation_out_of_fold", val_idx)]:
            split_labels = labels[idxs]
            row = {
                "model_name": MODEL_NAME,
                "run_id": RUN_ID,
                "seed": SEED,
                "fold": fold,
                "split": split_name,
                "n_samples": int(len(idxs)),
            }
            for c_idx, c_name in enumerate(base_dataset.classes):
                row[f"n_{c_name}"] = int((split_labels == c_idx).sum())
            rows.append(row)

    pd.DataFrame(rows).to_csv(SPLIT_SUMMARY_PATH, index=False)


# =============================================================================
# 9. TRAINING LOOP
# =============================================================================
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
write_split_summary(kfold)

history_rows = []
old_result_rows = []
best_summaries = []
all_prediction_dfs = []

indices = np.arange(len(base_dataset))

print(f"\nStarting {N_FOLDS}-Fold Cross Validation ({MODEL_NAME})...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(indices), start=1):
    print(f"\n{'=' * 22} FOLD {fold}/{N_FOLDS} {'=' * 22}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = FusionModel(num_classes=len(base_dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_metric = -np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    fold_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        loop = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}/{EPOCHS}", leave=True)
        model.train()
        running_loss = 0.0
        train_labels = []
        train_probs = []

        for raw, glcm, labels, _raw_paths, _glcm_paths in loop:
            raw = raw.to(DEVICE)
            glcm = glcm.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(raw, glcm)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            train_labels.extend(labels.detach().cpu().numpy())
            train_probs.extend(probs)

            current_probs_positive = extract_positive_probs(np.asarray(train_probs))
            current_metrics, _ = compute_binary_metrics(np.asarray(train_labels), current_probs_positive, 0.5)
            loop.set_postfix(loss=loss.item(), acc=current_metrics["accuracy"])

        train_probs_positive = extract_positive_probs(np.asarray(train_probs))
        train_metrics, _ = compute_binary_metrics(np.asarray(train_labels), train_probs_positive, 0.5)
        train_metrics["loss"] = running_loss / max(len(train_loader), 1)

        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step()

        # Use validation AUROC if valid; otherwise validation accuracy.
        val_selection_metric = val_metrics["auroc"]
        if np.isnan(val_selection_metric):
            val_selection_metric = val_metrics["accuracy"]

        is_best = val_selection_metric > best_val_metric
        if is_best:
            best_val_metric = float(val_selection_metric)
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH.format(fold))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_time = time.time() - epoch_start
        lr_current = optimizer.param_groups[0]["lr"]

        history_rows.append({
            "model_name": MODEL_NAME,
            "run_id": RUN_ID,
            "seed": SEED,
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_sensitivity": train_metrics["sensitivity"],
            "train_precision": train_metrics["precision"],
            "train_specificity": train_metrics["specificity"],
            "train_mcc": train_metrics["mcc"],
            "train_auroc": train_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_precision": val_metrics["precision"],
            "val_specificity": val_metrics["specificity"],
            "val_mcc": val_metrics["mcc"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "learning_rate": lr_current,
            "epoch_time_seconds": epoch_time,
            "is_best": int(is_best),
        })

        if EARLY_STOPPING_PATIENCE is not None and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    fold_time = time.time() - fold_start
    print(f"Evaluating best checkpoint for Fold {fold}...")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH.format(fold), map_location=DEVICE))
    model.eval()

    pred_df = evaluate_with_predictions(model, val_loader, fold=fold, threshold=0.5)
    all_prediction_dfs.append(pred_df)

    fold_metrics, _ = compute_binary_metrics(
        pred_df["true_label"].values,
        pred_df["prob_positive"].values,
        threshold=0.5,
    )

    best_summaries.append({
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "seed": SEED,
        "fold": fold,
        "best_epoch": int(best_epoch),
        "best_validation_selection_metric": float(best_val_metric),
        "selection_metric": "val_auroc_if_available_else_val_accuracy",
        "threshold": 0.5,
        "checkpoint_path": MODEL_SAVE_PATH.format(fold),
        "fold_training_time_seconds": fold_time,
        "fold_test_accuracy": fold_metrics["accuracy"],
        "fold_test_f1": fold_metrics["f1"],
        "fold_test_mcc": fold_metrics["mcc"],
        "fold_test_auroc": fold_metrics["auroc"],
        "fold_test_auprc": fold_metrics["auprc"],
    })

    old_result_rows.append({
        "Fold": fold,
        "Accuracy": fold_metrics["accuracy"],
        "F1 Score": fold_metrics["f1_weighted"],
        "Sensitivity": fold_metrics["sensitivity"],
        "Precision": fold_metrics["precision"],
        "Specificity": fold_metrics["specificity"],
        "MCC": fold_metrics["mcc"],
        "AUC": fold_metrics["auroc"],
        "AUPRC": fold_metrics["auprc"],
        "ECE": fold_metrics["ece"],
        "Brier Score": fold_metrics["brier_score"],
        "TP": fold_metrics["tp"],
        "TN": fold_metrics["tn"],
        "FP": fold_metrics["fp"],
        "FN": fold_metrics["fn"],
        "Training Time (s)": fold_time,
    })

    print(f"Fold {fold} finished. Acc: {fold_metrics['accuracy']:.4f}, AUROC: {fold_metrics['auroc']:.4f}")

# Save history and checkpoint summaries.
history_df = pd.DataFrame(history_rows)
history_df.to_csv(TRAINING_HISTORY_PATH, index=False)
# Preserve original training history file name.
history_df.rename(columns={
    "fold": "Fold",
    "epoch": "Epoch",
    "train_loss": "Train Loss",
    "train_accuracy": "Train Acc",
    "val_loss": "Val Loss",
    "val_accuracy": "Val Acc",
    "epoch_time_seconds": "Time (s)",
}).to_csv(OLD_HISTORY_CSV_PATH, index=False)

save_json(best_summaries, BEST_CHECKPOINT_SUMMARY_PATH)
pd.DataFrame(old_result_rows).to_csv(OLD_RESULTS_CSV_PATH, index=False)

# Out-of-fold predictions across all folds.
predictions_df = pd.concat(all_prediction_dfs, ignore_index=True)
predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)


# =============================================================================
# 10. POINT METRICS, BOOTSTRAP CI, CONFUSION MATRIX
# =============================================================================
point_rows = []

# Fold-level rows
for fold in sorted(predictions_df["fold"].unique()):
    df_fold = predictions_df[predictions_df["fold"] == fold]
    metrics, _ = compute_binary_metrics(df_fold["true_label"].values, df_fold["prob_positive"].values, 0.5)
    row = {
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "row_type": "fold",
        "fold": int(fold),
        "n_samples": len(df_fold),
        **metrics,
    }
    point_rows.append(row)

# Overall out-of-fold row
overall_metrics, overall_preds = compute_binary_metrics(
    predictions_df["true_label"].values,
    predictions_df["prob_positive"].values,
    0.5,
)
point_rows.append({
    "model_name": MODEL_NAME,
    "run_id": RUN_ID,
    "row_type": "overall_out_of_fold",
    "fold": "all",
    "n_samples": len(predictions_df),
    **overall_metrics,
})

# Mean/std across folds
fold_metric_df = pd.DataFrame([r for r in point_rows if r["row_type"] == "fold"])
for stat_name, func in [("mean_across_folds", np.mean), ("std_across_folds", np.std)]:
    row = {
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "row_type": stat_name,
        "fold": "all",
        "n_samples": int(fold_metric_df["n_samples"].sum()) if stat_name == "mean_across_folds" else "",
    }
    for col in [
        "accuracy", "f1", "f1_weighted", "sensitivity", "precision", "specificity",
        "mcc", "auroc", "auprc", "brier_score", "ece", "tp", "tn", "fp", "fn"
    ]:
        if col in fold_metric_df.columns:
            row[col] = float(func(fold_metric_df[col].astype(float)))
    point_rows.append(row)

pd.DataFrame(point_rows).to_csv(TEST_METRICS_POINT_PATH, index=False)

ci_df = bootstrap_metric_ci(
    predictions_df["true_label"].values,
    predictions_df["prob_positive"].values,
    threshold=0.5,
    n_bootstrap=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED,
)
ci_df.to_csv(TEST_METRICS_BOOTSTRAP_CI_PATH, index=False)

# Confusion matrix CSV and image
cm = confusion_matrix(
    predictions_df["true_label"].values,
    predictions_df["pred_label"].values,
    labels=[negative_label, positive_label],
)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{base_dataset.classes[negative_label]}", f"true_{base_dataset.classes[positive_label]}"],
    columns=[f"pred_{base_dataset.classes[negative_label]}", f"pred_{base_dataset.classes[positive_label]}"],
)
cm_df.to_csv(CONFUSION_MATRIX_CSV_PATH)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title(f"Confusion Matrix - {MODEL_NAME}")
plt.xticks([0, 1], [base_dataset.classes[negative_label], base_dataset.classes[positive_label]], rotation=20)
plt.yticks([0, 1], [base_dataset.classes[negative_label], base_dataset.classes[positive_label]])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PNG_PATH, dpi=300)
plt.close()


# =============================================================================
# 11. CALIBRATION
# =============================================================================
y_true_binary = (predictions_df["true_label"].values == positive_label).astype(int)
probs_positive = predictions_df["prob_positive"].values
ece, cal_bins_df = compute_calibration_bins(y_true_binary, probs_positive, n_bins=10)
cal_bins_df.insert(0, "model_name", MODEL_NAME)
cal_bins_df.insert(1, "run_id", RUN_ID)
cal_bins_df["ece_overall"] = ece
cal_bins_df.to_csv(CALIBRATION_BINS_PATH, index=False)

plt.figure(figsize=(5, 5))
valid_bins = cal_bins_df[cal_bins_df["count"] > 0]
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.plot(valid_bins["mean_confidence"], valid_bins["fraction_positive"], marker="o", label=MODEL_NAME)
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive fraction")
plt.title(f"Reliability Curve (ECE={ece:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig(RELIABILITY_CURVE_PATH, dpi=300)
plt.close()


# =============================================================================
# 12. MODEL COMPLEXITY
# =============================================================================
def estimate_model_complexity(model):
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops = np.nan
    flops_source = "not_available"
    dummy_raw = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)
    dummy_glcm = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)

    try:
        from thop import profile
        macs, params = profile(model, inputs=(dummy_raw, dummy_glcm), verbose=False)
        flops = float(macs * 2)  # approximate FLOPs from MACs
        flops_source = "thop_profile_macs_x2"
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            flops = float(FlopCountAnalysis(model, (dummy_raw, dummy_glcm)).total())
            flops_source = "fvcore"
        except Exception:
            flops = np.nan
            flops_source = "not_installed_thop_or_fvcore"

    # Inference time per image, measured on a small repeated dummy batch.
    batch_raw = torch.randn(BATCH_SIZE, 3, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)
    batch_glcm = torch.randn(BATCH_SIZE, 3, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)
    n_warmup = 5
    n_repeat = 20
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(batch_raw, batch_glcm)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_repeat):
            _ = model(batch_raw, batch_glcm)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
    inference_time_ms_per_image = (elapsed / (n_repeat * BATCH_SIZE)) * 1000

    checkpoint_sizes = [os.path.getsize(p) / (1024 ** 2) for p in glob(os.path.join(CHECKPOINT_DIR, "*.pth"))]
    model_size_mb = float(np.mean(checkpoint_sizes)) if checkpoint_sizes else np.nan

    gpu_memory_mb = np.nan
    if torch.cuda.is_available():
        gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

    return {
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "input_size": f"raw=1x3x{IMG_SIZE[0]}x{IMG_SIZE[1]}, glcm=1x3x{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "num_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "flops": flops,
        "flops_source": flops_source,
        "model_size_mb": model_size_mb,
        "inference_time_ms_per_image": float(inference_time_ms_per_image),
        "gpu_memory_mb": gpu_memory_mb,
        "device": str(DEVICE),
        "gpu_name": get_gpu_name(),
    }


complexity_model = FusionModel(num_classes=len(base_dataset.classes)).to(DEVICE)
# Load fold 1 only for inference timing; architecture is identical across folds.
complexity_model.load_state_dict(torch.load(MODEL_SAVE_PATH.format(1), map_location=DEVICE))
complexity_row = estimate_model_complexity(complexity_model)
pd.DataFrame([complexity_row]).to_csv(MODEL_COMPLEXITY_PATH, index=False)


# =============================================================================
# 13. FUSION WEIGHT AND BRANCH-COMPARISON RECORDS
# =============================================================================
def add_case_type(df):
    return np.where(
        (df["true_label"] == positive_label) & (df["pred_label"] == positive_label), "TP",
        np.where(
            (df["true_label"] == negative_label) & (df["pred_label"] == negative_label), "TN",
            np.where(
                (df["true_label"] == negative_label) & (df["pred_label"] == positive_label), "FP", "FN"
            )
        )
    )


fusion_df = predictions_df[[
    "model_name", "run_id", "seed", "fold", "image_path", "raw_image_path", "glcm_image_path",
    "true_label", "true_class", "pred_label", "pred_class", "prob_positive", "prob_pneumonia", "correct",
    "raw_feature_l2_norm", "texture_feature_l2_norm", "fused_feature_l2_norm",
    "raw_feature_norm_ratio", "texture_feature_norm_ratio",
    "se_raw_weight", "se_texture_weight", "se_raw_channel_mean", "se_texture_channel_mean"
]].copy()
fusion_df["w_raw"] = fusion_df["se_raw_weight"]
fusion_df["w_texture"] = fusion_df["se_texture_weight"]
fusion_df["fusion_type"] = "SE_after_feature_concatenation"
fusion_df["case_type"] = add_case_type(fusion_df)
fusion_df["note"] = "SE produces channel-wise attention over concatenated raw+GLCM features. w_raw and w_texture are normalized averages of SE channel weights over raw and texture channel groups, saved for branch-emphasis auditing."
fusion_df.to_csv(FUSION_WEIGHTS_PATH, index=False)

branch_df = fusion_df[[
    "model_name", "run_id", "fold", "image_path", "raw_image_path", "glcm_image_path",
    "true_label", "true_class", "pred_label", "pred_class", "correct", "case_type",
    "raw_feature_l2_norm", "texture_feature_l2_norm", "raw_feature_norm_ratio", "texture_feature_norm_ratio",
    "se_raw_weight", "se_texture_weight", "se_raw_channel_mean", "se_texture_channel_mean"
]].copy()
branch_df["raw_branch_prob_positive"] = np.nan
branch_df["texture_branch_prob_positive"] = np.nan
branch_df["fused_prob_positive"] = predictions_df["prob_positive"].values
branch_df["branch_agreement"] = "not_available_no_auxiliary_branch_classifiers"
branch_df["note"] = "This SE model has no separate raw-only or texture-only classification heads. Branch-emphasis auditing is provided through se_raw_weight and se_texture_weight."
branch_df.to_csv(BRANCH_COMPARISON_PATH, index=False)


# =============================================================================
# 14. ERROR ANALYSIS
# =============================================================================
err_df = predictions_df.copy()
err_df["case_type"] = add_case_type(err_df)
err_df["confidence"] = np.maximum(err_df["prob_positive"].values, 1.0 - err_df["prob_positive"].values)
err_df["abs_margin_from_threshold"] = np.abs(err_df["prob_positive"].values - 0.5)

# Save all errors plus a small group of low-confidence correct cases.
errors_only = err_df[err_df["correct"] == 0].copy()
low_conf_correct = err_df[err_df["correct"] == 1].sort_values("abs_margin_from_threshold").head(30).copy()
low_conf_correct["case_type"] = "low_confidence_correct"
error_analysis = pd.concat([errors_only, low_conf_correct], ignore_index=True)
error_analysis.to_csv(ERROR_ANALYSIS_PATH, index=False)


# =============================================================================
# 15. XAI: GRAD-CAM, SALIENCY, INTEGRATED GRADIENTS, DELETION/INSERTION
# =============================================================================
def unnormalize_tensor(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    x = img_tensor * std + mean
    return torch.clamp(x, 0, 1)


def normalize_map(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmin(x)
    denom = np.nanmax(x) + 1e-8
    return x / denom


def save_overlay(img_tensor, heatmap, out_path, title=""):
    img_np = unnormalize_tensor(img_tensor.detach().cpu()).permute(1, 2, 0).numpy()
    heatmap = normalize_map(heatmap)
    plt.figure(figsize=(4, 4))
    plt.imshow(img_np)
    plt.imshow(heatmap, alpha=0.45)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def load_pair_from_paths(raw_path, glcm_path):
    raw_img = Image.open(raw_path).convert("RGB")
    glcm_img = Image.open(glcm_path).convert("RGB")
    return val_transform(raw_img), val_transform(glcm_img)


class DualInputGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, raw, glcm, class_idx):
        self.model.zero_grad()
        output = self.model(raw, glcm)
        score = output[:, class_idx].sum()
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMG_SIZE[0], IMG_SIZE[1]),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam[0, 0].detach().cpu().numpy()
        return normalize_map(cam_np)

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def saliency_map_dual(model, raw, glcm, class_idx, target_input="raw"):
    raw_req = raw.clone().detach().requires_grad_(target_input == "raw")
    glcm_req = glcm.clone().detach().requires_grad_(target_input == "glcm")
    model.zero_grad()
    output = model(raw_req, glcm_req)
    score = output[:, class_idx].sum()
    score.backward()
    grad = raw_req.grad if target_input == "raw" else glcm_req.grad
    sal = grad.detach().abs().max(dim=1)[0][0].cpu().numpy()
    return normalize_map(sal)


def integrated_gradients_dual(model, raw, glcm, class_idx, target_input="raw", steps=20):
    baseline_raw = torch.zeros_like(raw)
    baseline_glcm = torch.zeros_like(glcm)

    if target_input == "raw":
        total_grad = torch.zeros_like(raw)
        for alpha in torch.linspace(0, 1, steps, device=raw.device):
            scaled_raw = (baseline_raw + alpha * (raw - baseline_raw)).clone().detach().requires_grad_(True)
            fixed_glcm = glcm.clone().detach()
            model.zero_grad()
            output = model(scaled_raw, fixed_glcm)
            score = output[:, class_idx].sum()
            score.backward()
            total_grad += scaled_raw.grad.detach()
        avg_grad = total_grad / steps
        ig = (raw - baseline_raw) * avg_grad
    else:
        total_grad = torch.zeros_like(glcm)
        for alpha in torch.linspace(0, 1, steps, device=glcm.device):
            fixed_raw = raw.clone().detach()
            scaled_glcm = (baseline_glcm + alpha * (glcm - baseline_glcm)).clone().detach().requires_grad_(True)
            model.zero_grad()
            output = model(fixed_raw, scaled_glcm)
            score = output[:, class_idx].sum()
            score.backward()
            total_grad += scaled_glcm.grad.detach()
        avg_grad = total_grad / steps
        ig = (glcm - baseline_glcm) * avg_grad

    ig_map = ig.detach().abs().max(dim=1)[0][0].cpu().numpy()
    return normalize_map(ig_map)


def prob_for_class(model, raw, glcm, class_idx):
    model.eval()
    with torch.no_grad():
        return float(torch.softmax(model(raw, glcm), dim=1)[0, class_idx].detach().cpu().item())


def deletion_insertion_test_dual(model, raw, glcm, importance_map, class_idx, target_input="raw", steps=10):
    importance = normalize_map(importance_map).reshape(-1)
    order = np.argsort(-importance)
    h, w = IMG_SIZE
    total_pixels = h * w
    fractions = np.linspace(0, 1, steps + 1)

    raw_baseline = torch.zeros_like(raw)
    glcm_baseline = torch.zeros_like(glcm)

    raw_deletion = raw.clone().detach()
    raw_insertion = raw_baseline.clone().detach()
    glcm_deletion = glcm.clone().detach()
    glcm_insertion = glcm_baseline.clone().detach()

    deletion_probs = []
    insertion_probs = []

    for frac in fractions:
        k = int(frac * total_pixels)
        if k > 0:
            selected = order[:k]
            rows = selected // w
            cols = selected % w
            if target_input == "raw":
                raw_deletion[:, :, rows, cols] = raw_baseline[:, :, rows, cols]
                raw_insertion[:, :, rows, cols] = raw[:, :, rows, cols]
            else:
                glcm_deletion[:, :, rows, cols] = glcm_baseline[:, :, rows, cols]
                glcm_insertion[:, :, rows, cols] = glcm[:, :, rows, cols]

        if target_input == "raw":
            deletion_probs.append(prob_for_class(model, raw_deletion, glcm, class_idx))
            insertion_probs.append(prob_for_class(model, raw_insertion, glcm, class_idx))
        else:
            deletion_probs.append(prob_for_class(model, raw, glcm_deletion, class_idx))
            insertion_probs.append(prob_for_class(model, raw, glcm_insertion, class_idx))

    deletion_auc = float(np.trapz(deletion_probs, fractions))
    insertion_auc = float(np.trapz(insertion_probs, fractions))
    return fractions, deletion_probs, insertion_probs, deletion_auc, insertion_auc


def select_xai_cases(pred_df):
    df = pred_df.copy()
    df["case_type"] = add_case_type(df)
    df["abs_margin_from_threshold"] = np.abs(df["prob_positive"] - 0.5)

    selected = []
    for case_type in ["TP", "TN", "FP", "FN"]:
        part = df[df["case_type"] == case_type].sort_values("abs_margin_from_threshold")
        selected.append(part.head(XAI_MAX_PER_CASE_TYPE))
    low_conf = df[df["correct"] == 1].sort_values("abs_margin_from_threshold").head(XAI_MAX_PER_CASE_TYPE)
    low_conf = low_conf.copy()
    low_conf["case_type"] = "low_confidence_correct"
    selected.append(low_conf)

    selected_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    selected_df = selected_df.drop_duplicates(subset=["image_path"])
    return selected_df


def plot_curve(fractions, probs, out_path, title, ylabel="Probability"):
    plt.figure(figsize=(5, 4))
    plt.plot(fractions, probs, marker="o")
    plt.xlabel("Fraction of most salient pixels")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


xai_summary_rows = []

if GENERATE_XAI:
    print("\nGenerating XAI examples and deletion/insertion tests...")
    xai_model = FusionModel(num_classes=len(base_dataset.classes)).to(DEVICE)
    xai_model.load_state_dict(torch.load(MODEL_SAVE_PATH.format(1), map_location=DEVICE))
    xai_model.eval()

    # Important: XAI uses the fold-1 checkpoint for representative examples only.
    # For strict fold-specific XAI, load each sample's own fold checkpoint here.
    raw_gradcam = DualInputGradCAM(xai_model, xai_model.raw_branch[0][-1])
    glcm_gradcam = DualInputGradCAM(xai_model, xai_model.faster_glcm_branch[7][-1])
    selected_cases = select_xai_cases(predictions_df)

    for idx, row in selected_cases.iterrows():
        try:
            raw_path = row["raw_image_path"]
            glcm_path = row["glcm_image_path"]
            raw_tensor, glcm_tensor = load_pair_from_paths(raw_path, glcm_path)
            raw_input = raw_tensor.to(DEVICE).unsqueeze(0)
            glcm_input = glcm_tensor.to(DEVICE).unsqueeze(0)
            class_idx = int(row["pred_label"])
            case_type = row["case_type"]
            base_name = f"{idx:03d}_{case_type}_fold{row['fold']}_{os.path.splitext(os.path.basename(raw_path))[0]}"

            # Grad-CAM for both branches
            cam_raw = raw_gradcam(raw_input, glcm_input, class_idx)
            cam_glcm = glcm_gradcam(raw_input, glcm_input, class_idx)

            # Gradient-based saliency and integrated gradients for both inputs.
            sal_raw = saliency_map_dual(xai_model, raw_input, glcm_input, class_idx, target_input="raw")
            sal_glcm = saliency_map_dual(xai_model, raw_input, glcm_input, class_idx, target_input="glcm")
            ig_raw = integrated_gradients_dual(
                xai_model, raw_input, glcm_input, class_idx, target_input="raw",
                steps=XAI_INTEGRATED_GRADIENT_STEPS,
            )
            ig_glcm = integrated_gradients_dual(
                xai_model, raw_input, glcm_input, class_idx, target_input="glcm",
                steps=XAI_INTEGRATED_GRADIENT_STEPS,
            )

            gradcam_raw_path = os.path.join(XAI_DIR, "gradcam_raw", f"{base_name}_raw_gradcam.png")
            gradcam_glcm_path = os.path.join(XAI_DIR, "gradcam_glcm", f"{base_name}_glcm_gradcam.png")
            saliency_raw_path = os.path.join(XAI_DIR, "saliency_raw", f"{base_name}_raw_saliency.png")
            saliency_glcm_path = os.path.join(XAI_DIR, "saliency_glcm", f"{base_name}_glcm_saliency.png")
            ig_raw_path = os.path.join(XAI_DIR, "integrated_gradients_raw", f"{base_name}_raw_ig.png")
            ig_glcm_path = os.path.join(XAI_DIR, "integrated_gradients_glcm", f"{base_name}_glcm_ig.png")

            save_overlay(raw_tensor, cam_raw, gradcam_raw_path, title=f"Raw Grad-CAM {case_type}")
            save_overlay(glcm_tensor, cam_glcm, gradcam_glcm_path, title=f"GLCM Grad-CAM {case_type}")
            save_overlay(raw_tensor, sal_raw, saliency_raw_path, title=f"Raw Saliency {case_type}")
            save_overlay(glcm_tensor, sal_glcm, saliency_glcm_path, title=f"GLCM Saliency {case_type}")
            save_overlay(raw_tensor, ig_raw, ig_raw_path, title=f"Raw IG {case_type}")
            save_overlay(glcm_tensor, ig_glcm, ig_glcm_path, title=f"GLCM IG {case_type}")

            # Deletion/insertion uses saliency maps as the importance maps.
            fractions, del_raw_probs, ins_raw_probs, del_raw_auc, ins_raw_auc = deletion_insertion_test_dual(
                xai_model,
                raw_input,
                glcm_input,
                sal_raw,
                class_idx=class_idx,
                target_input="raw",
                steps=DELETION_INSERTION_STEPS,
            )
            fractions, del_glcm_probs, ins_glcm_probs, del_glcm_auc, ins_glcm_auc = deletion_insertion_test_dual(
                xai_model,
                raw_input,
                glcm_input,
                sal_glcm,
                class_idx=class_idx,
                target_input="glcm",
                steps=DELETION_INSERTION_STEPS,
            )

            deletion_raw_curve_path = os.path.join(XAI_DIR, "deletion_curves_raw", f"{base_name}_raw_deletion.png")
            insertion_raw_curve_path = os.path.join(XAI_DIR, "insertion_curves_raw", f"{base_name}_raw_insertion.png")
            deletion_glcm_curve_path = os.path.join(XAI_DIR, "deletion_curves_glcm", f"{base_name}_glcm_deletion.png")
            insertion_glcm_curve_path = os.path.join(XAI_DIR, "insertion_curves_glcm", f"{base_name}_glcm_insertion.png")
            plot_curve(fractions, del_raw_probs, deletion_raw_curve_path, f"Raw deletion curve {case_type}")
            plot_curve(fractions, ins_raw_probs, insertion_raw_curve_path, f"Raw insertion curve {case_type}")
            plot_curve(fractions, del_glcm_probs, deletion_glcm_curve_path, f"GLCM deletion curve {case_type}")
            plot_curve(fractions, ins_glcm_probs, insertion_glcm_curve_path, f"GLCM insertion curve {case_type}")

            xai_summary_rows.append({
                "model_name": MODEL_NAME,
                "run_id": RUN_ID,
                "image_path": raw_path,
                "raw_image_path": raw_path,
                "glcm_image_path": glcm_path,
                "fold": int(row["fold"]),
                "true_label": int(row["true_label"]),
                "true_class": row["true_class"],
                "pred_label": int(row["pred_label"]),
                "pred_class": row["pred_class"],
                "case_type": case_type,
                "prob_positive": float(row["prob_positive"]),
                "gradcam_raw_path": gradcam_raw_path,
                "gradcam_glcm_path": gradcam_glcm_path,
                "saliency_raw_path": saliency_raw_path,
                "saliency_glcm_path": saliency_glcm_path,
                "integrated_gradients_raw_path": ig_raw_path,
                "integrated_gradients_glcm_path": ig_glcm_path,
                "deletion_raw_curve_path": deletion_raw_curve_path,
                "insertion_raw_curve_path": insertion_raw_curve_path,
                "deletion_glcm_curve_path": deletion_glcm_curve_path,
                "insertion_glcm_curve_path": insertion_glcm_curve_path,
                "deletion_raw_auc": del_raw_auc,
                "insertion_raw_auc": ins_raw_auc,
                "deletion_glcm_auc": del_glcm_auc,
                "insertion_glcm_auc": ins_glcm_auc,
                "xai_checkpoint_note": "fold_1_checkpoint_used_for_representative_xai",
            })
        except Exception as e:
            xai_summary_rows.append({
                "model_name": MODEL_NAME,
                "run_id": RUN_ID,
                "image_path": row.get("image_path", "unknown"),
                "case_type": row.get("case_type", "unknown"),
                "error": str(e),
            })

    raw_gradcam.close()
    glcm_gradcam.close()
else:
    xai_summary_rows.append({
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "note": "GENERATE_XAI=False; no XAI/deletion/insertion outputs generated.",
    })

pd.DataFrame(xai_summary_rows).to_csv(DELETION_INSERTION_PATH, index=False)


# =============================================================================
# 16. STATISTICAL TESTS AGAINST OTHER MODEL PREDICTION FILES
# =============================================================================
def exact_mcnemar_pvalue(b, c):
    n = b + c
    if n == 0:
        return 1.0
    if binomtest is not None:
        return float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    return np.nan


def paired_bootstrap_accuracy_diff(df_a, df_b, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    a_correct = df_a["correct"].values.astype(int)
    b_correct = df_b["correct"].values.astype(int)
    n = len(a_correct)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(a_correct[idx].mean() - b_correct[idx].mean()))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def load_and_standardize_prediction_file(path):
    df = pd.read_csv(path)
    if "prob_positive" not in df.columns:
        if "prob_pneumonia" in df.columns:
            df["prob_positive"] = df["prob_pneumonia"]
        elif safe_class_col_name(base_dataset.classes[positive_label]) in df.columns:
            df["prob_positive"] = df[safe_class_col_name(base_dataset.classes[positive_label])]
    if "correct" not in df.columns and {"true_label", "pred_label"}.issubset(df.columns):
        df["correct"] = (df["true_label"].astype(int) == df["pred_label"].astype(int)).astype(int)
    return df


def find_comparison_prediction_files():
    files = list(COMPARISON_PREDICTION_FILES)
    sibling_root = os.path.dirname(OUTPUT_PATH)
    auto_files = glob(os.path.join(sibling_root, "*", "test_predictions.csv"))
    auto_files += glob(os.path.join(sibling_root, "*", "*_test_predictions.csv"))
    for f in auto_files:
        if os.path.abspath(f) != os.path.abspath(TEST_PREDICTIONS_PATH) and f not in files:
            files.append(f)
    return files


stat_rows = []
comparison_files = find_comparison_prediction_files()
self_df = predictions_df.copy()

for comp_path in comparison_files:
    try:
        comp_df = load_and_standardize_prediction_file(comp_path)
        required = {"image_path", "true_label", "pred_label", "correct"}
        if not required.issubset(comp_df.columns):
            stat_rows.append({
                "reference_model": MODEL_NAME,
                "comparison_file": comp_path,
                "test_name": "not_run_missing_required_columns",
                "note": f"Required columns: {sorted(required)}",
            })
            continue

        merged = self_df.merge(
            comp_df,
            on="image_path",
            suffixes=("_reference", "_comparison"),
        )
        if len(merged) == 0:
            stat_rows.append({
                "reference_model": MODEL_NAME,
                "comparison_file": comp_path,
                "test_name": "not_run_no_overlapping_image_paths",
            })
            continue

        ref_correct = merged["correct_reference"].astype(int).values
        cmp_correct = merged["correct_comparison"].astype(int).values
        b = int(((ref_correct == 1) & (cmp_correct == 0)).sum())
        c = int(((ref_correct == 0) & (cmp_correct == 1)).sum())
        p = exact_mcnemar_pvalue(b, c)
        mean_diff, lo_diff, hi_diff = paired_bootstrap_accuracy_diff(
            merged.rename(columns={"correct_reference": "correct"}),
            merged.rename(columns={"correct_comparison": "correct"}),
            n_bootstrap=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )

        comp_model = comp_df["model_name"].iloc[0] if "model_name" in comp_df.columns and len(comp_df) else os.path.basename(os.path.dirname(comp_path))
        stat_rows.append({
            "reference_model": MODEL_NAME,
            "comparison_model": comp_model,
            "comparison_file": comp_path,
            "n_paired_samples": int(len(merged)),
            "metric": "accuracy",
            "test_name": "McNemar_exact_binomial",
            "b_reference_correct_comparison_wrong": b,
            "c_reference_wrong_comparison_correct": c,
            "p_value": p,
            "mean_difference_reference_minus_comparison": mean_diff,
            "lower_95ci_difference": lo_diff,
            "upper_95ci_difference": hi_diff,
            "significant_at_0.05": bool(p < 0.05) if not np.isnan(p) else None,
        })
    except Exception as e:
        stat_rows.append({
            "reference_model": MODEL_NAME,
            "comparison_file": comp_path,
            "test_name": "not_run_error",
            "error": str(e),
        })

if not stat_rows:
    stat_rows.append({
        "reference_model": MODEL_NAME,
        "comparison_model": "none_found",
        "metric": "accuracy",
        "test_name": "not_run",
        "note": "No paired comparison prediction files found. Run other models first, then rerun this section or use a separate comparison script.",
    })

pd.DataFrame(stat_rows).to_csv(STATISTICAL_TESTS_PATH, index=False)


# =============================================================================
# 17. FINAL SUMMARY
# =============================================================================
print("\nTraining and reviewer-ready logging complete.")
print("=" * 70)
print(f"Output folder: {OUTPUT_PATH}")
print("Saved files:")
for path in [
    EXPERIMENT_CONFIG_PATH,
    GLCM_CONFIG_PATH,
    MODEL_CONFIG_PATH,
    SPLIT_SUMMARY_PATH,
    TRAINING_HISTORY_PATH,
    BEST_CHECKPOINT_SUMMARY_PATH,
    TEST_PREDICTIONS_PATH,
    TEST_METRICS_POINT_PATH,
    TEST_METRICS_BOOTSTRAP_CI_PATH,
    CONFUSION_MATRIX_CSV_PATH,
    CONFUSION_MATRIX_PNG_PATH,
    STATISTICAL_TESTS_PATH,
    CALIBRATION_BINS_PATH,
    RELIABILITY_CURVE_PATH,
    MODEL_COMPLEXITY_PATH,
    FUSION_WEIGHTS_PATH,
    BRANCH_COMPARISON_PATH,
    ERROR_ANALYSIS_PATH,
    DELETION_INSERTION_PATH,
    XAI_DIR,
]:
    print(f" - {path}")
print("=" * 70)
