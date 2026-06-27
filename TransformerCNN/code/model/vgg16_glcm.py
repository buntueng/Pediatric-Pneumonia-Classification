import os
import csv
import json
import time
import math
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models

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

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
GLCM_DATASET_PATH = '/home/eecommu06/Desktop/Bee/Pediatric_Pneumonia/glcm_dataset'
OUTPUT_PATH = '/home/eecommu06/Desktop/Bee/Pediatric_Pneumonia/output/glcm_vgg16'

# Original filenames retained
MODEL_SAVE_PATH = os.path.join(OUTPUT_PATH, 'glcm_vgg16_fold{}.pth')
HISTORY_CSV_PATH = os.path.join(OUTPUT_PATH, 'training_history_glcm_vgg16.csv')
RESULTS_CSV_PATH = os.path.join(OUTPUT_PATH, 'glcm_vgg16_results.csv')

# Reviewer-requested standardized filenames
CHECKPOINT_DIR = os.path.join(OUTPUT_PATH, 'checkpoints')
XAI_DIR = os.path.join(OUTPUT_PATH, 'xai_outputs')
FIGURE_DIR = os.path.join(OUTPUT_PATH, 'figures')

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
EPOCHS = 30
IMG_SIZE = (224, 224)
N_FOLDS = 5
SEED = 42
NUM_WORKERS = 2
PIN_MEMORY = True
N_BOOTSTRAP = 1000
N_CALIBRATION_BINS = 10
POSITIVE_LABEL = 1
THRESHOLD = 0.5
MODEL_NAME = 'glcm_vgg16'
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(XAI_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

print(f"Running GLCM Branch (VGG16) on: {DEVICE}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ==========================================
# 2. HELPERS
# ==========================================
def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_float(x):
    if x is None:
        return None
    try:
        if np.isnan(x) or np.isinf(x):
            return None
    except Exception:
        pass
    return float(x)


def class_counts_from_indices(dataset, indices):
    rows = []
    labels = [dataset.samples[i][1] for i in indices]
    for cls_name, cls_idx in dataset.class_to_idx.items():
        rows.append({
            'class_name': cls_name,
            'class_index': cls_idx,
            'count': int(np.sum(np.array(labels) == cls_idx)),
        })
    return rows


def compute_metrics(y_true, y_pred, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    out = {
        'accuracy': safe_float(accuracy_score(y_true, y_pred)),
        'f1': safe_float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        'f1_weighted': safe_float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'sensitivity': safe_float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        'recall': safe_float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        'precision': safe_float(precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        'specificity': safe_float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        'mcc': safe_float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'threshold': float(threshold),
    }
    try:
        out['auroc'] = safe_float(roc_auc_score(y_true, y_prob))
    except Exception:
        out['auroc'] = None
    try:
        out['auprc'] = safe_float(average_precision_score(y_true, y_prob))
    except Exception:
        out['auprc'] = None
    try:
        out['brier_score'] = safe_float(brier_score_loss(y_true, y_prob))
    except Exception:
        out['brier_score'] = None
    return out


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        if count > 0:
            mean_conf = float(y_prob[mask].mean())
            observed_pos = float(y_true[mask].mean())
            acc_bin = float(((y_prob[mask] >= 0.5).astype(int) == y_true[mask]).mean())
            gap = abs(observed_pos - mean_conf)
            ece += (count / n) * gap
        else:
            mean_conf = np.nan
            observed_pos = np.nan
            acc_bin = np.nan
            gap = np.nan
        rows.append({
            'bin': i + 1,
            'bin_lower': lo,
            'bin_upper': hi,
            'count': count,
            'mean_predicted_probability': mean_conf,
            'observed_positive_fraction': observed_pos,
            'bin_accuracy_at_0.5': acc_bin,
            'absolute_calibration_gap': gap,
        })
    return float(ece), pd.DataFrame(rows)


def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    metric_values = {}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        for k, v in m.items():
            if k in ['tp', 'tn', 'fp', 'fn', 'threshold'] or v is None:
                continue
            metric_values.setdefault(k, []).append(v)
    rows = []
    point = compute_metrics(y_true, y_pred, y_prob)
    for k in ['accuracy', 'f1', 'sensitivity', 'precision', 'specificity', 'mcc', 'auroc', 'auprc', 'brier_score']:
        vals = np.array(metric_values.get(k, []), dtype=float)
        if vals.size > 0:
            rows.append({
                'model_name': MODEL_NAME,
                'run_id': RUN_ID,
                'metric': k,
                'point_estimate': point.get(k),
                'bootstrap_mean': float(np.mean(vals)),
                'lower_95ci': float(np.percentile(vals, 2.5)),
                'upper_95ci': float(np.percentile(vals, 97.5)),
                'n_bootstrap_requested': n_bootstrap,
                'n_bootstrap_valid': int(vals.size),
                'seed': seed,
            })
    return pd.DataFrame(rows)


def plot_confusion_matrix(cm, classes, save_path, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(cm)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_reliability_curve(calib_df, ece, save_path):
    import matplotlib.pyplot as plt

    df = calib_df[calib_df['count'] > 0].copy()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    ax.plot(df['mean_predicted_probability'], df['observed_positive_fraction'], marker='o', label=MODEL_NAME)
    ax.set_title(f'Reliability Curve (ECE={ece:.4f})', fontsize=12)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed positive fraction')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb_from_state_dict(model):
    total_bytes = 0
    for v in model.state_dict().values():
        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024 ** 2)


def estimate_model_complexity(model, device):
    model.eval()
    total_params, trainable_params = count_parameters(model)
    model_size_mb = model_size_mb_from_state_dict(model)
    dummy = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)

    flops = None
    macs = None
    flops_note = 'not_available'
    try:
        from thop import profile
        macs, params_thop = profile(model, inputs=(dummy,), verbose=False)
        flops = 2 * macs
        flops_note = 'estimated_by_thop; FLOPs approximated as 2*MACs'
    except Exception as e:
        flops_note = f'thop_not_available_or_failed: {repr(e)}'

    # Inference time
    n_warmup = 10
    n_runs = 50
    try:
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_runs):
                _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_ms = (time.time() - t0) * 1000.0 / n_runs
    except Exception:
        inference_ms = None

    gpu_memory_mb = None
    if device.type == 'cuda':
        try:
            gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        except Exception:
            gpu_memory_mb = None

    return {
        'model_name': MODEL_NAME,
        'input_type': 'precomputed_glcm_texture_image',
        'num_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'model_size_mb': safe_float(model_size_mb),
        'macs': safe_float(macs),
        'flops': safe_float(flops),
        'flops_note': flops_note,
        'inference_time_ms_per_image_batch1': safe_float(inference_ms),
        'gpu_memory_mb': safe_float(gpu_memory_mb),
        'device': str(device),
        'image_size': f'{IMG_SIZE[0]}x{IMG_SIZE[1]}',
    }


def exact_mcnemar_pvalue(b, c):
    # b = current correct / comparator wrong; c = current wrong / comparator correct
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n=n, p=0.5, alternative='two-sided').pvalue)
    except Exception:
        # Normal approximation fallback
        stat = (abs(b - c) - 1) ** 2 / n if n > 0 else 0
        try:
            from scipy.stats import chi2
            return float(1 - chi2.cdf(stat, 1))
        except Exception:
            return None


def create_statistical_tests(current_pred_path):
    current = pd.read_csv(current_pred_path)
    current['image_key'] = current['image_path'].astype(str)
    parent = Path(OUTPUT_PATH).parent
    rows = []
    for candidate in parent.glob('*/test_predictions.csv'):
        if candidate.resolve() == Path(current_pred_path).resolve():
            continue
        try:
            other = pd.read_csv(candidate)
        except Exception:
            continue
        if 'image_path' not in other.columns or 'true_label' not in other.columns or 'pred_label' not in other.columns:
            continue
        other['image_key'] = other['image_path'].astype(str)
        merged = current.merge(
            other[['image_key', 'pred_label', 'prob_pneumonia', 'model_name']].rename(columns={
                'pred_label': 'pred_label_other',
                'prob_pneumonia': 'prob_pneumonia_other',
                'model_name': 'other_model_name',
            }),
            on='image_key',
            how='inner'
        )
        if len(merged) == 0:
            # Try matching by basename if paths differ between raw and GLCM folders
            current['basename'] = current['image_path'].map(lambda x: os.path.basename(str(x)))
            other['basename'] = other['image_path'].map(lambda x: os.path.basename(str(x)))
            merged = current.merge(
                other[['basename', 'pred_label', 'prob_pneumonia', 'model_name']].rename(columns={
                    'pred_label': 'pred_label_other',
                    'prob_pneumonia': 'prob_pneumonia_other',
                    'model_name': 'other_model_name',
                }),
                on='basename',
                how='inner'
            )
        if len(merged) == 0:
            continue
        other_name = str(merged['other_model_name'].iloc[0]) if 'other_model_name' in merged else candidate.parent.name
        true = merged['true_label'].astype(int).to_numpy()
        cur_correct = (merged['pred_label'].astype(int).to_numpy() == true)
        oth_correct = (merged['pred_label_other'].astype(int).to_numpy() == true)
        b = int(np.sum(cur_correct & ~oth_correct))
        c = int(np.sum(~cur_correct & oth_correct))
        p = exact_mcnemar_pvalue(b, c)
        acc_cur = float(cur_correct.mean())
        acc_oth = float(oth_correct.mean())
        rows.append({
            'current_model': MODEL_NAME,
            'comparison_model': other_name,
            'comparison_file': str(candidate),
            'n_paired_samples': int(len(merged)),
            'test_name': 'McNemar exact/binomial if scipy available',
            'b_current_correct_other_wrong': b,
            'c_current_wrong_other_correct': c,
            'p_value': p,
            'current_accuracy_on_paired': acc_cur,
            'comparison_accuracy_on_paired': acc_oth,
            'accuracy_difference_current_minus_comparison': acc_cur - acc_oth,
            'significant_at_0.05': bool(p is not None and p < 0.05),
            'note': 'Use only when predictions are paired on the same images. Path or basename matching was used.'
        })
    if not rows:
        rows.append({
            'current_model': MODEL_NAME,
            'comparison_model': None,
            'comparison_file': None,
            'n_paired_samples': 0,
            'test_name': None,
            'p_value': None,
            'note': 'No paired test_predictions.csv files found in sibling output folders. Re-run baselines with standardized logging to enable tests.'
        })
    return pd.DataFrame(rows)


# ==========================================
# 3. DATASET
# ==========================================
class SingleInputDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        print(f"Scanning GLCM dataset at: {root_dir}")
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        classes = set()

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_ext):
                    img_path = os.path.join(root, file)
                    class_name = os.path.basename(root)
                    self.samples.append((img_path, class_name))
                    classes.add(class_name)

        self.classes = sorted(list(classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.samples = [(p, self.class_to_idx[c]) for p, c in self.samples]
        self.samples = sorted(self.samples, key=lambda x: x[0])

        print(f"Detected Classes: {self.classes}")
        print(f"Total GLCM images found: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Skip corrupted images deterministically by moving to next index
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = SingleInputDataset(GLCM_DATASET_PATH, transform=transform)

# Save configs after class discovery
save_json({
    'model_name': MODEL_NAME,
    'run_id': RUN_ID,
    'glcm_dataset_path': GLCM_DATASET_PATH,
    'output_path': OUTPUT_PATH,
    'seed': SEED,
    'device': str(DEVICE),
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'image_size': list(IMG_SIZE),
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'epochs': EPOCHS,
    'early_stopping_patience': None,
    'n_folds': N_FOLDS,
    'num_workers': NUM_WORKERS,
    'pin_memory': PIN_MEMORY,
    'optimizer': 'Adam',
    'loss': 'CrossEntropyLoss',
    'augmentation': {
        'used': False,
        'details': 'Original vgg16_glcm.py used resize + tensor conversion + ImageNet normalization only.',
        'rotation_degrees': 0,
        'translation': 0,
        'brightness_jitter': 0,
        'contrast_jitter': 0,
        'horizontal_flip': False,
    },
    'normalization': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'source': 'ImageNet',
        'applied_to': ['precomputed_glcm_map_image'],
    },
    'created_at': RUN_ID,
}, os.path.join(OUTPUT_PATH, 'experiment_config.json'))

save_json({
    'used_in_this_model': True,
    'model_name': MODEL_NAME,
    'glcm_input_path': GLCM_DATASET_PATH,
    'note': 'This script uses precomputed GLCM map images. Generation parameters must match the separate GLCM data-preparation script.',
    'image_size_used_by_training': list(IMG_SIZE),
    'texture_maps_expected_from_manuscript': ['contrast', 'dissimilarity', 'entropy'],
    'compute_glcm_after_resize': 'not_specified_in_original_vgg16_glcm_py',
    'gray_levels': 'not_specified_in_original_vgg16_glcm_py',
    'window_size': 'not_specified_in_original_vgg16_glcm_py',
    'distances': 'not_specified_in_original_vgg16_glcm_py',
    'angles_degrees': 'not_specified_in_original_vgg16_glcm_py',
    'offsets': 'not_specified_in_original_vgg16_glcm_py',
    'direction_aggregation': 'not_specified_in_original_vgg16_glcm_py',
    'entropy_formula': 'not_specified_in_original_vgg16_glcm_py',
    'border_handling': 'not_specified_in_original_vgg16_glcm_py',
    'map_channel_order': 'read_as_RGB_from_precomputed_GLCM_image',
    'normalization': 'ImageNet mean/std after reading the precomputed GLCM image as RGB',
}, os.path.join(OUTPUT_PATH, 'glcm_config.json'))

save_json({
    'model_name': MODEL_NAME,
    'input_type': 'precomputed_glcm_texture_image',
    'num_classes': len(dataset.classes),
    'classes': dataset.classes,
    'positive_label': POSITIVE_LABEL,
    'texture_branch': {
        'backbone': 'VGG16',
        'pretrained_weights': 'torchvision.models.VGG16_Weights.DEFAULT',
        'input': 'precomputed_GLCM_texture_image_read_as_RGB',
        'classifier': 'Original VGG16 classifier with final Linear(4096,num_classes)',
    },
    'fusion': {
        'used': False,
        'type': 'not_applicable_single_texture_branch_baseline',
        'attention_or_gate_used': False,
        'fusion_weight_type': 'placeholder_only; w_raw=0 and w_texture=1',
    },
}, os.path.join(OUTPUT_PATH, 'model_config.json'))


# ==========================================
# 4. MODEL
# ==========================================
class GLCMBranchModel(nn.Module):
    def __init__(self, num_classes=2):
        super(GLCMBranchModel, self).__init__()
        weights = models.VGG16_Weights.DEFAULT
        self.backbone = models.vgg16(weights=weights)
        in_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# Save initial complexity before training
try:
    tmp_model = GLCMBranchModel(num_classes=len(dataset.classes)).to(DEVICE)
    complexity = estimate_model_complexity(tmp_model, DEVICE)
    pd.DataFrame([complexity]).to_csv(os.path.join(OUTPUT_PATH, 'model_complexity.csv'), index=False)
    del tmp_model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
except Exception as e:
    pd.DataFrame([{
        'model_name': MODEL_NAME,
        'error': repr(e),
        'note': 'Model complexity calculation failed.'
    }]).to_csv(os.path.join(OUTPUT_PATH, 'model_complexity.csv'), index=False)


# ==========================================
# 5. K-FOLD TRAINING LOOP
# ==========================================
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

old_history_headers = ['Fold', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Time (s)']
old_results_headers = ['Fold', 'Accuracy', 'F1 Score', 'Sensitivity', 'Precision', 'Specificity', 'MCC', 'AUC', 'TP', 'TN', 'FP', 'FN', 'Training Time (s)']
new_history_headers = [
    'model_name', 'run_id', 'fold', 'epoch', 'train_loss', 'train_acc',
    'val_loss', 'val_acc', 'val_f1', 'val_sensitivity', 'val_precision',
    'val_specificity', 'val_mcc', 'val_auroc', 'val_auprc', 'learning_rate',
    'epoch_time_seconds', 'is_best_checkpoint'
]

with open(HISTORY_CSV_PATH, 'w', newline='') as f:
    csv.writer(f).writerow(old_history_headers)
with open(RESULTS_CSV_PATH, 'w', newline='') as f:
    csv.writer(f).writerow(old_results_headers)
with open(os.path.join(OUTPUT_PATH, 'training_history.csv'), 'w', newline='') as f:
    csv.writer(f).writerow(new_history_headers)

all_prediction_rows = []
best_checkpoint_rows = []
split_rows = []
all_samples_split_rows = []

print(f"\nStarting {N_FOLDS}-Fold Cross Validation (GLCM Branch - VGG16)...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), start=1):
    print(f"\n{'='*20} FOLD {fold}/{N_FOLDS} {'='*20}")

    for split_name, indices in [('train', train_idx), ('validation_out_of_fold', val_idx)]:
        counts = class_counts_from_indices(dataset, indices)
        for row in counts:
            split_rows.append({
                'model_name': MODEL_NAME,
                'run_id': RUN_ID,
                'fold': fold,
                'split': split_name,
                **row,
            })
        split_rows.append({
            'model_name': MODEL_NAME,
            'run_id': RUN_ID,
            'fold': fold,
            'split': split_name,
            'class_name': 'ALL',
            'class_index': -1,
            'count': int(len(indices)),
        })
    for idx in train_idx:
        p, y = dataset.samples[idx]
        all_samples_split_rows.append({'fold': fold, 'split': 'train', 'image_path': p, 'label': y, 'class_name': dataset.idx_to_class[y]})
    for idx in val_idx:
        p, y = dataset.samples[idx]
        all_samples_split_rows.append({'fold': fold, 'split': 'validation_out_of_fold', 'image_path': p, 'label': y, 'class_name': dataset.idx_to_class[y]})

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if DEVICE.type == 'cuda' else False,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if DEVICE.type == 'cuda' else False,
    )

    model = GLCMBranchModel(num_classes=len(dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    fold_start = time.time()
    best_metric = -np.inf
    best_epoch = -1
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME}_fold{fold}.pth')

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}/{EPOCHS}", leave=True)
        for images, labels, paths in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=100 * train_correct / max(1, train_total))

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for images, labels, paths in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)[:, POSITIVE_LABEL]
                preds = torch.argmax(outputs, dim=1)

                val_loss_sum += loss.item() * labels.size(0)
                val_total += labels.size(0)
                val_preds.extend(preds.cpu().numpy().tolist())
                val_labels.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs.cpu().numpy().tolist())

        val_loss = val_loss_sum / max(1, val_total)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs, threshold=THRESHOLD)
        val_acc = val_metrics['accuracy']
        val_auroc = val_metrics['auroc']
        selection_metric = val_auroc if val_auroc is not None else val_acc
        is_best = bool(selection_metric > best_metric)
        if is_best:
            best_metric = selection_metric
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)
            # Retain original checkpoint naming too
            torch.save(model.state_dict(), MODEL_SAVE_PATH.format(fold))

        epoch_dur = time.time() - epoch_start
        with open(HISTORY_CSV_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([fold, epoch, train_loss, train_acc, val_loss, val_acc, epoch_dur])
        with open(os.path.join(OUTPUT_PATH, 'training_history.csv'), 'a', newline='') as f:
            csv.writer(f).writerow([
                MODEL_NAME, RUN_ID, fold, epoch, train_loss, train_acc,
                val_loss, val_acc, val_metrics.get('f1'), val_metrics.get('sensitivity'),
                val_metrics.get('precision'), val_metrics.get('specificity'), val_metrics.get('mcc'),
                val_metrics.get('auroc'), val_metrics.get('auprc'), LEARNING_RATE,
                epoch_dur, is_best,
            ])

    # Evaluate best checkpoint on out-of-fold validation split
    print(f"Evaluating Fold {fold} best checkpoint (epoch {best_epoch})...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
    model.eval()

    fold_preds, fold_labels, fold_probs, fold_logits, fold_paths = [], [], [], [], []
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs_all = torch.softmax(outputs, dim=1)
            probs = probs_all[:, POSITIVE_LABEL]
            preds = torch.argmax(outputs, dim=1)

            fold_preds.extend(preds.cpu().numpy().tolist())
            fold_labels.extend(labels.numpy().tolist())
            fold_probs.extend(probs.cpu().numpy().tolist())
            fold_logits.extend(outputs.cpu().numpy().tolist())
            fold_paths.extend(list(paths))

    fold_metrics = compute_metrics(fold_labels, fold_preds, fold_probs, threshold=THRESHOLD)
    fold_time = time.time() - fold_start

    with open(RESULTS_CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([
            fold,
            fold_metrics['accuracy'],
            fold_metrics['f1_weighted'],
            fold_metrics['sensitivity'],
            fold_metrics['precision'],
            fold_metrics['specificity'],
            fold_metrics['mcc'],
            fold_metrics['auroc'],
            fold_metrics['tp'],
            fold_metrics['tn'],
            fold_metrics['fp'],
            fold_metrics['fn'],
            fold_time,
        ])

    best_checkpoint_rows.append({
        'model_name': MODEL_NAME,
        'run_id': RUN_ID,
        'seed': SEED,
        'fold': fold,
        'best_epoch': best_epoch,
        'best_validation_selection_metric': safe_float(best_metric),
        'selection_metric': 'val_auroc_if_available_else_val_accuracy',
        'threshold': THRESHOLD,
        'checkpoint_path': best_checkpoint_path,
        'fold_training_time_seconds': safe_float(fold_time),
        'fold_test_accuracy': fold_metrics['accuracy'],
        'fold_test_f1': fold_metrics['f1'],
        'fold_test_mcc': fold_metrics['mcc'],
        'fold_test_auroc': fold_metrics['auroc'],
        'fold_test_auprc': fold_metrics['auprc'],
    })

    for pth, y, pred, prob, logits in zip(fold_paths, fold_labels, fold_preds, fold_probs, fold_logits):
        row = {
            'model_name': MODEL_NAME,
            'run_id': RUN_ID,
            'seed': SEED,
            'fold': fold,
            'split': 'validation_out_of_fold',
            'image_path': pth,
            'true_label': int(y),
            'true_class': dataset.idx_to_class[int(y)],
            'pred_label': int(pred),
            'pred_class': dataset.idx_to_class[int(pred)],
            'prob_normal': float(1.0 - prob) if len(dataset.classes) == 2 else None,
            'prob_pneumonia': float(prob),
            'logit_normal': float(logits[0]) if len(logits) > 0 else None,
            'logit_pneumonia': float(logits[1]) if len(logits) > 1 else None,
            'correct': bool(int(y) == int(pred)),
        }
        all_prediction_rows.append(row)

    print(f"Fold {fold} Finished. Acc: {fold_metrics['accuracy']:.4f}, AUROC: {fold_metrics['auroc']}")

# ==========================================
# 6. SAVE AGGREGATED OUTPUTS
# ==========================================
pd.DataFrame(split_rows).to_csv(os.path.join(OUTPUT_PATH, 'split_summary.csv'), index=False)
pd.DataFrame(all_samples_split_rows).to_csv(os.path.join(OUTPUT_PATH, 'all_samples_split.csv'), index=False)
save_json(best_checkpoint_rows, os.path.join(OUTPUT_PATH, 'best_checkpoint_summary.json'))

pred_df = pd.DataFrame(all_prediction_rows)
pred_path = os.path.join(OUTPUT_PATH, 'test_predictions.csv')
pred_df.to_csv(pred_path, index=False)

y_true = pred_df['true_label'].astype(int).to_numpy()
y_pred = pred_df['pred_label'].astype(int).to_numpy()
y_prob = pred_df['prob_pneumonia'].astype(float).to_numpy()

metrics = compute_metrics(y_true, y_pred, y_prob, threshold=THRESHOLD)
ece, calib_df = expected_calibration_error(y_true, y_prob, n_bins=N_CALIBRATION_BINS)
metrics['ece'] = ece
metrics['model_name'] = MODEL_NAME
metrics['run_id'] = RUN_ID
metrics['seed'] = SEED
metrics['n_samples'] = int(len(pred_df))
metrics['evaluation_protocol'] = '5-fold out-of-fold cross-validation'
metrics['positive_class'] = dataset.idx_to_class.get(POSITIVE_LABEL, str(POSITIVE_LABEL))
pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_PATH, 'test_metrics_point.csv'), index=False)

boot_df = bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=N_BOOTSTRAP, seed=SEED)
# Add ECE bootstrap separately
rng = np.random.default_rng(SEED)
ece_vals = []
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, len(y_true), size=len(y_true))
    if len(np.unique(y_true[idx])) < 2:
        continue
    ece_i, _ = expected_calibration_error(y_true[idx], y_prob[idx], n_bins=N_CALIBRATION_BINS)
    ece_vals.append(ece_i)
if ece_vals:
    boot_df = pd.concat([boot_df, pd.DataFrame([{
        'model_name': MODEL_NAME,
        'run_id': RUN_ID,
        'metric': 'ece',
        'point_estimate': ece,
        'bootstrap_mean': float(np.mean(ece_vals)),
        'lower_95ci': float(np.percentile(ece_vals, 2.5)),
        'upper_95ci': float(np.percentile(ece_vals, 97.5)),
        'n_bootstrap_requested': N_BOOTSTRAP,
        'n_bootstrap_valid': int(len(ece_vals)),
        'seed': SEED,
    }])], ignore_index=True)
boot_df.to_csv(os.path.join(OUTPUT_PATH, 'test_metrics_bootstrap_ci.csv'), index=False)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
pd.DataFrame(cm, index=[f'true_{c}' for c in dataset.classes], columns=[f'pred_{c}' for c in dataset.classes]).to_csv(os.path.join(OUTPUT_PATH, 'confusion_matrix.csv'))
plot_confusion_matrix(cm, dataset.classes, os.path.join(OUTPUT_PATH, 'confusion_matrix.png'), 'Confusion Matrix: GLCM VGG16')
# Also keep copy in figures
plot_confusion_matrix(cm, dataset.classes, os.path.join(FIGURE_DIR, 'confusion_matrix.png'), 'Confusion Matrix: GLCM VGG16')

calib_df.insert(0, 'model_name', MODEL_NAME)
calib_df.insert(1, 'run_id', RUN_ID)
calib_df['ece'] = ece
calib_df.to_csv(os.path.join(OUTPUT_PATH, 'calibration_bins.csv'), index=False)
plot_reliability_curve(calib_df, ece, os.path.join(OUTPUT_PATH, 'reliability_curve.png'))
plot_reliability_curve(calib_df, ece, os.path.join(FIGURE_DIR, 'reliability_curve.png'))

# Statistical tests against other standardized outputs if available
create_statistical_tests(pred_path).to_csv(os.path.join(OUTPUT_PATH, 'statistical_tests.csv'), index=False)

# Placeholder fusion weight file for single-branch GLCM baseline
fusion_weight_rows = []
for _, r in pred_df.iterrows():
    fusion_weight_rows.append({
        'model_name': MODEL_NAME,
        'run_id': RUN_ID,
        'fold': r['fold'],
        'image_path': r['image_path'],
        'true_label': r['true_label'],
        'pred_label': r['pred_label'],
        'prob_pneumonia': r['prob_pneumonia'],
        'correct': r['correct'],
        'w_raw': 0.0,
        'w_texture': 1.0,
        'note': 'Placeholder for GLCM-only single-branch baseline. No learned fusion weights are present.'
    })
pd.DataFrame(fusion_weight_rows).to_csv(os.path.join(OUTPUT_PATH, 'fusion_weights_test.csv'), index=False)

# Branch prediction comparison placeholder
branch_rows = []
for _, r in pred_df.iterrows():
    branch_rows.append({
        'model_name': MODEL_NAME,
        'run_id': RUN_ID,
        'fold': r['fold'],
        'image_path': r['image_path'],
        'true_label': r['true_label'],
        'pred_label': r['pred_label'],
        'raw_branch_prob_pneumonia': np.nan,
        'texture_branch_prob_pneumonia': r['prob_pneumonia'],
        'fused_prob_pneumonia': r['prob_pneumonia'],
        'branch_agreement': 'not_applicable_single_texture_branch',
        'note': 'GLCM-only baseline. Texture branch probability equals final probability.'
    })
pd.DataFrame(branch_rows).to_csv(os.path.join(OUTPUT_PATH, 'branch_prediction_comparison.csv'), index=False)

# Error analysis cases
err_df = pred_df.copy()
err_df['case_type'] = np.select(
    [
        (err_df['true_label'] == 1) & (err_df['pred_label'] == 1),
        (err_df['true_label'] == 0) & (err_df['pred_label'] == 0),
        (err_df['true_label'] == 0) & (err_df['pred_label'] == 1),
        (err_df['true_label'] == 1) & (err_df['pred_label'] == 0),
    ],
    ['TP', 'TN', 'FP', 'FN'],
    default='unknown'
)
err_df['confidence'] = np.where(err_df['pred_label'] == 1, err_df['prob_pneumonia'], 1 - err_df['prob_pneumonia'])
err_df['uncertainty_abs_prob_minus_0.5'] = np.abs(err_df['prob_pneumonia'] - 0.5)
err_df.sort_values(['correct', 'uncertainty_abs_prob_minus_0.5'], ascending=[True, True]).to_csv(os.path.join(OUTPUT_PATH, 'error_analysis_cases.csv'), index=False)


# ==========================================
# 7. REPRESENTATIVE XAI AND DELETION/INSERTION
# ==========================================
def denormalize_for_plot(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    x = tensor * std + mean
    return torch.clamp(x, 0, 1)


def load_single_tensor(path):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


def make_gradcam(model, input_tensor, target_class):
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # VGG16 target layer: use the last convolutional layer in backbone.features.
    target_layer = None
    for m in reversed(list(model.backbone.features.modules())):
        if isinstance(m, nn.Conv2d):
            target_layer = m
            break
    if target_layer is None:
        return None
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    out = model(input_tensor)
    score = out[:, target_class].sum()
    score.backward()

    h1.remove()
    h2.remove()

    if not activations or not gradients:
        return None
    act = activations[0]
    grad = gradients[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(cam, size=IMG_SIZE, mode='bilinear', align_corners=False)
    cam = cam[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def make_saliency(model, input_tensor, target_class):
    x = input_tensor.clone().detach().requires_grad_(True)
    model.zero_grad()
    out = model(x)
    score = out[:, target_class].sum()
    score.backward()
    sal = x.grad.detach().abs().max(dim=1)[0][0]
    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8)
    return sal.cpu().numpy()


def make_integrated_gradients(model, input_tensor, target_class, steps=24):
    baseline = torch.zeros_like(input_tensor)
    total_grad = torch.zeros_like(input_tensor)
    for alpha in torch.linspace(0, 1, steps, device=input_tensor.device):
        x = (baseline + alpha * (input_tensor - baseline)).clone().detach().requires_grad_(True)
        model.zero_grad()
        out = model(x)
        score = out[:, target_class].sum()
        score.backward()
        total_grad += x.grad.detach()
    avg_grad = total_grad / steps
    ig = (input_tensor - baseline) * avg_grad
    ig_map = ig.detach().abs().sum(dim=1)[0]
    ig_map = ig_map - ig_map.min()
    ig_map = ig_map / (ig_map.max() + 1e-8)
    return ig_map.cpu().numpy()


def save_xai_figure(input_tensor, heatmap, save_path, title):
    import matplotlib.pyplot as plt

    img = denormalize_for_plot(input_tensor[0]).detach().cpu().permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.imshow(img)
    if heatmap is not None:
        ax.imshow(heatmap, alpha=0.45)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def deletion_insertion_curve(model, input_tensor, saliency_map, target_class, steps=10):
    model.eval()
    _, c, h, w = input_tensor.shape
    sal = torch.tensor(saliency_map.reshape(-1), device=input_tensor.device)
    order = torch.argsort(sal, descending=True)
    flat_input = input_tensor.clone().view(1, c, -1)
    baseline = torch.zeros_like(flat_input)
    n_pixels = h * w
    del_probs = []
    ins_probs = []
    fractions = []
    with torch.no_grad():
        for s in range(steps + 1):
            frac = s / steps
            k = int(frac * n_pixels)
            deletion_flat = flat_input.clone()
            insertion_flat = baseline.clone()
            if k > 0:
                idx = order[:k]
                deletion_flat[:, :, idx] = 0.0
                insertion_flat[:, :, idx] = flat_input[:, :, idx]
            deletion_img = deletion_flat.view(1, c, h, w)
            insertion_img = insertion_flat.view(1, c, h, w)
            del_prob = torch.softmax(model(deletion_img), dim=1)[0, target_class].item()
            ins_prob = torch.softmax(model(insertion_img), dim=1)[0, target_class].item()
            fractions.append(frac)
            del_probs.append(del_prob)
            ins_probs.append(ins_prob)
    deletion_auc = float(np.trapz(del_probs, fractions))
    insertion_auc = float(np.trapz(ins_probs, fractions))
    return fractions, del_probs, ins_probs, deletion_auc, insertion_auc


xai_rows = []
try:
    representatives = []
    for case_type in ['TP', 'TN', 'FP', 'FN']:
        subset = err_df[err_df['case_type'] == case_type].copy()
        if len(subset) == 0:
            continue
        # Use most confident correct cases and most confident wrong cases for visibility
        subset = subset.sort_values('confidence', ascending=False)
        representatives.append(subset.iloc[0])

    for rep in representatives:
        fold = int(rep['fold'])
        checkpoint = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME}_fold{fold}.pth')
        model = GLCMBranchModel(num_classes=len(dataset.classes)).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        model.eval()

        input_tensor = load_single_tensor(rep['image_path']).to(DEVICE)
        target_class = int(rep['pred_label'])
        case_dir = os.path.join(XAI_DIR, str(rep['case_type']))
        os.makedirs(case_dir, exist_ok=True)
        stem = Path(rep['image_path']).stem

        gradcam = make_gradcam(model, input_tensor, target_class)
        saliency = make_saliency(model, input_tensor, target_class)
        ig = make_integrated_gradients(model, input_tensor, target_class, steps=24)

        gradcam_path = os.path.join(case_dir, f'{stem}_gradcam.png')
        saliency_path = os.path.join(case_dir, f'{stem}_saliency.png')
        ig_path = os.path.join(case_dir, f'{stem}_integrated_gradients.png')
        save_xai_figure(input_tensor, gradcam, gradcam_path, f'{rep["case_type"]} Grad-CAM')
        save_xai_figure(input_tensor, saliency, saliency_path, f'{rep["case_type"]} Saliency')
        save_xai_figure(input_tensor, ig, ig_path, f'{rep["case_type"]} Integrated Gradients')

        fractions, del_probs, ins_probs, del_auc, ins_auc = deletion_insertion_curve(
            model, input_tensor, saliency, target_class, steps=10
        )
        curve_csv = os.path.join(case_dir, f'{stem}_deletion_insertion_curve.csv')
        pd.DataFrame({
            'fraction_salient_pixels_modified': fractions,
            'deletion_probability': del_probs,
            'insertion_probability': ins_probs,
        }).to_csv(curve_csv, index=False)

        xai_rows.append({
            'model_name': MODEL_NAME,
            'run_id': RUN_ID,
            'fold': fold,
            'case_type': rep['case_type'],
            'image_path': rep['image_path'],
            'true_label': int(rep['true_label']),
            'pred_label': int(rep['pred_label']),
            'prob_pneumonia': float(rep['prob_pneumonia']),
            'gradcam_path': gradcam_path,
            'saliency_path': saliency_path,
            'integrated_gradients_path': ig_path,
            'deletion_insertion_curve_csv': curve_csv,
            'deletion_auc': del_auc,
            'insertion_auc': ins_auc,
            'note': 'Representative examples only; use as qualitative model auditing, not proof of clinical interpretability.'
        })
except Exception as e:
    xai_rows.append({
        'model_name': MODEL_NAME,
        'run_id': RUN_ID,
        'error': repr(e),
        'note': 'XAI generation failed. Metrics files were still saved.'
    })

pd.DataFrame(xai_rows).to_csv(os.path.join(OUTPUT_PATH, 'deletion_insertion_results.csv'), index=False)

print("\nGLCM Branch (VGG16) Training Complete.")
print(f"Output folder: {OUTPUT_PATH}")
print("Main files saved:")
for f in [
    'experiment_config.json', 'glcm_config.json', 'model_config.json', 'split_summary.csv',
    'training_history.csv', 'best_checkpoint_summary.json', 'test_predictions.csv',
    'test_metrics_point.csv', 'test_metrics_bootstrap_ci.csv', 'confusion_matrix.csv',
    'confusion_matrix.png', 'statistical_tests.csv', 'calibration_bins.csv',
    'reliability_curve.png', 'model_complexity.csv', 'fusion_weights_test.csv',
    'branch_prediction_comparison.csv', 'error_analysis_cases.csv',
    'deletion_insertion_results.csv', 'xai_outputs/'
]:
    print(" -", f)
