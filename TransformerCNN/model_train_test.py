#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_train_test.py
===================
Model construction, training (stratified 10-fold CV on the training split)
and final held-out test-set evaluation for the hybrid pediatric-pneumonia
classification framework described in the manuscript.

It implements every component reported in the paper:

  * CXP            - already baked into the processed images by
                     dataset_preparation.py (resize + CLAHE). A runtime
                     toggle is kept only so the "ensemble" (no-CXP) ablation
                     can read the *original* raw images instead.
  * Dual branch    - Swin-T (timm: swin_tiny_patch4_window7_224) and
                     ResNet50 (torchvision), both ImageNet pre-trained.
  * WAM            - Wavelet Attention Module: Haar DWT (fixed depthwise
                     conv filters) -> frequency-component descriptor ->
                     FC-ReLU-FC-Sigmoid attention -> residual recalibration.
  * LEA            - Lightweight Efficient Attention: channel attention
                     (GAP -> MLP -> sigmoid) then spatial attention
                     (avg/max pool -> 7x7 conv -> sigmoid), followed by
                     GAP + linear projection to a 512-d branch embedding.
  * DHFFC          - Dual-Head Feature Fusion Classifier: a concat head
                     ([s; r] in R^1024 -> 512 -> 2 logits) and a Hadamard
                     head (s (.) r in R^512 -> 256 -> 2 logits) whose logits
                     are summed before softmax.

Ablation variants (selectable with --variant) follow Table 2 of the paper:

  ensemble                Swin-T + ResNet50, simple averaged heads, raw images
  cxp                     + CXP preprocessed images
  wam                     + CXP + WAM
  lea                     + CXP + WAM + LEA
  proposed                + CXP + WAM + LEA + DHFFC      (the full model)

Baselines (selectable with --variant baseline --backbone <name>) cover
Table 1: alexnet, efficientnet_b0, densenet121, resnet50, swin_t.

Outputs (written to --out_dir/<run_name>/):
  metrics.json            all scalar metrics on the held-out test set
  cv_metrics.csv          per-fold validation metrics from stratified CV
  cv_history.csv          per-epoch train/val loss & accuracy for every CV fold
  cv_predictions.csv      out-of-fold validation predictions
  cv_summary.csv          mean/std validation metrics across folds
  history.json            per-epoch train/validation loss & accuracy (final fit)
  predictions.csv         path,y_true,y_pred,prob_pneumonia for the test set
  embeddings.npy          (N, 512|fused) test embeddings for t-SNE
  embeddings_labels.npy   matching labels
  confusion_matrix.npy    2x2 confusion matrix on the test set
  model.pt                best model weights (final fit)

When --run_all is used, the script also writes:
  all_runs_summary.csv     final held-out test metrics for all runs
  statistical_tests.csv    Wilcoxon signed-rank and paired t-test results
                           comparing proposed vs. baselines/ablations using
                           paired fold-level CV metrics.

The result_analysis.ipynb notebook consumes these files.

Usage
-----
    # Full proposed model
    python model_train_test.py --manifest ./data_processed/manifest.csv \
        --variant proposed --epochs 20 --batch_size 32 --folds 10

    # A baseline
    python model_train_test.py --manifest ./data_processed/manifest.csv \
        --variant baseline --backbone swin_t --epochs 20

    # Reproduce the whole study (all baselines + all ablation steps)
    python model_train_test.py --manifest ./data_processed/manifest.csv --run_all
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score,
)

try:
    from scipy.stats import wilcoxon, ttest_rel
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

SEED = 42
NUM_CLASSES = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class CXRDataset(Dataset):
    """Reads images listed in the manifest produced by dataset_preparation.py."""

    def __init__(self, rows, train=False, img_size=224):
        self.rows = rows.reset_index(drop=True)
        self.train = train
        if train:
            # Augmentation applied to the TRAIN split only (paper: rotation,
            # horizontal flip, zoom, brightness).
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
                transforms.ColorJitter(brightness=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        x = self.tf(img)
        y = int(row["label"])
        return x, y, row["path"]


# --------------------------------------------------------------------------- #
# WAM - Wavelet Attention Module
# --------------------------------------------------------------------------- #
class HaarDWT(nn.Module):
    """Single-level 2-D Haar DWT implemented as a fixed depthwise conv.

    Returns the four sub-bands (LL, LH, HL, HH) stacked along a new axis so
    each input channel produces four frequency components.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Haar analysis filters (normalised).
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filt = torch.stack([ll, lh, hl, hh], dim=0)          # (4,2,2)
        filt = filt.unsqueeze(1)                             # (4,1,2,2)
        # depthwise: repeat each filter across channels
        weight = filt.repeat(channels, 1, 1, 1)              # (4C,1,2,2)
        self.register_buffer("weight", weight)

    def forward(self, x):
        b, c, h, w = x.shape
        # pad to even spatial size
        if h % 2:
            x = F.pad(x, (0, 0, 0, 1))
        if w % 2:
            x = F.pad(x, (0, 1, 0, 0))
        out = F.conv2d(x, self.weight, stride=2, groups=c)   # (B,4C,H/2,W/2)
        out = out.view(b, c, 4, out.shape[-2], out.shape[-1])
        return out  # (B, C, 4, H/2, W/2)


class WAM(nn.Module):
    """Frequency-aware channel attention with residual recalibration."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.dwt = HaarDWT(channels)
        # descriptor: per-channel energy of each of the 4 sub-bands -> 4C
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        sub = self.dwt(x)                       # (B,C,4,h,w)
        energy = sub.abs().mean(dim=(-1, -2))   # (B,C,4)
        desc = energy.view(b, c * 4)            # (B,4C)
        attn = self.mlp(desc).view(b, c, 1, 1)  # (B,C,1,1)
        return x + x * attn                     # residual recalibration


# --------------------------------------------------------------------------- #
# LEA - Lightweight Efficient Attention
# --------------------------------------------------------------------------- #
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class LEA(nn.Module):
    """Channel + spatial attention, then GAP + linear projection to 512-d."""

    def __init__(self, channels, out_dim=512, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(7)
        self.proj = nn.Linear(channels, out_dim)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        v = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B,C)
        return self.proj(v)                          # (B,512)


# --------------------------------------------------------------------------- #
# DHFFC - Dual-Head Feature Fusion Classifier
# --------------------------------------------------------------------------- #
class DHFFC(nn.Module):
    def __init__(self, dim=512, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.concat_head = nn.Sequential(
            nn.Linear(dim * 2, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.hadamard_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, s, r):
        z1 = self.concat_head(torch.cat([s, r], dim=1))
        z2 = self.hadamard_head(s * r)
        return z1 + z2


# --------------------------------------------------------------------------- #
# Backbones (feature-map outputs)
# --------------------------------------------------------------------------- #
def build_swin_features():
    if not _HAS_TIMM:
        raise RuntimeError("timm is required for the Swin-T branch: pip install timm")
    m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True,
                           features_only=True)
    ch = m.feature_info.channels()[-1]   # 768 for swin-tiny
    return m, ch


def build_resnet_features():
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    modules = list(net.children())[:-2]   # drop avgpool + fc -> (B,2048,7,7)
    return nn.Sequential(*modules), 2048


class SwinFeat(nn.Module):
    """Wrap timm Swin features_only output into a (B,C,H,W) map."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model(x)[-1]          # (B,H,W,C) for swin in timm
        if feats.dim() == 4 and feats.shape[1] != feats.shape[-1]:
            # timm swin returns NHWC; convert to NCHW
            feats = feats.permute(0, 3, 1, 2).contiguous()
        return feats


# --------------------------------------------------------------------------- #
# Full dual-branch model with ablation switches
# --------------------------------------------------------------------------- #
class DualBranchModel(nn.Module):
    """Configurable dual-branch model covering every ablation step.

    use_wam / use_lea / use_dhffc toggle the corresponding modules. When LEA
    is disabled a plain GAP + linear projection produces the 512-d embedding;
    when DHFFC is disabled the two branch logits are simply averaged.
    """

    def __init__(self, use_wam=True, use_lea=True, use_dhffc=True,
                 emb_dim=512, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.use_wam = use_wam
        self.use_lea = use_lea
        self.use_dhffc = use_dhffc

        swin_model, swin_ch = build_swin_features()
        self.swin = SwinFeat(swin_model)
        self.resnet, res_ch = build_resnet_features()

        if use_wam:
            self.wam_s = WAM(swin_ch)
            self.wam_r = WAM(res_ch)

        if use_lea:
            self.embed_s = LEA(swin_ch, emb_dim)
            self.embed_r = LEA(res_ch, emb_dim)
        else:
            self.embed_s = nn.Linear(swin_ch, emb_dim)
            self.embed_r = nn.Linear(res_ch, emb_dim)

        if use_dhffc:
            self.classifier = DHFFC(emb_dim, num_classes, dropout)
        else:
            # simple averaged-head ensemble
            self.head_s = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, num_classes))
            self.head_r = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, num_classes))

        self.last_embedding = None  # cached for t-SNE export

    def _embed(self, feat, wam, embed):
        if self.use_wam:
            feat = wam(feat)
        if self.use_lea:
            return embed(feat)
        v = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return embed(v)

    def forward(self, x):
        fs = self.swin(x)
        fr = self.resnet(x)
        s = self._embed(fs, getattr(self, "wam_s", None), self.embed_s)
        r = self._embed(fr, getattr(self, "wam_r", None), self.embed_r)
        self.last_embedding = torch.cat([s, r], dim=1).detach()
        if self.use_dhffc:
            return self.classifier(s, r)
        return 0.5 * (self.head_s(s) + self.head_r(r))


# --------------------------------------------------------------------------- #
# Baseline single-backbone models (Table 1)
# --------------------------------------------------------------------------- #
def build_baseline(name, num_classes=NUM_CLASSES):
    name = name.lower()
    if name == "alexnet":
        m = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "densenet121":
        m = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "swin_t":
        if not _HAS_TIMM:
            raise RuntimeError("timm required for swin_t baseline")
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True,
                              num_classes=num_classes)
    else:
        raise ValueError(f"unknown baseline {name}")
    return m


VARIANT_FLAGS = {
    # variant      : (use_wam, use_lea, use_dhffc)
    "ensemble":  (False, False, False),
    "cxp":       (False, False, False),
    "wam":       (True,  False, False),
    "lea":       (True,  True,  False),
    "proposed":  (True,  True,  True),
}


def build_model(variant, backbone=None):
    if variant == "baseline":
        return build_baseline(backbone)
    flags = VARIANT_FLAGS[variant]
    return DualBranchModel(*flags)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train)
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    torch.set_grad_enabled(True)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, collect_embeddings=False):
    model.eval()
    ys, ps, probs, paths, embs = [], [], [], [], []
    t0 = time.time()
    for x, y, p in loader:
        x = x.to(device)
        out = model(x)
        prob = F.softmax(out, dim=1)[:, 1]
        ys.extend(y.numpy().tolist())
        ps.extend(out.argmax(1).cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())
        paths.extend(list(p))
        if collect_embeddings and getattr(model, "last_embedding", None) is not None:
            embs.append(model.last_embedding.cpu().numpy())
    infer_time = time.time() - t0
    ys, ps, probs = np.array(ys), np.array(ps), np.array(probs)
    embeddings = np.concatenate(embs, axis=0) if embs else None
    return ys, ps, probs, paths, embeddings, infer_time


def compute_metrics(y_true, y_pred, y_prob, infer_time=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = float("nan")
    if infer_time is not None:
        metrics["inference_time_s"] = float(infer_time)
    return metrics


def make_loader(rows, train, batch_size, img_size, workers):
    ds = CXRDataset(rows, train=train, img_size=img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=True)



def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_manifest_splits(df, args):
    """Return train/validation/test DataFrames without leaking the test set.

    Expected manifest columns: path, label, split. If a validation split exists,
    it is used for final-model early stopping. If not, a small stratified
    validation subset is carved out of the training split.
    """
    required = {"path", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["split"] = df["split"].astype(str).str.lower()

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"].isin(["val", "valid", "validation"])].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("No rows with split == 'train' were found in the manifest.")
    if len(test_df) == 0:
        raise ValueError("No rows with split == 'test' were found in the manifest.")

    if len(val_df) == 0:
        # Create a final-fit validation set only from train_df.
        train_df, val_df = train_test_split(
            train_df,
            test_size=args.final_val_size,
            stratify=train_df["label"],
            random_state=SEED,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        print(
            f"No validation split found. Created a stratified final-fit validation "
            f"subset from training data: train={len(train_df)}, val={len(val_df)}."
        )
    else:
        print(
            f"Using manifest splits for final fit: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}."
        )

    return train_df, val_df, test_df


def class_balance(rows):
    counts = rows["label"].value_counts().sort_index().to_dict()
    return ";".join(f"{int(k)}:{int(v)}" for k, v in counts.items())


def train_one_run(args, variant, backbone, device):
    run_name = variant if variant != "baseline" else f"baseline_{backbone}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Run: {run_name} ===")

    df = pd.read_csv(args.manifest)
    full_train_df = df[df["split"].astype(str).str.lower() == "train"].reset_index(drop=True)
    final_train_df, final_val_df, test_df = get_manifest_splits(df, args)

    if len(full_train_df) == 0:
        raise ValueError("No training rows available for cross-validation.")
    if full_train_df["label"].nunique() < 2:
        raise ValueError("Cross-validation requires at least two classes in the training split.")

    criterion = nn.CrossEntropyLoss()

    # ----- Stratified k-fold CV on the training split only ----- #
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    cv_rows, cv_history_rows, cv_pred_rows = [], [], []
    print(f"Starting stratified {args.folds}-fold cross-validation on training split only.")

    for fold, (tr_idx, va_idx) in enumerate(
            skf.split(full_train_df, full_train_df["label"]), start=1):
        tr_rows = full_train_df.iloc[tr_idx].reset_index(drop=True)
        va_rows = full_train_df.iloc[va_idx].reset_index(drop=True)
        tr_loader = make_loader(tr_rows, True, args.batch_size, args.img_size, args.workers)
        va_loader = make_loader(va_rows, False, args.batch_size, args.img_size, args.workers)

        model = build_model(variant, backbone).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_metric, best_state, bad = -np.inf, None, 0
        for epoch in range(1, args.epochs + 1):
            tl, ta = run_epoch(model, tr_loader, criterion, optim, device, train=True)
            vl, va = run_epoch(model, va_loader, criterion, optim, device, train=False)
            cv_history_rows.append({
                "variant": run_name,
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(tl),
                "train_acc": float(ta),
                "val_loss": float(vl),
                "val_acc": float(va),
                "train_class_balance": class_balance(tr_rows),
                "val_class_balance": class_balance(va_rows),
            })
            print(
                f"  fold {fold:02d} epoch {epoch:02d}: "
                f"train_acc={ta:.4f} val_acc={va:.4f}"
            )
            if va > best_metric:
                best_metric, bad = va, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"  fold {fold:02d}: early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        yv, pv, prv, paths, _, _ = evaluate(model, va_loader, device)
        fm = compute_metrics(yv, pv, prv)
        fm.update({
            "variant": run_name,
            "fold": fold,
            "n_train": int(len(tr_rows)),
            "n_val": int(len(va_rows)),
            "train_class_balance": class_balance(tr_rows),
            "val_class_balance": class_balance(va_rows),
            "best_val_acc": float(best_metric),
        })
        cv_rows.append(fm)
        cv_pred_rows.extend({
            "variant": run_name,
            "fold": fold,
            "path": path,
            "y_true": int(yt),
            "y_pred": int(yp),
            "prob_pneumonia": float(pr),
        } for path, yt, yp, pr in zip(paths, yv, pv, prv))
        print(f"  fold {fold:02d}: val_acc={fm['accuracy']:.4f} f1={fm['f1']:.4f} mcc={fm['mcc']:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cv_df = pd.DataFrame(cv_rows)
    cv_history_df = pd.DataFrame(cv_history_rows)
    cv_pred_df = pd.DataFrame(cv_pred_rows)
    cv_df.to_csv(out_dir / "cv_metrics.csv", index=False)
    cv_history_df.to_csv(out_dir / "cv_history.csv", index=False)
    cv_pred_df.to_csv(out_dir / "cv_predictions.csv", index=False)

    metric_cols = ["accuracy", "precision", "sensitivity", "specificity", "f1", "mcc", "auc"]
    cv_summary = []
    for m in metric_cols:
        cv_summary.append({
            "variant": run_name,
            "metric": m,
            "mean": float(cv_df[m].mean()),
            "std": float(cv_df[m].std(ddof=1)),
            "folds": int(args.folds),
        })
    pd.DataFrame(cv_summary).to_csv(out_dir / "cv_summary.csv", index=False)

    # ----- Final fit: train + validation only; held-out test used once at end ----- #
    tr_loader = make_loader(final_train_df, True, args.batch_size, args.img_size, args.workers)
    va_loader = make_loader(final_val_df, False, args.batch_size, args.img_size, args.workers)
    te_loader = make_loader(test_df, False, args.batch_size, args.img_size, args.workers)

    model = build_model(variant, backbone).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "val_split_used": "manifest_val" if df["split"].astype(str).str.lower().isin(["val", "valid", "validation"]).any() else "stratified_from_train",
        "test_set_used_for_early_stopping": False,
    }
    best_acc, best_state, bad = -np.inf, None, 0
    print("Starting final training. Validation is used for early stopping; test is held out.")
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, tr_loader, criterion, optim, device, train=True)
        vl, va = run_epoch(model, va_loader, criterion, optim, device, train=False)
        history["train_loss"].append(float(tl)); history["train_acc"].append(float(ta))
        history["val_loss"].append(float(vl)); history["val_acc"].append(float(va))
        print(f"  epoch {epoch:02d}: train_acc={ta:.4f} val_acc={va:.4f}")
        if va > best_acc:
            best_acc, bad = va, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stopping.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    y, p, prob, paths, emb, infer_time = evaluate(
        model, te_loader, device, collect_embeddings=True)
    metrics = compute_metrics(y, p, prob, infer_time)
    metrics.update({
        "variant": run_name,
        "cv_folds": int(args.folds),
        "cv_type": "StratifiedKFold",
        "test_set_used_for_early_stopping": False,
        "n_final_train": int(len(final_train_df)),
        "n_final_val": int(len(final_val_df)),
        "n_test": int(len(test_df)),
        "final_train_class_balance": class_balance(final_train_df),
        "final_val_class_balance": class_balance(final_val_df),
        "test_class_balance": class_balance(test_df),
    })
    print(f"  HELD-OUT TEST: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} mcc={metrics['mcc']:.4f}")

    # ----- save artefacts ----- #
    save_json(metrics, out_dir / "metrics.json")
    save_json(history, out_dir / "history.json")
    pd.DataFrame({
        "path": paths, "y_true": y, "y_pred": p, "prob_pneumonia": prob,
    }).to_csv(out_dir / "predictions.csv", index=False)
    np.save(out_dir / "confusion_matrix.npy",
            confusion_matrix(y, p, labels=[0, 1]))
    if emb is not None:
        np.save(out_dir / "embeddings.npy", emb)
        np.save(out_dir / "embeddings_labels.npy", y)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"  saved -> {out_dir}")
    return metrics


def paired_statistical_tests(results_root, reference="proposed",
                             metrics=("accuracy", "f1", "mcc", "auc")):
    """Compare reference model against all other runs using paired CV folds.

    The Wilcoxon signed-rank test and paired t-test are run on fold-level
    metric values. This is suitable because every model uses the same
    deterministic StratifiedKFold splits (same seed, same fold IDs).
    """
    results_root = Path(results_root)
    ref_path = results_root / reference / "cv_metrics.csv"
    if not ref_path.exists():
        print(f"Statistical testing skipped: {ref_path} not found.")
        return pd.DataFrame()

    ref = pd.read_csv(ref_path)
    rows = []
    for model_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        comp_name = model_dir.name
        if comp_name == reference:
            continue
        comp_path = model_dir / "cv_metrics.csv"
        if not comp_path.exists():
            continue
        comp = pd.read_csv(comp_path)
        merged = ref.merge(comp, on="fold", suffixes=("_reference", "_comparison"))
        if merged.empty:
            continue
        for metric in metrics:
            ref_col = f"{metric}_reference"
            comp_col = f"{metric}_comparison"
            if ref_col not in merged.columns or comp_col not in merged.columns:
                continue
            x = merged[ref_col].astype(float).to_numpy()
            y = merged[comp_col].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            diff = x - y
            n = int(len(diff))
            if n < 2:
                continue

            wilcoxon_stat, wilcoxon_p = np.nan, np.nan
            paired_t_stat, paired_t_p = np.nan, np.nan
            if _HAS_SCIPY:
                try:
                    if np.allclose(diff, 0):
                        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
                    else:
                        w = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                        wilcoxon_stat, wilcoxon_p = float(w.statistic), float(w.pvalue)
                except Exception:
                    pass
                try:
                    t = ttest_rel(x, y, nan_policy="omit")
                    paired_t_stat, paired_t_p = float(t.statistic), float(t.pvalue)
                except Exception:
                    pass

            sd = float(np.std(diff, ddof=1)) if n > 1 else np.nan
            cohen_dz = float(np.mean(diff) / sd) if sd and not np.isclose(sd, 0) else np.nan
            rows.append({
                "reference": reference,
                "comparison": comp_name,
                "metric": metric,
                "n_pairs": n,
                "reference_mean": float(np.mean(x)),
                "comparison_mean": float(np.mean(y)),
                "mean_difference_reference_minus_comparison": float(np.mean(diff)),
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p_value": wilcoxon_p,
                "paired_t_statistic": paired_t_stat,
                "paired_t_p_value": paired_t_p,
                "cohen_dz": cohen_dz,
                "significant_at_0_05": bool(np.isfinite(wilcoxon_p) and wilcoxon_p < 0.05),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(results_root / "statistical_tests.csv", index=False)
        print(f"Wrote statistical tests -> {results_root / 'statistical_tests.csv'}")
    else:
        print("No statistical tests were written because no paired CV metrics were available.")
    return out


def main():
    ap = argparse.ArgumentParser(description="Train/test hybrid pneumonia model with stratified 10-fold CV and statistical testing.")
    ap.add_argument("--manifest", required=True, type=str,
                    help="manifest.csv from dataset_preparation.py")
    ap.add_argument("--out_dir", default="./results", type=str)
    ap.add_argument("--variant", default="proposed",
                    choices=["ensemble", "cxp", "wam", "lea", "proposed", "baseline"])
    ap.add_argument("--backbone", default="swin_t",
                    help="baseline backbone when --variant baseline")
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--weight_decay", default=1e-4, type=float)
    ap.add_argument("--folds", default=10, type=int,
                    help="number of stratified CV folds; default is 10")
    ap.add_argument("--final_val_size", default=0.10, type=float,
                    help="validation fraction carved from train if manifest has no val split")
    ap.add_argument("--patience", default=5, type=int)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--workers", default=4, type=int)
    ap.add_argument("--run_all", action="store_true",
                    help="reproduce all baselines + all ablation variants")
    ap.add_argument("--no_stats", action="store_true",
                    help="skip statistical tests after --run_all")
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CV setting: StratifiedKFold with n_splits={args.folds}, shuffle=True, random_state={SEED}")

    summary = []
    if args.run_all:
        for bb in ["alexnet", "efficientnet_b0", "densenet121", "resnet50", "swin_t"]:
            summary.append(train_one_run(args, "baseline", bb, device))
        for v in ["ensemble", "cxp", "wam", "lea", "proposed"]:
            summary.append(train_one_run(args, v, None, device))
        out_path = Path(args.out_dir) / "all_runs_summary.csv"
        pd.DataFrame(summary).to_csv(out_path, index=False)
        print(f"\nWrote summary -> {out_path}")
        if not args.no_stats:
            paired_statistical_tests(Path(args.out_dir), reference="proposed")
    else:
        train_one_run(args, args.variant, args.backbone, device)
        print("\nSingle run complete. Statistical tests require paired fold metrics from multiple models; use --run_all to generate statistical_tests.csv.")


if __name__ == "__main__":
    main()
