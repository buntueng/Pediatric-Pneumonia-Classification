#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_preparation.py
======================
Stage-0 data preparation for the hybrid pediatric-pneumonia classification
framework described in the manuscript.

What this script does
---------------------
1. Collects every image of the Kermany et al. pediatric chest X-ray dataset
   (the original release ships three folders: train/ val/ test/, each with
   NORMAL/ and PNEUMONIA/ subfolders) into one pool.
2. Re-splits the pool into TRAIN / VALIDATION / TEST partitions using an
   approximate 80:10:10 stratified ratio (the exact counts reported in the
   paper are reproduced when --seed 42 is used on the standard release).
3. Applies the "CXP" preprocessing step: resize to 224x224 and Contrast
   Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast
   while limiting noise over-amplification.
4. Writes the enhanced images to <out_dir>/<split>/<class>/ and a manifest
   CSV (path,label,split) that the training script consumes.

Data augmentation (rotation, horizontal flip, zoom, brightness) is NOT baked
in here; it is applied on-the-fly to the TRAIN split only inside
model_train_test.py, exactly as stated in the paper.

Dataset
-------
Kermany, D. S. et al., "Identifying medical diagnoses and treatable diseases
by image-based deep learning," Cell 172(5), 1122-1131 (2018).
Public mirror: https://data.mendeley.com/datasets/rscbjbr9sj

Usage
-----
    python dataset_preparation.py \
        --raw_dir /path/to/chest_xray \
        --out_dir ./data_processed \
        --img_size 224 --val_frac 0.10 --test_frac 0.10 --seed 42
"""

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CLASS_MAP = {"normal": 0, "pneumonia": 1}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- #
# Collection and splitting
# --------------------------------------------------------------------------- #
def collect_images(raw_dir: Path):
    """Walk the raw dataset and return a list of (path, label_int) tuples.

    Works both for the canonical Kermany layout (train/val/test -> NORMAL/
    PNEUMONIA) and for a flat layout with only NORMAL/ and PNEUMONIA/ folders.
    Class is inferred from any path component containing 'normal'/'pneumonia'.
    """
    samples = []
    for path in raw_dir.rglob("*"):
        if path.suffix.lower() not in IMG_EXTS:
            continue
        parts = [p.lower() for p in path.parts]
        if any("pneumonia" in p for p in parts):
            label = CLASS_MAP["pneumonia"]
        elif any("normal" in p for p in parts):
            label = CLASS_MAP["normal"]
        else:
            continue  # skip images whose class cannot be inferred
        samples.append((path, label))
    if not samples:
        raise RuntimeError(f"No labelled images found under {raw_dir}")
    return samples


def stratified_split(samples, val_frac, test_frac, seed):
    """Stratified train/val/test split, returning three lists of samples."""
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for s in samples:
        by_class[s[1]].append(s)

    train, val, test = [], [], []
    for label, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# --------------------------------------------------------------------------- #
# CXP preprocessing: resize + CLAHE
# --------------------------------------------------------------------------- #
def cxp_preprocess(img_bgr, img_size, clip_limit, tile_grid):
    """Resize to (img_size, img_size) and apply CLAHE.

    CLAHE is applied on the luminance channel so colour (if any) is preserved;
    pediatric CXRs are grayscale, so this reduces to single-channel CLAHE.
    """
    img = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced


def process_split(split_name, samples, out_dir, args, manifest_rows):
    inv_class = {v: k for k, v in CLASS_MAP.items()}
    for src_path, label in tqdm(samples, desc=f"{split_name:10s}", ncols=80):
        cls = inv_class[label]
        dst_dir = out_dir / split_name / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:  # unreadable file
            continue
        enhanced = cxp_preprocess(
            img, args.img_size, args.clahe_clip, (args.clahe_tile, args.clahe_tile)
        )
        dst_path = dst_dir / f"{src_path.stem}.png"
        cv2.imwrite(str(dst_path), enhanced)
        manifest_rows.append(
            {"path": str(dst_path), "label": label, "split": split_name}
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Prepare Kermany pediatric CXR data.")
    ap.add_argument("--raw_dir", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("./data_processed"), type=Path)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--val_frac", default=0.10, type=float)
    ap.add_argument("--test_frac", default=0.10, type=float)
    ap.add_argument("--clahe_clip", default=2.0, type=float,
                    help="CLAHE clip limit (selected on validation folds).")
    ap.add_argument("--clahe_tile", default=8, type=int,
                    help="CLAHE square tile-grid size.")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Scanning {args.raw_dir} ...")
    samples = collect_images(args.raw_dir)
    train, val, test = stratified_split(
        samples, args.val_frac, args.test_frac, args.seed
    )

    def counts(split):
        n_norm = sum(1 for _, y in split if y == 0)
        n_pneu = sum(1 for _, y in split if y == 1)
        return len(split), n_norm, n_pneu

    for name, sp in (("train", train), ("val", val), ("test", test)):
        t, n, p = counts(sp)
        print(f"  {name:5s}: {t:5d} images  (normal={n}, pneumonia={p})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    process_split("train", train, args.out_dir, args, manifest_rows)
    process_split("val", val, args.out_dir, args, manifest_rows)
    process_split("test", test, args.out_dir, args, manifest_rows)

    manifest_path = args.out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} processed images.")
    print(f"Manifest: {manifest_path}")
    print("Done. Use model_train_test.py next.")


if __name__ == "__main__":
    main()
