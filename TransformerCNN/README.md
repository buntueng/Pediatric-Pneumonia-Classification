# Transformer--CNN Fusion for Pediatric Pneumonia Classification

This folder contains the reproducibility code for the study:

**Transformer--CNN fusion with wavelet and efficient attention for interpretable pediatric pneumonia classification**

The project implements a hybrid deep learning framework for binary pediatric chest X-ray classification using a Swin Transformer branch, a ResNet50 branch, contrast-limited adaptive histogram equalization (CLAHE) preprocessing, a wavelet attention module (WAM), a lightweight efficient attention module (LEA), and a dual-head feature fusion classifier (DHFFC).

## Repository contents

```text
TransformerCNN/
├── dataset_preparation.py   # prepares Kermany pediatric CXR images and manifest.csv
├── model_train_test.py      # trains baselines, ablations, and the proposed model
├── result_analysis.ipynb    # generates result tables, plots, and statistical summaries
├── README.md                # this file
├── requirements.txt         # Python package requirements
└── LICENSE                  # software license
```

## Dataset

This code uses the publicly available pediatric chest X-ray dataset from:

Kermany, D. S. et al. **Identifying medical diagnoses and treatable diseases by image-based deep learning.** *Cell* 172(5), 1122--1131.e9 (2018).

Dataset DOI: `10.17632/rscbjbr9sj.2`

The dataset should be downloaded from the original public repository and extracted locally. The expected input layout is the standard Kermany chest X-ray directory structure:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The code also supports a simplified layout containing only `NORMAL/` and `PNEUMONIA/` folders. Class labels are inferred from path names containing `normal` or `pneumonia`.

No private patient data are included in this repository.

## Environment setup

Create a clean Python environment, then install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version if the default `pip install -r requirements.txt` does not install a CUDA-enabled PyTorch build.

## Step 1: Prepare the dataset

Run the preprocessing script after downloading and extracting the Kermany dataset.

```bash
python dataset_preparation.py \
  --raw_dir /path/to/chest_xray \
  --out_dir ./data_processed \
  --img_size 224 \
  --val_frac 0.10 \
  --test_frac 0.10 \
  --seed 42
```

This script:

- collects normal and pneumonia images,
- creates stratified train, validation, and test partitions,
- resizes images to 224 x 224 pixels,
- applies CLAHE preprocessing,
- writes processed images to `./data_processed/`, and
- creates `./data_processed/manifest.csv`.

## Step 2: Train and evaluate the proposed model

```bash
python model_train_test.py \
  --manifest ./data_processed/manifest.csv \
  --variant proposed \
  --epochs 20 \
  --batch_size 32 \
  --folds 10 \
  --out_dir ./results
```

The proposed model uses:

- Swin-T branch,
- ResNet50 branch,
- wavelet attention module,
- lightweight efficient attention module,
- dual-head feature fusion classifier.

## Step 3: Run all baselines and ablation models

```bash
python model_train_test.py \
  --manifest ./data_processed/manifest.csv \
  --run_all \
  --epochs 20 \
  --batch_size 32 \
  --folds 10 \
  --out_dir ./results
```

This command runs the single-backbone baselines and the progressive ablation variants:

```text
baseline_alexnet
baseline_efficientnet_b0
baseline_densenet121
baseline_resnet50
baseline_swin_t
ensemble
cxp
wam
lea
proposed
```

## Step 4: Analyze results

Open the notebook:

```bash
jupyter notebook result_analysis.ipynb
```

The notebook reads outputs from `./results/` and generates tables and figures under:

```text
figures/
tables/
```

## Main output files

Each model run writes outputs to `results/<run_name>/`, including:

```text
metrics.json
cv_metrics.csv
cv_history.csv
cv_predictions.csv
cv_summary.csv
history.json
predictions.csv
confusion_matrix.npy
embeddings.npy
embeddings_labels.npy
model.pt
```

When `--run_all` is used, the script also writes:

```text
results/all_runs_summary.csv
results/statistical_tests.csv
```

## Reproducibility notes

- Random seed: `42`
- Input size: `224 x 224`
- Optimizer: Adam
- Learning rate: `1e-4`
- Batch size: `32`
- Maximum epochs: `20`
- Early stopping patience: `5`
- Cross-validation: stratified K-fold, controlled by the `--folds` argument

Exact numerical results may vary slightly depending on the GPU, PyTorch version, CUDA/cuDNN version, and installed package versions.

## Citation

If this code is used, please cite the associated manuscript after publication. Until then, please cite the dataset article:

```bibtex
@article{kermany2018identifying,
  title   = {Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning},
  author  = {Kermany, Daniel S. and Goldbaum, Michael and Cai, Wenjia and others},
  journal = {Cell},
  volume  = {172},
  number  = {5},
  pages   = {1122--1131.e9},
  year    = {2018},
  doi     = {10.1016/j.cell.2018.02.010}
}
```

## License

The source code in this folder is released under the MIT License. See `LICENSE` for details.

The dataset is not redistributed in this repository. Users should obtain it from the original public source and follow the dataset license and citation requirements.
