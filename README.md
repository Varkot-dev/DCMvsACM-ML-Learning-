# DCMvsACM

A machine learning classifier that distinguishes Dilated Cardiomyopathy (DCM) from Arrhythmogenic Cardiomyopathy (ACM) using single-cell RNA sequencing data.

## What It Does

Uses Random Forest, XGBoost, and Logistic Regression with SMOTE oversampling to classify cardiac cell types from scRNA-seq expression profiles. Outputs classification reports, ROC curves, confusion matrices, and feature importance plots.

## Stack

- Python (scanpy, scikit-learn, XGBoost, imbalanced-learn)
- Jupyter Notebook for exploration

## Usage

```bash
python cardiomyopathy-ml/dcm_vs_acm.py
```

## Files

- `cardiomyopathy-ml/dcm_vs_acm.py` — full ML pipeline
- `cardiomyopathy-ml/Welcome_To_Colab.ipynb` — exploratory notebook
