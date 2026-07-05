"""
runpod_full_suite.py — the full battery, run once, on a machine with real RAM.

Everything that was previously blocked, chunked, or subsampled for memory
reasons on a laptop (16GB) or Colab free tier (12.7GB) is run here at full
scale on a RunPod A40 instance (503GB RAM, 46GB VRAM). No shortcuts.

Stages:
  1. Load the FULL 880k-cell atlas, filter to DCM+ACM (166,519 cells)
  2. Proper seurat_v3 HVG selection (the loess-based method that crashed
     everywhere else) on the full 32,383-gene panel
  3. scVI pretraining from scratch on the full filtered dataset
  4. UMAP of the learned latent space
  5. Honest (patient-level) classification: raw genes vs scVI latent
  6. Hugging Face pretrained model (scvi-tools/heart-cell-atlas-scvi) via
     query adaptation, as a separate comparison path
  7. Full pseudobulk differential expression across all 32,383 genes
  8. Save every intermediate artifact to /workspace/results/ so nothing
     is lost if the pod is stopped early
"""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DCM = "dilated cardiomyopathy"
ACM = "arrhythmogenic right ventricular cardiomyopathy"
H5AD_PATH = "/workspace/data/cardiomyocytes.h5ad"

log_lines = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


def save_log():
    with open(f"{RESULTS_DIR}/run_log.txt", "w") as f:
        f.write("\n".join(log_lines))


# ── Stage 1: load full atlas, filter ────────────────────────────────────────
log("Stage 1: loading full 880k-cell atlas...")
adata_full = ad.read_h5ad(H5AD_PATH)
log(f"Full atlas: {adata_full.n_obs:,} cells x {adata_full.n_vars:,} genes")

mask = adata_full.obs["disease"].isin([DCM, ACM])
adata = adata_full[mask].copy()
log(f"Filtered to DCM+ACM: {adata.n_obs:,} cells")
log(adata.obs["disease"].value_counts().to_string())

n_donors = adata.obs["donor_id"].nunique()
log(f"Donors: {n_donors}")

# ── Stage 2: proper seurat_v3 HVG selection (full method, no shortcuts) ────
log("Stage 2: seurat_v3 HVG selection on full 32,383-gene panel (the real method)...")
t0 = time.time()
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", subset=False)
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
log(f"HVG selection done in {time.time()-t0:.0f}s. HVG matrix: {adata_hvg.shape}")
adata_hvg.write_h5ad(f"{RESULTS_DIR}/adata_hvg_seurat_v3.h5ad")

# ── Stage 3: scVI pretraining from scratch, full dataset ───────────────────
log("Stage 3: scVI pretraining on full filtered dataset (no subsampling)...")
import scvi
scvi.settings.seed = 42

scvi.model.SCVI.setup_anndata(adata_hvg, batch_key="donor_id")
model = scvi.model.SCVI(adata_hvg, n_latent=20, n_layers=2, n_hidden=128)
log(str(model))

t0 = time.time()
model.train(max_epochs=100, train_size=0.9, early_stopping=True, early_stopping_patience=10)
log(f"scVI training done in {time.time()-t0:.0f}s")

train_hist = model.history["train_loss_epoch"]
val_hist = model.history["validation_loss"]
train_hist.to_csv(f"{RESULTS_DIR}/scvi_train_loss.csv")
val_hist.to_csv(f"{RESULTS_DIR}/scvi_val_loss.csv")

model.save(f"{RESULTS_DIR}/scvi_model", overwrite=True)
log("scVI model saved.")

adata_hvg.obsm["X_scVI"] = model.get_latent_representation()
latent_own = adata_hvg.obsm["X_scVI"]
np.save(f"{RESULTS_DIR}/latent_own_scvi.npy", latent_own)
log(f"Own-scVI latent shape: {latent_own.shape}")

# ── Stage 4: UMAP of the latent space ───────────────────────────────────────
log("Stage 4: UMAP of scVI latent space...")
t0 = time.time()
sc.pp.neighbors(adata_hvg, use_rep="X_scVI", n_neighbors=15)
sc.tl.umap(adata_hvg)
log(f"UMAP done in {time.time()-t0:.0f}s")
adata_hvg.obs[["disease", "donor_id"]].to_csv(f"{RESULTS_DIR}/umap_obs.csv")
np.save(f"{RESULTS_DIR}/umap_coords.npy", adata_hvg.obsm["X_umap"])

# ── Stage 5: honest evaluation — raw genes vs own-scVI latent ──────────────
log("Stage 5: honest patient-level evaluation, raw genes vs scVI latent...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

y = (adata_hvg.obs["disease"] == DCM).astype(int).values
groups = adata_hvg.obs["donor_id"].astype("category").cat.codes.values
scale = y.sum() / (1 - y).sum()

X_raw = adata_hvg.X
if sp.issparse(X_raw):
    X_raw = X_raw.toarray()

feature_sets = {
    "raw_genes_seurat_v3": X_raw,
    "scvi_latent_own": latent_own,
}

all_results = {}
for feat_name, X in feature_sets.items():
    for clf_name, clf in [
        ("RF", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=42, n_jobs=-1)),
        ("XGB", xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale,
                                   random_state=42, eval_metric="logloss", verbosity=0)),
    ]:
        aucs = []
        for fold, (tr, val) in enumerate(StratifiedGroupKFold(n_splits=5).split(X, y, groups)):
            clf.fit(X[tr], y[tr])
            auc = roc_auc_score(y[val], clf.predict_proba(X[val])[:, 1])
            aucs.append(auc)
        key = f"{feat_name}_{clf_name}"
        all_results[key] = aucs
        log(f"{key}: folds={['%.4f' % a for a in aucs]}  mean={np.mean(aucs):.4f}")

with open(f"{RESULTS_DIR}/honest_eval_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ── Stage 6: Hugging Face pretrained model comparison ───────────────────────
log("Stage 6: Hugging Face pretrained scVI model (query adaptation)...")
try:
    from scvi.hub import HubModel

    hub_model = HubModel.pull_from_huggingface_hub(
        repo_name="scvi-tools/heart-cell-atlas-scvi",
        cache_dir="/workspace/hf_cache",
    )
    log("Downloaded scvi-tools/heart-cell-atlas-scvi from Hugging Face.")

    query_model = scvi.model.SCVI.load_query_data(adata_hvg, hub_model.model)
    query_model.train(max_epochs=20, plan_kwargs={"weight_decay": 0.0})
    latent_hf = query_model.get_latent_representation()
    np.save(f"{RESULTS_DIR}/latent_hf_pretrained.npy", latent_hf)
    log(f"HF latent shape: {latent_hf.shape}")

    hf_results = {}
    for clf_name, clf in [
        ("RF", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=42, n_jobs=-1)),
        ("XGB", xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale,
                                   random_state=42, eval_metric="logloss", verbosity=0)),
    ]:
        aucs = []
        for fold, (tr, val) in enumerate(StratifiedGroupKFold(n_splits=5).split(latent_hf, y, groups)):
            clf.fit(latent_hf[tr], y[tr])
            auc = roc_auc_score(y[val], clf.predict_proba(latent_hf[val])[:, 1])
            aucs.append(auc)
        key = f"hf_pretrained_{clf_name}"
        hf_results[key] = aucs
        log(f"{key}: folds={['%.4f' % a for a in aucs]}  mean={np.mean(aucs):.4f}")

    with open(f"{RESULTS_DIR}/hf_eval_results.json", "w") as f:
        json.dump(hf_results, f, indent=2)

except Exception as e:
    log(f"Hugging Face comparison FAILED (non-fatal, continuing): {e}")

# ── Stage 7: full pseudobulk DE across ALL 32,383 genes ─────────────────────
log("Stage 7: pseudobulk DE across all 32,383 genes (no gene pre-filtering)...")
from scipy import stats
from scipy.stats import rankdata

donors = adata.obs["donor_id"].unique()
rows = []
X_full = adata.X
if not sp.issparse(X_full):
    X_full = sp.csr_matrix(X_full)

pseudo_rows = []
pseudo_meta = []
for donor in donors:
    m = (adata.obs["donor_id"] == donor).values
    X_d = X_full[m]
    disease = adata.obs.loc[m, "disease"].iloc[0]
    sex_cols = [c for c in adata.obs.columns if "sex" in c.lower()]
    sex = adata.obs.loc[m, sex_cols[0]].iloc[0] if sex_cols else "unknown"
    pseudo_rows.append(np.asarray(X_d.mean(axis=0)).ravel())
    pseudo_meta.append({"donor": donor, "label": "DCM" if disease == DCM else "ACM", "sex": sex})

pseudo_X = np.vstack(pseudo_rows)
pseudo_df = pd.DataFrame(pseudo_meta)
log(f"Pseudobulk matrix: {pseudo_X.shape[0]} patients x {pseudo_X.shape[1]} genes")

dcm_m = (pseudo_df["label"] == "DCM").values
acm_m = (pseudo_df["label"] == "ACM").values

t0 = time.time()
pvals = np.array([
    stats.ranksums(pseudo_X[dcm_m, g], pseudo_X[acm_m, g])[1]
    for g in range(pseudo_X.shape[1])
])
log(f"Wilcoxon across all genes done in {time.time()-t0:.0f}s")

n = len(pvals)
fdr = np.minimum(pvals * n / rankdata(pvals), 1.0)
gene_names = (adata.var["feature_name"].values
              if "feature_name" in adata.var.columns else adata.var_names.values)

de_results = pd.DataFrame({
    "gene": gene_names, "pval": pvals, "fdr": fdr,
    "mean_dcm": pseudo_X[dcm_m].mean(0),
    "mean_acm": pseudo_X[acm_m].mean(0),
}).sort_values("pval")
de_results["log2fc"] = (np.log2(de_results["mean_acm"] + 1e-6) -
                        np.log2(de_results["mean_dcm"] + 1e-6))
de_results.to_csv(f"{RESULTS_DIR}/pseudobulk_de_full.csv", index=False)
pseudo_df.to_csv(f"{RESULTS_DIR}/pseudobulk_obs_full.csv", index=False)

log(f"FDR < 0.05: {(de_results['fdr']<0.05).sum()} genes")
log(f"FDR < 0.10: {(de_results['fdr']<0.10).sum()} genes")
log("Top 15 genes:")
log(de_results.head(15)[["gene", "pval", "fdr", "log2fc"]].to_string())

# ── Stage 8: cell-substate stratified breakdown (only possible with real RAM) ──
# NOTE: this atlas is pre-filtered to cardiomyocytes only, so the generic
# 'cell_type' column is uninformative (every cell is "cardiac muscle cell").
# 'cell_states' has real signal instead: 9 ventricular cardiomyocyte substates
# (vCM1.0 .. vCM5) with meaningfully different group sizes. Checked this
# directly against the h5ad before launching — confirmed on 2026-07-05.
log("Stage 8: checking for cell-substate annotation to stratify results...")
type_col = "cell_states" if "cell_states" in adata.obs.columns else None
if type_col is None:
    for col in adata.obs.columns:
        if "cell_type" in col.lower() or "cluster" in col.lower() or "annotation" in col.lower():
            type_col = col
            break

if type_col is not None:
    log(f"Found cell-type column: '{type_col}'. Stratifying honest AUC + pseudobulk by type.")
    cell_types = adata.obs[type_col].value_counts()
    cell_types.to_csv(f"{RESULTS_DIR}/cell_type_counts.csv")
    log(cell_types.to_string())

    stratified_results = {}
    for ctype in cell_types.index:
        ctype_mask = (adata_hvg.obs[type_col] == ctype).values
        n_cells_ctype = ctype_mask.sum()
        n_donors_ctype = adata_hvg.obs.loc[ctype_mask, "donor_id"].nunique()
        n_acm_ctype = (adata_hvg.obs.loc[ctype_mask, "disease"] == ACM).sum()
        n_dcm_ctype = (adata_hvg.obs.loc[ctype_mask, "disease"] == DCM).sum()

        # Skip cell types with too few cells/donors for a stable 5-fold CV
        if n_cells_ctype < 500 or n_acm_ctype < 20 or n_dcm_ctype < 20:
            log(f"  Skipping '{ctype}': too few cells for stable CV "
                f"(n={n_cells_ctype}, ACM={n_acm_ctype}, DCM={n_dcm_ctype})")
            continue

        X_ctype = latent_own[ctype_mask]
        y_ctype = y[ctype_mask]
        groups_ctype = groups[ctype_mask]

        try:
            clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          random_state=42, n_jobs=-1)
            aucs = []
            for tr, val in StratifiedGroupKFold(n_splits=3).split(X_ctype, y_ctype, groups_ctype):
                clf.fit(X_ctype[tr], y_ctype[tr])
                aucs.append(roc_auc_score(y_ctype[val], clf.predict_proba(X_ctype[val])[:, 1]))
            stratified_results[ctype] = {
                "n_cells": int(n_cells_ctype), "n_donors": int(n_donors_ctype),
                "mean_auc": float(np.mean(aucs)), "fold_aucs": [float(a) for a in aucs],
            }
            log(f"  {ctype}: n={n_cells_ctype}, donors={n_donors_ctype}, "
                f"AUC={np.mean(aucs):.4f}")
        except Exception as e:
            log(f"  '{ctype}' failed: {e}")

    with open(f"{RESULTS_DIR}/cell_type_stratified_auc.json", "w") as f:
        json.dump(stratified_results, f, indent=2)
else:
    log("No cell-type annotation column found — skipping stratified breakdown.")

save_log()
log("ALL STAGES COMPLETE. Results in /workspace/results/")
