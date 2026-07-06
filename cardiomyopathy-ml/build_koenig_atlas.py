"""
build_koenig_atlas.py — Assemble GSE183852 (Koenig/Lavine 2022, "Cellular Atlas of
Human Heart Failure") into an AnnData matching the data.py contract.

This is the Phase 2 cross-cohort dataset. The spec assumed the Chaffin 2022 DCM/HCM
atlas would be available via CELLxGENE; it is not present there. GSE183852 was
substituted (see journal): it is DCM vs non-diseased donor, not DCM vs HCM, so Phase
2 becomes a replication test of the vCM3.0 stressed-state signature in a DCM-vs-donor
contrast, not a DCM-vs-HCM one.

Source layout (unusual, no h5ad or per-sample files exist for this series):
  - GSE183852_Integrated_Counts.csv.gz: one series-level file, genes as rows, cells
    as columns, 45,069 genes x ~269,795 cells. Column names are
    "<sample>_<16bp-barcode>-1". Sample names are the raw library identifiers, not
    GEO's per-GSM titles, so cross-referencing was needed (see SAMPLE_TITLE_MAP).
  - No obs metadata ships with the counts file. Disease state and sample identity
    were scraped from each of the 45 GSM pages individually and joined in here.

Output: data/chaffin_koenig/koenig_atlas.h5ad with obs columns 'disease' and
'donor_id' matching data.py's DCM_LABEL / ACM_LABEL-style contract (adapted to
DCM / non-diseased donor labels for this cohort), and raw integer counts in .X
(no existing normalization to preserve, unlike the Reichart atlas).
"""

import csv
import logging
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "chaffin_koenig"
# Decompressed once with `gunzip -k`: pandas' row-oriented CSV parser was ~3hr on
# this file's 45,069-row x 270,000-column shape, and polars' read_csv_batched spent
# minutes just constructing its reader (gathering stats/chunk offsets) before
# yielding a single batch, regardless of compression. Plain line-by-line parsing
# on the decompressed file is linear and predictable: ~10ms/row, ~7.5 min total.
COUNTS_PATH = DATA_DIR / "GSE183852_Integrated_Counts.csv"
GSM_METADATA_PATH = DATA_DIR / "gsm_metadata.csv"
OUTPUT_PATH = DATA_DIR / "koenig_atlas.h5ad"

DCM_LABEL = "dilated cardiomyopathy"
DONOR_LABEL = "non-diseased donor"

# GEO's per-GSM "Sample_title" field does not always match the library identifier
# used as the column prefix in the counts CSV. Discovered by diffing the 45 counts-
# file sample prefixes against the 45 GSM titles; every mismatch was one of the
# "H" single-cell samples (GSM title "TWCM-H1" -> CSV column prefix "HDCM1", with
# "-libN" suffixes on the GSM side dropped in the CSV) or a trailing "-1" on
# "H_ZC-LVAD-1" (GSM) vs "H_ZC-LVAD" (CSV).
GSM_TITLE_TO_CSV_PREFIX = {
    "TWCM-H1": "HDCM1",
    "TWCM-H3": "HDCM3",
    "TWCM-H4": "HDCM4",
    "TWCM-H5": "HDCM5",
    "TWCM-H6-lib1": "HDCM6",
    "TWCM-H7-lib1": "HDCM7",
    "TWCM-H8-lib1": "HDCM8",
    "H_ZC-LVAD-1": "H_ZC-LVAD",
}


def load_sample_metadata() -> dict[str, dict]:
    """Map counts-CSV sample prefix -> {disease, tissue, technology}."""
    with open(GSM_METADATA_PATH) as f:
        gsm_rows = list(csv.DictReader(f))

    prefix_to_meta = {}
    for row in gsm_rows:
        title = row["title"]
        csv_prefix = GSM_TITLE_TO_CSV_PREFIX.get(title, title)
        prefix_to_meta[csv_prefix] = {
            "disease": DCM_LABEL if row["disease_state"] == "Dilated cardiomyopathy" else DONOR_LABEL,
            "tissue": row["tissue"],
            "technology": row["technology"],
        }
    return prefix_to_meta


def split_barcode(column_name: str) -> str:
    """Column names are '<sample>_<16bp-barcode>-1'. Sample names themselves can
    contain underscores (e.g. 'H_ZC-11-292'), so split on the LAST underscore."""
    return column_name.rsplit("_", 1)[0]


def main() -> None:
    prefix_to_meta = load_sample_metadata()
    logger.info(f"Loaded metadata for {len(prefix_to_meta)} samples")

    logger.info("Reading header to build cell -> sample/disease/donor mapping...")
    with open(COUNTS_PATH) as f:
        header_line = f.readline()
    header = header_line.rstrip("\n").split(",")
    cell_columns = header[1:]  # first column is the gene name
    sample_prefixes = [split_barcode(c) for c in cell_columns]

    unknown = set(sample_prefixes) - set(prefix_to_meta)
    if unknown:
        raise ValueError(f"Sample prefixes with no metadata match: {unknown}")

    disease_col = [prefix_to_meta[p]["disease"] for p in sample_prefixes]
    n_dcm = sum(d == DCM_LABEL for d in disease_col)
    n_donor = sum(d == DONOR_LABEL for d in disease_col)
    logger.info(f"{len(cell_columns)} cells: DCM={n_dcm}, non-diseased donor={n_donor}, "
                f"{len(set(sample_prefixes))} unique donors")

    logger.info("Reading counts matrix line by line (plain Python, no pandas/polars "
                "overhead), building sparse matrix in row-blocks of 500 genes at a "
                "time (a dense read would need ~48GB RAM for this shape).")
    t_start = time.time()
    gene_names = []
    row_blocks = []  # each block: sparse (block_genes x n_cells)
    n_genes_total = 45069
    block_size = 500
    block_rows = []

    with open(COUNTS_PATH) as f:
        f.readline()  # skip header, already parsed above
        for i, line in enumerate(f):
            fields = line.rstrip("\n").split(",")
            gene_names.append(fields[0])
            block_rows.append(np.array(fields[1:], dtype=np.float32))

            if len(block_rows) == block_size:
                row_blocks.append(sp.csr_matrix(np.vstack(block_rows)))
                block_rows = []
                n_done = i + 1
                elapsed = time.time() - t_start
                rate = n_done / elapsed
                eta_s = (n_genes_total - n_done) / rate if rate > 0 else float("nan")
                logger.info(f"  {n_done}/{n_genes_total} genes, "
                            f"{elapsed:.0f}s elapsed, ETA {eta_s:.0f}s")

    if block_rows:
        row_blocks.append(sp.csr_matrix(np.vstack(block_rows)))

    gene_names = np.array(gene_names)
    X_genes_by_cells = sp.vstack(row_blocks, format="csr")
    logger.info(f"Loaded {X_genes_by_cells.shape[0]} genes x {X_genes_by_cells.shape[1]} cells")
    X = X_genes_by_cells.T.tocsr()

    obs = pd.DataFrame({
        "donor_id": sample_prefixes,
        "disease": disease_col,
        "tissue": [prefix_to_meta[p]["tissue"] for p in sample_prefixes],
        "technology": [prefix_to_meta[p]["technology"] for p in sample_prefixes],
    }, index=cell_columns)

    var = pd.DataFrame(index=gene_names)
    var["feature_name"] = gene_names

    adata = ad.AnnData(X=X, obs=obs, var=var)
    logger.info(f"Assembled AnnData: {adata.n_obs} cells x {adata.n_vars} genes, "
                f"{obs['donor_id'].nunique()} donors")

    adata.write_h5ad(OUTPUT_PATH)
    logger.info(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
