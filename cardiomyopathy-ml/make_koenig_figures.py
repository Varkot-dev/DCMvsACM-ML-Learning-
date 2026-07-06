"""
make_koenig_figures.py — figures L through O for the 2026-07-06 journal entry,
covering the Koenig/Lavine replication, classifier, confound checks, and
permutation test.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle

figstyle.apply()

RESULTS_DIR = "../results"
OUT_DIR = "../journal/2026-07-06"


def fig_l_replication_sweep():
    df = pd.read_csv(f"{RESULTS_DIR}/koenig_vcm3_replication.csv")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(df["percentile"], df["dcm_mean_stress"], marker="o", color=figstyle.DCM,
            linewidth=2, label="DCM donors")
    ax.plot(df["percentile"], df["donor_mean_stress"], marker="o", color=figstyle.ACCENT_2,
            linewidth=2, label="Non-diseased donors")

    for _, row in df.iterrows():
        if row["pval"] < 0.05:
            ax.annotate("*", (row["percentile"], row["dcm_mean_stress"] + 0.05),
                        ha="center", fontsize=13, color=figstyle.ACCENT)

    ax.set_xlabel("Cardiomyocyte-score percentile threshold (stricter, left to right)")
    ax.set_ylabel("Mean stress-program score")
    ax.legend(loc="upper left")
    figstyle.clean(ax)
    figstyle.title(ax, "The vCM3.0 stress program replicates in an independent cohort",
                    "Koenig/Lavine 2022 (GSE183852) — DCM vs non-diseased donor, six CM-score thresholds")
    figstyle.figure_caption(fig, "* = p<0.05 (Wilcoxon, donor-level). DCM scores higher at every threshold; "
                                  "gap widens as cell selection gets stricter. Asterisks mark 60th-95th percentile.")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"{OUT_DIR}/figL_koenig_replication_sweep.png")
    plt.close(fig)


def fig_m_confound_checks():
    orig = pd.read_csv(f"{RESULTS_DIR}/koenig_classifier_auc.csv")
    tech = pd.read_csv(f"{RESULTS_DIR}/koenig_confound_check_auc.csv")
    ckd = pd.read_csv(f"{RESULTS_DIR}/koenig_ckd_confound_check_auc.csv")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(orig["percentile"], orig["mean_auc"], marker="o", color=figstyle.MUTED,
            linewidth=2, label="Original (all samples)", linestyle="--")
    ax.plot(tech["percentile"], tech["mean_auc"], marker="o", color=figstyle.ACCENT_2,
            linewidth=2, label="Technology-controlled (Single Nuclei only)")
    ax.plot(ckd["percentile"], ckd["mean_auc"], marker="o", color=figstyle.ACCENT,
            linewidth=2, label="CKD-controlled (CKD-negative only)")

    ax.axhline(0.5, color=figstyle.MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Cardiomyocyte-score percentile threshold")
    ax.set_ylabel("Honest patient-level AUC")
    ax.legend(loc="lower right")
    figstyle.clean(ax)
    figstyle.title(ax, "Classification signal survives two confound checks",
                    "DCM vs non-diseased donor, Koenig/Lavine cohort")
    figstyle.figure_caption(fig, "Restricting to Single Nuclei only (removes a 4x technology imbalance) barely "
                                  "moves the AUC. Restricting to CKD-negative patients (removes a ~10x clinical "
                                  "imbalance) drops it somewhat, but it stays far above chance (dotted line).")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(f"{OUT_DIR}/figM_koenig_confound_checks.png")
    plt.close(fig)


def fig_n_permutation_test():
    perm = pd.read_csv(f"{RESULTS_DIR}/koenig_permutation_test.csv")
    with open(f"{RESULTS_DIR}/koenig_permutation_test_summary.txt") as f:
        summary = dict(line.strip().split("=") for line in f if line.strip())
    real_auc = float(summary["real_auc"])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(perm["shuffled_auc"], bins=10, color=figstyle.MUTED, edgecolor=figstyle.PAPER,
            linewidth=1.2, alpha=0.85, label="Shuffled-label AUCs (n=20)")
    ax.axvline(real_auc, color=figstyle.ACCENT, linewidth=2.5, label=f"Real AUC = {real_auc:.3f}")
    ax.axvline(0.5, color=figstyle.INK, linewidth=0.8, linestyle=":", label="Chance (0.5)")

    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")
    ax.set_xlim(0.3, 1.0)
    ax.legend(loc="upper left")
    figstyle.clean(ax)
    figstyle.title(ax, "The classifier is not a small-sample artifact",
                    "20 donor-level label shuffles vs the real result, 80th-percentile threshold")
    figstyle.figure_caption(fig, "Null distribution centers on chance (mean 0.496) as expected with no real "
                                  "signal. The real AUC sits far outside even the widest random shuffle (max 0.661).")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(f"{OUT_DIR}/figN_permutation_test.png")
    plt.close(fig)


def fig_o_domain_shift():
    labels = ["Reichart\n(trained + tested\non Reichart)", "Koenig\n(trained + tested\non Koenig)",
              "Reichart to Koenig\n(trained on Reichart,\ntested on Koenig)"]
    aucs = [0.9668, 0.9413, 0.6071]
    colors = [figstyle.DCM, figstyle.ACCENT_2, figstyle.ACCENT]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    bars = ax.bar(labels, aucs, color=colors, width=0.55)
    ax.axhline(0.5, color=figstyle.MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC")
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.02, f"{auc:.2f}",
                ha="center", fontsize=11, fontweight="bold", color=figstyle.INK)
    figstyle.clean(ax)
    figstyle.title(ax, "Domain shift: same model, different cohort, collapsed performance",
                    "DCM vs healthy/donor classification, honest patient-level AUC")
    figstyle.figure_caption(fig, "The Reichart-trained model stays near-perfect at home (0.97) but nearly "
                                  "collapses to chance (0.61) on a cohort it never saw during training — "
                                  "despite the disease biology being identical.")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(f"{OUT_DIR}/figO_domain_shift_collapse.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_l_replication_sweep()
    fig_m_confound_checks()
    fig_n_permutation_test()
    fig_o_domain_shift()
    print("Wrote figL through figO to", OUT_DIR)
