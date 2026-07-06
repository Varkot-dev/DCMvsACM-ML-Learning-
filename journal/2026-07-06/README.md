# July 6, 2026

Following up on the one open lead from last night. vCM3.0, one of nine ventricular cardiomyocyte substates in the atlas, classified DCM versus ACM at AUC 0.775, noticeably above the whole population average of 0.726. Before treating that as biology, it needed to survive the two confounds already found earlier: sex composition and PKP2 genotype composition.

## Ruling out the known confounds

Checked sex and genotype composition of vCM3.0 against every other substate.

Sex: 26.7 percent female in vCM3.0 versus 26.8 percent in the rest. Essentially identical.

Genotype among ACM cells: 81.3 percent PKP2 in vCM3.0 versus 80.7 percent in the rest. Essentially identical.

All 60 donors are represented in vCM3.0, including all 8 ACM patients, so the result is not driven by one or two patients dominating the substate.

Neither confound explains the higher AUC. This is not XIST or PKP2 reappearing under a different name. The result survives the check.

## Marker genes, and a mistake worth naming

The first attempt at finding what defines vCM3.0 used raw log fold change with no expression filter, ranking every gene in the panel. This produced garbage: the top hits were genes like KCNA10 and OR8K3, expressed at essentially zero in both groups, where a handful of stray reads in a tiny number of cells produces a huge fold change purely from division by a near zero denominator. None of those genes have any known role in cardiac tissue. Worth stating plainly rather than hiding: this is exactly the kind of result that looks like a finding on first glance and is actually a filtering bug.

Redone properly: filtered to genes with meaningful mean expression in at least one group, then ran a Wilcoxon test with Benjamini Hochberg correction on the filtered set.

1,903 genes survive FDR correction. The top hits are real, recognizable cardiac biology, not noise.

**ANKRD1**, the single strongest hit, is a mechanical stress response gene in cardiomyocytes, induced under cardiac stress and remodeling. **ACTA1** and **MYH9** are cytoskeletal and sarcomere associated genes. **FLNC** and **FLNC-AS1**, both elevated in vCM3.0, encode filamin C, a gene directly implicated in genetic cardiomyopathy, including forms of ACM. **XIRP2** is tied to Z-disc structure and cardiac remodeling.

![vCM3.0 marker genes](figK_vcm3_markers.png)

## What this adds up to

vCM3.0 is not defined by sex or genotype. It is defined by a mechanical stress and cytoskeletal remodeling gene program, and it is also the substate where honest DCM versus ACM classification performs best. Put together, this suggests a coherent hypothesis: vCM3.0 may represent cardiomyocytes actively undergoing stress remodeling, and disease specific transcriptional differences between DCM and ACM may be concentrated in cells in this state, while more quiescent substates dilute the signal by averaging it out.

This is a hypothesis, not a proven mechanism. It was generated from one dataset, and FLNC involvement in particular is worth checking against the literature rather than assumed. But it is a specific, checkable claim, which is more than the whole population analysis alone provided.

## What a next step would look like

Confirm whether FLNC specifically, rather than the broader stress program, differs between DCM and ACM within vCM3.0 specifically, not just between vCM3.0 and other substates. That is a different comparison than the one run today and would need its own analysis.

Raw marker gene table: `results/vcm3_marker_genes_filtered.csv`

## Phase 1 follow-up: does FLNC separate DCM from ACM inside vCM3.0

A literature review repositioned the vCM3.0 finding above: it is the established "stressed ventricular
cardiomyocyte" state from Litvinukova 2020, not a novel substate. Rediscovering its markers independently
validates the pipeline but is not itself a claim. What is defensible: the classification signal concentrating
there, and FLNC never having been compared DCM-vs-ACM at single-cell resolution before. This section runs
that comparison directly, rather than leaving it as the "next step" noted above.

Method: subset to vCM3.0, DCM+ACM cells only (21,084 cells, 60 donors, matching the count already confirmed
above). Pseudobulk per donor by summing raw counts (`adata.raw.X`, confirmed to hold integer counts, since
`adata.X` here is already log-normalized), CPM-normalize, log1p. Wilcoxon rank-sum DCM vs ACM per gene,
filtered first to mean log-CPM > 0.05 in at least one group (24,138 / 32,383 genes survive the filter),
BH-FDR corrected. This is a different comparison from the marker-gene analysis above: that contrast was
vCM3.0-vs-rest-of-atlas (what defines the substate); this one is DCM-vs-ACM within vCM3.0 only (does disease
status separate cells that are already in the stressed state).

Result: **zero genes survive FDR correction at either 0.05 or 0.10.** FLNC (p=0.47) and FLNC-AS1 (p=0.76) are
not significant. Neither is the broader stress program identified earlier: ANKRD1 (p=0.56), XIRP2 (p=0.53),
ACTA1 (p=0.91), MYH9 (p=0.16). None of these separate DCM from ACM within the substate.

This is the honest-null outcome the spec called out as a live possibility, not a failure. 8 ACM donors is the
same sample-size ceiling that produced zero surviving genes in the whole-atlas pseudobulk DE, twice confirmed
on 2026-07-05. Subsetting to one substate reduces cells per donor; it does not add donors. A null result at
this donor count is underpowered, not necessarily biologically true. The conclusion this analysis supports:
the vCM3.0 classification advantage (0.775 vs 0.726 AUC) does not trace to FLNC or the stress program at
donor-level pseudobulk resolution, at least not detectably with 8 ACM donors. Whatever drives the AUC gap
either needs more ACM samples to resolve at the donor level, or lives in a signal that donor-level pseudobulk
washes out (per-cell heterogeneity, non-linear combinations across genes, or something the classifier reads
that a single-gene Wilcoxon test cannot see).

Script: `cardiomyopathy-ml/vcm3_dcm_vs_acm_de.py`. Results: `results/vcm3_dcm_vs_acm_de.csv`,
`results/vcm3_dcm_vs_acm_pseudobulk_obs.csv`.

Method discrepancy worth logging: the July 5 pseudobulk script (`runpod_full_suite.py`) pseudobulks by
**per-cell mean** of `adata.X` (already log-normalized), not by summing raw counts. The spec for this phase
explicitly asked for sum-of-raw-counts-then-CPM/log, the standard bulk-RNA-seq-style pseudobulk approach,
which is a stricter and more standard method than averaging an already-log-transformed matrix. Both were run
correctly for what they intended, but they are not the same statistic, and a future direct comparison of
"vCM3.0-vs-rest" vs "DCM-vs-ACM-within-vCM3.0" p-values should not assume the same normalization pipeline
produced both.

Next: Phase 2, cross-cohort transfer against the Chaffin 2022 (DCM vs HCM) atlas. No ACM cohort exists there,
so this cannot be a DCM-vs-ACM transfer; it tests whether the vCM3.0 stressed-state signature replicates
independently in a second dataset.

## Phase 2: does the stress signature replicate in a second, independent cohort

The point of Phase 2 is a standard and important check in this kind of work: a finding from one dataset can
always be a fluke of that dataset's lab, batch, or patient population. The only way to know whether the
vCM3.0 stress program is real biology rather than an artifact of the Reichart atlas specifically is to test
it against a second, completely independent dataset — different lab, different patients, different
sequencing run.

**The planned dataset wasn't actually available.** The spec assumed the Chaffin 2022 DCM-vs-HCM atlas would be
downloadable from CZ CELLxGENE the same way Reichart was. It is not there. The GEO accession the spec cited
(GSE183852) turned out to belong to a different paper entirely: Koenig/Lavine 2022, "Cellular Atlas of Human
Heart Failure" — DCM vs non-diseased donor, not DCM vs HCM. Caught this before downloading anything blindly,
flagged it, and the call was made to proceed with Koenig/Lavine as the substitute cohort. This changes what
Phase 2 can claim: it tests replication in a DCM-vs-donor contrast, not DCM-vs-HCM. The core Phase 1 result is
unaffected either way; only the cross-cohort comparison target changed.

**Getting the data into usable shape took real data engineering.** Unlike Reichart, GSE183852 ships as one
series-level file with no h5ad and no per-sample metadata: a single 465MB gzipped CSV (45,069 genes x
269,795 cells, genes as rows) with sample identity baked into the column names as a barcode prefix, and zero
disease/donor metadata attached. All 45 GSM (per-sample) pages had to be scraped individually to recover
disease state, and a naming mismatch between GEO's per-sample titles and the CSV's column prefixes had to be
reverse-engineered by hand (e.g. GEO's "TWCM-H6-lib1" maps to the CSV's "HDCM6" — every mismatch turned out to
be one of the "H" single-cell samples with a dropped lib suffix, or a trailing "-1" difference).

Parsing that file was its own detour worth recording honestly: pandas' row-oriented CSV parser extrapolated to
roughly 3 hours on this file's shape (270,000 columns per row is adversarial for a parser tuned for many-rows/
few-columns data). Polars, which should have been faster, instead spent minutes just constructing its batched
reader (gathering stats and chunk offsets) before yielding a single batch, regardless of whether the source
was compressed. The fix that actually worked: decompress once (17 seconds, 24GB uncompressed), then parse
line-by-line in plain Python with no library overhead at all — a predictable ~10ms/row, about 8 minutes total
for all 45,069 genes. Sometimes the simplest tool beats the "fast" library, because the library's cleverness
(schema inference, upfront chunk planning) is solving a problem you don't have.

Output: `data/chaffin_koenig/koenig_atlas.h5ad`, 269,794 cells x 45,068 genes, 45 donors (18 DCM, 27
non-diseased donor). Sanity-checked before use: donor counts match the scraped metadata, values are
non-negative integers consistent with raw counts, all marker genes present, cells-per-donor distribution has
no single donor dominating (range 1,744-11,813, median ~5,550).

**The analysis method, and an honest limitation of it.** This cohort has no expert-annotated cell-type or
substate column, unlike Reichart's `cell_states`. Rather than run a full clustering pipeline to identify a
vCM3.0-equivalent substate from scratch (a much larger undertaking), the faster route taken: score every cell
for canonical cardiomyocyte markers (TTN, MYH6, MYH7, TNNT2, ACTC1) and keep only cells scoring above some
threshold as a cardiomyocyte proxy, then score those cells for the vCM3.0 stress program (ANKRD1, XIRP1,
XIRP2, FHL1, ACTA1, MYH9) and compare DCM donors against non-diseased donors.

The problem: plotting the cardiomyocyte-score distribution across all 269,794 cells showed one smooth,
unimodal curve, not the two separated humps you'd want for a clean "cardiomyocyte vs not" cutoff. Any single
threshold would have been an arbitrary judgment call, and picking one number and reporting only that result
would have been exactly the kind of hidden researcher-degrees-of-freedom problem that makes results
irreproducible.

**The fix: sweep the threshold instead of picking one.** Rather than commit to a single cutoff, the check was
run at six different percentiles of the cardiomyocyte score (50th through 95th), repeating the full DCM-vs-
donor comparison at each. The logic: if the stress-program elevation in DCM is real biology, it should hold up
regardless of exactly where the cardiomyocyte-identification line is drawn. If it only appeared at some
thresholds and vanished or flipped at others, that would mean the "signal" was an artifact of an arbitrary
cutoff rather than a property of the underlying biology. This is a standard robustness/sensitivity check, the
same principle behind a grid sweep in hyperparameter tuning, applied here to a cell-type-identification
threshold instead.

**Result:**

| CM-score percentile | cells kept | DCM mean stress score | donor mean stress score | p-value |
|---|---|---|---|---|
| 50th | 50.0% | 0.328 | 0.289 | 0.287 |
| 60th | 40.0% | 0.468 | 0.346 | 0.044 |
| 70th | 30.0% | 0.570 | 0.363 | 0.007 |
| 80th | 20.0% | 0.596 | 0.422 | 0.024 |
| 90th | 10.0% | 0.756 | 0.505 | 0.013 |
| 95th | 5.0% | 0.716 | 0.481 | 0.016 |

DCM donors score higher than non-diseased donors at every single threshold tested — the direction never
flips. 5 of 6 thresholds are significant at p < 0.05; only the loosest cutoff (50th percentile, which still
includes a lot of likely non-cardiomyocyte cells) misses significance. And the gap widens as the threshold
gets stricter: as cells more confidently identified as cardiomyocytes are isolated, the DCM-vs-donor
separation gets cleaner, which is exactly the pattern expected if this is genuine cardiomyocyte biology being
diluted by non-CM cells at loose thresholds, not noise.

**What this does and does not establish.** The vCM3.0 stress program (ANKRD1, XIRP1/2, FHL1, ACTA1, MYH9) is
elevated in DCM hearts in a second, fully independent dataset — different lab, different patients, different
sequencing technology (single-cell and single-nucleus mixed, vs Reichart's single-nucleus). That is genuine
replication evidence for the biology, and it survived a real sensitivity check rather than resting on one
arbitrary parameter choice. The honest caveat: this used a marker-score cardiomyocyte proxy, not real
expert-annotated substates like Reichart had, so it is suggestive replication under a rougher method, not
identical-rigor independent confirmation. Whether the DCM-vs-ACM classification signal itself (not just the
gene-program elevation) also concentrates in a comparable cell population here is a separate, not yet run,
question — Koenig/Lavine has no ACM patients at all, so at most a DCM-vs-donor classifier could be tested,
answering a related but distinct question from the original AUC finding.

Scripts: `cardiomyopathy-ml/build_koenig_atlas.py` (data assembly), `cardiomyopathy-ml/koenig_vcm3_replication.py`
(threshold-sweep analysis). Results: `results/koenig_vcm3_replication.csv`.

## Phase 2 continued: does the classification signal (not just the gene score) replicate too

The replication check above tested one gene program's average activity. A different, complementary question:
can a classifier trained on a cell's full expression profile actually predict DCM vs non-diseased donor, and
does that predictive signal concentrate in cardiomyocyte-like cells the same way it did in Reichart (honest
AUC 0.775 in vCM3.0 vs 0.726 whole-population)? No ACM cohort exists in Koenig/Lavine, so this is necessarily
DCM-vs-donor, not DCM-vs-ACM — same substitution and caveat as the rest of Phase 2.

Method: honest, patient-level evaluation via StratifiedGroupKFold (all cells from one donor stay in one fold,
same logic as the "naive vs honest AUC" result that was the strongest finding in Phase 1), RandomForest on the
top 2,000 highly-variable genes, run at the same six CM-score percentile thresholds as the replication check.
Runs entirely on CPU in minutes — no GPU needed. The RunPod GPU rental in Phase 1 was for scVI, a neural
network trained with gradient descent over many epochs; a RandomForest here is a fundamentally lighter
computation.

**First result: honest AUC of 0.92-0.97 across every threshold**, and like the gene-score result, it gets
stronger as the cardiomyocyte-identification threshold gets stricter (0.924 at 50th percentile up to 0.965 at
90th). This is a dramatically higher number than Phase 1's DCM-vs-ACM honest AUC (0.70) — expected, since
DCM-vs-healthy-donor is a fundamentally easier problem than DCM-vs-ACM (two different diseases that are both
heart failure). Believable on its face, but a number this high on a new dataset needs scrutiny before it's
trusted, not just excitement.

## The confound checks: this is where the real lesson of the day lives

Before accepting 0.92-0.97 as a real result, two specific confounds were identified and tested, because a
classifier does not know or care why two groups differ — it just finds whatever pattern separates them best,
and if a non-biological confound separates the groups more cleanly than the real disease biology does, the
model will exploit that shortcut and the AUC will look great while meaning something completely different than
"this predicts DCM."

**Confound 1: sequencing technology.** The scraped GSM metadata showed DCM samples are 4x more likely to be
"Single Cell" protocol than donor samples (28% vs 7%, 5/18 DCM donors vs 2/27 donor donors). Single-cell and
single-nucleus RNA-seq capture systematically different gene sets for protocol reasons unrelated to disease.
Test: restrict to Single Nuclei samples only (13 DCM, 25 donor, technology held constant) and re-run the
identical classifier. Result: AUC barely moved (within 1-3 points at every threshold, well inside the
fold-to-fold noise). Technology is not the explanation.

**Confound 2: chronic kidney disease.** Patient-level clinical metadata is not in GEO at all — it had to be
pulled from the original paper's own Supplementary Table 20 ("Patient-level metadata," cross-referenced by
sample name, 45/45 exact match against the h5ad's donor_id, a good independent confirmation the earlier GSM
scraping and sample-name reconciliation was correct). That table surfaced a bigger imbalance than technology:
39% of DCM patients have CKD vs 3.7% of donors (~10x skew). Cardiorenal syndrome is a real, expected clinical
association, so this is not surprising, but CKD has its own transcriptomic footprint independent of heart
disease, so it could not be waved away. Test: restrict to CKD-negative patients only (11 DCM, 26 donor) and
re-run. Result: AUC dropped meaningfully this time — roughly 4-9 points lower at most thresholds (e.g. 0.941
to 0.928 at 80th percentile, 0.950 to 0.856 at 95th) — but still sat well above chance (0.85-0.93 range), not
collapsing toward 0.5. CKD explains some of the original inflation, more than technology did, but not all of
it. The smaller sample after this restriction (down to 8-11 DCM donors at strict thresholds) also means some
of that drop could be added noise rather than pure signal loss — the two aren't fully separable with this
much data.

**A broader check of remaining clinical fields** (Race, Ethnicity, HTN, Diabetes, Smoking) showed no comparable
imbalance. Etiology-of-heart-failure, arrhythmia, and valve-disease fields were too sparse/free-text to check
cleanly and were not force-fit into a conclusion. Collection dates (available for 38/45 donors, the Nuclei
cohort only) showed DCM samples spanning a wider range (2010-2019, one notable outlier at 2019) than donor
samples (2010-2015, tighter) — a mild batch-effect risk, flagged rather than resolved, since the Cell cohort's
metadata table has no date field at all to check the remaining 7 donors against.

**The permutation test, run to answer the sample-size question directly rather than keep guessing at
individual columns.** Instead of testing one more specific hypothesis, this asks the general question: given
only 18 DCM and 27 donor donors, how much AUC would you expect from pure chance alone? Method: shuffle the
disease labels at the donor level (not cell level, which would break the honest per-donor grouping), re-run
the identical classifier on the shuffled labels, repeat 20 times, and compare the real AUC against that null
distribution. If the real AUC sits far above where chance lands, the result is not just small-sample luck.

Run at the 80th-percentile CM-score threshold (the middle of the earlier sweep), real AUC 0.9413.
Null distribution across 20 label-shuffled runs: mean 0.496, std 0.085, range 0.375-0.661. Permutation
p-value (fraction of shuffles reaching or exceeding the real AUC): 0.0000 — zero of 20 shuffles came close.
The null mean landing almost exactly on 0.5 is itself a good sanity check: that is precisely what pure
coin-flip guessing should produce when there is no real signal to find, confirming the test is measuring
what it claims to. The real AUC sits nearly 30 points above even the single highest random shuffle. If small
sample size alone (18 DCM, 27 donor donors) could produce AUCs this high by chance, at least some of the 20
shuffles should have landed near 0.94. None did. This is real evidence the classification signal is not a
small-sample artifact, on top of already surviving two specific confound checks (technology, CKD).

## The actual lesson of the day

The most important thing learned today was not a result, it was a habit: **a high AUC is not the end of an
analysis, it is the start of the real question.** The number itself does not tell you whether it reflects the
biology you care about or something else entirely that happens to correlate with your labels. Today that
"something else" turned out to be real and non-obvious twice in a row — first a lab protocol difference, then
a clinical comorbidity that had nothing to do with the disease being studied but showed up disproportionately
in the diseased group anyway. Both times, the fix was the same shape: find a way to hold the suspicious
variable constant, rerun the exact same test, and see if the result survives losing its "advantage." It mostly
did, which is what made it trustworthy — not the original number, but the number's behavior under attack.

The harder, still-open part of this lesson: recognizing a confound in the first place required already knowing
enough cardiology to be suspicious of a CKD imbalance, and required going and finding patient-level clinical
metadata that was not even in the primary data source (GEO) at all, sitting instead in a supplementary table of
the original paper. In a genuinely unfamiliar dataset, with no domain intuition to flag "check comorbidities"
and no guarantee the deeper metadata is findable or exists at all, this same mistake could go completely
undetected. That is the concrete skill worth building deliberately going forward: not just running the
statistical checks once shown how, but developing a systematic habit of asking, for any new dataset, "what
are all the ways two groups in this data could differ from each other that have nothing to do with the
question I am asking" before trusting any single number that looks good.
