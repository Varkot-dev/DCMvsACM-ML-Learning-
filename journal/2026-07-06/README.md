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
