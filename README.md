# ISOT–FakeNewsNet External-Validity and Dataset-Cue Audit

Version `1.0.0` — 2026-07-24

This public-release package accompanies the manuscript *Auditing Duplicate and
Source Cues in TF-IDF Fake News Classification: External Validation on
FakeNewsNet–PolitiFact and GossipCop*. It contains reproducible scripts,
protocol records, aggregate results, and text-free item-level predictions. It
does **not** redistribute the six original ISOT/FakeNewsNet CSV files.

All analyses predict the supplied binary corpus labels from normalized text.
They do not establish claim-level factual verification. The analyses added for
v1.0.0 were requested after review and are explicitly treated as post hoc
robustness evidence rather than preregistered confirmation.

## New v1.0.0 robustness analyses

### Title-level near-duplicate audit

- Exact Jaccard search used sets of boundary-padded hashed character 5-grams
  and reported thresholds 0.70, 0.80, and 0.90.
- At the principal 0.80 threshold, ISOT contained 56 verified title-title
  edges forming 47 multi-title components with 99 titles; no component mixed
  fake and real labels.
- Only one ISOT–PolitiFact and one ISOT–GossipCop title pair reached 0.80.
  Thus, the external findings cannot be explained by broad title-level
  duplication from the training corpus.
- Rebuilding five folds from the title-specific 0.80 components gave fixed-SVM
  outer macro-F1 94.09% and selected-SVM outer macro-F1 94.56%. The selected
  bigram SVM reached 66.93% balanced accuracy on PolitiFact and 50.98% on
  GossipCop, preserving the transfer conclusion.

### Source-clustered uncertainty

- Confidence intervals resampled normalized hostnames, moving all records from
  a sampled hostname together. The primary policy pooled missing hostnames;
  a secondary policy treated each missing-hostname item as its own cluster.
- For the fixed Linear SVM, source-clustered 95% balanced-accuracy intervals
  were 64.77–71.72% on PolitiFact and 50.21–53.05% on GossipCop under the
  primary policy.
- In the fold-based hostname diagnostic, the SVM remained near chance on
  GossipCop both for seen hostnames (51.69% balanced accuracy) and previously
  unseen hostnames (50.21%). The hostname-only rule fell to exactly 50.00% on
  unseen hostnames, confirming that its strong aggregate score is an
  in-corpus source-association diagnostic rather than portable validation.

### External-cleaning sensitivity

- The fixed SVM was re-evaluated on all nonempty external rows, on same-label
  exact-deduplicated rows with conflicting titles retained, and on the main
  unique/nonconflicting population.
- PolitiFact balanced accuracy ranged only from 67.94% to 68.28%.
- GossipCop balanced accuracy ranged only from 51.53% to 51.61%.
- The exclusions therefore do not drive the substantive external-validity
  conclusion.

The complete post-hoc specification is in
`protocol/additional_robustness_specification.json`.

## Earlier revision analyses

This package reproduces the external-validity analyses in Sections 3.12, 4.6, and 4.10 of the revised manuscript. It defines the estimand as prediction of the supplied binary corpus label from normalized title text—not claim-level factual verification—reports a held-out second external evaluation, and includes the post hoc source-hostname diagnostic requested during review.

Revision 10 additionally addresses the decision-threshold concern through the post-hoc analysis in Sections 3.13 and 4.11. The original no-retuning GossipCop result remains primary. The new analysis reports threshold-free ROC-AUC, a clearly labeled full-sample oracle diagnostic, and 200 repeated stratified target-adaptation/evaluation splits at 5%, 10%, and 20% adaptation fractions. Target labels used for threshold selection or Platt scaling never enter the paired evaluation subset.

Revision 11 adds the reviewer-motivated equal-budget tuning sensitivity in Sections 3.11 and 4.12. Multinomial NB, Logistic Regression, Linear SVM, and Random Forest each receive exactly 12 candidate configurations under the same inherited near-duplicate group folds and the same inner macro-F1 selection rule. Globally selected ISOT pipelines are then applied without external retuning to both PolitiFact and GossipCop. Because both external corpora had informed earlier analyses before this experiment, these comparisons are explicitly post hoc rather than confirmatory.

## Equal-budget tuning result

- Each model received four shared TF-IDF representations: word unigrams or unigrams+bigrams crossed with `min_df` 2 or 5.
- Each representation was crossed with three model-specific values: NB `alpha` {0.1, 1, 10}; LR/SVM `C` {0.1, 1, 10}; RF `max_depth` {unlimited, 50, 100}. Random Forest retained 200 trees.
- Nested outer-fold macro-F1 changed from 93.44% to 93.76% for NB, 93.62% to 94.64% for LR, 94.14% to 94.62% for SVM, and remained 92.50% for RF.
- On PolitiFact, selected-minus-fixed balanced-accuracy changes were −1.32 points for NB, +0.39 for LR, −1.35 for SVM, and zero for RF; all paired-bootstrap intervals for nonzero changes included zero.
- On GossipCop, the corresponding changes were −0.01 points for NB, +0.70 for LR, −0.63 for SVM, and zero for RF. Selected SVM balanced accuracy was 50.98%, below the fixed model's 51.61%; selected LR reached 50.67%.

Equalized tuning removes a strong tuning-invariant claim that one classifier is uniquely best, but it does not change the central external-validity conclusion: every selected pipeline remained near the 50% balanced-accuracy reference on GossipCop, and no model improved consistently across both external corpora.

## Global multiplicity sensitivity

Revision 12 addresses the multiple-testing concern with a conservative post-hoc audit of every unique exact paired McNemar contrast reported in the manuscript or associated Revision-11 results.

- Global family: 19 unique exact two-sided McNemar tests.
- Corrections: Holm family-wise error-rate control and Benjamini–Hochberg false-discovery-rate control, both at 0.05.
- Raw significant findings: 12 of 19.
- Significant after Holm: 11 of 19.
- Significant after Benjamini–Hochberg: 12 of 19.
- The only decision changed by Holm was the secondary GossipCop selected-versus-fixed Linear SVM accuracy comparison: raw p=0.02599, Holm-adjusted p=0.2079, and BH-adjusted p=0.0412.

All significant original source, temporal, and model diagnostics survived both corrections. None of the four PolitiFact tuning contrasts was significant before or after adjustment. Confidence intervals remain explicitly labeled as marginal effect-size uncertainty summaries; simultaneous 95% coverage is not claimed. Because the global family was defined after review, this is a sensitivity audit and not a confirmatory testing hierarchy.

## External source-brand and style-cue masking

Revision 13 addresses whether publisher names or recognizable reporting phrases embedded in external titles explain title-only transfer.

- A label-blind source-brand lexicon is derived from URL hostnames occurring in at least 10 cleaned records in each external corpus.
- Distinctive brand aliases are matched as complete normalized token sequences; ambiguous single words such as `people`, `time`, `today`, `medium`, `deadline`, and `variety` are excluded.
- A separate predeclared phrase set covers overt attribution and sensational presentation, including `according to`, `reportedly`, `breaking`, `exclusive`, `rumor`, and related forms.
- The locked SVM is evaluated after publisher masking, style masking, and combined masking without refitting. A symmetric combined-mask sensitivity refits the same SVM on masked ISOT titles.
- Paired changes use 2,000 class-stratified bootstrap resamples. No new null-hypothesis tests are added to the multiplicity family.

Explicit source-brand cues occurred in 2.67% of PolitiFact and 0.97% of GossipCop titles; only 0.92% and 0.24%, respectively, contained the title's own hostname brand. Publisher masking changed balanced accuracy by 0.00 points on PolitiFact and −0.05 points on GossipCop. Style phrases occurred in 9.23% and 6.13% of titles and were label-associated, but masking them reduced PolitiFact balanced accuracy from 68.28% to 66.54% and left GossipCop essentially unchanged at 51.58%. Explicit source names therefore do not explain the external scores, while visible style phrases carry some label signal without accounting for the broader transfer failure.

## Post-hoc threshold-sensitivity result

- Locked zero-threshold GossipCop balanced accuracy: 51.61%.
- Threshold-free ROC-AUC: 50.92% (95% class-stratified-bootstrap CI 49.97–51.84).
- Descriptive full-sample oracle balanced accuracy: 51.92%; this reuses all labels and is not a valid external-performance estimate.
- With 20% labeled target adaptation and 80% disjoint evaluation, mean balanced accuracy across 200 splits was 51.50% after threshold selection versus 51.60% for the locked threshold on the same subsets.
- Target-domain Platt scaling at the 20% adaptation fraction predicted every evaluation item as real and therefore matched the always-real reference: 76.86% raw accuracy and 50.00% balanced accuracy.

These results do not support a threshold-shift-only explanation for the near-chance transfer. They do not rule out richer supervised domain adaptation; they show that simple target-prior or threshold correction does not recover useful ranking.

## Evidential status and implemented design

No dated preregistration, OSF record, or repository commit predating the GossipCop predictions is available. The GossipCop analysis is therefore not presented as protocol-frozen, prospectively specified, or confirmatory. The specification in this package was documented retrospectively.

- Training data: 38,681 exact-deduplicated ISOT titles.
- Original external test: 975 cleaned FakeNewsNet–PolitiFact titles.
- Held-out second external evaluation: 20,587 cleaned FakeNewsNet–GossipCop titles.
- Main reported model: the original fixed title-only Linear SVM.
- Main reported metric: balanced accuracy, emphasized because 76.86% of cleaned GossipCop labels are real.
- Supporting metrics: accuracy, macro-F1, fake recall, and real recall.
- Reference: always predict the external majority class.
- Uncertainty: 2,000 class-stratified bootstrap resamples.
- Implemented workflow separation: no GossipCop feature, hyperparameter, model, or threshold selection.

The retrospectively documented specification is stored in `protocol/second_external_evaluation_specification.json`. The code verifies operational separation from GossipCop fitting and selection; it cannot prove preregistration or pre-outcome freezing.

## Source-hostname diagnostic

The added analysis is intentionally separate from the title classifier:

- URLs are parsed after the same title cleaning and conflict exclusions used for the external tests.
- Standard Wayback Machine URLs are unwrapped before extracting the archived target hostname.
- Hostnames are lowercased and common presentation prefixes are removed; missing or unparseable URLs form one explicit category.
- A five-fold stratified out-of-fold diagnostic learns each hostname's majority label from the training folds only.
- A hostname absent from a training fold receives that fold's majority-class prediction.
- No title text or TF-IDF features are used by this diagnostic.
- Confidence intervals use 2,000 class-stratified bootstrap resamples with seed 42.

This is an in-corpus association diagnostic. It is not source-disjoint validation, does not identify a causal publisher effect, and does not reconstruct the complete original FakeNewsNet collection process. The complete diagnostic specification is stored in `protocol/source_hostname_audit_protocol.json`.

## Cleaning and overlap audit

- GossipCop raw: 5,323 fake and 16,817 real records (22,140 total).
- Same-label duplicate rows removed: 1,309.
- Conflicting cleaned titles: 122, excluded with 244 rows.
- GossipCop final: 4,764 fake and 15,823 real titles (20,587 total).
- Exact cleaned-title overlap: ISOT–GossipCop 0; ISOT–PolitiFact 0; PolitiFact–GossipCop 2.
- All-zero ISOT TF-IDF vectors: 9 PolitiFact titles and 137 GossipCop titles.

## Main results

For the fixed Linear SVM on GossipCop:

- Balanced accuracy: 51.61% (95% CI 50.86–52.36).
- Accuracy: 43.56% (42.92–44.22).
- Macro-F1: 42.63% (42.03–43.25).
- Fake recall: 66.60%; real recall: 36.62%.
- Always-real reference: 76.86% accuracy, 50.00% balanced accuracy, and 43.46% macro-F1.

GossipCop minus PolitiFact differences for the fixed Linear SVM were −24.03 percentage points in accuracy, −16.67 points in balanced accuracy, and −24.93 points in macro-F1. These results quantify label-transfer failure under the observed domain shift; they do not measure factual truth.

For the source-hostname diagnostic:

- PolitiFact contained 519 valid normalized hostnames plus 58 missing or unparseable URLs. Of the valid hostnames, 503 occurred under only one label and covered 731 titles (74.97%). The out-of-fold diagnostic obtained 69.44% accuracy (95% CI 67.49–71.38), 65.10% balanced accuracy (62.87–67.26), and 62.83% macro-F1 (59.64–65.71).
- GossipCop contained 2,071 valid normalized hostnames plus 268 missing or unparseable URLs. Of the valid hostnames, 1,668 occurred under only one label and covered 4,923 titles (23.91%). The out-of-fold diagnostic obtained 84.78% accuracy (95% CI 84.40–85.17), 70.84% balanced accuracy (70.16–71.56), and 74.22% macro-F1 (73.49–74.99).
- The supplied cleaned GossipCop real-class URLs include no `eonline.com` hostname, whereas 136 cleaned fake-class rows use it. This prevents reconstructing the original repository's collection rule from these minimal CSV files and is reported as a provenance limitation rather than reinterpreted as a causal source effect.

## Files

- `CITATION.cff`: release citation metadata; add the repository URL and DOI
  after the public archive is created.
- `LICENSE`: MIT terms for package-authored software and documentation; the
  source datasets and manuscript are excluded.
- `CHANGELOG.md`: v1.0.0 change record.
- `RELEASE_CHECKLIST.md`: final repository/tag/archive actions.
- `MANIFEST.sha256`: SHA-256 checksums for every other released package file.
- `scripts/second_external_validity.py`: complete analysis.
- `scripts/fake_news_experiments.py`: ISOT loading, normalization, and hashing utilities.
- `scripts/external_error_analysis.py`: shared FakeNewsNet title cleaning utility.
- `scripts/source_hostname_audit.py`: source-hostname cleaning, association summary, out-of-fold diagnostic, bootstrap intervals, and hashes.
- `scripts/gossipcop_threshold_sensitivity.py`: threshold-free ranking, oracle diagnostic, repeated disjoint target-threshold selection, and target-domain Platt sensitivity.
- `scripts/reconstruct_isot_groups.py`: exact Jaccard ≥0.80 component reconstruction and the original combined-text group-fold assignment inherited by title rows.
- `scripts/equalized_multi_model_tuning.py`: equal-budget nested tuning and fixed-versus-selected evaluation for all four classifiers.
- `scripts/external_paired_bootstrap.py`: paired class-stratified bootstrap intervals for external selected-minus-fixed differences.
- `scripts/multiplicity_sensitivity.py`: reconstructs the 19-test family and computes raw, Holm-adjusted, and Benjamini–Hochberg-adjusted exact p values.
- `scripts/external_title_cue_masking.py`: builds the label-blind hostname lexicon, audits title cues, performs locked and symmetric masking experiments, and computes paired bootstrap differences.
- `scripts/title_near_duplicate_audit.py`: exact character-5-gram Jaccard
  audit within and across all three title corpora and title-specific ISOT fold
  construction.
- `scripts/source_clustered_bootstrap.py`: hostname-clustered intervals plus
  seen/unseen-hostname subgroup summaries.
- `scripts/external_cleaning_sensitivity.py`: fixed-SVM sensitivity to each
  external cleaning exclusion.
- `protocol/second_external_evaluation_specification.json`: retrospective specification and evidential-status declaration for the held-out GossipCop evaluation.
- `protocol/source_hostname_audit_protocol.json`: fixed specification and interpretation limits for the post hoc hostname diagnostic.
- `protocol/gossipcop_threshold_sensitivity_specification.json`: retrospective specification and interpretation limits for the post-hoc threshold analysis.
- `protocol/equalized_multi_model_tuning_specification.json`: grids, folds, selection rule, uncertainty analysis, and evidential status for the equal-budget experiment.
- `protocol/multiplicity_sensitivity_specification.json`: global-family definition, correction procedures, decision rule, interval policy, and interpretation limits.
- `protocol/external_title_cue_masking_specification.json`: cue definitions, masking conditions, uncertainty analysis, and interpretation limits.
- `protocol/additional_robustness_specification.json`: title-duplicate,
  title-specific-fold, source-clustered, seen/unseen-hostname, and cleaning
  sensitivity definitions and post-hoc status.
- `results/external_cleaning_audit.csv`: raw-to-clean counts.
- `results/cross_corpus_overlap_audit.csv`: exact cleaned-title overlap.
- `results/two_external_corpus_metrics.csv`: all model and baseline metrics with intervals.
- `results/svm_cross_corpus_difference.csv`: independent stratified-bootstrap differences.
- `results/gossipcop_svm_vs_majority_mcnemar.csv`: paired accuracy comparison.
- `results/external_predictions_without_text.csv`: IDs, labels, hashes, feature counts, and predictions; raw title text is not redistributed.
- `results/second_external_validity_config.json`: hashes, counts, software, and embedded protocol.
- `results/source_hostname_cleaning_audit.csv`: external raw-to-clean counts used by the diagnostic.
- `results/source_hostname_corpus_summary.csv`: valid/missing and label-exclusive hostname counts.
- `results/source_hostname_counts.csv`: per-corpus hostname class counts without title text.
- `results/source_hostname_oof_metrics.csv`: diagnostic metrics, class recalls, confusion counts, and bootstrap intervals.
- `results/source_hostname_oof_predictions_without_text.csv`: identifiers, cleaned-title hashes, hostnames, folds, labels, and out-of-fold predictions; raw title text is not redistributed.
- `results/source_hostname_audit_config.json`: input hashes, protocol hash, counts, metrics, and software versions.
- `results/gossipcop_full_sample_threshold_metrics.csv`: locked and descriptive oracle full-sample metrics.
- `results/gossipcop_threshold_adaptation_repeats.csv`: every method, fraction, and repeated split.
- `results/gossipcop_threshold_adaptation_summary.csv`: means, standard deviations, and split-stability percentiles.
- `results/gossipcop_threshold_paired_differences.csv`: paired adapted-minus-locked differences on identical evaluation subsets.
- `results/gossipcop_svm_scores_without_text.csv`: identifiers, labels, title hashes, SVM margins, and locked predictions without raw title text.
- `results/gossipcop_threshold_sensitivity_config.json`: hashes, counts, design, results, and software versions.
- `results/isot_title_group_fold_assignments.csv`: 38,681 title rows with cleaned-title hashes, near-duplicate groups, and inherited outer folds.
- `results/isot_near_duplicate_edges_jaccard_0_80.csv`: verified near-duplicate edges without raw text.
- `results/isot_group_reconstruction_summary.json`: reconstruction counts and fold-construction provenance.
- `results/equalized_tuning_grid_scores.csv`: every inner-fold score for all 48 candidate definitions.
- `results/equalized_tuning_selected_configs.csv`: nested outer and global ISOT selections.
- `results/equalized_tuning_outer_fold_metrics.csv`: fixed and selected outer-fold metrics.
- `results/equalized_tuning_nested_summary.csv`: model-level outer-fold means and standard deviations.
- `results/equalized_tuning_oof_predictions_without_text.csv`: paired ISOT outer predictions without title text.
- `results/equalized_tuning_external_metrics.csv`: fixed and selected PolitiFact/GossipCop metrics.
- `results/equalized_tuning_external_predictions_without_text.csv`: paired external predictions, labels, IDs, and cleaned-title hashes without raw text.
- `results/equalized_tuning_mcnemar.csv`: exact two-sided paired McNemar tests.
- `results/equalized_tuning_external_paired_bootstrap.csv`: 2,000-resample paired class-stratified intervals.
- `results/equalized_tuning_config.json`: grids, hashes, cleaning counts, runtime, and software versions.
- `results/multiple_testing_mcnemar_adjustments.csv`: all 19 paired contrasts with raw, Holm-adjusted, and BH-adjusted exact p values.
- `results/multiple_testing_family_summary.csv`: test and significance counts by analysis group and globally.
- `results/multiple_testing_config.json`: hashes, family definition, correction policy, results, and software versions.
- `results/external_title_cue_prevalence.csv`: any-brand, own-host-brand, style, and combined cue prevalence by corpus and label.
- `results/external_title_cue_counts.csv`: per-cue occurrence counts by corpus and label without title text.
- `results/external_title_cue_masking_metrics.csv`: metrics and prediction-change counts for all five masking conditions.
- `results/external_title_cue_masking_paired_bootstrap.csv`: paired masked-minus-baseline differences with marginal 95% intervals.
- `results/external_title_style_rule_metrics.csv`: deterministic style-cue-only diagnostic metrics.
- `results/external_title_cue_predictions_without_text.csv`: IDs, labels, title/masked hashes, cue indicators, and predictions without redistributed title text.
- `results/external_title_cue_masking_config.json`: lexicon, phrase patterns, hashes, counts, design, and software versions.
- `results/title_near_duplicate_edges_without_text.csv`: verified title-pair
  edges at Jaccard at least 0.70 without redistributed title text.
- `results/title_near_duplicate_pair_summary.csv`: pair counts at 0.70, 0.80,
  and 0.90 within and across corpora.
- `results/isot_title_specific_group_fold_assignments.csv`: title-specific
  Jaccard-0.80 component groups and five fold assignments.
- `results/title_near_duplicate_audit_config.json`: title-audit hashes,
  algorithm details, counts, and software versions.
- `results/title_specific_svm_*.csv` and
  `results/title_specific_svm_config.json`: nested and external Linear SVM
  sensitivity using the title-specific folds.
- `results/source_clustered_bootstrap_intervals.csv`: hostname-clustered
  intervals under both missing-hostname policies.
- `results/hostname_seen_unseen_performance.csv`: descriptive subgroup
  performance for fixed title models and the hostname-only rule.
- `results/hostname_seen_unseen_assignments_without_text.csv`: text-free
  subgroup assignment audit.
- `results/source_clustered_bootstrap_config.json`: clustered-bootstrap inputs,
  seed, policies, and software.
- `results/external_cleaning_sensitivity_metrics.csv`: three-population fixed
  SVM metrics and item-stratified intervals.
- `results/external_cleaning_sensitivity_predictions_without_text.csv`:
  text-free predictions for all sensitivity populations.
- `results/external_cleaning_sensitivity_config.json`: cleaning-population
  counts, hashes, model settings, and software.

## Reproduction

Obtain the original files from the
[ISOT dataset page](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
and the [FakeNewsNet project](https://github.com/KaiDMML/FakeNewsNet), keep
them outside this package, and verify their SHA-256 hashes against
`results/second_external_validity_config.json`. Run from the package root:

```bash
python scripts/second_external_validity.py \
  --isot-fake /path/to/Fake\(2\).csv \
  --isot-true /path/to/True\(2\).csv \
  --politifact-fake /path/to/politifact_fake.csv \
  --politifact-real /path/to/politifact_real.csv \
  --gossipcop-fake /path/to/gossipcop_fake.csv \
  --gossipcop-real /path/to/gossipcop_real.csv \
  --protocol protocol/second_external_evaluation_specification.json \
  --output reproduced_results
```

Confirm all source hashes against `results/second_external_validity_config.json` before comparing results.

Run the source-hostname diagnostic independently with:

```bash
python scripts/source_hostname_audit.py \
  --politifact-fake /path/to/politifact_fake.csv \
  --politifact-real /path/to/politifact_real.csv \
  --gossipcop-fake /path/to/gossipcop_fake.csv \
  --gossipcop-real /path/to/gossipcop_real.csv \
  --protocol protocol/source_hostname_audit_protocol.json \
  --output reproduced_source_hostname_results
```

Confirm the four external source hashes and the protocol hash against `results/source_hostname_audit_config.json`. The numeric title-only and hostname-audit outputs remain unchanged from Revision 8; Revision 9 corrects the evidential-status language and removes the unverifiable claim of prospective protocol freezing.

Run the post-hoc threshold-sensitivity analysis independently with:

```bash
python scripts/gossipcop_threshold_sensitivity.py \
  --isot-fake /path/to/Fake\(2\).csv \
  --isot-true /path/to/True\(2\).csv \
  --gossipcop-fake /path/to/gossipcop_fake.csv \
  --gossipcop-real /path/to/gossipcop_real.csv \
  --output reproduced_threshold_results \
  --seed 42 \
  --bootstrap-resamples 2000 \
  --split-repeats 200
```

The 2.5th and 97.5th percentiles across repeated splits describe split stability and are not sampling confidence intervals. The full-sample oracle is a descriptive in-sample upper bound only.

Reconstruct the original combined-text near-duplicate groups and inherited title folds with:

```bash
python scripts/reconstruct_isot_groups.py \
  --fake /path/to/Fake.csv \
  --true /path/to/True.csv \
  --output reproduced_groups \
  --threshold 0.80 \
  --seed 42
```

The resulting `isot_title_group_fold_assignments.csv` should match the hash recorded in `results/equalized_tuning_config.json`.

Run the equal-budget nested experiment with:

```bash
python scripts/equalized_multi_model_tuning.py \
  --isot-fake /path/to/Fake.csv \
  --isot-true /path/to/True.csv \
  --politifact-fake /path/to/politifact_fake.csv \
  --politifact-real /path/to/politifact_real.csv \
  --gossipcop-fake /path/to/gossipcop_fake.csv \
  --gossipcop-real /path/to/gossipcop_real.csv \
  --fold-assignments reproduced_groups/isot_title_group_fold_assignments.csv \
  --output reproduced_equalized_tuning \
  --models nb,lr,svm,rf \
  --seed 42 \
  --rf-estimators 200 \
  --rf-parallel-configs 3 \
  --rf-jobs-per-fit 3

python scripts/external_paired_bootstrap.py \
  --predictions reproduced_equalized_tuning/equalized_tuning_external_predictions_without_text.csv \
  --output reproduced_equalized_tuning/equalized_tuning_external_paired_bootstrap.csv \
  --resamples 2000 \
  --seed 4211
```

Run the global multiplicity audit with:

```bash
python scripts/multiplicity_sensitivity.py \
  --equalized-tests results/equalized_tuning_mcnemar.csv \
  --majority-test results/gossipcop_svm_vs_majority_mcnemar.csv \
  --output reproduced_multiplicity_results \
  --alpha 0.05
```

The script recomputes exact p values from discordant paired counts rather than trusting rounded manuscript values. Its output should reproduce 12 nominal findings, 11 Holm-retained findings, and 12 Benjamini–Hochberg-retained findings.

Run the external title-cue audit with:

```bash
python scripts/external_title_cue_masking.py \
  --isot-fake /path/to/Fake.csv \
  --isot-true /path/to/True.csv \
  --politifact-fake /path/to/politifact_fake.csv \
  --politifact-real /path/to/politifact_real.csv \
  --gossipcop-fake /path/to/gossipcop_fake.csv \
  --gossipcop-real /path/to/gossipcop_real.csv \
  --output reproduced_title_cue_results \
  --minimum-source-frequency 10 \
  --bootstrap-resamples 2000 \
  --seed 42 \
  --bootstrap-seed 5213
```

The outputs should reproduce source-brand prevalence of 2.67% for PolitiFact and 0.97% for GossipCop, style-cue prevalence of 9.23% and 6.13%, and locked combined-mask balanced accuracies of 66.54% and 51.53%.

Run the v1.0.0 title-level audit and construct title-specific folds with:

```bash
python scripts/title_near_duplicate_audit.py \
  --isot-fake /path/to/Fake.csv \
  --isot-true /path/to/True.csv \
  --politifact-fake /path/to/politifact_fake.csv \
  --politifact-real /path/to/politifact_real.csv \
  --gossipcop-fake /path/to/gossipcop_fake.csv \
  --gossipcop-real /path/to/gossipcop_real.csv \
  --output reproduced_title_audit \
  --principal-threshold 0.80 \
  --seed 42
```

Pass
`reproduced_title_audit/isot_title_specific_group_fold_assignments.csv` to
`scripts/equalized_multi_model_tuning.py` with `--models svm` to reproduce the
title-specific nested SVM sensitivity.

Run the source-clustered analysis from the released prediction files with:

```bash
python scripts/source_clustered_bootstrap.py \
  --external-predictions results/external_predictions_without_text.csv \
  --hostname-predictions results/source_hostname_oof_predictions_without_text.csv \
  --output reproduced_source_clustered \
  --resamples 2000 \
  --seed 6217
```

Run the external-cleaning sensitivity with:

```bash
python scripts/external_cleaning_sensitivity.py \
  --isot-fake /path/to/Fake.csv \
  --isot-true /path/to/True.csv \
  --politifact-fake /path/to/politifact_fake.csv \
  --politifact-real /path/to/politifact_real.csv \
  --gossipcop-fake /path/to/gossipcop_fake.csv \
  --gossipcop-real /path/to/gossipcop_real.csv \
  --output reproduced_cleaning_sensitivity \
  --seed 42 \
  --bootstrap-resamples 2000
```
