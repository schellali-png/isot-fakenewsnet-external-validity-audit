# Changelog

## 1.1.0 — 2026-07-31

- Added `group_aware_internal_validation.py` as the complete executable path
  for group reconstruction, inherited title folds, fixed-model internal
  validation, conventional holdouts, source-rule diagnostics, and feature
  coefficients.
- Added `chronological_evaluation.py` with deterministic date parsing and full
  removal of near-duplicate components that cross the temporal cutoff.
- Added `calibration_transfer.py` with nested ISOT group-fold Platt calibration
  and locked ISOT-to-PolitiFact transfer, reliability, ECE, Brier, log-loss,
  and selective-prediction outputs.
- Added a synthetic end-to-end smoke test for the three workflows.
- Updated the package and preferred manuscript titles, citation metadata,
  reproduction commands, and release checklist.

## 1.0.0 — 2026-07-24

- Prepared a clean public-release package with citation and licensing metadata.
- Added an exact title-level character-5-gram Jaccard near-duplicate audit at
  thresholds 0.70, 0.80, and 0.90 across ISOT, PolitiFact, and GossipCop.
- Added title-specific Jaccard-0.80 group folds and a nested Linear SVM
  sensitivity analysis using those folds.
- Added source-hostname clustered bootstrap intervals with pooled-missing and
  item-unique-missing policies.
- Added seen-versus-unseen hostname subgroup summaries for the fixed title
  models and the hostname-only diagnostic.
- Added external cleaning sensitivity on all nonempty rows, same-label
  exact-deduplicated rows with conflicts retained, and the main
  unique/nonconflicting population.
- Retained all Revision-13 external title-cue, multiplicity, tuning,
  threshold, hostname, and two-corpus validation artifacts.
