# Changelog

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
