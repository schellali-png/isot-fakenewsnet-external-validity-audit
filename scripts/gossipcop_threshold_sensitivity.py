#!/usr/bin/env python3
"""Post-hoc GossipCop threshold and target-calibration sensitivity analysis.

The locked zero-threshold ISOT-to-GossipCop result remains the primary external
evaluation. This script asks whether its class-recall asymmetry can be explained
by a target-domain decision-threshold mismatch. It reports threshold-free
ranking, a descriptive full-sample oracle bound, and repeated stratified
adaptation/evaluation splits. Target labels used to select a threshold or fit a
Platt sigmoid are confined to each adaptation subset; the paired evaluation
subset is untouched. The analysis is retrospective and exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from external_error_analysis import load_politifact  # noqa: E402
from fake_news_experiments import load_dataset, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isot-fake", type=Path, required=True)
    parser.add_argument("--isot-true", type=Path, required=True)
    parser.add_argument("--gossipcop-fake", type=Path, required=True)
    parser.add_argument("--gossipcop-real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--split-repeats", type=int, default=200)
    parser.add_argument(
        "--adaptation-fractions", default="0.05,0.10,0.20",
        help="Comma-separated target-label fractions used only for adaptation.",
    )
    parser.add_argument("--isot-encoding", default="utf-8")
    return parser.parse_args()


def make_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        lowercase=False,
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "fake_recall": float(recall_score(labels, predictions, pos_label=0)),
        "real_recall": float(recall_score(labels, predictions, pos_label=1)),
        "predicted_real_rate": float(np.mean(predictions == 1)),
    }


def select_balanced_accuracy_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> float:
    """Maximize Youden J; deterministic ties prefer the threshold nearest zero."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, scores, pos_label=1, drop_intermediate=False
    )
    youden_j = true_positive_rate - false_positive_rate
    maximum = float(np.max(youden_j))
    candidates = np.flatnonzero(
        np.isclose(youden_j, maximum, rtol=0.0, atol=1e-15)
    )
    return float(thresholds[candidates[np.argmin(np.abs(thresholds[candidates]))]])


def stratified_auc_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    fake_indices = np.flatnonzero(labels == 0)
    real_indices = np.flatnonzero(labels == 1)
    values = np.empty(resamples, dtype=float)
    for index in range(resamples):
        fake_draw = rng.choice(fake_indices, len(fake_indices), replace=True)
        real_draw = rng.choice(real_indices, len(real_indices), replace=True)
        draw = np.concatenate([fake_draw, real_draw])
        values[index] = roc_auc_score(labels[draw], scores[draw])
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def file_cleaned_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def repeated_target_adaptation(
    labels: np.ndarray,
    scores: np.ndarray,
    fractions: list[float],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fraction in fractions:
        splitter = StratifiedShuffleSplit(
            n_splits=repeats,
            train_size=fraction,
            random_state=seed + int(round(fraction * 10_000)),
        )
        for repeat, (adaptation, evaluation) in enumerate(
            splitter.split(scores, labels), start=1
        ):
            adaptation_labels = labels[adaptation]
            adaptation_scores = scores[adaptation]
            evaluation_labels = labels[evaluation]
            evaluation_scores = scores[evaluation]

            locked_predictions = (evaluation_scores >= 0.0).astype(int)
            selected_threshold = select_balanced_accuracy_threshold(
                adaptation_labels, adaptation_scores
            )
            selected_predictions = (evaluation_scores >= selected_threshold).astype(int)

            sigmoid = LogisticRegression(
                C=1_000_000.0,
                solver="lbfgs",
                max_iter=1_000,
                random_state=seed,
            )
            sigmoid.fit(adaptation_scores.reshape(-1, 1), adaptation_labels)
            probabilities = sigmoid.predict_proba(
                evaluation_scores.reshape(-1, 1)
            )[:, 1]
            platt_predictions = (probabilities >= 0.5).astype(int)

            common = {
                "adaptation_fraction": fraction,
                "repeat": repeat,
                "adaptation_n": int(len(adaptation)),
                "evaluation_n": int(len(evaluation)),
            }
            methods = (
                (
                    "locked_zero_threshold",
                    locked_predictions,
                    0.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ),
                (
                    "adaptation_balanced_accuracy_threshold",
                    selected_predictions,
                    selected_threshold,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ),
                (
                    "adaptation_platt_probability_0.5",
                    platt_predictions,
                    np.nan,
                    float(sigmoid.coef_[0, 0]),
                    float(sigmoid.intercept_[0]),
                    float(brier_score_loss(evaluation_labels, probabilities)),
                    float(log_loss(evaluation_labels, probabilities)),
                ),
            )
            for (
                method,
                predictions,
                score_threshold,
                platt_slope,
                platt_intercept,
                brier,
                logarithmic_loss,
            ) in methods:
                row = dict(common)
                row.update(
                    {
                        "method": method,
                        "score_threshold": score_threshold,
                        "platt_slope": platt_slope,
                        "platt_intercept": platt_intercept,
                        "brier": brier,
                        "log_loss": logarithmic_loss,
                    }
                )
                row.update(classification_metrics(evaluation_labels, predictions))
                records.append(row)
    return pd.DataFrame.from_records(records)


def summarize_repeats(repeats: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "score_threshold",
        "platt_slope",
        "platt_intercept",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "fake_recall",
        "real_recall",
        "predicted_real_rate",
        "brier",
        "log_loss",
    ]
    records: list[dict[str, object]] = []
    for (fraction, method), group in repeats.groupby(
        ["adaptation_fraction", "method"], sort=True
    ):
        row: dict[str, object] = {
            "adaptation_fraction": float(fraction),
            "method": method,
            "repeats": int(len(group)),
            "adaptation_n": int(group["adaptation_n"].iloc[0]),
            "evaluation_n": int(group["evaluation_n"].iloc[0]),
        }
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            row[f"{metric}_split_p2_5"] = float(np.percentile(values, 2.5))
            row[f"{metric}_split_p97_5"] = float(np.percentile(values, 97.5))
        records.append(row)
    return pd.DataFrame.from_records(records)


def summarize_paired_threshold_differences(repeats: pd.DataFrame) -> pd.DataFrame:
    """Compare adapted and locked thresholds on each identical evaluation split."""
    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    records: list[dict[str, object]] = []
    for fraction, group in repeats.groupby("adaptation_fraction", sort=True):
        wide = group.pivot(index="repeat", columns="method", values=metrics)
        for metric in metrics:
            difference = (
                wide[metric]["adaptation_balanced_accuracy_threshold"]
                - wide[metric]["locked_zero_threshold"]
            ).to_numpy(dtype=float)
            low, high = np.percentile(difference, [2.5, 97.5])
            records.append(
                {
                    "adaptation_fraction": float(fraction),
                    "comparison": (
                        "adaptation balanced-accuracy threshold minus locked zero"
                    ),
                    "metric": metric,
                    "repeats": int(len(difference)),
                    "mean_paired_difference": float(np.mean(difference)),
                    "sd_paired_difference": float(np.std(difference, ddof=1)),
                    "paired_difference_split_p2_5": float(low),
                    "paired_difference_split_p97_5": float(high),
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fractions = [float(item) for item in args.adaptation_fractions.split(",")]
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("Every adaptation fraction must be between zero and one.")

    isot, isot_counts = load_dataset(
        args.isot_fake,
        args.isot_true,
        drop_duplicates=True,
        encoding=args.isot_encoding,
        text_field="title",
        mask_reuters=False,
    )
    gossipcop, gossipcop_counts = load_politifact(
        args.gossipcop_fake, args.gossipcop_real
    )
    if len(isot) != 38_681:
        raise ValueError(f"Expected 38,681 ISOT titles, found {len(isot):,}")
    if len(gossipcop) != 20_587:
        raise ValueError(f"Expected 20,587 GossipCop titles, found {len(gossipcop):,}")

    vectorizer = make_vectorizer(args.max_features)
    isot_matrix = vectorizer.fit_transform(isot["content"])
    gossipcop_matrix = vectorizer.transform(gossipcop["content"])
    model = LinearSVC(C=1.0, random_state=args.seed)
    model.fit(isot_matrix, isot["label"].to_numpy())
    labels = gossipcop["label"].to_numpy(dtype=int)
    scores = model.decision_function(gossipcop_matrix)
    locked_predictions = (scores >= 0.0).astype(int)

    locked_metrics = classification_metrics(labels, locked_predictions)
    auc = float(roc_auc_score(labels, scores))
    auc_low, auc_high = stratified_auc_bootstrap(
        labels,
        scores,
        args.bootstrap_resamples,
        args.seed + 1_000,
    )
    average_precision_real = float(average_precision_score(labels, scores))
    average_precision_fake = float(average_precision_score(1 - labels, -scores))

    oracle_threshold = select_balanced_accuracy_threshold(labels, scores)
    oracle_predictions = (scores >= oracle_threshold).astype(int)
    oracle_metrics = classification_metrics(labels, oracle_predictions)

    full_records: list[dict[str, object]] = []
    for status, threshold, metrics in (
        ("locked_zero_threshold_primary", 0.0, locked_metrics),
        (
            "full_sample_oracle_descriptive_upper_bound",
            oracle_threshold,
            oracle_metrics,
        ),
    ):
        record = {"analysis": status, "score_threshold": threshold}
        record.update(metrics)
        full_records.append(record)
    pd.DataFrame(full_records).to_csv(
        args.output / "gossipcop_full_sample_threshold_metrics.csv", index=False
    )

    repetitions = repeated_target_adaptation(
        labels, scores, fractions, args.split_repeats, args.seed
    )
    repetitions.to_csv(
        args.output / "gossipcop_threshold_adaptation_repeats.csv", index=False
    )
    summary = summarize_repeats(repetitions)
    summary.to_csv(
        args.output / "gossipcop_threshold_adaptation_summary.csv", index=False
    )
    paired_differences = summarize_paired_threshold_differences(repetitions)
    paired_differences.to_csv(
        args.output / "gossipcop_threshold_paired_differences.csv", index=False
    )

    prediction_audit = pd.DataFrame(
        {
            "id": gossipcop["id"].astype(str),
            "cleaned_title_sha256": gossipcop["content"].map(file_cleaned_hash),
            "label": labels,
            "linear_svm_decision_score": scores,
            "locked_zero_threshold_prediction": locked_predictions,
        }
    )
    prediction_audit.to_csv(
        args.output / "gossipcop_svm_scores_without_text.csv", index=False
    )

    config = {
        "analysis_status": "post-hoc target-adaptation sensitivity analysis",
        "interpretation": (
            "The locked zero-threshold result remains primary. Target-label "
            "adaptation is exploratory and is evaluated on disjoint subsets."
        ),
        "design": {
            "adaptation_fractions": fractions,
            "split_repeats": args.split_repeats,
            "split_percentiles": (
                "2.5th and 97.5th percentiles across repeated splits; not a "
                "sampling confidence interval"
            ),
            "threshold_objective": "maximum balanced accuracy (Youden J)",
            "threshold_tie_break": "threshold closest to locked zero",
            "platt_model": "near-unregularized one-dimensional logistic sigmoid",
            "target_label_separation": (
                "adaptation labels are not used to evaluate the paired holdout"
            ),
        },
        "counts": {
            "isot_titles": int(len(isot)),
            "gossipcop_titles": int(len(gossipcop)),
            "gossipcop_fake": int(np.sum(labels == 0)),
            "gossipcop_real": int(np.sum(labels == 1)),
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "isot_cleaning": isot_counts,
        "gossipcop_cleaning": gossipcop_counts,
        "threshold_free": {
            "roc_auc": auc,
            "roc_auc_stratified_bootstrap_95_low": auc_low,
            "roc_auc_stratified_bootstrap_95_high": auc_high,
            "bootstrap_resamples": args.bootstrap_resamples,
            "average_precision_real": average_precision_real,
            "average_precision_fake": average_precision_fake,
            "macro_average_precision": (
                average_precision_real + average_precision_fake
            )
            / 2.0,
        },
        "locked_zero_threshold": locked_metrics,
        "full_sample_oracle_descriptive_upper_bound": {
            "score_threshold": oracle_threshold,
            **oracle_metrics,
        },
        "input_hashes": {
            args.isot_fake.name: sha256_file(args.isot_fake),
            args.isot_true.name: sha256_file(args.isot_true),
            args.gossipcop_fake.name: sha256_file(args.gossipcop_fake),
            args.gossipcop_real.name: sha256_file(args.gossipcop_real),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output / "gossipcop_threshold_sensitivity_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(json.dumps(config["threshold_free"], indent=2))
    print(json.dumps(config["locked_zero_threshold"], indent=2))
    print(json.dumps(config["full_sample_oracle_descriptive_upper_bound"], indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
