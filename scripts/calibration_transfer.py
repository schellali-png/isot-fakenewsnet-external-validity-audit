#!/usr/bin/env python3
"""Audit probability calibration from ISOT titles to PolitiFact titles.

The internal ISOT estimate uses nested inherited group folds: each outer-fold
probability is produced by a Platt calibrator fitted only to inner out-of-fold
Linear-SVM margins from the remaining groups. For transfer, one Platt mapping
is fitted to five-fold ISOT out-of-fold margins, while the base SVM is refitted
on all ISOT titles and applied unchanged to PolitiFact. No PolitiFact label is
used for fitting, threshold choice, or calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from equalized_multi_model_tuning import (  # noqa: E402
    fixed_config,
    make_model,
    make_vectorizer,
)
from external_error_analysis import load_politifact  # noqa: E402
from fake_news_experiments import load_dataset, sha256_file  # noqa: E402
from group_aware_internal_validation import cleaned_hash, wilson_interval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isot-fake", type=Path, required=True)
    parser.add_argument("--isot-true", type=Path, required=True)
    parser.add_argument("--politifact-fake", type=Path, required=True)
    parser.add_argument("--politifact-real", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--confidence-thresholds", default="0.50,0.60,0.70,0.80,0.90"
    )
    parser.add_argument("--isot-encoding", default="utf-8")
    parser.add_argument(
        "--strict-manuscript-snapshot",
        action="store_true",
        help="Require the published ISOT and PolitiFact cleaned-title counts.",
    )
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted(
        {float(item.strip()) for item in value.split(",") if item.strip()}
    )
    if not thresholds or any(item < 0.5 or item > 1 for item in thresholds):
        raise ValueError("Confidence thresholds must lie in [0.5, 1]")
    return thresholds


def validate_and_attach_folds(
    titles: pd.DataFrame,
    path: Path,
) -> tuple[pd.DataFrame, list[int]]:
    assignments = pd.read_csv(path)
    required = {"row_id", "label", "outer_fold"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Fold assignments are missing columns: {sorted(missing)}")
    if assignments["row_id"].duplicated().any():
        raise ValueError("Fold assignments contain duplicate row_id values")

    frame = titles.reset_index(drop=True).copy()
    expected_ids = set(frame["row_id"].astype(int))
    supplied_ids = set(assignments["row_id"].astype(int))
    if expected_ids != supplied_ids:
        raise ValueError(
            "Fold assignments do not exactly match the cleaned ISOT-title rows: "
            f"missing={len(expected_ids - supplied_ids)}, "
            f"extra={len(supplied_ids - expected_ids)}"
        )
    selected_columns = ["row_id", "label", "outer_fold"]
    hash_column = None
    for candidate in ("cleaned_title_sha256", "content_sha256"):
        if candidate in assignments.columns:
            hash_column = candidate
            selected_columns.append(candidate)
            break
    attached = frame.merge(
        assignments[selected_columns],
        on="row_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_assignment"),
    )
    if not np.array_equal(
        attached["label"].to_numpy(dtype=int),
        attached["label_assignment"].to_numpy(dtype=int),
    ):
        raise ValueError("Assignment labels disagree with the cleaned ISOT titles")
    if hash_column is not None:
        observed_hashes = attached["content"].map(cleaned_hash)
        if not observed_hashes.equals(attached[hash_column].astype(str)):
            mismatch = int((observed_hashes != attached[hash_column].astype(str)).sum())
            raise ValueError(f"Assignment title hashes disagree for {mismatch} rows")
    attached["outer_fold"] = attached["outer_fold"].astype(int)
    folds = sorted(attached["outer_fold"].unique().tolist())
    if len(folds) < 3:
        raise ValueError("Nested calibration requires at least three outer folds")
    return attached, folds


def fit_margin_model(
    train_text: pd.Series,
    train_labels: np.ndarray,
    test_text: pd.Series,
    max_features: int,
    seed: int,
) -> tuple[np.ndarray, object, object]:
    config = fixed_config("svm")
    vectorizer = make_vectorizer(config, max_features)
    train_matrix = vectorizer.fit_transform(train_text)
    test_matrix = vectorizer.transform(test_text)
    estimator = make_model("svm", config.model_parameter, seed, 1, 1)
    estimator.fit(train_matrix, train_labels)
    margins = np.asarray(estimator.decision_function(test_matrix), dtype=float)
    return margins, vectorizer, estimator


def make_platt(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=1_000,
        random_state=seed,
    )


def probabilities(platt: LogisticRegression, margins: np.ndarray) -> np.ndarray:
    return platt.predict_proba(np.asarray(margins).reshape(-1, 1))[:, 1]


def expected_calibration_error(
    labels: np.ndarray,
    probabilities_: np.ndarray,
    bins: int,
) -> float:
    indices = np.minimum((probabilities_ * bins).astype(int), bins - 1)
    total = len(labels)
    return float(
        sum(
            np.sum(indices == bin_index)
            / total
            * abs(
                float(np.mean(probabilities_[indices == bin_index]))
                - float(np.mean(labels[indices == bin_index]))
            )
            for bin_index in range(bins)
            if np.any(indices == bin_index)
        )
    )


def calibration_metrics(
    dataset: str,
    method: str,
    labels: np.ndarray,
    probabilities_: np.ndarray,
    bins: int,
) -> dict[str, object]:
    predictions = (probabilities_ >= 0.5).astype(int)
    return {
        "dataset": dataset,
        "method": method,
        "n": int(len(labels)),
        "fake_n": int(np.sum(labels == 0)),
        "real_n": int(np.sum(labels == 1)),
        "brier_score": float(brier_score_loss(labels, probabilities_)),
        "log_loss": float(log_loss(labels, probabilities_, labels=[0, 1])),
        "ece_equal_width": expected_calibration_error(labels, probabilities_, bins),
        "ece_bins": bins,
        "roc_auc": (
            float(roc_auc_score(labels, probabilities_))
            if len(np.unique(labels)) == 2
            else math.nan
        ),
        "accuracy_at_0_5": float(accuracy_score(labels, predictions)),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1_at_0_5": float(f1_score(labels, predictions, average="macro")),
        "mean_predicted_real_probability": float(np.mean(probabilities_)),
        "empirical_real_prevalence": float(np.mean(labels)),
    }


def reliability_rows(
    dataset: str,
    method: str,
    labels: np.ndarray,
    probabilities_: np.ndarray,
    bins: int,
) -> list[dict[str, object]]:
    indices = np.minimum((probabilities_ * bins).astype(int), bins - 1)
    rows: list[dict[str, object]] = []
    for bin_index in range(bins):
        selected = indices == bin_index
        count = int(np.sum(selected))
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "bin": bin_index + 1,
                "lower_inclusive": bin_index / bins,
                "upper_inclusive_only_for_final_bin": (bin_index + 1) / bins,
                "n": count,
                "mean_predicted_real_probability": (
                    float(np.mean(probabilities_[selected])) if count else math.nan
                ),
                "empirical_real_frequency": (
                    float(np.mean(labels[selected])) if count else math.nan
                ),
                "absolute_calibration_gap": (
                    abs(
                        float(np.mean(probabilities_[selected]))
                        - float(np.mean(labels[selected]))
                    )
                    if count
                    else math.nan
                ),
            }
        )
    return rows


def selective_rows(
    dataset: str,
    method: str,
    labels: np.ndarray,
    probabilities_: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, object]]:
    confidence = np.maximum(probabilities_, 1 - probabilities_)
    predictions = (probabilities_ >= 0.5).astype(int)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        retained = confidence >= threshold
        count = int(np.sum(retained))
        correct = int(np.sum(predictions[retained] == labels[retained]))
        low, high = wilson_interval(correct, count)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "confidence_threshold": threshold,
                "retained_n": count,
                "coverage": count / len(labels),
                "errors": count - correct,
                "accuracy": correct / count if count else math.nan,
                "accuracy_wilson_95_low": low,
                "accuracy_wilson_95_high": high,
                "balanced_accuracy": (
                    float(
                        balanced_accuracy_score(
                            labels[retained], predictions[retained]
                        )
                    )
                    if count and len(np.unique(labels[retained])) == 2
                    else math.nan
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    thresholds = parse_thresholds(args.confidence_thresholds)
    args.output.mkdir(parents=True, exist_ok=True)

    isot, isot_counts = load_dataset(
        args.isot_fake,
        args.isot_true,
        drop_duplicates=True,
        encoding=args.isot_encoding,
        text_field="title",
        mask_reuters=False,
    )
    isot, folds = validate_and_attach_folds(isot, args.fold_assignments)
    politifact, politifact_counts = load_politifact(
        args.politifact_fake, args.politifact_real
    )
    if args.strict_manuscript_snapshot:
        observed = {"isot_titles": len(isot), "politifact_titles": len(politifact)}
        expected = {"isot_titles": 38_681, "politifact_titles": 975}
        if observed != expected:
            raise ValueError(
                f"Snapshot audit mismatch: expected {expected}, observed {observed}"
            )

    labels = isot["label"].to_numpy(dtype=int)
    fold_ids = isot["outer_fold"].to_numpy(dtype=int)
    nested_margins = np.full(len(isot), np.nan, dtype=float)
    nested_probabilities = np.full(len(isot), np.nan, dtype=float)
    parameter_rows: list[dict[str, object]] = []

    for outer_fold in folds:
        outer_test = fold_ids == outer_fold
        outer_train = ~outer_test
        inner_margins = np.full(int(np.sum(outer_train)), np.nan, dtype=float)
        outer_train_positions = np.flatnonzero(outer_train)
        inner_fold_ids = fold_ids[outer_train]
        inner_labels = labels[outer_train]
        for inner_fold in folds:
            if inner_fold == outer_fold:
                continue
            inner_validation = inner_fold_ids == inner_fold
            inner_training = ~inner_validation
            margins, _, _ = fit_margin_model(
                isot.iloc[outer_train_positions[inner_training]]["content"],
                inner_labels[inner_training],
                isot.iloc[outer_train_positions[inner_validation]]["content"],
                args.max_features,
                args.seed,
            )
            inner_margins[inner_validation] = margins
        if np.isnan(inner_margins).any():
            raise RuntimeError(f"Incomplete inner margins for outer fold {outer_fold}")
        platt = make_platt(args.seed)
        platt.fit(inner_margins.reshape(-1, 1), inner_labels)
        outer_margins, _, _ = fit_margin_model(
            isot.loc[outer_train, "content"],
            labels[outer_train],
            isot.loc[outer_test, "content"],
            args.max_features,
            args.seed,
        )
        nested_margins[outer_test] = outer_margins
        nested_probabilities[outer_test] = probabilities(platt, outer_margins)
        parameter_rows.append(
            {
                "calibrator": "nested_outer",
                "outer_fold": outer_fold,
                "training_oof_n": int(len(inner_labels)),
                "slope": float(platt.coef_[0, 0]),
                "intercept": float(platt.intercept_[0]),
            }
        )
        print(f"completed nested calibration outer_fold={outer_fold}", flush=True)

    if np.isnan(nested_probabilities).any():
        raise RuntimeError("Nested calibration did not cover every ISOT row")

    global_oof_margins = np.full(len(isot), np.nan, dtype=float)
    for fold in folds:
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        margins, _, _ = fit_margin_model(
            isot.loc[train_mask, "content"],
            labels[train_mask],
            isot.loc[test_mask, "content"],
            args.max_features,
            args.seed,
        )
        global_oof_margins[test_mask] = margins
    global_platt = make_platt(args.seed)
    global_platt.fit(global_oof_margins.reshape(-1, 1), labels)
    parameter_rows.append(
        {
            "calibrator": "global_transfer",
            "outer_fold": "all",
            "training_oof_n": int(len(labels)),
            "slope": float(global_platt.coef_[0, 0]),
            "intercept": float(global_platt.intercept_[0]),
        }
    )

    external_labels = politifact["label"].to_numpy(dtype=int)
    external_margins, _, _ = fit_margin_model(
        isot["content"],
        labels,
        politifact["content"],
        args.max_features,
        args.seed,
    )
    external_probabilities = probabilities(global_platt, external_margins)
    constant_probabilities = np.full(len(politifact), float(np.mean(labels)))

    evaluated = (
        (
            "ISOT",
            "nested_group_fold_platt_svm",
            labels,
            nested_probabilities,
        ),
        (
            "PolitiFact",
            "isot_oof_platt_svm_transfer",
            external_labels,
            external_probabilities,
        ),
        (
            "PolitiFact",
            "isot_real_prevalence_constant",
            external_labels,
            constant_probabilities,
        ),
    )
    metric_rows: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    selective: list[dict[str, object]] = []
    for dataset, method, y_true, probability_values in evaluated:
        metric_rows.append(
            calibration_metrics(
                dataset, method, y_true, probability_values, args.bins
            )
        )
        reliability.extend(
            reliability_rows(
                dataset, method, y_true, probability_values, args.bins
            )
        )
        selective.extend(
            selective_rows(
                dataset, method, y_true, probability_values, thresholds
            )
        )
    pd.DataFrame(metric_rows).to_csv(
        args.output / "calibration_metrics.csv", index=False
    )
    pd.DataFrame(reliability).to_csv(
        args.output / "calibration_reliability_bins.csv", index=False
    )
    pd.DataFrame(selective).to_csv(
        args.output / "calibration_selective_prediction.csv", index=False
    )
    pd.DataFrame(parameter_rows).to_csv(
        args.output / "calibration_platt_parameters.csv", index=False
    )

    prediction_rows: list[dict[str, object]] = []
    for index, row in isot.iterrows():
        prediction_rows.append(
            {
                "dataset": "ISOT",
                "record_id": str(int(row["row_id"])),
                "cleaned_title_sha256": cleaned_hash(row["content"]),
                "outer_fold": int(row["outer_fold"]),
                "label": int(row["label"]),
                "decision_margin": nested_margins[index],
                "real_probability": nested_probabilities[index],
                "prediction": int(nested_probabilities[index] >= 0.5),
            }
        )
    for index, row in politifact.iterrows():
        prediction_rows.append(
            {
                "dataset": "PolitiFact",
                "record_id": str(row["id"]),
                "cleaned_title_sha256": cleaned_hash(row["content"]),
                "outer_fold": "",
                "label": int(row["label"]),
                "decision_margin": external_margins[index],
                "real_probability": external_probabilities[index],
                "prediction": int(external_probabilities[index] >= 0.5),
            }
        )
    pd.DataFrame(prediction_rows).to_csv(
        args.output / "calibration_predictions_without_text.csv", index=False
    )

    config = {
        "analysis": "nested calibration and locked probability transfer",
        "class_mapping": {"0": "fake", "1": "real"},
        "base_model": {
            "name": "Linear SVM",
            "C": 1.0,
            "tfidf_ngrams": [1, 1],
            "tfidf_min_df": 2,
            "tfidf_max_df": 0.95,
            "tfidf_max_features": args.max_features,
        },
        "calibrator": {
            "method": "Platt logistic mapping",
            "C": 1_000_000.0,
            "internal_protocol": "nested inherited group folds",
            "external_protocol": "mapping fitted to all ISOT OOF margins",
            "external_labels_used_for_fitting": False,
        },
        "folds": folds,
        "bins": args.bins,
        "confidence_thresholds": thresholds,
        "counts": {"isot_titles": len(isot), "politifact_titles": len(politifact)},
        "cleaning": {"isot": isot_counts, "politifact": politifact_counts},
        "input_hashes": {
            args.isot_fake.name: sha256_file(args.isot_fake),
            args.isot_true.name: sha256_file(args.isot_true),
            args.politifact_fake.name: sha256_file(args.politifact_fake),
            args.politifact_real.name: sha256_file(args.politifact_real),
            args.fold_assignments.name: sha256_file(args.fold_assignments),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (args.output / "calibration_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved calibration transfer audit to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
