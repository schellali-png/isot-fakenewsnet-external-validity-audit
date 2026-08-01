#!/usr/bin/env python3
"""Reproduce group-aware ISOT validation and source-cue diagnostics.

The script reconstructs exact/Jaccard near-duplicate components from the
exact-deduplicated combined-text population, creates five
StratifiedGroupKFold partitions, and evaluates the four fixed TF-IDF models on
three representations:

* exact-deduplicated full text;
* exact-deduplicated full text with the token ``reuters`` masked; and
* exact-deduplicated titles inheriting the combined-text folds.

It also emits the conventional random-holdout audit, Reuters-only paired
comparison, text-free out-of-fold predictions, and the Linear-SVM coefficient
table used to draw the manuscript feature figure. Raw news text is never
written to the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from equalized_multi_model_tuning import (  # noqa: E402
    MODEL_NAMES,
    fixed_config,
    make_model,
    make_vectorizer,
    point_metrics,
)
from fake_news_experiments import load_dataset, sha256_file  # noqa: E402
from reconstruct_isot_groups import (  # noqa: E402
    component_summary,
    exact_jaccard_edges,
    hashed_shingle_set,
)


MODEL_ORDER = ("nb", "lr", "svm", "rf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isot-fake", type=Path, required=True)
    parser.add_argument("--isot-true", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--rf-jobs", type=int, default=-1)
    parser.add_argument("--isot-encoding", default="utf-8")
    parser.add_argument("--top-features", type=int, default=15)
    parser.add_argument(
        "--skip-conventional-holdout",
        action="store_true",
        help="Skip the raw/deduplicated 80/20 diagnostic holdouts.",
    )
    parser.add_argument(
        "--strict-manuscript-snapshot",
        action="store_true",
        help="Require the row and duplicate counts reported in the manuscript.",
    )
    return parser.parse_args()


def cleaned_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def mask_reuters(contents: pd.Series) -> pd.Series:
    return (
        contents.str.replace(r"\breuters\b", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    estimate = successes / total
    denominator = 1 + z * z / total
    centre = (estimate + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            (estimate * (1 - estimate) + z * z / (4 * total)) / total
        )
        / denominator
    )
    return centre - half, centre + half


def model_list(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(models).difference(MODEL_ORDER)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    if not models:
        raise ValueError("At least one model is required")
    return models


def reconstruct_groups_and_folds(
    combined: pd.DataFrame,
    threshold: float,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]], dict[str, object]]:
    shingle_sets = [hashed_shingle_set(value) for value in combined["content"]]
    edges, candidate_pairs = exact_jaccard_edges(shingle_sets, threshold)
    group_ids, multi, sizes, mixed = component_summary(
        len(combined), combined["label"].to_numpy(dtype=int), edges
    )
    group_ids = pd.factorize(group_ids, sort=False)[0].astype(np.int32)

    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    fold_ids = np.empty(len(combined), dtype=np.int16)
    dummy = csr_matrix((len(combined), 1))
    for fold, (_, test_indices) in enumerate(
        splitter.split(dummy, combined["label"], group_ids), start=1
    ):
        fold_ids[test_indices] = fold

    summary: dict[str, object] = {
        "jaccard_threshold": threshold,
        "candidate_pairs": candidate_pairs,
        "verified_edges": len(edges),
        "multi_document_groups": len(multi),
        "documents_in_multi_document_groups": int(sum(map(len, multi))),
        "component_size_counts": {
            str(key): int(value) for key, value in sorted(sizes.items())
        },
        "mixed_label_components": int(mixed),
        "combined_rows": int(len(combined)),
        "folds": folds,
        "fold_counts": {
            str(fold): int(np.sum(fold_ids == fold))
            for fold in range(1, folds + 1)
        },
    }
    return group_ids, fold_ids, edges, summary


def fit_fixed_predictions(
    model_name: str,
    train_contents: pd.Series,
    train_labels: np.ndarray,
    test_contents: pd.Series,
    args: argparse.Namespace,
) -> np.ndarray:
    config = fixed_config(model_name)
    vectorizer = make_vectorizer(config, args.max_features)
    train_matrix = vectorizer.fit_transform(train_contents)
    test_matrix = vectorizer.transform(test_contents)
    estimator = make_model(
        model_name,
        config.model_parameter,
        args.seed,
        args.rf_estimators,
        args.rf_jobs if model_name == "rf" else 1,
    )
    estimator.fit(train_matrix, train_labels)
    return estimator.predict(test_matrix).astype(int)


def evaluate_group_folds(
    name: str,
    data: pd.DataFrame,
    contents: pd.Series,
    fold_ids: np.ndarray,
    models: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    labels = data["label"].to_numpy(dtype=int)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for fold in range(1, args.folds + 1):
        train_indices = np.flatnonzero(fold_ids != fold)
        test_indices = np.flatnonzero(fold_ids == fold)
        if len(test_indices) == 0:
            raise ValueError(f"Representation {name} has an empty fold {fold}")
        for model_name in models:
            predictions = fit_fixed_predictions(
                model_name,
                contents.iloc[train_indices],
                labels[train_indices],
                contents.iloc[test_indices],
                args,
            )
            metrics = point_metrics(labels[test_indices], predictions)
            matrix = confusion_matrix(
                labels[test_indices], predictions, labels=[0, 1]
            )
            metric_rows.append(
                {
                    "representation": name,
                    "fold": fold,
                    "model": model_name,
                    "model_name": MODEL_NAMES[model_name],
                    "train_n": int(len(train_indices)),
                    "test_n": int(len(test_indices)),
                    **metrics,
                    "true_fake_pred_fake": int(matrix[0, 0]),
                    "true_fake_pred_real": int(matrix[0, 1]),
                    "true_real_pred_fake": int(matrix[1, 0]),
                    "true_real_pred_real": int(matrix[1, 1]),
                }
            )
            for local_index, row_index in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "representation": name,
                        "model": model_name,
                        "row_id": int(data.iloc[row_index]["row_id"]),
                        "content_sha256": cleaned_hash(contents.iloc[row_index]),
                        "outer_fold": fold,
                        "label": int(labels[row_index]),
                        "prediction": int(predictions[local_index]),
                    }
                )
        print(f"completed representation={name} fold={fold}", flush=True)
    return metric_rows, prediction_rows


def summarize_folds(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(
            ["representation", "model", "model_name"], sort=False
        )[["accuracy", "balanced_accuracy", "macro_f1", "fake_recall", "real_recall"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(filter(None, map(str, column))).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def evaluate_holdout_condition(
    condition: str,
    data: pd.DataFrame,
    contents: pd.Series,
    models: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    indices = np.arange(len(data), dtype=int)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=args.seed,
        stratify=data["label"].to_numpy(dtype=int),
    )
    labels = data["label"].to_numpy(dtype=int)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    stored: dict[str, np.ndarray] = {"test_indices": test_indices}

    for model_name in models:
        predictions = fit_fixed_predictions(
            model_name,
            contents.iloc[train_indices],
            labels[train_indices],
            contents.iloc[test_indices],
            args,
        )
        stored[model_name] = predictions
        metrics = point_metrics(labels[test_indices], predictions)
        low, high = wilson_interval(
            int(np.sum(predictions == labels[test_indices])), len(test_indices)
        )
        metric_rows.append(
            {
                "condition": condition,
                "model": model_name,
                "model_name": MODEL_NAMES[model_name],
                "train_n": int(len(train_indices)),
                "test_n": int(len(test_indices)),
                **metrics,
                "accuracy_wilson_95_low": low,
                "accuracy_wilson_95_high": high,
            }
        )
        for local_index, row_index in enumerate(test_indices):
            prediction_rows.append(
                {
                    "condition": condition,
                    "model": model_name,
                    "row_id": int(data.iloc[row_index]["row_id"]),
                    "content_sha256": cleaned_hash(contents.iloc[row_index]),
                    "label": int(labels[row_index]),
                    "prediction": int(predictions[local_index]),
                }
            )
    return metric_rows, prediction_rows, stored


def coefficient_table(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    config = fixed_config("svm")
    vectorizer = make_vectorizer(config, args.max_features)
    matrix = vectorizer.fit_transform(data["content"])
    estimator = make_model(
        "svm", config.model_parameter, args.seed, args.rf_estimators, 1
    )
    estimator.fit(matrix, data["label"].to_numpy(dtype=int))
    coefficients = np.asarray(estimator.coef_).ravel()
    feature_names = vectorizer.get_feature_names_out()
    count = min(args.top_features, len(feature_names))
    negative = np.argsort(coefficients)[:count]
    positive = np.argsort(coefficients)[-count:][::-1]
    rows: list[dict[str, object]] = []
    for direction, indices in (("fake", negative), ("real", positive)):
        for rank, feature_index in enumerate(indices, start=1):
            rows.append(
                {
                    "direction": direction,
                    "rank": rank,
                    "feature": str(feature_names[feature_index]),
                    "coefficient": float(coefficients[feature_index]),
                }
            )
    metadata = {
        "training_rows": int(len(data)),
        "vocabulary_size": int(len(feature_names)),
        "top_features_per_direction": count,
    }
    return pd.DataFrame(rows), metadata


def main() -> None:
    args = parse_args()
    if not 0 < args.jaccard_threshold <= 1:
        raise ValueError("--jaccard-threshold must be in (0, 1]")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    models = model_list(args.models)
    args.output.mkdir(parents=True, exist_ok=True)

    combined, combined_counts = load_dataset(
        args.isot_fake,
        args.isot_true,
        drop_duplicates=True,
        encoding=args.isot_encoding,
        text_field="combined",
        mask_reuters=False,
    )
    titles, title_counts = load_dataset(
        args.isot_fake,
        args.isot_true,
        drop_duplicates=True,
        encoding=args.isot_encoding,
        text_field="title",
        mask_reuters=False,
    )
    groups, combined_folds, edges, group_summary = reconstruct_groups_and_folds(
        combined,
        args.jaccard_threshold,
        args.folds,
        args.seed,
    )

    row_to_fold = dict(zip(combined["row_id"], combined_folds))
    row_to_group = dict(zip(combined["row_id"], groups))
    missing_title_rows = [
        int(row_id) for row_id in titles["row_id"] if row_id not in row_to_fold
    ]
    if missing_title_rows:
        raise ValueError(
            "Title rows are missing from the combined-text fold population: "
            f"{missing_title_rows[:10]}"
        )
    title_folds = np.array(
        [row_to_fold[row_id] for row_id in titles["row_id"]], dtype=np.int16
    )
    title_groups = np.array(
        [row_to_group[row_id] for row_id in titles["row_id"]], dtype=np.int32
    )

    if args.strict_manuscript_snapshot:
        expected = {
            "combined_rows": 38_826,
            "title_rows": 38_681,
            "verified_edges": 103,
            "multi_document_groups": 88,
            "documents_in_multi_document_groups": 185,
            "mixed_label_components": 1,
        }
        observed = {
            "combined_rows": int(len(combined)),
            "title_rows": int(len(titles)),
            **{
                key: group_summary[key]
                for key in (
                    "verified_edges",
                    "multi_document_groups",
                    "documents_in_multi_document_groups",
                    "mixed_label_components",
                )
            },
        }
        if observed != expected:
            raise ValueError(
                f"Snapshot audit mismatch: expected {expected}, observed {observed}"
            )

    assignment_frame = pd.DataFrame(
        {
            "row_id": combined["row_id"].to_numpy(dtype=int),
            "combined_text_sha256": combined["content"].map(cleaned_hash),
            "label": combined["label"].to_numpy(dtype=int),
            "near_duplicate_group": groups,
            "outer_fold": combined_folds,
        }
    )
    assignment_frame.to_csv(
        args.output / "isot_combined_group_fold_assignments.csv", index=False
    )
    pd.DataFrame(
        {
            "row_id": titles["row_id"].to_numpy(dtype=int),
            "cleaned_title_sha256": titles["content"].map(cleaned_hash),
            "label": titles["label"].to_numpy(dtype=int),
            "near_duplicate_group": title_groups,
            "outer_fold": title_folds,
        }
    ).to_csv(
        args.output / "isot_title_inherited_group_fold_assignments.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "first_row_id": int(combined.iloc[first]["row_id"]),
                "second_row_id": int(combined.iloc[second]["row_id"]),
                "jaccard": similarity,
            }
            for first, second, similarity in edges
        ]
    ).to_csv(args.output / "group_validation_near_duplicate_edges.csv", index=False)

    all_metric_rows: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []
    representations = (
        ("full_text", combined, combined["content"], combined_folds),
        (
            "full_text_reuters_masked",
            combined,
            mask_reuters(combined["content"]),
            combined_folds,
        ),
        ("title", titles, titles["content"], title_folds),
    )
    for name, frame, contents, fold_ids in representations:
        metrics, predictions = evaluate_group_folds(
            name, frame, contents.reset_index(drop=True), fold_ids, models, args
        )
        all_metric_rows.extend(metrics)
        all_prediction_rows.extend(predictions)

    fold_metrics = pd.DataFrame(all_metric_rows)
    fold_metrics.to_csv(args.output / "group_validation_fold_metrics.csv", index=False)
    summarize_folds(fold_metrics).to_csv(
        args.output / "group_validation_summary.csv", index=False
    )
    pd.DataFrame(all_prediction_rows).to_csv(
        args.output / "group_validation_oof_predictions_without_text.csv",
        index=False,
    )

    features, feature_metadata = coefficient_table(combined, args)
    features.to_csv(args.output / "linear_svm_top_coefficients.csv", index=False)

    holdout_metadata: dict[str, object] = {"skipped": True}
    if not args.skip_conventional_holdout:
        raw, raw_counts = load_dataset(
            args.isot_fake,
            args.isot_true,
            drop_duplicates=False,
            encoding=args.isot_encoding,
            text_field="combined",
            mask_reuters=False,
        )
        holdout_metrics: list[dict[str, object]] = []
        holdout_predictions: list[dict[str, object]] = []
        stored_by_condition: dict[str, dict[str, np.ndarray]] = {}
        for condition, frame, contents in (
            ("raw_nonempty_full_text", raw, raw["content"]),
            ("deduplicated_full_text", combined, combined["content"]),
            (
                "deduplicated_full_text_reuters_masked",
                combined,
                mask_reuters(combined["content"]),
            ),
        ):
            metrics, predictions, stored = evaluate_holdout_condition(
                condition,
                frame,
                contents.reset_index(drop=True),
                models,
                args,
            )
            holdout_metrics.extend(metrics)
            holdout_predictions.extend(predictions)
            stored_by_condition[condition] = stored

        deduplicated = stored_by_condition["deduplicated_full_text"]
        test_indices = deduplicated["test_indices"]
        test_labels = combined.iloc[test_indices]["label"].to_numpy(dtype=int)
        reuters_rule = (
            combined.iloc[test_indices]["content"]
            .str.contains(r"\breuters\b", regex=True)
            .to_numpy(dtype=bool)
            .astype(int)
        )
        rule_metrics = point_metrics(test_labels, reuters_rule)
        low, high = wilson_interval(
            int(np.sum(reuters_rule == test_labels)), len(test_labels)
        )
        holdout_metrics.append(
            {
                "condition": "deduplicated_full_text",
                "model": "reuters_rule",
                "model_name": "Reuters-only rule",
                "train_n": 0,
                "test_n": int(len(test_indices)),
                **rule_metrics,
                "accuracy_wilson_95_low": low,
                "accuracy_wilson_95_high": high,
            }
        )
        for local_index, row_index in enumerate(test_indices):
            holdout_predictions.append(
                {
                    "condition": "deduplicated_full_text",
                    "model": "reuters_rule",
                    "row_id": int(combined.iloc[row_index]["row_id"]),
                    "content_sha256": cleaned_hash(
                        combined.iloc[row_index]["content"]
                    ),
                    "label": int(test_labels[local_index]),
                    "prediction": int(reuters_rule[local_index]),
                }
            )

        if "svm" in deduplicated:
            svm_predictions = deduplicated["svm"]
            svm_only = int(
                np.sum((svm_predictions == test_labels) & (reuters_rule != test_labels))
            )
            rule_only = int(
                np.sum((svm_predictions != test_labels) & (reuters_rule == test_labels))
            )
            discordant = svm_only + rule_only
            p_value = (
                float(
                    binomtest(
                        min(svm_only, rule_only),
                        discordant,
                        0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if discordant
                else 1.0
            )
            pd.DataFrame(
                [
                    {
                        "comparison": "Linear SVM vs Reuters-only rule",
                        "svm_only_correct": svm_only,
                        "reuters_rule_only_correct": rule_only,
                        "discordant": discordant,
                        "exact_two_sided_mcnemar_p": p_value,
                    }
                ]
            ).to_csv(args.output / "source_rule_mcnemar.csv", index=False)

        pd.DataFrame(holdout_metrics).to_csv(
            args.output / "conventional_holdout_metrics.csv", index=False
        )
        pd.DataFrame(holdout_predictions).to_csv(
            args.output / "conventional_holdout_predictions_without_text.csv",
            index=False,
        )
        source_prevalence = (
            raw.assign(
                contains_reuters=raw["content"].str.contains(
                    r"\breuters\b", regex=True
                )
            )
            .groupby("label", sort=True)["contains_reuters"]
            .agg(["sum", "count"])
            .reset_index()
        )
        source_prevalence["proportion"] = (
            source_prevalence["sum"] / source_prevalence["count"]
        )
        source_prevalence.to_csv(
            args.output / "reuters_prevalence_by_label.csv", index=False
        )
        holdout_metadata = {
            "skipped": False,
            "raw_counts": raw_counts,
            "conditions": [
                "raw_nonempty_full_text",
                "deduplicated_full_text",
                "deduplicated_full_text_reuters_masked",
            ],
        }

    config = {
        "analysis": "group-aware internal validation",
        "input_hashes": {
            args.isot_fake.name: sha256_file(args.isot_fake),
            args.isot_true.name: sha256_file(args.isot_true),
        },
        "counts": {
            "combined_rows": int(len(combined)),
            "title_rows": int(len(titles)),
        },
        "cleaning": {
            "combined": combined_counts,
            "title": title_counts,
        },
        "group_reconstruction": group_summary,
        "models": models,
        "tfidf": {
            "ngram_range": [1, 1],
            "min_df": 2,
            "max_df": 0.95,
            "max_features": args.max_features,
            "sublinear_tf": True,
            "stop_words": "english",
        },
        "holdout": holdout_metadata,
        "feature_table": feature_metadata,
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
    (args.output / "group_validation_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved group-aware audit to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
