#!/usr/bin/env python3
"""Source-clustered uncertainty and seen/unseen hostname sensitivity analyses.

This post hoc analysis joins the fixed external title-model predictions to the
hostname audit by corpus, item id, title hash, and label. Confidence intervals
resample normalized hostnames (all observations from a sampled hostname move
together). The primary policy pools missing hostnames as one cluster; a
secondary policy treats each missing-hostname item as its own cluster.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fake_news_experiments import sha256_file  # noqa: E402


MODEL_COLUMNS = {
    "multinomial_nb": "multinomial_nb_prediction",
    "logistic_regression": "logistic_regression_prediction",
    "linear_svm": "linear_svm_prediction",
    "random_forest": "random_forest_prediction",
    "hostname_only": "hostname_only_prediction",
}
MISSING_HOSTNAME = "[missing]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument("--hostname-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=6217)
    return parser.parse_args()


def metric_values(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "fake_recall": float(
            recall_score(labels, predictions, pos_label=0, zero_division=0)
        ),
        "real_recall": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
    }


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = matrix.ravel().astype(float)
    total = tn + fp + fn + tp
    fake_recall = tn / (tn + fp) if tn + fp else np.nan
    real_recall = tp / (tp + fn) if tp + fn else np.nan
    fake_precision = tn / (tn + fn) if tn + fn else 0.0
    real_precision = tp / (tp + fp) if tp + fp else 0.0
    fake_f1 = (
        2.0 * fake_precision * fake_recall / (fake_precision + fake_recall)
        if fake_precision + fake_recall
        else 0.0
    )
    real_f1 = (
        2.0 * real_precision * real_recall / (real_precision + real_recall)
        if real_precision + real_recall
        else 0.0
    )
    return {
        "accuracy": (tn + tp) / total if total else np.nan,
        "balanced_accuracy": np.nanmean([fake_recall, real_recall]),
        "macro_f1": (fake_f1 + real_f1) / 2.0,
        "fake_recall": fake_recall,
        "real_recall": real_recall,
    }


def cluster_confusions(
    data: pd.DataFrame,
    prediction_column: str,
    cluster_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    clusters: list[str] = []
    matrices: list[np.ndarray] = []
    for cluster, part in data.groupby(cluster_column, sort=False):
        clusters.append(str(cluster))
        matrices.append(
            confusion_matrix(
                part["label"].to_numpy(dtype=int),
                part[prediction_column].to_numpy(dtype=int),
                labels=[0, 1],
            )
        )
    return np.asarray(clusters, dtype=object), np.asarray(matrices, dtype=np.int64)


def clustered_intervals(
    matrices: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    cluster_count = len(matrices)
    distributions = {
        "accuracy": np.empty(resamples, dtype=float),
        "balanced_accuracy": np.empty(resamples, dtype=float),
        "macro_f1": np.empty(resamples, dtype=float),
        "fake_recall": np.empty(resamples, dtype=float),
        "real_recall": np.empty(resamples, dtype=float),
    }
    for iteration in range(resamples):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        values = metrics_from_confusion(matrices[sampled].sum(axis=0))
        for metric, value in values.items():
            distributions[metric][iteration] = value
    return {
        metric: tuple(np.nanpercentile(values, [2.5, 97.5]).astype(float))
        for metric, values in distributions.items()
    }


def prepare_join(
    external_path: Path,
    hostname_path: Path,
) -> pd.DataFrame:
    external = pd.read_csv(external_path)
    hostname = pd.read_csv(hostname_path)
    keys = ["corpus", "id", "cleaned_title_sha256", "label"]
    required_external = set(keys).union(
        column for model, column in MODEL_COLUMNS.items() if model != "hostname_only"
    )
    required_hostname = set(keys).union(
        {"hostname", "fold", MODEL_COLUMNS["hostname_only"]}
    )
    missing_external = required_external.difference(external.columns)
    missing_hostname = required_hostname.difference(hostname.columns)
    if missing_external:
        raise ValueError(
            f"External predictions missing columns: {sorted(missing_external)}"
        )
    if missing_hostname:
        raise ValueError(
            f"Hostname predictions missing columns: {sorted(missing_hostname)}"
        )
    joined = external.merge(
        hostname[
            keys + ["hostname", "fold", MODEL_COLUMNS["hostname_only"]]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(external) or len(joined) != len(hostname):
        raise ValueError(
            "Prediction files do not have the same one-to-one item population."
        )
    joined["hostname"] = joined["hostname"].fillna(MISSING_HOSTNAME).astype(str)
    joined["cluster_missing_pooled"] = joined["hostname"]
    joined["cluster_missing_item_unique"] = joined["hostname"]
    missing = joined["hostname"].eq(MISSING_HOSTNAME)
    joined.loc[missing, "cluster_missing_item_unique"] = (
        MISSING_HOSTNAME
        + "::"
        + joined.loc[missing, "corpus"].astype(str)
        + "::"
        + joined.loc[missing, "id"].astype(str)
    )

    fold_diversity = joined.groupby(
        ["corpus", "hostname"], sort=False
    )["fold"].transform("nunique")
    joined["hostname_seen_in_training_fold"] = fold_diversity.gt(1)
    joined["hostname_status"] = np.where(
        joined["hostname_seen_in_training_fold"], "seen", "unseen"
    )
    return joined


def main() -> None:
    args = parse_args()
    if args.resamples < 1:
        raise ValueError("--resamples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    joined = prepare_join(
        args.external_predictions, args.hostname_predictions
    )

    interval_rows: list[dict[str, object]] = []
    policies = {
        "missing_pooled_primary": "cluster_missing_pooled",
        "missing_item_unique_secondary": "cluster_missing_item_unique",
    }
    seed_offset = 0
    for corpus, corpus_data in joined.groupby("corpus", sort=True):
        for policy_name, cluster_column in policies.items():
            for model, prediction_column in MODEL_COLUMNS.items():
                clusters, matrices = cluster_confusions(
                    corpus_data, prediction_column, cluster_column
                )
                intervals = clustered_intervals(
                    matrices,
                    resamples=args.resamples,
                    seed=args.seed + seed_offset,
                )
                seed_offset += 1
                point = metric_values(
                    corpus_data["label"].to_numpy(dtype=int),
                    corpus_data[prediction_column].to_numpy(dtype=int),
                )
                row: dict[str, object] = {
                    "corpus": corpus,
                    "model": model,
                    "cluster_policy": policy_name,
                    "n": int(len(corpus_data)),
                    "source_clusters": int(len(clusters)),
                    "missing_hostname_n": int(
                        corpus_data["hostname"].eq(MISSING_HOSTNAME).sum()
                    ),
                    "bootstrap_resamples": args.resamples,
                }
                for metric, value in point.items():
                    low, high = intervals[metric]
                    row[metric] = value
                    row[f"{metric}_cluster_bootstrap_95_low"] = low
                    row[f"{metric}_cluster_bootstrap_95_high"] = high
                interval_rows.append(row)

    subgroup_rows: list[dict[str, object]] = []
    for (corpus, status), part in joined.groupby(
        ["corpus", "hostname_status"], sort=True
    ):
        for model, prediction_column in MODEL_COLUMNS.items():
            labels = part["label"].to_numpy(dtype=int)
            predictions = part[prediction_column].to_numpy(dtype=int)
            values = metric_values(labels, predictions)
            subgroup_rows.append(
                {
                    "corpus": corpus,
                    "hostname_status": status,
                    "model": model,
                    "n": int(len(part)),
                    "fake_n": int((labels == 0).sum()),
                    "real_n": int((labels == 1).sum()),
                    "hostname_clusters": int(part["hostname"].nunique()),
                    **values,
                }
            )

    pd.DataFrame(interval_rows).to_csv(
        args.output / "source_clustered_bootstrap_intervals.csv", index=False
    )
    pd.DataFrame(subgroup_rows).to_csv(
        args.output / "hostname_seen_unseen_performance.csv", index=False
    )
    item_columns = [
        "corpus",
        "id",
        "cleaned_title_sha256",
        "label",
        "hostname",
        "fold",
        "hostname_status",
    ]
    joined[item_columns].to_csv(
        args.output / "hostname_seen_unseen_assignments_without_text.csv",
        index=False,
    )
    config = {
        "analysis_status": "post_hoc_sensitivity_analysis",
        "cluster_definition": "normalized source hostname",
        "primary_missing_hostname_policy": (
            "all missing hostnames pooled as one cluster"
        ),
        "secondary_missing_hostname_policy": (
            "each missing-hostname item treated as a unique cluster"
        ),
        "seen_definition": (
            "the item's normalized hostname occurs in at least one fold other "
            "than the item's own five-fold diagnostic fold"
        ),
        "resamples": args.resamples,
        "seed": args.seed,
        "external_predictions": {
            "file": args.external_predictions.name,
            "sha256": sha256_file(args.external_predictions),
        },
        "hostname_predictions": {
            "file": args.hostname_predictions.name,
            "sha256": sha256_file(args.hostname_predictions),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output / "source_clustered_bootstrap_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    primary = pd.DataFrame(interval_rows)
    print(
        primary.loc[
            (primary["cluster_policy"] == "missing_pooled_primary")
            & (primary["model"].isin(["linear_svm", "hostname_only"])),
            [
                "corpus",
                "model",
                "n",
                "source_clusters",
                "balanced_accuracy",
                "balanced_accuracy_cluster_bootstrap_95_low",
                "balanced_accuracy_cluster_bootstrap_95_high",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
