#!/usr/bin/env python3
"""Held-out ISOT-to-PolitiFact/GossipCop title-only external evaluation.

The target is explicitly the supplied binary corpus label, not claim-level
factual truth. Neither external subset is used for fitting or hyperparameter
choice in the implemented workflow. The accompanying specification is
retrospective and is not evidence of preregistration or pre-outcome freezing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from external_error_analysis import load_politifact  # noqa: E402
from fake_news_experiments import load_dataset, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isot-fake", type=Path, required=True)
    parser.add_argument("--isot-true", type=Path, required=True)
    parser.add_argument("--politifact-fake", type=Path, required=True)
    parser.add_argument("--politifact-real", type=Path, required=True)
    parser.add_argument("--gossipcop-fake", type=Path, required=True)
    parser.add_argument("--gossipcop-real", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
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


def make_models(seed: int) -> dict[str, object]:
    return {
        "multinomial_nb": MultinomialNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1_000,
            solver="liblinear",
            random_state=seed,
        ),
        "linear_svm": LinearSVC(C=1.0, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def metrics_from_counts(
    correct_fake: np.ndarray,
    correct_real: np.ndarray,
    fake_n: int,
    real_n: int,
) -> dict[str, np.ndarray]:
    fake_recall = correct_fake / fake_n
    real_recall = correct_real / real_n
    wrong_fake = fake_n - correct_fake
    wrong_real = real_n - correct_real
    fake_precision = safe_divide(correct_fake, correct_fake + wrong_real)
    real_precision = safe_divide(correct_real, correct_real + wrong_fake)
    fake_f1 = safe_divide(2 * fake_precision * fake_recall, fake_precision + fake_recall)
    real_f1 = safe_divide(2 * real_precision * real_recall, real_precision + real_recall)
    return {
        "accuracy": (correct_fake + correct_real) / (fake_n + real_n),
        "balanced_accuracy": (fake_recall + real_recall) / 2,
        "macro_f1": (fake_f1 + real_f1) / 2,
        "fake_recall": fake_recall,
        "real_recall": real_recall,
    }


def point_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    recalls = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )[1]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "fake_recall": float(recalls[0]),
        "real_recall": float(recalls[1]),
    }


def bootstrap_intervals(
    labels: np.ndarray,
    predictions: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fake_n = int(matrix[0].sum())
    real_n = int(matrix[1].sum())
    fake_correct_probability = matrix[0, 0] / fake_n
    real_correct_probability = matrix[1, 1] / real_n
    correct_fake = rng.binomial(fake_n, fake_correct_probability, size=resamples)
    correct_real = rng.binomial(real_n, real_correct_probability, size=resamples)
    distributions = metrics_from_counts(correct_fake, correct_real, fake_n, real_n)
    return {
        metric: tuple(np.percentile(values, [2.5, 97.5]).astype(float))
        for metric, values in distributions.items()
    }


def metric_record(
    corpus: str,
    model: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    values = point_metrics(labels, predictions)
    intervals = bootstrap_intervals(labels, predictions, resamples, rng)
    row: dict[str, object] = {
        "corpus": corpus,
        "model": model,
        "n": int(len(labels)),
        "fake_n": int(np.sum(labels == 0)),
        "real_n": int(np.sum(labels == 1)),
    }
    row.update(values)
    for metric, (low, high) in intervals.items():
        row[f"{metric}_bootstrap_95_low"] = low
        row[f"{metric}_bootstrap_95_high"] = high
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    row.update(
        {
            "true_fake_pred_fake": int(matrix[0, 0]),
            "true_fake_pred_real": int(matrix[0, 1]),
            "true_real_pred_fake": int(matrix[1, 0]),
            "true_real_pred_real": int(matrix[1, 1]),
        }
    )
    return row


def cross_corpus_bootstrap_difference(
    first_labels: np.ndarray,
    first_prediction: np.ndarray,
    second_labels: np.ndarray,
    second_prediction: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    def draws(labels: np.ndarray, predictions: np.ndarray) -> dict[str, np.ndarray]:
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        fake_n, real_n = int(matrix[0].sum()), int(matrix[1].sum())
        correct_fake = rng.binomial(fake_n, matrix[0, 0] / fake_n, size=resamples)
        correct_real = rng.binomial(real_n, matrix[1, 1] / real_n, size=resamples)
        return metrics_from_counts(correct_fake, correct_real, fake_n, real_n)

    first_draws = draws(first_labels, first_prediction)
    second_draws = draws(second_labels, second_prediction)
    first_point = point_metrics(first_labels, first_prediction)
    second_point = point_metrics(second_labels, second_prediction)
    rows: list[dict[str, object]] = []
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        difference = second_draws[metric] - first_draws[metric]
        low, high = np.percentile(difference, [2.5, 97.5])
        rows.append(
            {
                "comparison": "GossipCop minus PolitiFact",
                "model": "linear_svm",
                "metric": metric,
                "observed_difference": second_point[metric] - first_point[metric],
                "bootstrap_resamples": resamples,
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
            }
        )
    return rows


def cleaned_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("analysis_status") != "held-out second external evaluation":
        raise ValueError(
            "The supplied file is not the held-out GossipCop evaluation specification"
        )

    isot, isot_counts = load_dataset(
        args.isot_fake,
        args.isot_true,
        drop_duplicates=True,
        encoding=args.isot_encoding,
        text_field="title",
        mask_reuters=False,
    )
    politifact, politifact_counts = load_politifact(
        args.politifact_fake, args.politifact_real
    )
    gossipcop, gossipcop_counts = load_politifact(
        args.gossipcop_fake, args.gossipcop_real
    )
    if len(isot) != 38_681:
        raise ValueError(f"Expected 38,681 ISOT titles, found {len(isot):,}")
    if len(politifact) != 975:
        raise ValueError(f"Expected 975 PolitiFact titles, found {len(politifact):,}")

    corpora = {"PolitiFact": politifact, "GossipCop": gossipcop}
    audit_rows: list[dict[str, object]] = []
    for corpus, counts in (
        ("PolitiFact", politifact_counts),
        ("GossipCop", gossipcop_counts),
    ):
        audit_rows.append({"corpus": corpus, **counts})
    pd.DataFrame(audit_rows).to_csv(args.output / "external_cleaning_audit.csv", index=False)

    overlap_rows = []
    for first_name, first_data, second_name, second_data in (
        ("ISOT", isot, "PolitiFact", politifact),
        ("ISOT", isot, "GossipCop", gossipcop),
        ("PolitiFact", politifact, "GossipCop", gossipcop),
    ):
        first_titles = set(first_data["content"])
        second_titles = set(second_data["content"])
        overlap_rows.append(
            {
                "first_corpus": first_name,
                "second_corpus": second_name,
                "exact_cleaned_title_overlap": len(first_titles & second_titles),
                "first_unique_titles": len(first_titles),
                "second_unique_titles": len(second_titles),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(args.output / "cross_corpus_overlap_audit.csv", index=False)

    vectorizer = make_vectorizer(args.max_features)
    x_train = vectorizer.fit_transform(isot["content"])
    external_matrices = {
        name: vectorizer.transform(frame["content"]) for name, frame in corpora.items()
    }
    labels_train = isot["label"].to_numpy()
    models = make_models(args.seed)
    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    svm_predictions: dict[str, np.ndarray] = {}

    for corpus, frame in corpora.items():
        prediction_frames[corpus] = pd.DataFrame(
            {
                "corpus": corpus,
                "id": frame["id"].astype(str),
                "cleaned_title_sha256": frame["content"].map(cleaned_hash),
                "label": frame["label"].astype(int),
                "vector_nonzero_features": external_matrices[corpus].getnnz(axis=1),
            }
        )

    for model_name, model in models.items():
        model.fit(x_train, labels_train)
        for corpus, frame in corpora.items():
            prediction = model.predict(external_matrices[corpus]).astype(int)
            labels = frame["label"].to_numpy()
            metric_rows.append(
                metric_record(
                    corpus,
                    model_name,
                    labels,
                    prediction,
                    args.bootstrap_resamples,
                    rng,
                )
            )
            prediction_frames[corpus][f"{model_name}_prediction"] = prediction
            if model_name == "linear_svm":
                svm_predictions[corpus] = prediction

    for corpus, frame in corpora.items():
        labels = frame["label"].to_numpy()
        majority_label = int(np.mean(labels) >= 0.5)
        prediction = np.full(len(labels), majority_label, dtype=int)
        metric_rows.append(
            metric_record(
                corpus,
                "majority_class_reference",
                labels,
                prediction,
                args.bootstrap_resamples,
                rng,
            )
        )
        prediction_frames[corpus]["majority_class_prediction"] = prediction

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "two_external_corpus_metrics.csv", index=False)
    pd.concat(prediction_frames.values(), ignore_index=True).to_csv(
        args.output / "external_predictions_without_text.csv", index=False
    )

    difference_rows = cross_corpus_bootstrap_difference(
        politifact["label"].to_numpy(),
        svm_predictions["PolitiFact"],
        gossipcop["label"].to_numpy(),
        svm_predictions["GossipCop"],
        args.bootstrap_resamples,
        rng,
    )
    pd.DataFrame(difference_rows).to_csv(
        args.output / "svm_cross_corpus_difference.csv", index=False
    )

    gossip_labels = gossipcop["label"].to_numpy()
    gossip_svm = svm_predictions["GossipCop"]
    gossip_majority = np.full(len(gossip_labels), int(np.mean(gossip_labels) >= 0.5))
    svm_only = int(np.sum((gossip_svm == gossip_labels) & (gossip_majority != gossip_labels)))
    majority_only = int(np.sum((gossip_svm != gossip_labels) & (gossip_majority == gossip_labels)))
    pd.DataFrame(
        [
            {
                "comparison": "Linear SVM vs majority-class reference on GossipCop",
                "svm_only_correct": svm_only,
                "majority_only_correct": majority_only,
                "discordant_total": svm_only + majority_only,
                "exact_two_sided_mcnemar_p": float(
                    binomtest(
                        min(svm_only, majority_only),
                        svm_only + majority_only,
                        0.5,
                        alternative="two-sided",
                    ).pvalue
                ),
            }
        ]
    ).to_csv(args.output / "gossipcop_svm_vs_majority_mcnemar.csv", index=False)

    config = {
        "protocol": protocol,
        "counts": {
            "isot_titles": int(len(isot)),
            "politifact_titles": int(len(politifact)),
            "gossipcop_titles": int(len(gossipcop)),
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "isot_cleaning": isot_counts,
        "external_cleaning": {
            "PolitiFact": politifact_counts,
            "GossipCop": gossipcop_counts,
        },
        "input_hashes": {
            path.name: sha256_file(path)
            for path in (
                args.isot_fake,
                args.isot_true,
                args.politifact_fake,
                args.politifact_real,
                args.gossipcop_fake,
                args.gossipcop_real,
                args.protocol,
            )
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
    (args.output / "second_external_validity_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    primary = metrics.loc[
        (metrics["corpus"] == "GossipCop") & (metrics["model"] == "linear_svm")
    ].iloc[0]
    print("GossipCop cleaned titles:", len(gossipcop))
    print("Primary Linear SVM balanced accuracy:", primary["balanced_accuracy"])
    print("Primary Linear SVM accuracy:", primary["accuracy"])
    print("Primary Linear SVM macro-F1:", primary["macro_f1"])


if __name__ == "__main__":
    main()
