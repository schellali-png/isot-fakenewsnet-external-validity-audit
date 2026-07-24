#!/usr/bin/env python3
"""Locked Linear-SVM error analysis for ISOT-to-FakeNewsNet–PolitiFact transfer.

The analysis is descriptive and uses fixed, predeclared bins:

* cleaned-title length: 1–5, 6–10, 11–15, and >=16 words;
* ISOT-vocabulary coverage: 0, (0, 0.50], (0.50, 0.75], and (0.75, 1];
* absolute Linear-SVM decision margin: <0.5, [0.5, 1), [1, 2), and >=2.

PolitiFact is never used for fitting or parameter selection. The output does not
redistribute article titles; records are identified by their supplied IDs.
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
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from fake_news_experiments import clean_text, load_dataset, sha256_file  # noqa: E402


LENGTH_ORDER = ["1–5", "6–10", "11–15", "≥16"]
COVERAGE_ORDER = ["0", "(0, 0.50]", "(0.50, 0.75]", "(0.75, 1]"]
MARGIN_ORDER = ["<0.5", "[0.5, 1)", "[1, 2)", "≥2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isot-fake", type=Path, required=True)
    parser.add_argument("--isot-true", type=Path, required=True)
    parser.add_argument("--politifact-fake", type=Path, required=True)
    parser.add_argument("--politifact-real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isot-encoding", default="utf-8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--minimum-domain-size", type=int, default=10)
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return centre - half, centre + half


def source_domain(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "[missing]"
    raw = str(value).strip()
    archive = re.search(
        r"web\.archive\.org/web/[^/]+/(?:https?://)?([^/]+)", raw, flags=re.I
    )
    if archive:
        host = archive.group(1)
    else:
        parsed = urlparse(raw if "://" in raw else "//" + raw)
        host = parsed.hostname or "[unparsed]"
    host = host.lower().split(":", 1)[0].strip(".")
    return re.sub(r"^www\.", "", host) or "[unparsed]"


def load_politifact(fake_path: Path, real_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    fake = pd.read_csv(fake_path).copy()
    real = pd.read_csv(real_path).copy()
    required = {"id", "news_url", "title"}
    for path, frame in ((fake_path, fake), (real_path, real)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    fake["label"] = 0
    real["label"] = 1
    data = pd.concat([fake, real], ignore_index=True)
    data["content"] = data["title"].map(clean_text)
    empty = int(data["content"].eq("").sum())
    data = data.loc[data["content"].ne("")].copy()

    same_label_duplicates = int(data.duplicated(["content", "label"]).sum())
    data = data.drop_duplicates(["content", "label"], keep="first").copy()
    conflicting_titles = (
        data.groupby("content", sort=False)["label"].nunique().loc[lambda x: x > 1].index
    )
    conflicting_rows = int(data["content"].isin(conflicting_titles).sum())
    data = data.loc[~data["content"].isin(conflicting_titles)].copy()
    data["domain"] = data["news_url"].map(source_domain)
    data = data.reset_index(drop=True)
    counts = {
        "raw_fake": int(len(fake)),
        "raw_real": int(len(real)),
        "raw_total": int(len(fake) + len(real)),
        "empty_titles_removed": empty,
        "same_label_duplicate_rows_removed": same_label_duplicates,
        "conflicting_clean_titles": int(len(conflicting_titles)),
        "conflicting_rows_removed_after_same_label_deduplication": conflicting_rows,
        "final_fake": int(data["label"].eq(0).sum()),
        "final_real": int(data["label"].eq(1).sum()),
        "final_total": int(len(data)),
    }
    return data, counts


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


def length_bin(words: int) -> str:
    if words <= 5:
        return "1–5"
    if words <= 10:
        return "6–10"
    if words <= 15:
        return "11–15"
    return "≥16"


def coverage_bin(value: float) -> str:
    if value == 0:
        return "0"
    if value <= 0.50:
        return "(0, 0.50]"
    if value <= 0.75:
        return "(0.50, 0.75]"
    return "(0.75, 1]"


def margin_bin(value: float) -> str:
    if value < 0.5:
        return "<0.5"
    if value < 1:
        return "[0.5, 1)"
    if value < 2:
        return "[1, 2)"
    return "≥2"


def grouped_performance(data: pd.DataFrame, column: str, order: list[str] | None = None) -> pd.DataFrame:
    values = order if order is not None else sorted(data[column].dropna().unique().tolist())
    rows: list[dict[str, object]] = []
    for value in values:
        group = data.loc[data[column].eq(value)]
        if group.empty:
            continue
        correct = int(group["correct"].sum())
        lower, upper = wilson_interval(correct, len(group))
        labels = group["label"].to_numpy()
        predictions = group["prediction"].to_numpy()
        balanced = (
            float(balanced_accuracy_score(labels, predictions))
            if set(labels) == {0, 1}
            else math.nan
        )
        rows.append(
            {
                "dimension": column,
                "group": value,
                "n": int(len(group)),
                "fake_n": int(group["label"].eq(0).sum()),
                "real_n": int(group["label"].eq(1).sum()),
                "correct": correct,
                "errors": int(len(group) - correct),
                "accuracy": correct / len(group),
                "accuracy_wilson_95_low": lower,
                "accuracy_wilson_95_high": upper,
                "balanced_accuracy": balanced,
                "median_vocabulary_coverage": float(group["vocabulary_coverage"].median()),
                "median_absolute_margin": float(group["absolute_margin"].median()),
            }
        )
    return pd.DataFrame(rows)


def distribution_summary(name: str, data: pd.DataFrame) -> dict[str, object]:
    return {
        "corpus": name,
        "n": int(len(data)),
        "word_count_median": float(data["word_count"].median()),
        "word_count_q1": float(data["word_count"].quantile(0.25)),
        "word_count_q3": float(data["word_count"].quantile(0.75)),
        "model_token_count_median": float(data["model_token_count"].median()),
        "vector_nonzero_features_median": float(data["vector_nnz"].median()),
        "zero_vector_titles": int(data["vector_nnz"].eq(0).sum()),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    isot, isot_counts = load_dataset(
        args.isot_fake,
        args.isot_true,
        drop_duplicates=True,
        encoding=args.isot_encoding,
        text_field="title",
        mask_reuters=False,
    )
    external, external_counts = load_politifact(
        args.politifact_fake, args.politifact_real
    )
    if len(isot) != 38_681:
        raise ValueError(f"Expected 38,681 deduplicated ISOT titles, found {len(isot):,}")
    if external_counts["final_total"] != 975:
        raise ValueError(f"Expected 975 cleaned PolitiFact titles, found {len(external):,}")

    vectorizer = make_vectorizer(args.max_features)
    x_train = vectorizer.fit_transform(isot["content"])
    x_external = vectorizer.transform(external["content"])
    model = LinearSVC(C=1.0, random_state=args.seed)
    model.fit(x_train, isot["label"].to_numpy())
    decision = model.decision_function(x_external)
    prediction = (decision >= 0).astype(int)

    analyzer = vectorizer.build_analyzer()
    vocabulary = vectorizer.vocabulary_
    model_tokens = external["content"].map(analyzer)
    external["word_count"] = external["content"].str.split().map(len)
    external["model_token_count"] = model_tokens.map(len)
    external["in_vocabulary_token_count"] = model_tokens.map(
        lambda tokens: sum(token in vocabulary for token in tokens)
    )
    external["vocabulary_coverage"] = np.where(
        external["model_token_count"].gt(0),
        external["in_vocabulary_token_count"] / external["model_token_count"],
        0.0,
    )
    external["vector_nnz"] = np.diff(x_external.indptr)
    external["prediction"] = prediction
    external["decision_score_real_positive"] = decision
    external["absolute_margin"] = np.abs(decision)
    external["correct"] = external["label"].eq(external["prediction"])
    external["length_bin"] = external["word_count"].map(length_bin)
    external["coverage_bin"] = external["vocabulary_coverage"].map(coverage_bin)
    external["margin_bin"] = external["absolute_margin"].map(margin_bin)

    isot_tokens = isot["content"].map(analyzer)
    isot["word_count"] = isot["content"].str.split().map(len)
    isot["model_token_count"] = isot_tokens.map(len)
    isot["vector_nnz"] = np.diff(x_train.indptr)

    overall_accuracy = float(accuracy_score(external["label"], prediction))
    overall_balanced = float(balanced_accuracy_score(external["label"], prediction))
    matrix = confusion_matrix(external["label"], prediction, labels=[0, 1])
    if not math.isclose(overall_accuracy, 0.6758974358974359, abs_tol=1e-12):
        raise ValueError(f"Locked SVM accuracy mismatch: {overall_accuracy}")

    performance = pd.concat(
        [
            grouped_performance(external, "label", [0, 1]),
            grouped_performance(external, "length_bin", LENGTH_ORDER),
            grouped_performance(external, "coverage_bin", COVERAGE_ORDER),
            grouped_performance(external, "margin_bin", MARGIN_ORDER),
        ],
        ignore_index=True,
    )
    domain_counts = external["domain"].value_counts()
    eligible_domains = domain_counts.loc[domain_counts >= args.minimum_domain_size].index
    domain_performance = grouped_performance(
        external.loc[external["domain"].isin(eligible_domains)], "domain"
    ).sort_values(["n", "group"], ascending=[False, True])

    prediction_columns = [
        "id",
        "label",
        "prediction",
        "correct",
        "decision_score_real_positive",
        "absolute_margin",
        "word_count",
        "model_token_count",
        "in_vocabulary_token_count",
        "vocabulary_coverage",
        "vector_nnz",
        "length_bin",
        "coverage_bin",
        "margin_bin",
        "domain",
    ]
    external[prediction_columns].to_csv(
        args.output / "external_svm_prediction_audit.csv", index=False
    )
    external.loc[~external["correct"], prediction_columns].sort_values(
        "absolute_margin", ascending=False
    ).head(50).to_csv(args.output / "highest_margin_errors.csv", index=False)
    performance.to_csv(args.output / "external_error_groups.csv", index=False)
    domain_performance.to_csv(args.output / "source_domain_performance_min10.csv", index=False)
    pd.DataFrame(
        [distribution_summary("ISOT training titles", isot), distribution_summary("PolitiFact test titles", external)]
    ).to_csv(args.output / "corpus_title_distribution.csv", index=False)
    pd.DataFrame(
        matrix,
        index=["actual_fake", "actual_real"],
        columns=["predicted_fake", "predicted_real"],
    ).to_csv(args.output / "external_svm_confusion_matrix.csv")

    correct = external.loc[external["correct"]]
    errors = external.loc[~external["correct"]]
    summary = {
        "protocol_status": "locked; PolitiFact not used for fitting or tuning",
        "inputs": {
            path.name: sha256_file(path)
            for path in (
                args.isot_fake,
                args.isot_true,
                args.politifact_fake,
                args.politifact_real,
            )
        },
        "counts": {"isot": isot_counts, "politifact": external_counts},
        "model": {
            "vectorizer_vocabulary_size": int(len(vocabulary)),
            "linear_svm_C": 1.0,
            "seed": args.seed,
        },
        "overall": {
            "accuracy": overall_accuracy,
            "balanced_accuracy": overall_balanced,
            "fake_recall": float(matrix[0, 0] / matrix[0].sum()),
            "real_recall": float(matrix[1, 1] / matrix[1].sum()),
            "correct": int(external["correct"].sum()),
            "errors": int((~external["correct"]).sum()),
            "confusion_matrix_labels_fake_real": matrix.tolist(),
        },
        "coverage_and_margin": {
            "zero_vector_titles": int(external["vector_nnz"].eq(0).sum()),
            "median_vocabulary_coverage_all": float(external["vocabulary_coverage"].median()),
            "median_vocabulary_coverage_correct": float(correct["vocabulary_coverage"].median()),
            "median_vocabulary_coverage_error": float(errors["vocabulary_coverage"].median()),
            "median_absolute_margin_correct": float(correct["absolute_margin"].median()),
            "median_absolute_margin_error": float(errors["absolute_margin"].median()),
        },
        "source_domains": {
            "unique_domains_including_missing": int(external["domain"].nunique()),
            "missing_url_rows": int(external["domain"].eq("[missing]").sum()),
            "domains_with_at_least_10_records": int(len(eligible_domains)),
        },
        "fixed_bins": {
            "length": LENGTH_ORDER,
            "coverage": COVERAGE_ORDER,
            "absolute_margin": MARGIN_ORDER,
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
    (args.output / "external_error_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["overall"], indent=2))
    print(json.dumps(summary["coverage_and_margin"], indent=2))


if __name__ == "__main__":
    main()
