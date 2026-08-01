#!/usr/bin/env python3
"""Reproducible TF-IDF fake-news classification experiments.

This script reconstructs the methodology described in the thesis/article:

* Fake.csv (label 0) and True.csv (label 1)
* title and text are combined and cleaned
* stratified 80/20 train/test split
* TF-IDF is fitted on the training split only
* Multinomial Naive Bayes and Random Forest are evaluated

Important: the original work did not record every random seed, TF-IDF setting,
or model hyperparameter. The explicit defaults below are replication choices;
they must not be presented as the exact undocumented original configuration.

Install dependencies:
    python -m pip install pandas numpy scikit-learn joblib

Example:
    python fake_news_experiments.py \
        --fake Fake.csv --true True.csv --output results --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


LABEL_NAMES = {0: "fake", 1: "real"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible TF-IDF fake-news experiments."
    )
    parser.add_argument("--fake", type=Path, required=True, help="Path to Fake.csv")
    parser.add_argument("--true", type=Path, required=True, help="Path to True.csv")
    parser.add_argument(
        "--output", type=Path, default=Path("results"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--test-size", type=float, default=0.20, help="Test fraction (default: 0.20)"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="Maximum TF-IDF vocabulary size",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        choices=(1, 2),
        default=1,
        help="Use unigrams (1) or unigrams+bigrams (2)",
    )
    parser.add_argument(
        "--min-df", type=int, default=2, help="Minimum document frequency"
    )
    parser.add_argument(
        "--max-df", type=float, default=0.95, help="Maximum document-frequency ratio"
    )
    parser.add_argument(
        "--nb-alpha", type=float, default=1.0, help="MultinomialNB smoothing alpha"
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=200,
        help="Number of Random Forest trees",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop exact duplicate cleaned documents before splitting",
    )
    parser.add_argument(
        "--encoding",
        default="latin-1",
        help="CSV text encoding (uploaded files require latin-1)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Also run stratified k-fold CV; use 5 for the journal experiment",
    )
    parser.add_argument(
        "--text-field",
        choices=("combined", "title", "text"),
        default="combined",
        help="Document field used for modeling",
    )
    parser.add_argument(
        "--mask-reuters",
        action="store_true",
        help="Remove the source token 'reuters' after cleaning for robustness testing",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    """Apply the documented normalization without corpus-level fitting."""
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_news_csv(path: Path, encoding: str) -> tuple[pd.DataFrame, int]:
    """Read the supplied corpus and repair extra unquoted commas deterministically.

    The uploaded True file contains one logical record with eight CSV fields. The
    final two fields still validate as subject and date, so any extra middle
    fields are joined back into the article body. The repair count is reported.
    """
    repaired = 0
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["title", "text", "subject", "date"]:
            raise ValueError(f"Unexpected columns in {path.name}: {header}")
        rows: list[list[str]] = []
        for row in reader:
            if len(row) > 4:
                row = [row[0], ",".join(row[1:-2]), row[-2], row[-1]]
                repaired += 1
            elif len(row) < 4:
                raise ValueError(
                    f"Malformed row ending at physical line {reader.line_num} "
                    f"in {path.name}: expected 4 fields, found {len(row)}"
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=header), repaired


def load_dataset(
    fake_path: Path,
    true_path: Path,
    drop_duplicates: bool,
    encoding: str,
    text_field: str,
    mask_reuters: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    for path in (fake_path, true_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    fake, fake_repairs = read_news_csv(fake_path, encoding)
    true, true_repairs = read_news_csv(true_path, encoding)
    required = {"title", "text"}
    for name, frame in (("Fake.csv", fake), ("True.csv", true)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    fake = fake.copy()
    true = true.copy()
    fake["label"] = 0
    true["label"] = 1
    fake["source_file"] = "Fake.csv"
    true["source_file"] = "True.csv"

    data = pd.concat([fake, true], ignore_index=True)
    data.insert(0, "row_id", np.arange(len(data), dtype=int))
    if text_field == "title":
        selected_text = data["title"].fillna("").astype(str)
    elif text_field == "text":
        selected_text = data["text"].fillna("").astype(str)
    else:
        selected_text = (
            data["title"].fillna("").astype(str)
            + " "
            + data["text"].fillna("").astype(str)
        )
    data["content"] = selected_text.map(clean_text)
    reuters_documents = int(
        data["content"].str.contains(r"\breuters\b", regex=True).sum()
    )
    if mask_reuters:
        data["content"] = (
            data["content"]
            .str.replace(r"\breuters\b", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    empty_count = int(data["content"].eq("").sum())
    data = data.loc[data["content"].ne("")].copy()
    duplicate_count = int(data.duplicated(subset=["content", "label"]).sum())
    if drop_duplicates:
        data = data.drop_duplicates(subset=["content", "label"], keep="first").copy()

    counts = {
        "fake_rows_read": int(len(fake)),
        "real_rows_read": int(len(true)),
        "fake_malformed_rows_repaired": fake_repairs,
        "real_malformed_rows_repaired": true_repairs,
        "empty_documents_removed": empty_count,
        "exact_same_label_duplicates_detected": duplicate_count,
        "documents_containing_reuters_before_masking": reuters_documents,
        "rows_after_cleaning": int(len(data)),
    }
    return data, counts


def build_models(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "multinomial_nb": MultinomialNB(alpha=args.nb_alpha),
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1_000,
            solver="liblinear",
            random_state=args.seed,
        ),
        "linear_svm": LinearSVC(C=1.0, random_state=args.seed),
        "random_forest": RandomForestClassifier(
            n_estimators=args.rf_estimators,
            random_state=args.seed,
            n_jobs=-1,
        ),
    }


def run_cross_validation(
    data: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    models: dict[str, Any],
    folds: int,
    seed: int,
    output_dir: Path,
) -> None:
    if folds < 2:
        return
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    fold_rows: list[dict[str, Any]] = []
    for name, model in models.items():
        print(f"\nRunning {folds}-fold CV for {name} ...")
        pipeline = Pipeline([("tfidf", vectorizer), ("model", model)])
        scores = cross_validate(
            pipeline,
            data["content"],
            data["label"],
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        for fold in range(folds):
            row: dict[str, Any] = {"model": name, "fold": fold + 1}
            for metric in scoring:
                row[metric] = float(scores[f"test_{metric}"][fold])
            row["fit_time_seconds"] = float(scores["fit_time"][fold])
            fold_rows.append(row)

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    summary = fold_table.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_macro_mean=("precision_macro", "mean"),
        precision_macro_std=("precision_macro", "std"),
        recall_macro_mean=("recall_macro", "mean"),
        recall_macro_std=("recall_macro", "std"),
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
    )
    summary.to_csv(output_dir / "cv_summary.csv")
    print("\nCross-validation summary:")
    print(summary.round(4).to_string())


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {"accuracy": float(accuracy_score(y_true, y_pred))}

    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        metrics[f"precision_{average}"] = float(precision)
        metrics[f"recall_{average}"] = float(recall)
        metrics[f"f1_{average}"] = float(f1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    for index, label in enumerate((0, 1)):
        name = LABEL_NAMES[label]
        metrics[f"precision_{name}"] = float(precision[index])
        metrics[f"recall_{name}"] = float(recall[index])
        metrics[f"f1_{name}"] = float(f1[index])
        metrics[f"support_{name}"] = int(support[index])
    return metrics


def evaluate_model(
    name: str,
    model: Any,
    x_train: Any,
    x_test: Any,
    y_train: pd.Series,
    y_test: pd.Series,
    test_row_ids: pd.Series,
    output_dir: Path,
) -> dict[str, float]:
    print(f"\nTraining {name} ...")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = calculate_metrics(y_test, predictions)

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
        digits=4,
        zero_division=0,
    ))

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_fake", "actual_real"],
        columns=["predicted_fake", "predicted_real"],
    ).to_csv(output_dir / f"confusion_matrix_{name}.csv")

    pd.DataFrame({
        "row_id": test_row_ids.to_numpy(),
        "true_label": y_test.to_numpy(),
        "predicted_label": predictions,
    }).to_csv(output_dir / f"predictions_{name}.csv", index=False)

    joblib.dump(model, output_dir / f"model_{name}.joblib")
    return metrics


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1")

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    data, data_counts = load_dataset(
        args.fake,
        args.true,
        args.drop_duplicates,
        args.encoding,
        args.text_field,
        args.mask_reuters,
    )
    print("Dataset summary:", data_counts)
    print("Class counts:", data["label"].value_counts().sort_index().to_dict())

    train, test = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data["label"],
    )

    # Stop-word removal occurs inside the vectorizer. Crucially, vocabulary and
    # IDF weights are learned from the training split only to prevent leakage.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=False,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train["content"])
    x_test = vectorizer.transform(test["content"])
    print(f"TF-IDF shapes: train={x_train.shape}, test={x_test.shape}")

    models = build_models(args)

    all_metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        all_metrics[name] = evaluate_model(
            name,
            model,
            x_train,
            x_test,
            train["label"],
            test["label"],
            test["row_id"],
            args.output,
        )

    metrics_table = pd.DataFrame(all_metrics).T
    metrics_table.to_csv(args.output / "metrics.csv", index_label="model")
    test[["row_id", "source_file", "label"]].sort_values("row_id").to_csv(
        args.output / "test_split_manifest.csv", index=False
    )
    joblib.dump(vectorizer, args.output / "tfidf_vectorizer.joblib")

    run_cross_validation(
        data,
        TfidfVectorizer(
            stop_words="english",
            lowercase=False,
            ngram_range=(1, args.ngram_max),
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        build_models(args),
        args.cv_folds,
        args.seed,
        args.output,
    )

    config = {
        "warning": (
            "These explicit settings are replication choices because the original "
            "work did not document every seed, vectorizer setting, or hyperparameter."
        ),
        "class_mapping": {"0": "fake", "1": "real"},
        "input_files": {
            "fake": {"name": args.fake.name, "sha256": sha256_file(args.fake)},
            "true": {"name": args.true.name, "sha256": sha256_file(args.true)},
        },
        "data_counts": data_counts,
        "split": {
            "method": "stratified holdout",
            "test_size": args.test_size,
            "random_seed": args.seed,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        },
        "preprocessing": {
            "combined_fields": ["title", "text"],
            "lowercase": True,
            "urls_removed": True,
            "non_ascii_letters_removed": True,
            "stop_words": "scikit-learn English list",
            "drop_exact_duplicates": args.drop_duplicates,
            "csv_encoding": args.encoding,
            "text_field": args.text_field,
            "mask_reuters": args.mask_reuters,
        },
        "tfidf": {
            "ngram_range": [1, args.ngram_max],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "max_features": args.max_features,
            "sublinear_tf": True,
            "fit_on_training_only": True,
        },
        "models": {
            "multinomial_nb": {"alpha": args.nb_alpha},
            "logistic_regression": {
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 1000,
                "random_state": args.seed,
            },
            "linear_svm": {"C": 1.0, "random_state": args.seed},
            "random_forest": {
                "n_estimators": args.rf_estimators,
                "random_state": args.seed,
                "n_jobs": -1,
            },
        },
        "cross_validation": {
            "folds": args.cv_folds,
            "stratified": args.cv_folds >= 2,
            "shuffle": args.cv_folds >= 2,
            "random_seed": args.seed,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (args.output / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nFinal metrics:")
    print(metrics_table.round(4).to_string())
    print(f"\nSaved reproducibility artifacts to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
