#!/usr/bin/env python3
"""Equalized nested group-fold tuning for four classical title classifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from external_error_analysis import load_politifact  # noqa: E402
from fake_news_experiments import load_dataset, sha256_file  # noqa: E402


MODEL_NAMES = {
    "nb": "Multinomial NB",
    "lr": "Logistic Regression",
    "svm": "Linear SVM",
    "rf": "Random Forest",
}


@dataclass(frozen=True)
class Config:
    ngram_max: int
    min_df: int
    model_parameter: float | int | None

    def as_dict(self, model: str) -> dict[str, object]:
        parameter_name = {
            "nb": "alpha",
            "lr": "C",
            "svm": "C",
            "rf": "max_depth",
        }[model]
        return {
            "ngram_max": self.ngram_max,
            "min_df": self.min_df,
            parameter_name: self.model_parameter,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isot-fake", type=Path, required=True)
    parser.add_argument("--isot-true", type=Path, required=True)
    parser.add_argument("--politifact-fake", type=Path, required=True)
    parser.add_argument("--politifact-real", type=Path, required=True)
    parser.add_argument("--gossipcop-fake", type=Path, required=True)
    parser.add_argument("--gossipcop-real", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="nb,lr,svm,rf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--rf-parallel-configs", type=int, default=3)
    parser.add_argument("--rf-jobs-per-fit", type=int, default=3)
    parser.add_argument("--isot-encoding", default="utf-8")
    return parser.parse_args()


def cleaned_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parameter_values(model: str) -> list[float | int | None]:
    if model in {"lr", "svm"}:
        return [0.1, 1.0, 10.0]
    if model == "nb":
        return [0.1, 1.0, 10.0]
    if model == "rf":
        return [None, 50, 100]
    raise ValueError(model)


def all_configs(model: str) -> list[Config]:
    return [
        Config(ngram_max, min_df, parameter)
        for ngram_max in (1, 2)
        for min_df in (2, 5)
        for parameter in parameter_values(model)
    ]


def fixed_config(model: str) -> Config:
    return Config(1, 2, None if model == "rf" else 1.0)


def make_vectorizer(config: Config, max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        lowercase=False,
        ngram_range=(1, config.ngram_max),
        min_df=config.min_df,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def make_model(
    model: str,
    parameter: float | int | None,
    seed: int,
    rf_estimators: int,
    rf_jobs: int,
):
    if model == "nb":
        return MultinomialNB(alpha=float(parameter))
    if model == "lr":
        return LogisticRegression(
            C=float(parameter),
            max_iter=1_000,
            solver="liblinear",
            random_state=seed,
        )
    if model == "svm":
        return LinearSVC(C=float(parameter), random_state=seed)
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=parameter,
            random_state=seed,
            n_jobs=rf_jobs,
        )
    raise ValueError(model)


def fit_predict(
    model: str,
    parameter: float | int | None,
    train_matrix,
    train_labels: np.ndarray,
    validation_matrix,
    seed: int,
    rf_estimators: int,
    rf_jobs: int,
) -> np.ndarray:
    estimator = make_model(model, parameter, seed, rf_estimators, rf_jobs)
    estimator.fit(train_matrix, train_labels)
    return estimator.predict(validation_matrix)


def point_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    recalls = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )[1]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "fake_recall": float(recalls[0]),
        "real_recall": float(recalls[1]),
    }


def tie_key(model: str, record: dict[str, object]) -> tuple[object, ...]:
    parameter = record["model_parameter"]
    if model == "rf":
        parameter_default_distance = 0 if parameter is None else 1
        parameter_secondary = 0 if parameter is None else -int(parameter)
    else:
        value = float(parameter)
        parameter_default_distance = abs(np.log10(value))
        parameter_secondary = value
    return (
        -float(record["mean_inner_macro_f1"]),
        float(record["sd_inner_macro_f1"]),
        int(record["ngram_max"] != 1),
        -int(record["min_df"]),
        parameter_default_distance,
        parameter_secondary,
    )


def evaluate_grid(
    stage: str,
    held_out_fold: int | None,
    folds_for_validation: list[int],
    contents: pd.Series,
    labels: np.ndarray,
    folds: np.ndarray,
    models: list[str],
    args: argparse.Namespace,
    records: list[dict[str, object]],
) -> dict[str, Config]:
    score_store: dict[tuple[str, Config], list[float]] = {
        (model, config): [] for model in models for config in all_configs(model)
    }
    for validation_fold in folds_for_validation:
        train_mask = folds != validation_fold
        if held_out_fold is not None:
            train_mask &= folds != held_out_fold
        validation_mask = folds == validation_fold
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        train_labels = labels[train_indices]
        validation_labels = labels[validation_indices]
        print(
            f"{stage} held_out={held_out_fold} validation={validation_fold} "
            f"train={len(train_indices)} validation_n={len(validation_indices)}",
            flush=True,
        )
        for ngram_max in (1, 2):
            for min_df in (2, 5):
                representation = Config(ngram_max, min_df, None)
                vectorizer = make_vectorizer(representation, args.max_features)
                train_matrix = vectorizer.fit_transform(contents.iloc[train_indices])
                validation_matrix = vectorizer.transform(contents.iloc[validation_indices])
                for model in [item for item in models if item != "rf"]:
                    for parameter in parameter_values(model):
                        predictions = fit_predict(
                            model,
                            parameter,
                            train_matrix,
                            train_labels,
                            validation_matrix,
                            args.seed,
                            args.rf_estimators,
                            1,
                        )
                        config = Config(ngram_max, min_df, parameter)
                        score = f1_score(
                            validation_labels, predictions, average="macro"
                        )
                        score_store[(model, config)].append(float(score))
                if "rf" in models:
                    rf_parameters = parameter_values("rf")
                    rf_predictions = Parallel(
                        n_jobs=args.rf_parallel_configs,
                        backend="loky",
                    )(
                        delayed(fit_predict)(
                            "rf",
                            parameter,
                            train_matrix,
                            train_labels,
                            validation_matrix,
                            args.seed,
                            args.rf_estimators,
                            args.rf_jobs_per_fit,
                        )
                        for parameter in rf_parameters
                    )
                    for parameter, predictions in zip(rf_parameters, rf_predictions):
                        config = Config(ngram_max, min_df, parameter)
                        score = f1_score(
                            validation_labels, predictions, average="macro"
                        )
                        score_store[("rf", config)].append(float(score))
    selections: dict[str, Config] = {}
    for model in models:
        model_records: list[dict[str, object]] = []
        for config in all_configs(model):
            scores = score_store[(model, config)]
            row: dict[str, object] = {
                "stage": stage,
                "held_out_fold": held_out_fold,
                "model": model,
                "model_name": MODEL_NAMES[model],
                "ngram_max": config.ngram_max,
                "min_df": config.min_df,
                "model_parameter": config.model_parameter,
                "mean_inner_macro_f1": float(np.mean(scores)),
                "sd_inner_macro_f1": float(np.std(scores, ddof=1)),
                "validation_folds": ",".join(map(str, folds_for_validation)),
            }
            for index, score in enumerate(scores, start=1):
                row[f"fold_score_{index}"] = score
            records.append(row)
            model_records.append(row)
        selected_record = min(model_records, key=lambda row: tie_key(model, row))
        selected = Config(
            int(selected_record["ngram_max"]),
            int(selected_record["min_df"]),
            selected_record["model_parameter"],
        )
        selections[model] = selected
        print(
            f"selected {stage} held_out={held_out_fold} model={model}: "
            f"{selected.as_dict(model)} macro_f1={selected_record['mean_inner_macro_f1']:.6f}",
            flush=True,
        )
    return selections


def evaluate_config(
    model: str,
    config: Config,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    contents: pd.Series,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    vectorizer = make_vectorizer(config, args.max_features)
    train_matrix = vectorizer.fit_transform(contents.iloc[train_indices])
    test_matrix = vectorizer.transform(contents.iloc[test_indices])
    return fit_predict(
        model,
        config.model_parameter,
        train_matrix,
        labels[train_indices],
        test_matrix,
        args.seed,
        args.rf_estimators,
        -1 if model == "rf" else 1,
    )


def mcnemar_record(
    scope: str,
    model: str,
    labels: np.ndarray,
    fixed_predictions: np.ndarray,
    selected_predictions: np.ndarray,
) -> dict[str, object]:
    fixed_correct = fixed_predictions == labels
    selected_correct = selected_predictions == labels
    selected_only = int(np.sum(selected_correct & ~fixed_correct))
    fixed_only = int(np.sum(fixed_correct & ~selected_correct))
    discordant = selected_only + fixed_only
    p_value = (
        float(binomtest(min(selected_only, fixed_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "scope": scope,
        "model": model,
        "model_name": MODEL_NAMES[model],
        "selected_only_correct": selected_only,
        "fixed_only_correct": fixed_only,
        "discordant": discordant,
        "exact_two_sided_mcnemar_p": p_value,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = set(models).difference(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    started = time.time()
    title_data, title_counts = load_dataset(
        args.isot_fake,
        args.isot_true,
        drop_duplicates=True,
        encoding=args.isot_encoding,
        text_field="title",
        mask_reuters=False,
    )
    assignments = pd.read_csv(args.fold_assignments)
    if len(title_data) != 38_681 or len(assignments) != len(title_data):
        raise ValueError("Unexpected ISOT title or fold-assignment count")
    audit = pd.DataFrame(
        {
            "row_id": title_data["row_id"].to_numpy(dtype=int),
            "cleaned_title_sha256": title_data["content"].map(cleaned_hash),
            "label": title_data["label"].to_numpy(dtype=int),
        }
    ).merge(assignments, on=["row_id", "cleaned_title_sha256", "label"], how="left")
    if audit["outer_fold"].isna().any():
        raise ValueError("Fold assignments do not match cleaned ISOT titles")
    folds = audit["outer_fold"].to_numpy(dtype=int)
    labels = title_data["label"].to_numpy(dtype=int)
    contents = title_data["content"].reset_index(drop=True)

    politifact, politifact_counts = load_politifact(
        args.politifact_fake, args.politifact_real
    )
    gossipcop, gossipcop_counts = load_politifact(
        args.gossipcop_fake, args.gossipcop_real
    )
    if len(politifact) != 975 or len(gossipcop) != 20_587:
        raise ValueError("Unexpected external cleaned count")

    grid_records: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []
    outer_metric_records: list[dict[str, object]] = []
    oof_records: list[dict[str, object]] = []
    mcnemar_records: list[dict[str, object]] = []
    pooled_fixed = {model: np.empty(len(labels), dtype=int) for model in models}
    pooled_selected = {model: np.empty(len(labels), dtype=int) for model in models}

    for outer_fold in range(1, 6):
        inner_folds = [fold for fold in range(1, 6) if fold != outer_fold]
        selected = evaluate_grid(
            "nested_inner",
            outer_fold,
            inner_folds,
            contents,
            labels,
            folds,
            models,
            args,
            grid_records,
        )
        train_indices = np.flatnonzero(folds != outer_fold)
        test_indices = np.flatnonzero(folds == outer_fold)
        for model in models:
            for pipeline, config in (
                ("fixed", fixed_config(model)),
                ("selected", selected[model]),
            ):
                predictions = evaluate_config(
                    model,
                    config,
                    train_indices,
                    test_indices,
                    contents,
                    labels,
                    args,
                )
                if pipeline == "fixed":
                    pooled_fixed[model][test_indices] = predictions
                else:
                    pooled_selected[model][test_indices] = predictions
                metrics = point_metrics(labels[test_indices], predictions)
                outer_metric_records.append(
                    {
                        "outer_fold": outer_fold,
                        "model": model,
                        "model_name": MODEL_NAMES[model],
                        "pipeline": pipeline,
                        **config.as_dict(model),
                        **metrics,
                    }
                )
                if pipeline == "selected":
                    selection_records.append(
                        {
                            "stage": "nested_outer",
                            "outer_fold": outer_fold,
                            "model": model,
                            "model_name": MODEL_NAMES[model],
                            **config.as_dict(model),
                        }
                    )
        print(f"completed outer fold {outer_fold}", flush=True)

    for model in models:
        mcnemar_records.append(
            mcnemar_record(
                "pooled_nested_outer",
                model,
                labels,
                pooled_fixed[model],
                pooled_selected[model],
            )
        )
        for index in range(len(labels)):
            oof_records.append(
                {
                    "row_id": int(title_data.iloc[index]["row_id"]),
                    "cleaned_title_sha256": cleaned_hash(contents.iloc[index]),
                    "label": int(labels[index]),
                    "outer_fold": int(folds[index]),
                    "model": model,
                    "fixed_prediction": int(pooled_fixed[model][index]),
                    "selected_prediction": int(pooled_selected[model][index]),
                }
            )

    global_selected = evaluate_grid(
        "global_group_cv",
        None,
        [1, 2, 3, 4, 5],
        contents,
        labels,
        folds,
        models,
        args,
        grid_records,
    )

    external_metric_records: list[dict[str, object]] = []
    external_prediction_records: list[dict[str, object]] = []
    all_train_indices = np.arange(len(labels), dtype=int)
    for corpus_name, external in (
        ("PolitiFact", politifact),
        ("GossipCop", gossipcop),
    ):
        external_labels = external["label"].to_numpy(dtype=int)
        for model in models:
            predictions_by_pipeline: dict[str, np.ndarray] = {}
            for pipeline, config in (
                ("fixed", fixed_config(model)),
                ("selected", global_selected[model]),
            ):
                vectorizer = make_vectorizer(config, args.max_features)
                train_matrix = vectorizer.fit_transform(contents)
                external_matrix = vectorizer.transform(external["content"])
                predictions = fit_predict(
                    model,
                    config.model_parameter,
                    train_matrix,
                    labels,
                    external_matrix,
                    args.seed,
                    args.rf_estimators,
                    -1 if model == "rf" else 1,
                )
                predictions_by_pipeline[pipeline] = predictions
                external_metric_records.append(
                    {
                        "corpus": corpus_name,
                        "model": model,
                        "model_name": MODEL_NAMES[model],
                        "pipeline": pipeline,
                        **config.as_dict(model),
                        **point_metrics(external_labels, predictions),
                    }
                )
            mcnemar_records.append(
                mcnemar_record(
                    corpus_name,
                    model,
                    external_labels,
                    predictions_by_pipeline["fixed"],
                    predictions_by_pipeline["selected"],
                )
            )
            selection_records.append(
                {
                    "stage": "global_group_cv",
                    "outer_fold": None,
                    "model": model,
                    "model_name": MODEL_NAMES[model],
                    **global_selected[model].as_dict(model),
                }
            )
            for index in range(len(external)):
                external_prediction_records.append(
                    {
                        "corpus": corpus_name,
                        "id": str(external.iloc[index]["id"]),
                        "cleaned_title_sha256": cleaned_hash(
                            external.iloc[index]["content"]
                        ),
                        "label": int(external_labels[index]),
                        "model": model,
                        "fixed_prediction": int(
                            predictions_by_pipeline["fixed"][index]
                        ),
                        "selected_prediction": int(
                            predictions_by_pipeline["selected"][index]
                        ),
                    }
                )

    pd.DataFrame(grid_records).to_csv(
        args.output / "equalized_tuning_grid_scores.csv", index=False
    )
    pd.DataFrame(selection_records).drop_duplicates().to_csv(
        args.output / "equalized_tuning_selected_configs.csv", index=False
    )
    outer_metrics = pd.DataFrame(outer_metric_records)
    outer_metrics.to_csv(
        args.output / "equalized_tuning_outer_fold_metrics.csv", index=False
    )
    summary = (
        outer_metrics.groupby(["model", "model_name", "pipeline"], sort=False)[
            ["accuracy", "balanced_accuracy", "macro_f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(filter(None, map(str, column))).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(
        args.output / "equalized_tuning_nested_summary.csv", index=False
    )
    pd.DataFrame(oof_records).to_csv(
        args.output / "equalized_tuning_oof_predictions_without_text.csv", index=False
    )
    pd.DataFrame(external_metric_records).to_csv(
        args.output / "equalized_tuning_external_metrics.csv", index=False
    )
    pd.DataFrame(external_prediction_records).to_csv(
        args.output / "equalized_tuning_external_predictions_without_text.csv",
        index=False,
    )
    pd.DataFrame(mcnemar_records).to_csv(
        args.output / "equalized_tuning_mcnemar.csv", index=False
    )

    config = {
        "analysis_status": "post-hoc equalized tuning sensitivity",
        "selection_metric": "mean inner macro-F1",
        "representations": [
            {"ngram_max": ngram_max, "min_df": min_df}
            for ngram_max in (1, 2)
            for min_df in (2, 5)
        ],
        "model_grids": {
            "nb_alpha": [0.1, 1.0, 10.0],
            "lr_C": [0.1, 1.0, 10.0],
            "svm_C": [0.1, 1.0, 10.0],
            "rf_max_depth": [None, 50, 100],
            "rf_estimators_fixed": args.rf_estimators,
        },
        "configurations_per_model": 12,
        "outer_folds": 5,
        "inner_rotations_per_outer_fold": 4,
        "counts": {
            "isot_titles": len(title_data),
            "politifact_titles": len(politifact),
            "gossipcop_titles": len(gossipcop),
        },
        "cleaning": {
            "isot": title_counts,
            "politifact": politifact_counts,
            "gossipcop": gossipcop_counts,
        },
        "input_hashes": {
            args.isot_fake.name: sha256_file(args.isot_fake),
            args.isot_true.name: sha256_file(args.isot_true),
            args.politifact_fake.name: sha256_file(args.politifact_fake),
            args.politifact_real.name: sha256_file(args.politifact_real),
            args.gossipcop_fake.name: sha256_file(args.gossipcop_fake),
            args.gossipcop_real.name: sha256_file(args.gossipcop_real),
            args.fold_assignments.name: sha256_file(args.fold_assignments),
        },
        "runtime_seconds": time.time() - started,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output / "equalized_tuning_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(external_metric_records).to_string(index=False), flush=True)
    print(pd.DataFrame(mcnemar_records).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
