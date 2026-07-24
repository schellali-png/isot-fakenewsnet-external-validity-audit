#!/usr/bin/env python3
"""Reproduce the post hoc source-hostname diagnostics in the revised manuscript.

The diagnostic uses only the supplied FakeNewsNet URL field and corpus label.
It does not use article titles as predictors and must not be interpreted as a
causal publisher effect or as source-disjoint external validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

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
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fake_news_experiments import clean_text, sha256_file  # noqa: E402


MISSING_HOSTNAME = "[missing]"
LABEL_NAMES = {0: "fake", 1: "real"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--politifact-fake", type=Path, required=True)
    parser.add_argument("--politifact-real", type=Path, required=True)
    parser.add_argument("--gossipcop-fake", type=Path, required=True)
    parser.add_argument("--gossipcop-real", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    return parser.parse_args()


def cleaned_title_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unwrap_wayback_url(raw: str) -> str:
    """Return the archived target URL when a standard Wayback URL is supplied."""
    decoded = unquote(raw)
    match = re.match(
        r"^(?:https?://)?(?:www\.)?web\.archive\.org/web/[^/]+/(https?://.+)$",
        decoded,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return decoded


def normalized_hostname(value: object) -> str:
    """Parse a deterministic lower-case hostname or the missing category."""
    if pd.isna(value):
        return MISSING_HOSTNAME
    raw = str(value).strip()
    if not raw:
        return MISSING_HOSTNAME
    raw = unwrap_wayback_url(raw)
    candidate = (
        raw
        if re.match(r"^[a-z][a-z0-9+.-]*://", raw, flags=re.IGNORECASE)
        else "http://" + raw
    )
    try:
        host = (urlparse(candidate).hostname or "").lower().rstrip(".")
    except ValueError:
        return MISSING_HOSTNAME
    for prefix in ("www.", "amp.", "m.", "mobile."):
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    return host or MISSING_HOSTNAME


def read_external_pair(
    corpus: str,
    fake_path: Path,
    real_path: Path,
) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    file_rows: list[dict[str, object]] = []
    for label, path in ((0, fake_path), (1, real_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        frame = pd.read_csv(path).copy()
        required = {"id", "news_url", "title"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame["label"] = label
        frame["source_file"] = path.name
        frames.append(frame)
        file_rows.append(
            {
                "corpus": corpus,
                "class": LABEL_NAMES[label],
                "file": path.name,
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
            }
        )

    raw = pd.concat(frames, ignore_index=True)
    raw["content"] = raw["title"].map(clean_text)
    empty_mask = raw["content"].eq("")
    nonempty = raw.loc[~empty_mask].copy()
    duplicate_mask = nonempty.duplicated(["content", "label"], keep="first")
    after_same_label = nonempty.loc[~duplicate_mask].copy()
    conflicting_titles = (
        after_same_label.groupby("content", sort=False)["label"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    conflict_mask = after_same_label["content"].isin(conflicting_titles)
    clean = after_same_label.loc[~conflict_mask].copy().reset_index(drop=True)
    clean["hostname"] = clean["news_url"].map(normalized_hostname)

    counts = {
        "raw_fake": int((raw["label"] == 0).sum()),
        "raw_real": int((raw["label"] == 1).sum()),
        "raw_total": int(len(raw)),
        "empty_titles_removed": int(empty_mask.sum()),
        "same_label_duplicate_rows_removed": int(duplicate_mask.sum()),
        "conflicting_clean_titles": int(len(conflicting_titles)),
        "conflicting_rows_removed_after_same_label_deduplication": int(
            conflict_mask.sum()
        ),
        "final_fake": int((clean["label"] == 0).sum()),
        "final_real": int((clean["label"] == 1).sum()),
        "final_total": int(len(clean)),
    }
    return clean, counts, file_rows


def out_of_fold_hostname_majority(
    data: pd.DataFrame,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict each held-out row from training-fold hostname majorities only."""
    labels = data["label"].to_numpy(dtype=int)
    hostnames = data["hostname"].astype(str).to_numpy()
    predictions = np.empty(len(data), dtype=int)
    fold_ids = np.empty(len(data), dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold_id, (train_index, test_index) in enumerate(
        splitter.split(hostnames, labels), start=1
    ):
        training = pd.DataFrame(
            {"hostname": hostnames[train_index], "label": labels[train_index]}
        )
        training_rates = training.groupby("hostname", sort=False)["label"].mean()
        fallback = int(training["label"].mean() >= 0.5)
        predictions[test_index] = np.fromiter(
            (
                int(training_rates[host] >= 0.5)
                if host in training_rates.index
                else fallback
                for host in hostnames[test_index]
            ),
            dtype=int,
            count=len(test_index),
        )
        fold_ids[test_index] = fold_id
    return predictions, fold_ids


def stratified_bootstrap_intervals(
    labels: np.ndarray,
    predictions: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    distributions: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }
    for _ in range(resamples):
        sample = np.concatenate(
            [rng.choice(index, len(index), replace=True) for index in class_indices]
        )
        sample_labels = labels[sample]
        sample_predictions = predictions[sample]
        distributions["accuracy"].append(
            accuracy_score(sample_labels, sample_predictions)
        )
        distributions["balanced_accuracy"].append(
            balanced_accuracy_score(sample_labels, sample_predictions)
        )
        distributions["macro_f1"].append(
            f1_score(
                sample_labels,
                sample_predictions,
                average="macro",
                zero_division=0,
            )
        )
    return {
        metric: tuple(np.percentile(values, [2.5, 97.5]).astype(float))
        for metric, values in distributions.items()
    }


def metric_row(
    corpus: str,
    data: pd.DataFrame,
    predictions: np.ndarray,
    folds: int,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    labels = data["label"].to_numpy(dtype=int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    intervals = stratified_bootstrap_intervals(
        labels, predictions, resamples=resamples, seed=seed
    )
    row: dict[str, object] = {
        "corpus": corpus,
        "diagnostic": "five_fold_out_of_fold_hostname_majority",
        "n": int(len(data)),
        "fake_n": int((labels == 0).sum()),
        "real_n": int((labels == 1).sum()),
        "folds": folds,
        "seed": seed,
        "unseen_hostname_fallback": "training_fold_majority_class",
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "fake_recall": float(
            recall_score(labels, predictions, pos_label=0, zero_division=0)
        ),
        "real_recall": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "true_fake_pred_fake": int(matrix[0, 0]),
        "true_fake_pred_real": int(matrix[0, 1]),
        "true_real_pred_fake": int(matrix[1, 0]),
        "true_real_pred_real": int(matrix[1, 1]),
        "bootstrap_resamples": resamples,
    }
    for metric, (low, high) in intervals.items():
        row[f"{metric}_bootstrap_95_low"] = low
        row[f"{metric}_bootstrap_95_high"] = high
    return row


def hostname_tables(
    corpus: str,
    data: pd.DataFrame,
    minimum_size: int = 10,
) -> tuple[dict[str, object], pd.DataFrame]:
    counts = (
        data.pivot_table(
            index="hostname",
            columns="label",
            values="id",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns={0: "fake_n", 1: "real_n"})
        .reset_index()
    )
    for column in ("fake_n", "real_n"):
        if column not in counts:
            counts[column] = 0
    counts["fake_n"] = counts["fake_n"].astype(int)
    counts["real_n"] = counts["real_n"].astype(int)
    counts["total"] = counts["fake_n"] + counts["real_n"]
    counts["real_rate"] = counts["real_n"] / counts["total"]
    counts["is_missing_category"] = counts["hostname"].eq(MISSING_HOSTNAME)
    counts["label_exclusive"] = (
        counts["fake_n"].eq(0) | counts["real_n"].eq(0)
    ) & ~counts["is_missing_category"]
    counts.insert(0, "corpus", corpus)
    counts = counts.sort_values(
        ["total", "hostname"], ascending=[False, True]
    ).reset_index(drop=True)

    valid = counts.loc[~counts["is_missing_category"]]
    exclusive_names = set(valid.loc[valid["label_exclusive"], "hostname"])
    exclusive_rows = data["hostname"].isin(exclusive_names)
    missing_rows = int(data["hostname"].eq(MISSING_HOSTNAME).sum())
    summary = {
        "corpus": corpus,
        "n": int(len(data)),
        "missing_or_unparseable_urls": missing_rows,
        "valid_normalized_hostnames": int(len(valid)),
        "hostname_categories_including_missing": int(len(counts)),
        "hostnames_seen_in_both_labels": int((~valid["label_exclusive"]).sum()),
        "label_exclusive_hostnames": int(valid["label_exclusive"].sum()),
        "rows_on_label_exclusive_hostnames": int(exclusive_rows.sum()),
        "share_rows_on_label_exclusive_hostnames": float(exclusive_rows.mean()),
        "valid_hostnames_with_at_least_10_rows": int(
            valid["total"].ge(minimum_size).sum()
        ),
        "missing_category_with_at_least_10_rows": bool(
            missing_rows >= minimum_size
        ),
    }
    return summary, counts


def main() -> None:
    args = parse_args()
    if args.folds != 5:
        raise ValueError("The manuscript specifies exactly five stratified folds")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("analysis_type") != "post_hoc_source_hostname_diagnostic":
        raise ValueError("Unexpected source-hostname protocol")
    args.output.mkdir(parents=True, exist_ok=True)

    corpus_paths = {
        "PolitiFact": (args.politifact_fake, args.politifact_real),
        "GossipCop": (args.gossipcop_fake, args.gossipcop_real),
    }
    cleaning_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    hostname_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for corpus, (fake_path, real_path) in corpus_paths.items():
        data, cleaning, files = read_external_pair(corpus, fake_path, real_path)
        cleaning_rows.append({"corpus": corpus, **cleaning})
        file_rows.extend(files)
        summary, hostname_counts = hostname_tables(corpus, data)
        summary_rows.append(summary)
        hostname_frames.append(hostname_counts)

        predictions, fold_ids = out_of_fold_hostname_majority(
            data, folds=args.folds, seed=args.seed
        )
        metric_rows.append(
            metric_row(
                corpus,
                data,
                predictions,
                folds=args.folds,
                resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "corpus": corpus,
                    "id": data["id"].astype(str),
                    "source_file": data["source_file"].astype(str),
                    "cleaned_title_sha256": data["content"].map(cleaned_title_hash),
                    "label": data["label"].astype(int),
                    "hostname": data["hostname"].astype(str),
                    "fold": fold_ids,
                    "hostname_only_prediction": predictions,
                    "correct": predictions == data["label"].to_numpy(dtype=int),
                }
            )
        )

    cleaning_table = pd.DataFrame(cleaning_rows)
    file_table = pd.DataFrame(file_rows)
    summary_table = pd.DataFrame(summary_rows)
    metric_table = pd.DataFrame(metric_rows)
    cleaning_table.to_csv(
        args.output / "source_hostname_cleaning_audit.csv", index=False
    )
    summary_table.to_csv(
        args.output / "source_hostname_corpus_summary.csv", index=False
    )
    metric_table.to_csv(
        args.output / "source_hostname_oof_metrics.csv", index=False
    )
    pd.concat(hostname_frames, ignore_index=True).to_csv(
        args.output / "source_hostname_counts.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output / "source_hostname_oof_predictions_without_text.csv", index=False
    )

    config = {
        "analysis": protocol,
        "input_files": file_table.to_dict(orient="records"),
        "cleaning": cleaning_table.to_dict(orient="records"),
        "hostname_summary": summary_table.to_dict(orient="records"),
        "diagnostic_metrics": metric_table.to_dict(orient="records"),
        "protocol_sha256": sha256_file(args.protocol),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output / "source_hostname_audit_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(summary_table.to_string(index=False))
    print(metric_table.to_string(index=False))


if __name__ == "__main__":
    main()
