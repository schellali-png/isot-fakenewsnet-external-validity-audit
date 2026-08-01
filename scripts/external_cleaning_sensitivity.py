#!/usr/bin/env python3
"""Post hoc sensitivity of external SVM performance to cleaning exclusions.

The fixed ISOT-trained title-only Linear SVM is evaluated on three nested
external populations: all nonempty rows, same-label exact-deduplicated rows
with conflicts retained, and the manuscript's main unique/nonconflicting
population. External data are never used to fit features or model parameters.
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
import sklearn
from sklearn.svm import LinearSVC


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fake_news_experiments import clean_text, load_dataset, sha256_file  # noqa: E402
from second_external_validity import (  # noqa: E402
    make_vectorizer,
    metric_record,
)


LABEL_NAMES = {0: "fake", 1: "real"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isot-fake", type=Path, required=True)
    parser.add_argument("--isot-true", type=Path, required=True)
    parser.add_argument("--politifact-fake", type=Path, required=True)
    parser.add_argument("--politifact-real", type=Path, required=True)
    parser.add_argument("--gossipcop-fake", type=Path, required=True)
    parser.add_argument("--gossipcop-real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--isot-encoding", default="utf-8")
    return parser.parse_args()


def title_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_external_rows(
    corpus: str,
    fake_path: Path,
    real_path: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    frames: list[pd.DataFrame] = []
    for label, path in ((0, fake_path), (1, real_path)):
        frame = pd.read_csv(path).copy()
        required = {"id", "title"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        frame["label"] = label
        frame["source_file"] = path.name
        frame["source_row"] = np.arange(len(frame), dtype=int)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["content"] = raw["title"].map(clean_text)
    raw["cleaned_title_sha256"] = raw["content"].map(title_hash)
    raw["source_row_key"] = (
        corpus
        + "::"
        + raw["source_file"].astype(str)
        + "::"
        + raw["source_row"].astype(str)
    )

    nonempty = raw.loc[raw["content"].ne("")].copy()
    same_label_duplicate = nonempty.duplicated(
        ["content", "label"], keep="first"
    )
    deduplicated = nonempty.loc[~same_label_duplicate].copy()
    conflicting_titles = (
        deduplicated.groupby("content", sort=False)["label"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    conflict_mask = deduplicated["content"].isin(conflicting_titles)
    main = deduplicated.loc[~conflict_mask].copy()
    populations = {
        "nonempty_all_rows": nonempty.reset_index(drop=True),
        "same_label_deduplicated_conflicts_retained": (
            deduplicated.reset_index(drop=True)
        ),
        "main_unique_nonconflicting": main.reset_index(drop=True),
    }
    counts = {
        "raw_rows": int(len(raw)),
        "empty_rows": int(raw["content"].eq("").sum()),
        "nonempty_all_rows": int(len(nonempty)),
        "same_label_duplicate_rows": int(same_label_duplicate.sum()),
        "same_label_deduplicated_conflicts_retained": int(len(deduplicated)),
        "conflicting_clean_titles": int(len(conflicting_titles)),
        "conflicting_rows_after_same_label_deduplication": int(
            conflict_mask.sum()
        ),
        "main_unique_nonconflicting": int(len(main)),
    }
    return populations, counts


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
    if len(isot) != 38_681:
        raise ValueError(f"Expected 38,681 ISOT titles, found {len(isot):,}")

    corpora = {
        "PolitiFact": load_external_rows(
            "PolitiFact", args.politifact_fake, args.politifact_real
        ),
        "GossipCop": load_external_rows(
            "GossipCop", args.gossipcop_fake, args.gossipcop_real
        ),
    }
    expected_main = {"PolitiFact": 975, "GossipCop": 20_587}
    for corpus, (populations, _) in corpora.items():
        observed = len(populations["main_unique_nonconflicting"])
        if observed != expected_main[corpus]:
            raise ValueError(
                f"Expected {expected_main[corpus]:,} main {corpus} rows, "
                f"found {observed:,}"
            )

    vectorizer = make_vectorizer(args.max_features)
    x_train = vectorizer.fit_transform(isot["content"])
    model = LinearSVC(C=1.0, random_state=args.seed)
    model.fit(x_train, isot["label"].to_numpy(dtype=int))

    rng = np.random.default_rng(args.seed + 881)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for corpus, (populations, _) in corpora.items():
        for population_name, data in populations.items():
            labels = data["label"].to_numpy(dtype=int)
            predictions = model.predict(vectorizer.transform(data["content"]))
            row = metric_record(
                corpus=corpus,
                model="linear_svm",
                labels=labels,
                predictions=predictions,
                resamples=args.bootstrap_resamples,
                rng=rng,
            )
            row["population"] = population_name
            metric_rows.append(row)

            output = data[
                [
                    "source_row_key",
                    "id",
                    "source_file",
                    "cleaned_title_sha256",
                    "label",
                ]
            ].copy()
            output.insert(0, "corpus", corpus)
            output.insert(1, "population", population_name)
            output["linear_svm_prediction"] = predictions
            prediction_rows.append(output)

    metrics = pd.DataFrame(metric_rows)
    ordered = [
        "corpus",
        "population",
        "model",
        "n",
        "fake_n",
        "real_n",
    ]
    metrics = metrics[
        ordered + [column for column in metrics.columns if column not in ordered]
    ]
    metrics.to_csv(
        args.output / "external_cleaning_sensitivity_metrics.csv", index=False
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output
        / "external_cleaning_sensitivity_predictions_without_text.csv",
        index=False,
    )

    input_paths = [
        args.isot_fake,
        args.isot_true,
        args.politifact_fake,
        args.politifact_real,
        args.gossipcop_fake,
        args.gossipcop_real,
    ]
    config = {
        "analysis_status": "post_hoc_sensitivity_analysis",
        "purpose": (
            "evaluate whether exclusion of same-label exact duplicates and "
            "conflicting cleaned titles materially changes fixed external SVM "
            "performance"
        ),
        "training": {
            "corpus": "exact-deduplicated ISOT titles",
            "n": int(len(isot)),
            "cleaning": isot_counts,
        },
        "model": {
            "representation": "TF-IDF word unigrams",
            "max_features": args.max_features,
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "linear_svm_C": 1.0,
            "seed": args.seed,
            "external_fitting_or_selection": False,
        },
        "populations": [
            "nonempty_all_rows",
            "same_label_deduplicated_conflicts_retained",
            "main_unique_nonconflicting",
        ],
        "counts": {
            corpus: counts for corpus, (_, counts) in corpora.items()
        },
        "bootstrap": {
            "method": "class-stratified item bootstrap",
            "resamples": args.bootstrap_resamples,
            "seed": args.seed + 881,
        },
        "input_hashes": {
            path.name: sha256_file(path) for path in input_paths
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (
        args.output / "external_cleaning_sensitivity_config.json"
    ).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(
        metrics[
            [
                "corpus",
                "population",
                "n",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "fake_recall",
                "real_recall",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
