#!/usr/bin/env python3
"""Run a leakage-controlled chronological evaluation on ISOT.

Rows dated before ``--cutoff`` form the training period and rows on or after
the cutoff form the future test period. Exact/Jaccard near-duplicate components
that cross the boundary are removed in full, so no reconstructed event group
can occur on both sides. The script evaluates the fixed TF-IDF classifiers,
a Reuters-only rule, and a Reuters-masked Linear SVM. It never writes article
text to disk.
"""

from __future__ import annotations

import argparse
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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from equalized_multi_model_tuning import MODEL_NAMES, point_metrics  # noqa: E402
from fake_news_experiments import load_dataset, sha256_file  # noqa: E402
from group_aware_internal_validation import (  # noqa: E402
    cleaned_hash,
    fit_fixed_predictions,
    mask_reuters,
    model_list,
    wilson_interval,
)
from reconstruct_isot_groups import (  # noqa: E402
    component_summary,
    exact_jaccard_edges,
    hashed_shingle_set,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isot-fake", type=Path, required=True)
    parser.add_argument("--isot-true", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", default="2017-01-01")
    parser.add_argument("--models", default="nb,lr,svm,rf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--rf-jobs", type=int, default=-1)
    parser.add_argument("--isot-encoding", default="utf-8")
    parser.add_argument(
        "--strict-manuscript-snapshot",
        action="store_true",
        help="Require the counts reported for the 2017-01-01 analysis.",
    )
    return parser.parse_args()


def parse_news_dates(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse known ISOT date forms deterministically and report the format."""
    stripped = values.fillna("").astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    formats = pd.Series("unparsed", index=values.index, dtype="object")
    candidates = (
        ("%B %d, %Y", "full_month_day_year"),
        ("%b %d, %Y", "short_month_day_year"),
        ("%Y-%m-%d", "iso_date"),
        ("%Y/%m/%d", "iso_slash_date"),
        ("%m/%d/%Y", "us_numeric_date"),
    )
    for date_format, label in candidates:
        missing = parsed.isna() & stripped.ne("")
        if not missing.any():
            break
        attempt = pd.to_datetime(
            stripped.loc[missing], format=date_format, errors="coerce"
        )
        accepted = attempt.notna()
        accepted_indices = attempt.index[accepted]
        parsed.loc[accepted_indices] = attempt.loc[accepted_indices]
        formats.loc[accepted_indices] = label
    formats.loc[stripped.eq("")] = "missing"
    return parsed, formats


def reconstruct_groups(
    contents: pd.Series,
    labels: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[tuple[int, int, float]], dict[str, object]]:
    shingle_sets = [hashed_shingle_set(value) for value in contents]
    edges, candidate_pairs = exact_jaccard_edges(shingle_sets, threshold)
    group_ids, multi, sizes, mixed = component_summary(len(contents), labels, edges)
    group_ids = pd.factorize(group_ids, sort=False)[0].astype(np.int32)
    summary = {
        "jaccard_threshold": threshold,
        "candidate_pairs": candidate_pairs,
        "verified_edges": len(edges),
        "multi_document_groups": len(multi),
        "documents_in_multi_document_groups": int(sum(map(len, multi))),
        "component_size_counts": {
            str(key): int(value) for key, value in sorted(sizes.items())
        },
        "mixed_label_components": int(mixed),
    }
    return group_ids, edges, summary


def paired_exact_mcnemar(
    name: str,
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, object]:
    first_only = int(np.sum((first == labels) & (second != labels)))
    second_only = int(np.sum((first != labels) & (second == labels)))
    discordant = first_only + second_only
    p_value = (
        float(
            binomtest(
                min(first_only, second_only),
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "comparison": name,
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant": discordant,
        "exact_two_sided_mcnemar_p": p_value,
    }


def metric_row(
    model: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    train_n: int,
) -> dict[str, object]:
    correct = int(np.sum(labels == predictions))
    low, high = wilson_interval(correct, len(labels))
    return {
        "model": model,
        "model_name": MODEL_NAMES.get(model, model),
        "train_n": train_n,
        "test_n": int(len(labels)),
        **point_metrics(labels, predictions),
        "accuracy_wilson_95_low": low,
        "accuracy_wilson_95_high": high,
    }


def main() -> None:
    args = parse_args()
    models = model_list(args.models)
    if not 0 < args.jaccard_threshold <= 1:
        raise ValueError("--jaccard-threshold must be in (0, 1]")
    cutoff = pd.Timestamp(args.cutoff)
    args.output.mkdir(parents=True, exist_ok=True)

    data, cleaning_counts = load_dataset(
        args.isot_fake,
        args.isot_true,
        drop_duplicates=True,
        encoding=args.isot_encoding,
        text_field="combined",
        mask_reuters=False,
    )
    labels = data["label"].to_numpy(dtype=int)
    dates, date_formats = parse_news_dates(data["date"])
    groups, edges, group_summary = reconstruct_groups(
        data["content"], labels, args.jaccard_threshold
    )

    dated = dates.notna().to_numpy()
    earlier = dated & (dates.to_numpy() < cutoff.to_datetime64())
    future = dated & (dates.to_numpy() >= cutoff.to_datetime64())
    boundary = pd.DataFrame(
        {"group": groups, "earlier": earlier, "future": future}
    ).groupby("group", sort=False)[["earlier", "future"]].any()
    crossing_groups = boundary.index[boundary["earlier"] & boundary["future"]]
    crossing = np.isin(groups, crossing_groups.to_numpy())
    train_indices = np.flatnonzero(earlier & ~crossing)
    test_indices = np.flatnonzero(future & ~crossing)
    unparsed_indices = np.flatnonzero(~dated)

    if not len(train_indices) or not len(test_indices):
        raise ValueError("The cutoff produced an empty training or test period")
    for name, indices in (("training", train_indices), ("future test", test_indices)):
        if len(np.unique(labels[indices])) < 2:
            raise ValueError(f"The {name} period must contain both labels")

    observed = {
        "combined_rows": int(len(data)),
        "unparseable_dates": int(len(unparsed_indices)),
        "cross_cutoff_groups": int(len(crossing_groups)),
        "cross_cutoff_rows_removed": int(crossing.sum()),
        "train_rows": int(len(train_indices)),
        "future_test_rows": int(len(test_indices)),
    }
    if args.strict_manuscript_snapshot:
        if cutoff != pd.Timestamp("2017-01-01"):
            raise ValueError(
                "--strict-manuscript-snapshot requires --cutoff 2017-01-01"
            )
        expected = {
            "combined_rows": 38_826,
            "unparseable_dates": 1,
            "cross_cutoff_groups": 7,
            "cross_cutoff_rows_removed": 15,
            "train_rows": 15_777,
            "future_test_rows": 23_033,
        }
        if observed != expected:
            raise ValueError(
                f"Snapshot audit mismatch: expected {expected}, observed {observed}"
            )

    test_labels = labels[test_indices]
    predictions: dict[str, np.ndarray] = {}
    for model in models:
        predictions[model] = fit_fixed_predictions(
            model,
            data.iloc[train_indices]["content"],
            labels[train_indices],
            data.iloc[test_indices]["content"],
            args,
        )
        print(f"completed chronological model={model}", flush=True)

    reuters_rule = (
        data.iloc[test_indices]["content"]
        .str.contains(r"\breuters\b", regex=True)
        .to_numpy(dtype=bool)
        .astype(int)
    )
    predictions["reuters_rule"] = reuters_rule
    masked_svm = fit_fixed_predictions(
        "svm",
        mask_reuters(data.iloc[train_indices]["content"]),
        labels[train_indices],
        mask_reuters(data.iloc[test_indices]["content"]),
        args,
    )
    predictions["svm_reuters_masked"] = masked_svm

    metrics = [
        metric_row(
            model,
            test_labels,
            values,
            0 if model == "reuters_rule" else len(train_indices),
        )
        for model, values in predictions.items()
    ]
    pd.DataFrame(metrics).to_csv(
        args.output / "chronological_metrics.csv", index=False
    )

    prediction_rows: list[dict[str, object]] = []
    for model, values in predictions.items():
        for local_index, row_index in enumerate(test_indices):
            prediction_rows.append(
                {
                    "model": model,
                    "row_id": int(data.iloc[row_index]["row_id"]),
                    "combined_text_sha256": cleaned_hash(
                        data.iloc[row_index]["content"]
                    ),
                    "date": dates.iloc[row_index].date().isoformat(),
                    "label": int(test_labels[local_index]),
                    "prediction": int(values[local_index]),
                }
            )
    pd.DataFrame(prediction_rows).to_csv(
        args.output / "chronological_predictions_without_text.csv", index=False
    )

    quarter_rows: list[dict[str, object]] = []
    test_dates = dates.iloc[test_indices]
    quarters = test_dates.dt.to_period("Q")
    for quarter in sorted(quarters.unique()):
        local_mask = (quarters == quarter).to_numpy()
        quarter_labels = test_labels[local_mask]
        for model, values in predictions.items():
            correct = int(np.sum(values[local_mask] == quarter_labels))
            low, high = wilson_interval(correct, len(quarter_labels))
            quarter_rows.append(
                {
                    "quarter": str(quarter),
                    "model": model,
                    "n": int(len(quarter_labels)),
                    "fake_n": int(np.sum(quarter_labels == 0)),
                    "real_n": int(np.sum(quarter_labels == 1)),
                    **point_metrics(quarter_labels, values[local_mask]),
                    "accuracy_wilson_95_low": low,
                    "accuracy_wilson_95_high": high,
                }
            )
    pd.DataFrame(quarter_rows).to_csv(
        args.output / "chronological_quarterly_metrics.csv", index=False
    )

    mcnemar_rows: list[dict[str, object]] = []
    if "svm" in predictions:
        mcnemar_rows.extend(
            [
                paired_exact_mcnemar(
                    "Linear SVM vs Reuters-only rule",
                    test_labels,
                    predictions["svm"],
                    reuters_rule,
                ),
                paired_exact_mcnemar(
                    "Linear SVM vs Reuters-masked Linear SVM",
                    test_labels,
                    predictions["svm"],
                    masked_svm,
                ),
            ]
        )
    pd.DataFrame(mcnemar_rows).to_csv(
        args.output / "chronological_mcnemar.csv", index=False
    )

    pd.DataFrame(
        {
            "date_parse_status": date_formats.value_counts(dropna=False).index,
            "n": date_formats.value_counts(dropna=False).values,
        }
    ).to_csv(args.output / "date_parsing_audit.csv", index=False)
    crossing_rows: list[dict[str, object]] = []
    for group in crossing_groups:
        member_indices = np.flatnonzero(groups == group)
        crossing_rows.append(
            {
                "near_duplicate_group": int(group),
                "n": int(len(member_indices)),
                "earliest_date": dates.iloc[member_indices].min().date().isoformat(),
                "latest_date": dates.iloc[member_indices].max().date().isoformat(),
                "fake_n": int(np.sum(labels[member_indices] == 0)),
                "real_n": int(np.sum(labels[member_indices] == 1)),
            }
        )
    pd.DataFrame(crossing_rows).to_csv(
        args.output / "chronological_removed_cross_cutoff_groups.csv", index=False
    )

    edge_frame = pd.DataFrame(
        [
            {
                "first_row_id": int(data.iloc[first]["row_id"]),
                "second_row_id": int(data.iloc[second]["row_id"]),
                "jaccard": similarity,
            }
            for first, second, similarity in edges
        ]
    )
    edge_frame.to_csv(
        args.output / "chronological_near_duplicate_edges.csv", index=False
    )

    config = {
        "analysis": "leakage-controlled chronological evaluation",
        "cutoff": cutoff.date().isoformat(),
        "counts": observed,
        "cleaning": cleaning_counts,
        "group_reconstruction": group_summary,
        "models": list(predictions),
        "input_hashes": {
            args.isot_fake.name: sha256_file(args.isot_fake),
            args.isot_true.name: sha256_file(args.isot_true),
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
    (args.output / "chronological_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved chronological audit to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
