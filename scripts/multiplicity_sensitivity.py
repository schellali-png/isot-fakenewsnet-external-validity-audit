#!/usr/bin/env python3
"""Global multiplicity sensitivity audit for exact paired McNemar tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(first_only, second_only), discordant, 0.5).pvalue)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    size = len(p_values)
    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order]
    adjusted_ordered = np.maximum.accumulate(
        np.array([(size - index) * value for index, value in enumerate(ordered)])
    )
    adjusted = np.empty(size, dtype=float)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    size = len(p_values)
    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order]
    ranks = np.arange(1, size + 1, dtype=float)
    scaled = ordered * size / ranks
    adjusted_ordered = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(size, dtype=float)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equalized-tests", type=Path, required=True)
    parser.add_argument("--majority-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    legacy = [
        (
            "legacy_group_holdout_svm_vs_rf",
            "Original diagnostics",
            "Group-aware holdout: Linear SVM vs Random Forest",
            46,
            11,
        ),
        (
            "legacy_c1_reuters_rule_vs_svm",
            "Original diagnostics",
            "Conventional C1 holdout: Reuters-only rule vs Linear SVM",
            27,
            36,
        ),
        (
            "legacy_c1_svm_vs_rf",
            "Original diagnostics",
            "Conventional C1 holdout: Linear SVM vs Random Forest",
            49,
            12,
        ),
        (
            "legacy_masked_svm_vs_rf",
            "Original diagnostics",
            "Reuters-masked holdout: Linear SVM vs Random Forest",
            164,
            25,
        ),
        (
            "legacy_temporal_reuters_rule_vs_svm",
            "Original diagnostics",
            "Future test: Reuters-only rule vs Linear SVM",
            489,
            84,
        ),
        (
            "legacy_temporal_svm_vs_masked",
            "Original diagnostics",
            "Future test: unmasked vs Reuters-masked Linear SVM",
            1064,
            9,
        ),
    ]
    for test_id, family, comparison, first, second in legacy:
        rows.append(
            {
                "test_id": test_id,
                "family": family,
                "comparison": comparison,
                "first_only_correct": first,
                "second_only_correct": second,
                "discordant": first + second,
                "raw_exact_mcnemar_p": exact_p(first, second),
                "source": "reported paired counts",
            }
        )

    equalized = pd.read_csv(args.equalized_tests)
    scope_family = {
        "pooled_nested_outer": "Equalized ISOT nested",
        "PolitiFact": "Equalized PolitiFact",
        "GossipCop": "Equalized GossipCop",
    }
    for record in equalized.to_dict(orient="records"):
        first = int(record["selected_only_correct"])
        second = int(record["fixed_only_correct"])
        scope = str(record["scope"])
        model = str(record["model"])
        rows.append(
            {
                "test_id": f"equalized_{scope.lower()}_{model}",
                "family": scope_family[scope],
                "comparison": f"{scope}: {record['model_name']} selected vs fixed",
                "first_only_correct": first,
                "second_only_correct": second,
                "discordant": first + second,
                "raw_exact_mcnemar_p": exact_p(first, second),
                "source": args.equalized_tests.name,
            }
        )

    majority = pd.read_csv(args.majority_test).iloc[0]
    first = int(majority["svm_only_correct"])
    second = int(majority["majority_only_correct"])
    rows.append(
        {
            "test_id": "gossipcop_fixed_svm_vs_majority",
            "family": "GossipCop majority comparison",
            "comparison": str(majority["comparison"]),
            "first_only_correct": first,
            "second_only_correct": second,
            "discordant": first + second,
            "raw_exact_mcnemar_p": exact_p(first, second),
            "source": args.majority_test.name,
        }
    )

    tests = pd.DataFrame(rows)
    if tests["test_id"].duplicated().any() or len(tests) != 19:
        raise ValueError("Expected 19 unique exact McNemar contrasts")
    raw = tests["raw_exact_mcnemar_p"].to_numpy(dtype=float)
    tests["holm_adjusted_p"] = holm_adjust(raw)
    tests["bh_fdr_adjusted_p"] = bh_adjust(raw)
    tests["nominal_significant_0_05"] = raw <= args.alpha
    tests["holm_significant_0_05"] = tests["holm_adjusted_p"] <= args.alpha
    tests["bh_fdr_significant_0_05"] = tests["bh_fdr_adjusted_p"] <= args.alpha
    tests.to_csv(args.output / "multiple_testing_mcnemar_adjustments.csv", index=False)

    family_order = [
        "Original diagnostics",
        "Equalized ISOT nested",
        "Equalized PolitiFact",
        "Equalized GossipCop",
        "GossipCop majority comparison",
    ]
    summary = (
        tests.groupby("family", sort=False)
        .agg(
            tests=("test_id", "size"),
            nominal_p_le_0_05=("nominal_significant_0_05", "sum"),
            holm_p_le_0_05=("holm_significant_0_05", "sum"),
            bh_fdr_p_le_0_05=("bh_fdr_significant_0_05", "sum"),
        )
        .reindex(family_order)
        .reset_index()
    )
    summary.loc[len(summary)] = {
        "family": "All tests",
        "tests": len(tests),
        "nominal_p_le_0_05": int(tests["nominal_significant_0_05"].sum()),
        "holm_p_le_0_05": int(tests["holm_significant_0_05"].sum()),
        "bh_fdr_p_le_0_05": int(tests["bh_fdr_significant_0_05"].sum()),
    }
    summary.to_csv(args.output / "multiple_testing_family_summary.csv", index=False)

    config = {
        "analysis_status": "post-hoc conservative multiplicity sensitivity audit",
        "global_family_size": len(tests),
        "family_definition": (
            "all unique exact paired McNemar contrasts reported in the manuscript "
            "or its Revision 11 associated results; duplicated SVM tuning contrasts "
            "are counted once"
        ),
        "alpha": args.alpha,
        "adjustments": {
            "holm": "family-wise error-rate control",
            "benjamini_hochberg": "false-discovery-rate control",
        },
        "interval_policy": (
            "Wilson and bootstrap intervals remain marginal effect-size uncertainty "
            "summaries; they are not claimed to provide simultaneous 95% coverage"
        ),
        "input_hashes": {
            args.equalized_tests.name: sha256_file(args.equalized_tests),
            args.majority_test.name: sha256_file(args.majority_test),
        },
        "results": {
            "nominal_significant": int(tests["nominal_significant_0_05"].sum()),
            "holm_significant": int(tests["holm_significant_0_05"].sum()),
            "bh_fdr_significant": int(tests["bh_fdr_significant_0_05"].sum()),
            "holm_change": (
                "GossipCop selected-vs-fixed Linear SVM raw p=0.02599 becomes "
                "Holm-adjusted p>0.05; all other nominally significant contrasts "
                "remain significant under Holm"
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    (args.output / "multiple_testing_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(
        tests.loc[
            tests["nominal_significant_0_05"] != tests["holm_significant_0_05"],
            ["test_id", "raw_exact_mcnemar_p", "holm_adjusted_p", "bh_fdr_adjusted_p"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
