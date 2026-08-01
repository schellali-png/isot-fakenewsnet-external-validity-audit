#!/usr/bin/env python3
"""Post-hoc title-level near-duplicate and cross-corpus overlap audit.

The audit uses hashed character five-gram sets and exact Jaccard verification.
Candidate generation is performed once at Jaccard >= 0.70; the reported
principal threshold is 0.80, with 0.70 and 0.90 sensitivity summaries.
Raw title text is never written to the output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import murmurhash3_32


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from external_error_analysis import load_politifact  # noqa: E402
from fake_news_experiments import load_dataset, sha256_file  # noqa: E402
from reconstruct_isot_groups import (  # noqa: E402
    component_summary,
    exact_jaccard_edges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isot-fake", type=Path, required=True)
    parser.add_argument("--isot-true", type=Path, required=True)
    parser.add_argument("--politifact-fake", type=Path, required=True)
    parser.add_argument("--politifact-real", type=Path, required=True)
    parser.add_argument("--gossipcop-fake", type=Path, required=True)
    parser.add_argument("--gossipcop-real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-threshold", type=float, default=0.70)
    parser.add_argument("--principal-threshold", type=float, default=0.80)
    parser.add_argument("--character-ngram", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def title_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hashed_character_shingles(text: str, ngram: int) -> set[int]:
    padded = f" {text} "
    values = (
        [padded]
        if len(padded) < ngram
        else [padded[index : index + ngram] for index in range(len(padded) - ngram + 1)]
    )
    return {
        int(murmurhash3_32(value, seed=17, positive=True))
        for value in values
    }


def corpus_pair(first: str, second: str) -> str:
    order = {"ISOT": 0, "PolitiFact": 1, "GossipCop": 2}
    return "–".join(sorted((first, second), key=order.__getitem__))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    isot, isot_counts = load_dataset(
        args.isot_fake,
        args.isot_true,
        drop_duplicates=True,
        encoding="utf-8",
        text_field="title",
        mask_reuters=False,
    )
    politifact, politifact_counts = load_politifact(
        args.politifact_fake, args.politifact_real
    )
    gossipcop, gossipcop_counts = load_politifact(
        args.gossipcop_fake, args.gossipcop_real
    )
    if (len(isot), len(politifact), len(gossipcop)) != (38_681, 975, 20_587):
        raise ValueError("Unexpected cleaned title population")

    frames: list[pd.DataFrame] = []
    for corpus, frame, identifier in (
        ("ISOT", isot, "row_id"),
        ("PolitiFact", politifact, "id"),
        ("GossipCop", gossipcop, "id"),
    ):
        part = pd.DataFrame(
            {
                "corpus": corpus,
                "item_id": frame[identifier].astype(str),
                "label": frame["label"].to_numpy(dtype=int),
                "content": frame["content"].astype(str),
            }
        )
        part["cleaned_title_sha256"] = part["content"].map(title_hash)
        frames.append(part)
    combined = pd.concat(frames, ignore_index=True)
    sets = [
        hashed_character_shingles(value, args.character_ngram)
        for value in combined["content"]
    ]
    edges, candidates = exact_jaccard_edges(sets, args.minimum_threshold)
    print(
        f"titles={len(combined)} candidates={candidates} "
        f"verified_at_{args.minimum_threshold:.2f}={len(edges)}",
        flush=True,
    )

    edge_records: list[dict[str, object]] = []
    for first, second, similarity in edges:
        left = combined.iloc[first]
        right = combined.iloc[second]
        edge_records.append(
            {
                "first_corpus": left["corpus"],
                "first_item_id": left["item_id"],
                "first_title_sha256": left["cleaned_title_sha256"],
                "first_label": int(left["label"]),
                "second_corpus": right["corpus"],
                "second_item_id": right["item_id"],
                "second_title_sha256": right["cleaned_title_sha256"],
                "second_label": int(right["label"]),
                "corpus_pair": corpus_pair(left["corpus"], right["corpus"]),
                "character_5gram_jaccard": float(similarity),
            }
        )
    edge_frame = pd.DataFrame(edge_records)
    edge_frame.to_csv(
        args.output / "title_near_duplicate_edges_without_text.csv", index=False
    )

    thresholds = sorted(
        {args.minimum_threshold, args.principal_threshold, 0.90}
    )
    pair_names = [
        "ISOT–ISOT",
        "PolitiFact–PolitiFact",
        "GossipCop–GossipCop",
        "ISOT–PolitiFact",
        "ISOT–GossipCop",
        "PolitiFact–GossipCop",
    ]
    summary_rows: list[dict[str, object]] = []
    for threshold in thresholds:
        selected = (
            edge_frame.loc[edge_frame["character_5gram_jaccard"].ge(threshold)]
            if not edge_frame.empty
            else edge_frame
        )
        for pair in pair_names:
            subset = selected.loc[selected["corpus_pair"].eq(pair)]
            summary_rows.append(
                {
                    "threshold": threshold,
                    "corpus_pair": pair,
                    "verified_pairs": int(len(subset)),
                    "cross_label_pairs": int(
                        subset["first_label"].ne(subset["second_label"]).sum()
                    )
                    if not subset.empty
                    else 0,
                    "exact_pairs": int(
                        subset["character_5gram_jaccard"].eq(1.0).sum()
                    )
                    if not subset.empty
                    else 0,
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        args.output / "title_near_duplicate_pair_summary.csv", index=False
    )

    isot_size = len(isot)
    principal_isot_edges = [
        (first, second, similarity)
        for first, second, similarity in edges
        if first < isot_size
        and second < isot_size
        and similarity >= args.principal_threshold
    ]
    group_ids, multi, sizes, mixed = component_summary(
        isot_size,
        isot["label"].to_numpy(dtype=int),
        principal_isot_edges,
    )
    group_ids = pd.factorize(group_ids, sort=False)[0].astype(np.int32)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=args.seed
    )
    folds = np.empty(isot_size, dtype=np.int8)
    dummy = csr_matrix((isot_size, 1))
    for fold, (_, test_indices) in enumerate(
        splitter.split(dummy, isot["label"], group_ids), start=1
    ):
        folds[test_indices] = fold
    assignments = pd.DataFrame(
        {
            "row_id": isot["row_id"].to_numpy(dtype=int),
            "cleaned_title_sha256": isot["content"].map(title_hash),
            "label": isot["label"].to_numpy(dtype=int),
            "near_duplicate_group": group_ids,
            "outer_fold": folds,
        }
    )
    assignments.to_csv(
        args.output / "isot_title_specific_group_fold_assignments.csv",
        index=False,
    )

    fold_summary = {
        str(fold): {
            "n": int(assignments["outer_fold"].eq(fold).sum()),
            "fake": int(
                assignments.loc[assignments["outer_fold"].eq(fold), "label"].eq(0).sum()
            ),
            "real": int(
                assignments.loc[assignments["outer_fold"].eq(fold), "label"].eq(1).sum()
            ),
            "groups": int(
                assignments.loc[
                    assignments["outer_fold"].eq(fold), "near_duplicate_group"
                ].nunique()
            ),
        }
        for fold in range(1, 6)
    }
    config = {
        "analysis_status": "post-hoc title-level near-duplicate sensitivity",
        "representation": {
            "normalization": "shared lowercased alphabetic title normalization",
            "shingles": f"hashed character {args.character_ngram}-grams with boundary spaces",
            "hash": "MurmurHash3 32-bit, seed 17",
            "candidate_threshold": args.minimum_threshold,
            "principal_threshold": args.principal_threshold,
            "sensitivity_thresholds": thresholds,
            "verification": "exact Jaccard on hashed shingle sets",
        },
        "candidate_pairs_checked": candidates,
        "verified_pairs_at_minimum_threshold": len(edges),
        "isot_principal_components": {
            "verified_edges": len(principal_isot_edges),
            "multi_title_groups": len(multi),
            "titles_in_multi_title_groups": int(sum(map(len, multi))),
            "component_size_counts": {
                str(key): int(value) for key, value in sorted(Counter(map(len, multi)).items())
            },
            "mixed_label_components": mixed,
        },
        "counts": {
            "isot": len(isot),
            "politifact": len(politifact),
            "gossipcop": len(gossipcop),
        },
        "cleaning": {
            "isot": isot_counts,
            "politifact": politifact_counts,
            "gossipcop": gossipcop_counts,
        },
        "folds": fold_summary,
        "input_hashes": {
            args.isot_fake.name: sha256_file(args.isot_fake),
            args.isot_true.name: sha256_file(args.isot_true),
            args.politifact_fake.name: sha256_file(args.politifact_fake),
            args.politifact_real.name: sha256_file(args.politifact_real),
            args.gossipcop_fake.name: sha256_file(args.gossipcop_fake),
            args.gossipcop_real.name: sha256_file(args.gossipcop_real),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output / "title_near_duplicate_audit_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config["isot_principal_components"], indent=2), flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
