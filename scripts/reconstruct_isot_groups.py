from __future__ import annotations

import argparse
import math
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import murmurhash3_32


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fake_news_experiments import load_dataset  # noqa: E402


class DisjointSet:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        parent = int(self.parent[item])
        while parent != int(self.parent[parent]):
            self.parent[parent] = self.parent[int(self.parent[parent])]
            parent = int(self.parent[parent])
        while item != parent:
            next_item = int(self.parent[item])
            self.parent[item] = parent
            item = next_item
        return parent

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def hashed_shingle_set(text: str) -> set[int]:
    tokens = text.split()
    if len(tokens) < 5:
        values = tokens
    else:
        values = (" ".join(tokens[index : index + 5]) for index in range(len(tokens) - 4))
    return {
        int(murmurhash3_32(value, seed=0, positive=True) % (2**30))
        for value in values
    }


def exact_jaccard_edges(sets: list[set[int]], threshold: float) -> tuple[list[tuple[int, int, float]], int]:
    frequencies = Counter(token for values in sets for token in values)
    ordered = [sorted(values, key=lambda token: (frequencies[token], token)) for values in sets]
    index: dict[int, list[int]] = defaultdict(list)
    edges: list[tuple[int, int, float]] = []
    checked: set[tuple[int, int]] = set()
    candidate_count = 0

    for current, values in enumerate(sets):
        size = len(values)
        prefix_length = size - math.ceil(threshold * size) + 1
        candidates: set[int] = set()
        for token in ordered[current][:prefix_length]:
            candidates.update(index[token])
        for previous in candidates:
            previous_size = len(sets[previous])
            if previous_size < math.ceil(threshold * size):
                continue
            if previous_size > math.floor(size / threshold):
                continue
            pair = (previous, current)
            if pair in checked:
                continue
            checked.add(pair)
            candidate_count += 1
            intersection = len(values.intersection(sets[previous]))
            union = size + previous_size - intersection
            similarity = intersection / union
            if similarity >= threshold:
                edges.append((previous, current, similarity))
        for token in ordered[current][:prefix_length]:
            index[token].append(current)
    return edges, candidate_count


def component_summary(size: int, labels: np.ndarray, edges: list[tuple[int, int, float]]):
    structure = DisjointSet(size)
    for first, second, _ in edges:
        structure.union(first, second)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(size):
        groups[structure.find(index)].append(index)
    multi = [members for members in groups.values() if len(members) > 1]
    sizes = Counter(len(members) for members in multi)
    mixed = sum(len(set(labels[members])) > 1 for members in multi)
    group_ids = np.array([structure.find(index) for index in range(size)], dtype=np.int32)
    return group_ids, multi, sizes, mixed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", type=Path, required=True)
    parser.add_argument("--true", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    combined, counts = load_dataset(
        args.fake,
        args.true,
        drop_duplicates=True,
        encoding="utf-8",
        text_field="combined",
        mask_reuters=False,
    )
    print("combined", len(combined), counts, flush=True)
    sets = [hashed_shingle_set(value) for value in combined["content"]]
    print(
        "shingles",
        sum(map(len, sets)),
        "median",
        float(np.median([len(item) for item in sets])),
        flush=True,
    )
    edges, candidates = exact_jaccard_edges(sets, args.threshold)
    group_ids, multi, sizes, mixed = component_summary(
        len(combined), combined["label"].to_numpy(), edges
    )
    # Canonicalize component identifiers over the complete C1 population before
    # title filtering. This preserves the historical identifiers (including gaps
    # left by combined-text rows with empty titles) in the published fold file.
    group_ids = pd.factorize(group_ids, sort=False)[0].astype(np.int32)
    print(
        "threshold",
        args.threshold,
        "candidates",
        candidates,
        "edges",
        len(edges),
        "groups",
        len(multi),
        "documents",
        sum(map(len, multi)),
        "sizes",
        dict(sorted(sizes.items())),
        "mixed",
        mixed,
        flush=True,
    )

    # Reconstruct the original five folds on the combined-text C1 population first.
    # Title-only rows then inherit those fold identifiers, exactly as in the study.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    combined_fold = np.empty(len(combined), dtype=np.int8)
    dummy = csr_matrix((len(combined), 1))
    for fold_index, (_, test) in enumerate(
        splitter.split(dummy, combined["label"], group_ids), start=1
    ):
        combined_fold[test] = fold_index

    title_data, title_counts = load_dataset(
        args.fake,
        args.true,
        drop_duplicates=True,
        encoding="utf-8",
        text_field="title",
        mask_reuters=False,
    )
    row_to_group = dict(zip(combined["row_id"], group_ids))
    row_to_fold = dict(zip(combined["row_id"], combined_fold))
    missing = [row_id for row_id in title_data["row_id"] if row_id not in row_to_group]
    print("titles", len(title_data), title_counts, "missing_groups", len(missing), flush=True)
    title_groups = np.array([row_to_group[row_id] for row_id in title_data["row_id"]])
    title_folds = np.array([row_to_fold[row_id] for row_id in title_data["row_id"]])
    for fold_index in range(1, 6):
        mask = title_folds == fold_index
        labels = title_data.loc[mask, "label"]
        print(
            "fold",
            fold_index,
            "n",
            int(mask.sum()),
            "fake",
            int(labels.eq(0).sum()),
            "real",
            int(labels.eq(1).sum()),
            "groups",
            len(np.unique(title_groups[mask])),
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    assignments = pd.DataFrame(
        {
            "row_id": title_data["row_id"].to_numpy(dtype=int),
            "cleaned_title_sha256": title_data["content"].map(
                lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
            ),
            "label": title_data["label"].to_numpy(dtype=int),
            "near_duplicate_group": title_groups,
            "outer_fold": title_folds,
        }
    )
    assignments.to_csv(args.output / "isot_title_group_fold_assignments.csv", index=False)
    edge_rows = [
        {
            "first_row_id": int(combined.iloc[first]["row_id"]),
            "second_row_id": int(combined.iloc[second]["row_id"]),
            "jaccard": similarity,
        }
        for first, second, similarity in edges
    ]
    pd.DataFrame(edge_rows).to_csv(
        args.output / "isot_near_duplicate_edges_jaccard_0_80.csv", index=False
    )
    (args.output / "isot_group_reconstruction_summary.json").write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "candidate_pairs": candidates,
                "verified_edges": len(edges),
                "multi_document_groups": len(multi),
                "documents_in_multi_document_groups": sum(map(len, multi)),
                "component_size_counts": dict(sorted(sizes.items())),
                "mixed_label_components": mixed,
                "combined_rows": len(combined),
                "title_rows": len(title_data),
                "fold_construction_population": "combined-text C1; title rows inherit folds",
                "fold_counts": {
                    str(key): int(value)
                    for key, value in assignments["outer_fold"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
