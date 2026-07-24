from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    accuracy = float(np.mean(y == pred))
    recalls = []
    f1s = []
    for label in (0, 1):
        tp = int(np.sum((y == label) & (pred == label)))
        fn = int(np.sum((y == label) & (pred != label)))
        fp = int(np.sum((y != label) & (pred == label)))
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return accuracy, float(np.mean(recalls)), float(np.mean(f1s))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4211)
    args = parser.parse_args()

    data = pd.read_csv(args.predictions)
    rows: list[dict[str, object]] = []
    for (corpus, model), part in data.groupby(["corpus", "model"], sort=True):
        y = part["label"].to_numpy(dtype=np.int8)
        fixed = part["fixed_prediction"].to_numpy(dtype=np.int8)
        selected = part["selected_prediction"].to_numpy(dtype=np.int8)
        observed_fixed = metrics(y, fixed)
        observed_selected = metrics(y, selected)
        rng = np.random.default_rng(args.seed)
        by_class = [np.flatnonzero(y == label) for label in (0, 1)]
        boot = np.empty((args.resamples, 3), dtype=np.float64)
        for b in range(args.resamples):
            sampled = np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in by_class]
            )
            fixed_b = metrics(y[sampled], fixed[sampled])
            selected_b = metrics(y[sampled], selected[sampled])
            boot[b] = np.subtract(selected_b, fixed_b)
        for metric_index, metric in enumerate(("accuracy", "balanced_accuracy", "macro_f1")):
            difference = observed_selected[metric_index] - observed_fixed[metric_index]
            lower, upper = np.quantile(boot[:, metric_index], [0.025, 0.975])
            rows.append(
                {
                    "corpus": corpus,
                    "model": model,
                    "metric": metric,
                    "fixed": observed_fixed[metric_index],
                    "selected": observed_selected[metric_index],
                    "selected_minus_fixed": difference,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "resamples": args.resamples,
                    "seed": args.seed,
                    "bootstrap": "paired class-stratified percentile",
                }
            )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
