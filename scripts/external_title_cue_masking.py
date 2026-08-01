#!/usr/bin/env python3
"""Post-hoc audit and masking sensitivity for source/style cues in external titles."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.svm import LinearSVC


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from external_error_analysis import load_politifact  # noqa: E402
from fake_news_experiments import load_dataset, sha256_file  # noqa: E402


# Aliases are label-blind and restricted to distinctive source names. Ambiguous
# single-word brands such as people, time, today, and medium are excluded.
BRANDS: dict[str, dict[str, list[str]]] = {
    "youtube": {"domains": ["youtube.com"], "aliases": ["youtube"]},
    "politifact": {"domains": ["politifact.com"], "aliases": ["politifact"]},
    "abc_news": {"domains": ["abcnews.go.com"], "aliases": ["abc news"]},
    "new_york_times": {"domains": ["nytimes.com"], "aliases": ["new york times", "ny times"]},
    "washington_post": {"domains": ["washingtonpost.com"], "aliases": ["washington post"]},
    "your_news_wire": {"domains": ["yournewswire.com"], "aliases": ["your news wire", "yournewswire"]},
    "fox_news": {"domains": ["foxnews.com"], "aliases": ["fox news"]},
    "cnn": {"domains": ["cnn.com"], "aliases": ["cnn"]},
    "cbs_news": {"domains": ["cbsnews.com"], "aliases": ["cbs news"]},
    "nbc_news": {"domains": ["nbcnews.com"], "aliases": ["nbc news"]},
    "politico": {"domains": ["politico.com"], "aliases": ["politico"]},
    "the_hill": {"domains": ["thehill.com"], "aliases": ["the hill"]},
    "breitbart": {"domains": ["breitbart.com"], "aliases": ["breitbart"]},
    "usa_today": {"domains": ["usatoday.com"], "aliases": ["usa today"]},
    "daily_mail": {"domains": ["dailymail.co.uk"], "aliases": ["daily mail"]},
    "us_weekly": {"domains": ["usmagazine.com"], "aliases": ["us weekly", "us magazine"]},
    "entertainment_tonight": {"domains": ["etonline.com"], "aliases": ["entertainment tonight", "et online"]},
    "long_room": {"domains": ["longroom.com"], "aliases": ["long room"]},
    "wikipedia": {"domains": ["wikipedia.org"], "aliases": ["wikipedia"]},
    "hollywood_life": {"domains": ["hollywoodlife.com"], "aliases": ["hollywood life"]},
    "hollywood_reporter": {"domains": ["hollywoodreporter.com"], "aliases": ["hollywood reporter"]},
    "entertainment_weekly": {"domains": ["ew.com"], "aliases": ["entertainment weekly"]},
    "msn": {"domains": ["msn.com"], "aliases": ["msn"]},
    "radar_online": {"domains": ["radaronline.com"], "aliases": ["radar online"]},
    "billboard": {"domains": ["billboard.com"], "aliases": ["billboard"]},
    "the_wrap": {"domains": ["thewrap.com"], "aliases": ["the wrap"]},
    "inquisitr": {"domains": ["inquisitr.com"], "aliases": ["inquisitr"]},
    "harpers_bazaar": {"domains": ["harpersbazaar.com"], "aliases": ["harper s bazaar", "harpers bazaar"]},
    "bravo_tv": {"domains": ["bravotv.com"], "aliases": ["bravo tv"]},
    "celebrity_insider": {"domains": ["celebrityinsider.org"], "aliases": ["celebrity insider"]},
    "imdb": {"domains": ["imdb.com"], "aliases": ["imdb"]},
    "cosmopolitan": {"domains": ["cosmopolitan.com"], "aliases": ["cosmopolitan"]},
    "vanity_fair": {"domains": ["vanityfair.com"], "aliases": ["vanity fair"]},
    "e_news": {"domains": ["eonline.com"], "aliases": ["e news", "e online"]},
    "page_six": {"domains": ["pagesix.com"], "aliases": ["page six"]},
    "instyle": {"domains": ["instyle.com"], "aliases": ["instyle", "in style"]},
    "refinery29": {"domains": ["refinery29.com"], "aliases": ["refinery 29", "refinery29"]},
    "new_idea": {"domains": ["newidea.com.au"], "aliases": ["new idea"]},
    "huffington_post": {"domains": ["huffingtonpost.com", "huffpost.com"], "aliases": ["huffington post", "huffpost"]},
}


STYLE_PATTERNS: dict[str, str] = {
    "according_to": r"\baccording to\b",
    "reportedly": r"\breportedly\b",
    "report_or_reports": r"\breports?\b",
    "sources_say": r"\bsources? (?:say|said|claim|claimed)\b",
    "insider_says": r"\binsiders? (?:say|said|claim|claimed)\b",
    "exclusive": r"\bexclusive\b",
    "breaking": r"\bbreaking\b",
    "claim_or_claims": r"\bclaims?\b",
    "allegedly": r"\ballegedly\b",
    "rumor_or_rumour": r"\brumou?rs?\b",
    "fact_check": r"\bfact check\b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isot-fake", type=Path, required=True)
    parser.add_argument("--isot-true", type=Path, required=True)
    parser.add_argument("--politifact-fake", type=Path, required=True)
    parser.add_argument("--politifact-real", type=Path, required=True)
    parser.add_argument("--gossipcop-fake", type=Path, required=True)
    parser.add_argument("--gossipcop-real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-source-frequency", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=5213)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def normalized_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def domain_matches(domain: str, candidate: str) -> bool:
    return domain == candidate or domain.endswith("." + candidate)


def compile_alias(alias: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])")


def eligible_brands(data: pd.DataFrame, minimum_frequency: int) -> dict[str, dict[str, object]]:
    domain_counts = data["domain"].value_counts().to_dict()
    selected: dict[str, dict[str, object]] = {}
    for name, specification in BRANDS.items():
        matched_domains = [
            domain
            for domain, count in domain_counts.items()
            if count >= minimum_frequency
            and any(domain_matches(domain, candidate) for candidate in specification["domains"])
        ]
        if matched_domains:
            selected[name] = {
                "domains": matched_domains,
                "aliases": specification["aliases"],
                "patterns": [compile_alias(alias) for alias in specification["aliases"]],
            }
    return selected


def find_brand_hits(text: str, brands: dict[str, dict[str, object]]) -> list[str]:
    return [
        name
        for name, specification in brands.items()
        if any(pattern.search(text) for pattern in specification["patterns"])
    ]


def find_style_hits(text: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    return [name for name, pattern in patterns.items() if pattern.search(text)]


def mask_patterns(text: str, patterns: list[re.Pattern[str]]) -> str:
    masked = text
    for pattern in patterns:
        masked = pattern.sub(" ", masked)
    return re.sub(r"\s+", " ", masked).strip()


def point_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    fake = labels == 0
    real = labels == 1
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "fake_recall": float(np.mean(predictions[fake] == 0)),
        "real_recall": float(np.mean(predictions[real] == 1)),
    }


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


def fit_svm(train_text: pd.Series, labels: np.ndarray, max_features: int, seed: int):
    vectorizer = make_vectorizer(max_features)
    matrix = vectorizer.fit_transform(train_text)
    model = LinearSVC(C=1.0, random_state=seed)
    model.fit(matrix, labels)
    return vectorizer, model


def bootstrap_differences(
    labels: np.ndarray,
    baseline: np.ndarray,
    alternative: np.ndarray,
    resamples: int,
    seed: int,
) -> list[dict[str, float | str]]:
    observed_base = point_metrics(labels, baseline)
    observed_alt = point_metrics(labels, alternative)
    keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    draws = {key: np.empty(resamples, dtype=float) for key in keys}
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    for index in range(resamples):
        sampled = np.concatenate(
            [rng.choice(items, size=len(items), replace=True) for items in class_indices]
        )
        base_metrics = point_metrics(labels[sampled], baseline[sampled])
        alt_metrics = point_metrics(labels[sampled], alternative[sampled])
        for key in keys:
            draws[key][index] = alt_metrics[key] - base_metrics[key]
    rows: list[dict[str, float | str]] = []
    for key in keys:
        lower, upper = np.quantile(draws[key], [0.025, 0.975])
        rows.append(
            {
                "metric": key,
                "baseline": observed_base[key],
                "alternative": observed_alt[key],
                "alternative_minus_baseline": observed_alt[key] - observed_base[key],
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
            }
        )
    return rows


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
    isot_text = isot["content"].reset_index(drop=True)
    isot_labels = isot["label"].to_numpy(dtype=int)
    externals = {}
    external_counts = {}
    for name, fake, real in (
        ("PolitiFact", args.politifact_fake, args.politifact_real),
        ("GossipCop", args.gossipcop_fake, args.gossipcop_real),
    ):
        externals[name], external_counts[name] = load_politifact(fake, real)

    style_patterns = {name: re.compile(value) for name, value in STYLE_PATTERNS.items()}
    all_style_patterns = list(style_patterns.values())
    baseline_vectorizer, baseline_model = fit_svm(
        isot_text, isot_labels, args.max_features, args.seed
    )

    prevalence_rows: list[dict[str, object]] = []
    cue_count_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    corpus_config: dict[str, object] = {}

    for corpus, data in externals.items():
        data = data.copy()
        brands = eligible_brands(data, args.minimum_source_frequency)
        brand_patterns = [
            pattern
            for specification in brands.values()
            for pattern in specification["patterns"]
        ]
        data["publisher_hits"] = data["content"].map(lambda text: find_brand_hits(text, brands))
        data["style_hits"] = data["content"].map(
            lambda text: find_style_hits(text, style_patterns)
        )
        data["publisher_cue"] = data["publisher_hits"].map(bool)
        data["style_cue"] = data["style_hits"].map(bool)
        data["combined_cue"] = data["publisher_cue"] | data["style_cue"]

        def is_self_brand(record: pd.Series) -> bool:
            return any(
                hit in brands
                and any(
                    domain_matches(str(record["domain"]), domain)
                    for domain in brands[hit]["domains"]
                )
                for hit in record["publisher_hits"]
            )

        data["self_publisher_cue"] = data.apply(is_self_brand, axis=1)
        data["publisher_masked"] = data["content"].map(
            lambda text: mask_patterns(text, brand_patterns)
        )
        data["style_masked"] = data["content"].map(
            lambda text: mask_patterns(text, all_style_patterns)
        )
        data["combined_masked"] = data["content"].map(
            lambda text: mask_patterns(mask_patterns(text, brand_patterns), all_style_patterns)
        )

        labels = data["label"].to_numpy(dtype=int)
        baseline_predictions = baseline_model.predict(
            baseline_vectorizer.transform(data["content"])
        )
        predictions: dict[str, np.ndarray] = {"baseline_locked": baseline_predictions}
        for condition, column in (
            ("publisher_masked_locked", "publisher_masked"),
            ("style_masked_locked", "style_masked"),
            ("combined_masked_locked", "combined_masked"),
        ):
            predictions[condition] = baseline_model.predict(
                baseline_vectorizer.transform(data[column])
            )

        symmetric_train = isot_text.map(
            lambda text: mask_patterns(mask_patterns(text, brand_patterns), all_style_patterns)
        )
        symmetric_vectorizer, symmetric_model = fit_svm(
            symmetric_train, isot_labels, args.max_features, args.seed
        )
        predictions["combined_masked_symmetric_refit"] = symmetric_model.predict(
            symmetric_vectorizer.transform(data["combined_masked"])
        )

        for cue_type, column in (
            ("publisher_any", "publisher_cue"),
            ("publisher_self", "self_publisher_cue"),
            ("style_any", "style_cue"),
            ("publisher_or_style", "combined_cue"),
        ):
            mask = data[column].to_numpy(dtype=bool)
            prevalence_rows.append(
                {
                    "corpus": corpus,
                    "cue_type": cue_type,
                    "n": int(mask.sum()),
                    "prevalence": float(mask.mean()),
                    "fake_n": int(np.sum(mask & (labels == 0))),
                    "fake_prevalence": float(mask[labels == 0].mean()),
                    "real_n": int(np.sum(mask & (labels == 1))),
                    "real_prevalence": float(mask[labels == 1].mean()),
                }
            )

        for cue_family, lists in (
            ("publisher", data["publisher_hits"]),
            ("style", data["style_hits"]),
        ):
            all_counts = Counter(item for values in lists for item in values)
            fake_counts = Counter(
                item
                for values in lists.loc[data["label"].eq(0)]
                for item in values
            )
            real_counts = Counter(
                item
                for values in lists.loc[data["label"].eq(1)]
                for item in values
            )
            for cue, count in sorted(all_counts.items(), key=lambda item: (-item[1], item[0])):
                cue_count_rows.append(
                    {
                        "corpus": corpus,
                        "cue_family": cue_family,
                        "cue": cue,
                        "total_occurrences": count,
                        "fake_occurrences": fake_counts[cue],
                        "real_occurrences": real_counts[cue],
                    }
                )

        for condition, condition_predictions in predictions.items():
            metric_rows.append(
                {
                    "corpus": corpus,
                    "condition": condition,
                    "titles_changed_by_mask": (
                        0
                        if condition == "baseline_locked"
                        else int(
                            data[
                                "combined_masked"
                                if condition == "combined_masked_symmetric_refit"
                                else condition.replace("_locked", "")
                            ].ne(data["content"]).sum()
                        )
                    ),
                    "prediction_changes_vs_baseline": int(
                        np.sum(condition_predictions != baseline_predictions)
                    ),
                    **point_metrics(labels, condition_predictions),
                }
            )
            if condition != "baseline_locked":
                for row in bootstrap_differences(
                    labels,
                    baseline_predictions,
                    condition_predictions,
                    args.bootstrap_resamples,
                    args.bootstrap_seed,
                ):
                    bootstrap_rows.append(
                        {"corpus": corpus, "condition": condition, **row}
                    )

        style_rule = np.where(data["style_cue"].to_numpy(dtype=bool), 0, 1)
        rule_rows.append(
            {
                "corpus": corpus,
                "rule": "predict fake iff a predeclared style cue is present",
                **point_metrics(labels, style_rule),
            }
        )

        for index, record in data.iterrows():
            row = {
                "corpus": corpus,
                "id": str(record["id"]),
                "label": int(record["label"]),
                "cleaned_title_sha256": normalized_hash(record["content"]),
                "publisher_masked_sha256": normalized_hash(record["publisher_masked"]),
                "style_masked_sha256": normalized_hash(record["style_masked"]),
                "combined_masked_sha256": normalized_hash(record["combined_masked"]),
                "publisher_cue": bool(record["publisher_cue"]),
                "self_publisher_cue": bool(record["self_publisher_cue"]),
                "style_cue": bool(record["style_cue"]),
                "publisher_cue_count": len(record["publisher_hits"]),
                "style_cue_count": len(record["style_hits"]),
            }
            for condition, condition_predictions in predictions.items():
                row[f"prediction_{condition}"] = int(condition_predictions[index])
            prediction_rows.append(row)

        corpus_config[corpus] = {
            "eligible_brand_count": len(brands),
            "eligible_brands": {
                name: {"domains": value["domains"], "aliases": value["aliases"]}
                for name, value in brands.items()
            },
            "publisher_titles_changed": int(data["publisher_masked"].ne(data["content"]).sum()),
            "style_titles_changed": int(data["style_masked"].ne(data["content"]).sum()),
            "combined_titles_changed": int(data["combined_masked"].ne(data["content"]).sum()),
            "empty_after_combined_mask": int(data["combined_masked"].eq("").sum()),
        }

    pd.DataFrame(prevalence_rows).to_csv(
        args.output / "external_title_cue_prevalence.csv", index=False
    )
    pd.DataFrame(cue_count_rows).to_csv(
        args.output / "external_title_cue_counts.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(
        args.output / "external_title_cue_masking_metrics.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output / "external_title_cue_masking_paired_bootstrap.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        args.output / "external_title_cue_predictions_without_text.csv", index=False
    )
    pd.DataFrame(rule_rows).to_csv(
        args.output / "external_title_style_rule_metrics.csv", index=False
    )

    config = {
        "analysis_status": "post-hoc reviewer-motivated source/style title-cue audit",
        "model": "fixed title-only TF-IDF + Linear SVM",
        "conditions": [
            "baseline locked model on original external titles",
            "locked model after external publisher-name masking",
            "locked model after external style-phrase masking",
            "locked model after combined external masking",
            "symmetric combined masking followed by ISOT refit",
        ],
        "publisher_lexicon_policy": {
            "minimum_unlabeled_hostname_frequency": args.minimum_source_frequency,
            "ambiguous_single_word_aliases_excluded": [
                "people",
                "time",
                "today",
                "medium",
                "deadline",
                "variety",
            ],
            "labels_used_to_select_aliases": False,
        },
        "style_patterns": STYLE_PATTERNS,
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "method": "paired class-stratified percentile",
            "interval_status": "marginal descriptive 95% intervals; no simultaneous coverage claim",
        },
        "counts": {"isot": isot_counts, **external_counts},
        "corpora": corpus_config,
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
    (args.output / "external_title_cue_masking_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(prevalence_rows).to_string(index=False))
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(pd.DataFrame(rule_rows).to_string(index=False))


if __name__ == "__main__":
    main()
