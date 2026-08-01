#!/usr/bin/env python3
"""Small end-to-end smoke test for the v1.1.0 workflows."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def alphabetic_id(value: int) -> str:
    letters: list[str] = []
    value += 1
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("a") + remainder))
    return "event" + "".join(reversed(letters))


def write_isot(path: Path, label: int, rows: int = 45) -> None:
    class_words = (
        "fabricated rumor doubtful invented" if label == 0 else "verified report official reuters"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["title", "text", "subject", "date"])
        for index in range(rows):
            token = alphabetic_id(index + label * rows)
            date = "December 15, 2016" if index < rows // 2 else "March 15, 2017"
            writer.writerow(
                [
                    f"{class_words} {token} headline",
                    f"{token} evidence context analysis details {class_words}",
                    "News",
                    date,
                ]
            )


def write_politifact(path: Path, label: int, rows: int = 15) -> None:
    class_words = "fabricated rumor" if label == 0 else "verified official"
    frame = pd.DataFrame(
        {
            "id": [f"p{label}-{index}" for index in range(rows)],
            "news_url": [f"https://example{label}.org/{index}" for index in range(rows)],
            "title": [
                f"{class_words} transfer {alphabetic_id(index + label * rows)}"
                for index in range(rows)
            ],
            "tweet_ids": [""] * rows,
        }
    )
    frame.to_csv(path, index=False)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="audit-smoke-") as directory:
        temporary = Path(directory)
        fake = temporary / "Fake.csv"
        true = temporary / "True.csv"
        politifact_fake = temporary / "politifact_fake.csv"
        politifact_real = temporary / "politifact_real.csv"
        write_isot(fake, 0)
        write_isot(true, 1)
        write_politifact(politifact_fake, 0)
        write_politifact(politifact_real, 1)

        group_output = temporary / "group"
        run(
            [
                sys.executable,
                str(SCRIPTS / "group_aware_internal_validation.py"),
                "--isot-fake",
                str(fake),
                "--isot-true",
                str(true),
                "--output",
                str(group_output),
                "--models",
                "svm",
                "--folds",
                "3",
                "--skip-conventional-holdout",
            ]
        )
        run(
            [
                sys.executable,
                str(SCRIPTS / "chronological_evaluation.py"),
                "--isot-fake",
                str(fake),
                "--isot-true",
                str(true),
                "--output",
                str(temporary / "chronological"),
                "--models",
                "svm",
            ]
        )
        run(
            [
                sys.executable,
                str(SCRIPTS / "calibration_transfer.py"),
                "--isot-fake",
                str(fake),
                "--isot-true",
                str(true),
                "--politifact-fake",
                str(politifact_fake),
                "--politifact-real",
                str(politifact_real),
                "--fold-assignments",
                str(group_output / "isot_title_inherited_group_fold_assignments.csv"),
                "--output",
                str(temporary / "calibration"),
                "--bins",
                "5",
            ]
        )

        required = (
            group_output / "group_validation_summary.csv",
            temporary / "chronological" / "chronological_metrics.csv",
            temporary / "calibration" / "calibration_metrics.csv",
        )
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Expected non-empty smoke-test output: {path}")
        group_summary = pd.read_csv(required[0])
        if set(group_summary["representation"]) != {
            "full_text",
            "full_text_reuters_masked",
            "title",
        }:
            raise RuntimeError("Group-validation smoke output is incomplete")
        chronological = pd.read_csv(required[1])
        if set(chronological["model"]) != {
            "svm",
            "reuters_rule",
            "svm_reuters_masked",
        }:
            raise RuntimeError("Chronological smoke output is incomplete")
        calibration = pd.read_csv(required[2])
        if len(calibration) != 3 or not calibration["brier_score"].between(0, 1).all():
            raise RuntimeError("Calibration smoke output failed range checks")
    print("v1.1.0 workflow smoke test passed")


if __name__ == "__main__":
    main()
