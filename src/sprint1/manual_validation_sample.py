"""Generate a manual validation sample from documents_enriched.csv.

Sprint 1 helper for quick human QA of extracted metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META_CSV = ROOT / "data" / "metadata" / "documents_enriched.csv"
OUT_CSV = ROOT / "outputs" / "sprint1" / "manual_validation_sample.csv"


def build_sample(sample_size: int, seed: int) -> pd.DataFrame:
    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {META_CSV}")

    df = pd.read_csv(META_CSV)
    if df.empty:
        raise ValueError("documents_enriched.csv is empty")

    selected_columns = [
        "name",
        "url",
        "ministry",
        "category",
        "filename",
        "rel_path",
        "char_count",
        "is_scanned",
        "date",
        "doc_type",
        "success",
    ]
    existing = [c for c in selected_columns if c in df.columns]
    sample_n = min(sample_size, len(df))

    # Prefer a mixed sample so handwritten/scanned PDFs are always represented.
    scanned = df[df["is_scanned"] == True] if "is_scanned" in df.columns else pd.DataFrame()  # noqa: E712
    regular = df[df["is_scanned"] != True] if "is_scanned" in df.columns else df  # noqa: E712

    scanned_n = min(max(1, sample_n // 2) if len(scanned) else 0, len(scanned))
    regular_n = min(sample_n - scanned_n, len(regular))

    sampled_parts = []
    if scanned_n:
        sampled_parts.append(scanned[existing].sample(n=scanned_n, random_state=seed))
    if regular_n:
        sampled_parts.append(regular[existing].sample(n=regular_n, random_state=seed + 1))

    sample = pd.concat(sampled_parts, axis=0).drop_duplicates()
    if len(sample) < sample_n:
        remaining = df.loc[~df.index.isin(sample.index), existing]
        extra = remaining.sample(n=min(sample_n - len(sample), len(remaining)), random_state=seed + 2)
        sample = pd.concat([sample, extra], axis=0)

    sample = sample.sort_index().copy()

    # Columns for manual QA round by the team.
    sample["manual_date"] = ""
    sample["manual_doc_type"] = ""
    sample["date_ok"] = ""
    sample["doc_type_ok"] = ""
    sample["needs_ocr"] = sample["is_scanned"] if "is_scanned" in sample.columns else ""
    sample["notes"] = ""

    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manual validation sample")
    parser.add_argument("--n", type=int, default=15, help="Rows to sample (default: 15)")
    parser.add_argument("--seed", type=int, default=23, help="Random seed (default: 23)")
    args = parser.parse_args()

    sample = build_sample(sample_size=args.n, seed=args.seed)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"Saved sample: {OUT_CSV}")
    print(f"Rows: {len(sample)}")
    print("Next: fill manual_* and *_ok columns for QA report.")


if __name__ == "__main__":
    main()
