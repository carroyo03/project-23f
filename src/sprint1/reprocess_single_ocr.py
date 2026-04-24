"""Reprocess a single Moncloa PDF with OCR and update documents_enriched.csv.

Usage:
    python src/sprint1/reprocess_single_ocr.py --filename Documento_38_R.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pdfplumber

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src" / "data_etl") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "data_etl"))

from ocr_extractor import _get_reader, detect_ocr_device  # noqa: E402

META_CSV = ROOT / "data" / "metadata" / "documents_enriched.csv"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reprocess one PDF with OCR")
    parser.add_argument("--filename", required=True, help="Exact PDF filename")
    parser.add_argument("--resolution", type=int, default=180, help="OCR render DPI (default: 180)")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional max pages to OCR")
    args = parser.parse_args()

    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {META_CSV}")

    df = pd.read_csv(META_CSV)
    rows = df[df["filename"] == args.filename]
    if rows.empty:
        raise ValueError(f"Filename not found in documents_enriched.csv: {args.filename}")

    row = rows.iloc[0]
    rel_path = Path(str(row["rel_path"]))
    pdf_path = RAW_DIR / rel_path
    txt_path = PROCESSED_DIR / rel_path.with_suffix(".txt")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    backend = detect_ocr_device()
    use_gpu = backend in {"cuda", "mps"}
    print(f"OCR backend: {backend}")
    print(f"Processing: {args.filename}")

    reader = _get_reader(gpu=use_gpu, download_enabled=True)
    pages_text: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        max_pages = min(total_pages, args.max_pages) if args.max_pages else total_pages
        for i, page in enumerate(pdf.pages[:max_pages], start=1):
            img = np.array(page.to_image(resolution=args.resolution).original)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            parts = reader.readtext(img, detail=0, paragraph=True)
            text = "\n".join(parts)
            pages_text.append(f"--- Page {i} ---\n{text}")
            print(f"Page {i}/{max_pages}: chars={len(text)}")

    full_text = "\n\n".join(pages_text)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(full_text, encoding="utf-8")

    chars = len(full_text)
    mask = df["filename"] == args.filename
    df.loc[mask, "char_count"] = chars
    df.loc[mask, "is_scanned"] = chars < 200
    df.loc[mask, "success"] = True
    df.to_csv(META_CSV, index=False, encoding="utf-8")

    text = txt_path.read_text(encoding="utf-8", errors="ignore") if txt_path.exists() else ""
    print(f"Updated char_count={chars}, nonspace={len(text.strip())}")
    print(f"Saved metadata: {META_CSV}")


if __name__ == "__main__":
    main()
