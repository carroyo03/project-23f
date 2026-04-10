
"""
OCR Extractor: Extracts text from scanned PDFs using EasyOCR (GPU/MPS acceleration)

Processes PDFs marked as scanned (char_count == 0) by applying OCR.
- Parallel processing using ProcessPoolExecutor (CPU-bound)
- Each process runs EasyOCR independently (GPU/MPS acceleration if available)
- Updates the metadata CSV upon completion

Usage:
    python src/ocr_extractor.py              # Process all scanned PDFs
    python src/ocr_extractor.py --dry-run    # Only show what would be processed
    python src/ocr_extractor.py --limit 10   # Limit to 10 for testing
"""

import argparse
import os
import ssl
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


import pandas as pd
import pdfplumber
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import easyocr

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"


OCR_LANG = "es"  # EasyOCR uses 'es' for Spanish
RESOLUTION = 300  # DPI for page-to-image conversion


_OCR_READER = None


def _allow_easyocr_model_downloads() -> None:
    """Allow EasyOCR to fetch its model files in environments with broken CA bundles."""
    ssl._create_default_https_context = ssl._create_unverified_context


def _has_accelerator() -> bool:
    """Detect whether a GPU or MPS device is available for EasyOCR."""
    if torch.cuda.is_available():
        return True
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return True
    return False


def _get_reader(gpu: bool = False):
    """Creates one EasyOCR reader per process and reuses it across tasks."""
    global _OCR_READER
    if _OCR_READER is None:
        _allow_easyocr_model_downloads()
        _OCR_READER = easyocr.Reader([OCR_LANG], gpu=gpu)
    return _OCR_READER


def _warm_up_easyocr_cache(gpu: bool = False) -> None:
    """Downloads EasyOCR models once in the parent process before spawning workers."""
    _get_reader(gpu=gpu)


# ---------------------------------------------------------------------------
# Worker — must be at module level to be picklable by ProcessPoolExecutor
# ---------------------------------------------------------------------------


def _ocr_worker(args: tuple[str, str]) -> dict:
    """
    Extracts text from a scanned PDF using EasyOCR (GPU/MPS acceleration if available).

    Args:
        args: (pdf_path_str, txt_path_str)

    Returns:
        dict with {archivo, status, chars, error}.
    """
    pdf_path_str, txt_path_str = args
    pdf_path = Path(pdf_path_str)
    txt_path = Path(txt_path_str)

    if not pdf_path.exists():
        return {"filename": pdf_path.name, "status": "pdf_not_found", "chars": 0, "error": "PDF not found"}

    try:
        # Reuse one EasyOCR reader per process.
        reader = _get_reader()
        pages_text: list[str] = []

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                img: Image.Image = page.to_image(resolution=RESOLUTION).original
                img_np = np.array(img)
                # EasyOCR expects RGB images
                if img_np.ndim == 2:
                    img_np = np.stack([img_np]*3, axis=-1)
                result = reader.readtext(img_np, detail=0, paragraph=True)
                text = "\n".join(result)
                pages_text.append(f"--- Page {i + 1} ---\n{text}")

        full_text = "\n\n".join(pages_text)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(full_text, encoding="utf-8")

        return {"filename": pdf_path.name, "status": "success", "chars": len(full_text), "error": None}

    except Exception as e:
        return {"filename": pdf_path.name, "status": "error", "chars": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_scanned_pdfs(
    dry_run: bool = False,
    limit: int | None = None,
    update_csv: bool = True,
    max_workers: int | None = None,
) -> pd.DataFrame | None:
    """
    Applies OCR to all PDFs with char_count == 0.

    Args:
        dry_run:     Only shows what would be processed, without executing.
        limit:       Limit to N PDFs (useful for testing).
        update_csv:  Update char_count and is_scanned in the metadata CSV.
        max_workers: Parallel processes. Default: os.cpu_count() // 2.
    """
    print("=" * 60)
    print("OCR EXTRACTOR - Scanned PDFs (EasyOCR GPU/MPS)")
    print("=" * 60)

    csv_path = METADATA_DIR / "documents_enriched.csv"
    if not csv_path.exists():
        print(f"ERROR: Does not exist {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    pending = df[(df["is_scanned"] == True) | (df["char_count"] == 0)].copy()  # noqa: E712

    print(f"\nPending OCR: {len(pending)}")

    if len(pending) == 0:
        print("All PDFs have been processed. Nothing left to do.")
        return None

    if dry_run:
        print("\n--- DRY RUN: First 10 to be processed ---")
        for _, row in pending.head(10).iterrows():
            print(f"  {row['rel_path']}")
        print("(EasyOCR, GPU/MPS acceleration if available)")
        return None

    if limit:
        pending = pending.head(limit)
        print(f"Limiting to {limit} PDFs for testing\n")

    print("Pre-warming EasyOCR model cache...")
    _warm_up_easyocr_cache()

    # Build task list (pdf_path, txt_path)
    tasks: list[tuple[str, str]] = []
    for _, row in pending.iterrows():
        rel_path = Path(str(row["rel_path"]))
        pdf_path = RAW_DIR / rel_path
        txt_path = PROCESSED_DIR / rel_path.with_suffix(".txt")
        tasks.append((str(pdf_path), str(txt_path)))

    workers = max_workers or max(1, (os.cpu_count() or 2) // 2)
    print(f"Parallel processes : {workers}")
    print(f"Starting OCR...\n")

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_ocr_worker, task): task for task in tasks}
        for future in tqdm(as_completed(future_map), total=len(tasks), desc="OCR"):
            results.append(future.result())

    df_results = pd.DataFrame(results)
    success = df_results[df_results["status"] == "success"]
    errors = df_results[df_results["status"] == "error"]

    print("\n" + "=" * 60)
    print(f"✓ Successful: {len(success)}")
    print(f"✗ Errors    : {len(errors)}")
    if not success.empty:
        print(f"  Extracted chars (total)  : {success['chars'].sum():,}")
        print(f"  Extracted chars (mean)   : {success['chars'].mean():.0f}")
    if not errors.empty:
        print("\nFirst errors:")
        for _, e in errors.head(5).iterrows():
            print(f"  {e['archivo']}: {e.get('error', '?')}")

    if update_csv and not success.empty:
        df_meta = pd.read_csv(csv_path)
        for _, res in success.iterrows():
            mask = df_meta["filename"] == res["filename"]
            df_meta.loc[mask, "char_count"] = res["chars"]
            df_meta.loc[mask, "is_scanned"] = False
        df_meta.to_csv(csv_path, index=False)
        print(f"\nCSV updated: {csv_path}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR for 23-F scanned PDFs")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be processed")
    parser.add_argument("--limit", type=int, help="Limit to N PDFs")
    parser.add_argument("--workers", type=int, default=None, help="Parallel processes (default: cpu_count//2)")
    parser.add_argument("--no-update", action="store_true", help="Do not update CSV")
    args = parser.parse_args()

    process_scanned_pdfs(
        dry_run=args.dry_run,
        limit=args.limit,
        update_csv=not args.no_update,
        max_workers=args.workers,
    )
