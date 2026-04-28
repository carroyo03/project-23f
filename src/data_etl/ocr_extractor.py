
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
import re
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

MIN_ALPHA_CHARS = 40
MAX_PAGE_MARKER_RATIO = 0.7


_OCR_READER = None
_PAGE_MARKER_RE = re.compile(
    r"^\s*(?:[-–—\s]*)?(?:page|p[aá]gina)\s*[:\-]?\s*\d+(?:\s*(?:/|de)\s*\d+)?\s*(?:[-–—\s]*)$",
    flags=re.IGNORECASE,
)


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


def detect_ocr_device() -> str:
    """Detect and return the device type: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _assess_ocr_quality(page_texts: list[str]) -> tuple[bool, float, str]:
    """Returns (is_bad, page_marker_ratio, reason) for OCR text quality checks."""
    joined = "\n".join(t for t in page_texts if isinstance(t, str))
    lines = [ln.strip() for ln in joined.splitlines() if ln.strip()]

    if not lines:
        return True, 1.0, "empty_text"

    marker_lines = sum(1 for ln in lines if _PAGE_MARKER_RE.match(ln))
    marker_ratio = marker_lines / len(lines)
    alpha_chars = sum(ch.isalpha() for ch in joined)

    if marker_ratio >= MAX_PAGE_MARKER_RATIO and alpha_chars < MIN_ALPHA_CHARS:
        return True, marker_ratio, "mostly_page_markers"
    return False, marker_ratio, "ok"


# ---------------------------------------------------------------------------
# Worker — must be at module level to be picklable by ProcessPoolExecutor
# ---------------------------------------------------------------------------


def _ocr_worker(args: tuple[str, str]) -> dict:
    """
    Extracts text from a scanned PDF using EasyOCR (GPU/MPS acceleration if available).

    Args:
        args: (pdf_path_str, txt_path_str)

    Returns:
        dict with {filename, status, chars, error}.
    """
    pdf_path_str, txt_path_str = args
    pdf_path = Path(pdf_path_str)
    txt_path = Path(txt_path_str)

    if not pdf_path.exists():
        return {"filename": pdf_path.name, "status": "pdf_not_found", "chars": 0, "error": "PDF not found"}

    try:
        # Reuse one EasyOCR reader per process with GPU acceleration if available.
        reader = _get_reader(gpu=_has_accelerator())
        pages_text: list[str] = []
        raw_ocr_texts: list[str] = []

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                img: Image.Image = page.to_image(resolution=RESOLUTION).original
                img_np = np.array(img)
                # EasyOCR expects RGB images
                if img_np.ndim == 2:
                    img_np = np.stack([img_np]*3, axis=-1)
                result = reader.readtext(img_np, detail=0, paragraph=True)
                text = "\n".join(result)
                raw_ocr_texts.append(text)
                pages_text.append(f"--- Page {i + 1} ---\n{text}")

        full_text = "\n\n".join(pages_text)
        is_bad, marker_ratio, reason = _assess_ocr_quality(raw_ocr_texts)

        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(full_text, encoding="utf-8")

        if is_bad:
            return {
                "filename": pdf_path.name,
                "status": "bad_extraction",
                "chars": len(full_text),
                "error": reason,
                "page_marker_ratio": marker_ratio,
            }

        return {
            "filename": pdf_path.name,
            "status": "success",
            "chars": len(full_text),
            "error": None,
            "page_marker_ratio": marker_ratio,
        }

    except Exception as e:
        return {
            "filename": pdf_path.name,
            "status": "error",
            "chars": 0,
            "error": str(e),
            "page_marker_ratio": None,
        }


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
    device = detect_ocr_device()
    print(f"Using device: {device}")
    _warm_up_easyocr_cache(gpu=_has_accelerator())

    # Build task list (pdf_path, txt_path)
    tasks: list[tuple[str, str]] = []
    for _, row in pending.iterrows():
        rel_path = Path(str(row["rel_path"]))
        pdf_path = RAW_DIR / rel_path
        txt_path = PROCESSED_DIR / rel_path.with_suffix(".txt")
        tasks.append((str(pdf_path), str(txt_path)))

    has_accel = _has_accelerator()
    if has_accel:
        workers = 1 if max_workers is None else max_workers
        if workers > 1 and os.getenv("OCR_FORCE_MULTI_GPU", "0") != "1":
            print("WARNING: Forcing workers=1 for GPU/MPS OCR stability.")
            workers = 1
    else:
        workers = max_workers or max(1, (os.cpu_count() or 2) // 2)

    print(f"Parallel processes : {workers}")
    print(f"GPU acceleration   : {'enabled' if has_accel else 'disabled (CPU only)'}")
    print(f"Starting OCR...\n")

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_ocr_worker, task): task for task in tasks}
        for future in tqdm(as_completed(future_map), total=len(tasks), desc="OCR"):
            results.append(future.result())

    df_results = pd.DataFrame(results)
    success = df_results[df_results["status"] == "success"]
    bad = df_results[df_results["status"] == "bad_extraction"]
    errors = df_results[df_results["status"] == "error"]

    print("\n" + "=" * 60)
    print(f"✓ Successful: {len(success)}")
    print(f"⚠ Bad OCR    : {len(bad)}")
    print(f"✗ Errors    : {len(errors)}")
    if not success.empty:
        print(f"  Extracted chars (total)  : {success['chars'].sum():,}")
        print(f"  Extracted chars (mean)   : {success['chars'].mean():.0f}")
    if not errors.empty:
        print("\nFirst errors:")
        for _, e in errors.head(5).iterrows():
            print(f"  {e['filename']}: {e.get('error', '?')}")
    if not bad.empty:
        print("\nFirst bad extractions:")
        for _, b in bad.head(5).iterrows():
            ratio = b.get("page_marker_ratio")
            ratio_str = f"{ratio:.2f}" if pd.notna(ratio) else "n/a"
            print(f"  {b['filename']}: reason={b.get('error', '?')} marker_ratio={ratio_str}")

    if update_csv and (not success.empty or not bad.empty):
        df_meta = pd.read_csv(csv_path)

        if "ocr_quality" not in df_meta.columns:
            df_meta["ocr_quality"] = ""
        if "ocr_page_marker_ratio" not in df_meta.columns:
            df_meta["ocr_page_marker_ratio"] = np.nan

        for _, res in success.iterrows():
            mask = df_meta["filename"] == res["filename"]
            df_meta.loc[mask, "char_count"] = res["chars"]
            df_meta.loc[mask, "is_scanned"] = False
            df_meta.loc[mask, "success"] = True
            df_meta.loc[mask, "ocr_quality"] = "ok"
            df_meta.loc[mask, "ocr_page_marker_ratio"] = res.get("page_marker_ratio", np.nan)

        for _, res in bad.iterrows():
            mask = df_meta["filename"] == res["filename"]
            df_meta.loc[mask, "is_scanned"] = True
            df_meta.loc[mask, "success"] = False
            df_meta.loc[mask, "ocr_quality"] = "bad_page_markers"
            df_meta.loc[mask, "ocr_page_marker_ratio"] = res.get("page_marker_ratio", np.nan)

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
