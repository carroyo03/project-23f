"""
23-F Document Pipeline
======================
End-to-end pipeline: PDF extraction → OCR → corpus build → df_final.

Assumes PDFs are already in data/raw/ and RTVE metadata is in
data/metadata/rtve_documents.csv. Run scraper/downloader first if needed.

Usage:
    python pipeline.py              # full run, saves document_corpus.csv
    python pipeline.py --dry-run   # show what would be processed, no writes
    python pipeline.py --ocr       # also run OCR on scanned PDFs

    from pipeline import run
    df_final = run()
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

ENRICHED_CSV = METADATA_DIR / "documents_enriched.csv"
RTVE_CSV     = METADATA_DIR / "rtve_documents.csv"
CORPUS_CSV   = METADATA_DIR / "document_corpus.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1 — Text extraction (pdfplumber, I/O-bound → ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _extract_one(pdf_path: Path, txt_path: Path) -> dict:
    """Extracts text from a single PDF with pdfplumber."""
    import pdfplumber

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(f"--- Page {i + 1} ---\n{text}")
            full_text = _clean_text("\n\n".join(pages))

        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(full_text, encoding="utf-8")

        return {
            "filename": pdf_path.name,
            "char_count": len(full_text),
            "is_scanned": len(full_text) < 200,
            "error": None,
        }
    except Exception as exc:
        return {"filename": pdf_path.name, "char_count": 0, "is_scanned": True, "error": str(exc)}


def run_extraction(df_meta: pd.DataFrame, max_workers: int = 4, force: bool = False) -> pd.DataFrame:
    """
    Extracts text from all PDFs in df_meta that haven't been processed yet.
    Idempotent: skips PDFs whose .txt already exists (unless force=True).
    """
    log.info("=== Step 1: PDF text extraction (pdfplumber) ===")

    tasks: list[tuple[Path, Path]] = []
    for _, row in df_meta.iterrows():
        pdf_path = RAW_DIR / row["filename"]
        txt_path = PROCESSED_DIR / Path(row["filename"]).with_suffix(".txt").name

        if not pdf_path.exists():
            continue
        if txt_path.exists() and not force:
            continue
        tasks.append((pdf_path, txt_path))

    if not tasks:
        log.info("  All PDFs already extracted. Skipping.")
        return df_meta

    log.info(f"  {len(tasks)} PDFs to extract ({max_workers} threads)")
    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_extract_one, pdf, txt): pdf for pdf, txt in tasks}
        for future in tqdm(as_completed(future_map), total=len(tasks), desc="Extracting"):
            r = future.result()
            results[r["filename"]] = r

    # Update df_meta in place
    df_meta = df_meta.copy()
    for fname, r in results.items():
        mask = df_meta["filename"] == fname
        df_meta.loc[mask, "char_count"] = r["char_count"]
        df_meta.loc[mask, "is_scanned"] = r["is_scanned"]

    ok    = sum(1 for r in results.values() if r["error"] is None)
    errs  = sum(1 for r in results.values() if r["error"])
    log.info(f"  Done: {ok} extracted, {errs} errors")
    return df_meta


# ---------------------------------------------------------------------------
# Step 2 — OCR (EasyOCR, CPU-bound → ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def run_ocr(df_meta: pd.DataFrame, max_workers: int | None = None, force: bool = False) -> pd.DataFrame:
    """
    Applies OCR to PDFs flagged as scanned (char_count == 0 or is_scanned).
    Idempotent: skips PDFs that already have non-empty .txt (unless force=True).
    """
    from concurrent.futures import ProcessPoolExecutor

    # Import at module level is not possible for the worker (needs to be picklable)
    # so we import inside the function and use a module-level worker via src/data_etl/ocr_extractor.py
    sys.path.insert(0, str(ROOT / 'src' / 'data_etl'))
    from ocr_extractor import _ocr_worker

    log.info("=== Step 2: OCR for scanned PDFs (EasyOCR) ===")

    pending = df_meta[(df_meta["is_scanned"] == True) | (df_meta["char_count"] == 0)].copy()  # noqa: E712

    if pending.empty:
        log.info("  No scanned PDFs pending. Skipping OCR.")
        return df_meta

    tasks: list[tuple[str, str]] = []
    for _, row in pending.iterrows():
        pdf_path = RAW_DIR / row["filename"]
        txt_path = PROCESSED_DIR / Path(row["filename"]).with_suffix(".txt").name

        if not pdf_path.exists():
            continue
        if txt_path.exists() and txt_path.stat().st_size > 100 and not force:
            continue
        tasks.append((str(pdf_path), str(txt_path)))

    if not tasks:
        log.info("  All scanned PDFs already OCR'd. Skipping.")
        return df_meta

    workers = max_workers or max(1, (os.cpu_count() or 2) // 2)
    log.info(f"  {len(tasks)} PDFs to OCR ({workers} processes, EasyOCR GPU/MPS)")

    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_ocr_worker, task): task for task in tasks}
        for future in tqdm(as_completed(future_map), total=len(tasks), desc="OCR"):
            r = future.result()
            results[r["filename"]] = r

    df_meta = df_meta.copy()
    for fname, r in results.items():
        if r["status"] == "success":
            mask = df_meta["filename"] == fname
            df_meta.loc[mask, "char_count"] = r["chars"]
            df_meta.loc[mask, "is_scanned"] = False

    ok   = sum(1 for r in results.values() if r["status"] == "success")
    errs = sum(1 for r in results.values() if r["status"] == "error")
    log.info(f"  Done: {ok} OCR'd, {errs} errors")
    return df_meta


# ---------------------------------------------------------------------------
# Step 3 — Load extracted texts into the Moncloa DataFrame
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def _read_txt(txt_path: Path) -> str:
    """Reads a .txt file, returns empty string on any failure."""
    try:
        return txt_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_moncloa_texts(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Reads the extracted .txt files and attaches them to df_meta."""
    log.info("=== Step 3: Loading extracted texts ===")
    texts = []
    missing = 0

    for _, row in df_meta.iterrows():
        txt_path = PROCESSED_DIR / Path(row["filename"]).with_suffix(".txt").name
        if txt_path.exists():
            texts.append(_clean_text(_read_txt(txt_path)))
        else:
            texts.append(None)
            missing += 1

    df_meta = df_meta.copy()
    df_meta["extracted_text"] = texts

    log.info(f"  {len(df_meta) - missing}/{len(df_meta)} texts loaded ({missing} missing)")
    return df_meta


# ---------------------------------------------------------------------------
# Step 4 — Build the final corpus
# ---------------------------------------------------------------------------

FINAL_COLUMNS = [
    "doc_id",
    "source",
    "title",
    "url",
    "ministry",
    "category",
    "filename",
    "date",
    "doc_type",
    "tags",
    "extracted_text",
    "extracted_text_length",
    "rtve_summary",
]


def build_final_corpus(df_moncloa: pd.DataFrame, df_rtve: pd.DataFrame) -> pd.DataFrame:
    """Merges Moncloa and RTVE into the final document corpus."""
    log.info("=== Step 4: Building final corpus ===")

    moncloa_rows = [
        {
            "doc_id":                f"M{i + 1:03d}",
            "source":                "Moncloa",
            "title":                 row.get("name") or None,
            "url":                   row.get("url") or None,
            "ministry":              row.get("ministry") or None,
            "category":              row.get("category") or None,
            "filename":              row.get("filename") or None,
            "date":                  row.get("date") or None,
            "doc_type":              row.get("doc_type") or None,
            "tags":                  None,
            "extracted_text":        row.get("extracted_text") or None,
            "extracted_text_length": len(row.get("extracted_text") or "") if isinstance(row.get("extracted_text"), str) else 0,
            "rtve_summary":          None,
        }
        for i, (_, row) in enumerate(df_moncloa.iterrows())
    ]

    rtve_rows = [
        {
            "doc_id":                f"R{i + 1:03d}",
            "source":                "RTVE",
            "title":                 row.get("name") or None,
            "url":                   row.get("link") or None,
            "ministry":              None,
            "category":              None,
            "filename":              None,
            "date":                  None,
            "doc_type":              None,
            "tags":                  row.get("tags") or None,
            "extracted_text":        None,
            "extracted_text_length": 0,
            "rtve_summary":          _clean_text(str(row.get("summary", ""))) or None,
        }
        for i, (_, row) in enumerate(df_rtve.iterrows())
    ]

    df_corpus = pd.DataFrame(moncloa_rows + rtve_rows)[FINAL_COLUMNS]

    log.info(f"  Total: {len(df_corpus)} documents "
             f"(Moncloa: {len(moncloa_rows)}, RTVE: {len(rtve_rows)})")

    empty_moncloa = sum(1 for r in moncloa_rows if not r["extracted_text"])
    if empty_moncloa:
        log.warning(f"  {empty_moncloa} Moncloa docs have empty extracted_text (may need OCR)")

    return df_corpus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    apply_ocr: bool = False,
    max_extraction_workers: int = 4,
    max_ocr_workers: int | None = None,
    force_reextract: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """
    Runs the full pipeline and returns df_final.

    Args:
        apply_ocr:               Run EasyOCR on scanned PDFs. Default False.
        max_extraction_workers:  Threads for pdfplumber extraction.
        max_ocr_workers:         Processes for EasyOCR (default: cpu_count//2).
        force_reextract:         Re-extract even if .txt already exists.
        save:                    Save document_corpus.csv. Default True.

    Returns:
        df_final (pd.DataFrame) with columns defined in FINAL_COLUMNS.
    """
    sys.path.insert(0, str(ROOT / 'src' / 'data_etl'))

    if not ENRICHED_CSV.exists():
        raise FileNotFoundError(
            f"{ENRICHED_CSV} not found. Run the scraper + downloader first:\n"
            "  python main.py --scrape\n"
            "  python main.py --download"
        )
    if not RTVE_CSV.exists():
        raise FileNotFoundError(
            f"{RTVE_CSV} not found. Run the RTVE scraper first:\n"
            "  python src/data_etl/rtve_scraper.py"
        )

    df_meta  = pd.read_csv(ENRICHED_CSV)
    df_rtve  = pd.read_csv(RTVE_CSV)

    log.info(f"Loaded: {len(df_meta)} Moncloa docs, {len(df_rtve)} RTVE docs")

    df_meta = run_extraction(df_meta, max_workers=max_extraction_workers, force=force_reextract)

    if apply_ocr:
        df_meta = run_ocr(df_meta, max_workers=max_ocr_workers, force=force_reextract)
    else:
        scanned = int(((df_meta["is_scanned"] == True) | (df_meta["char_count"] == 0)).sum())  # noqa: E712
        if scanned:
            log.warning(f"  {scanned} scanned PDFs skipped (pass apply_ocr=True to process them)")

    df_meta = load_moncloa_texts(df_meta)

    df_final = build_final_corpus(df_meta, df_rtve)

    if save:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(CORPUS_CSV, index=False, encoding="utf-8")
        log.info(f"Saved: {CORPUS_CSV}  ({len(df_final)} rows)")

    return df_final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="23-F end-to-end pipeline")
    parser.add_argument("--ocr",       action="store_true", help="Run OCR on scanned PDFs (slow)")
    parser.add_argument("--force",     action="store_true", help="Re-extract even if .txt exists")
    parser.add_argument("--no-save",   action="store_true", help="Do not save document_corpus.csv")
    parser.add_argument("--workers",   type=int, default=4, help="Extraction threads (default: 4)")
    parser.add_argument("--dry-run",   action="store_true", help="Only show counts, no processing")
    args = parser.parse_args()

    if args.dry_run:
        if ENRICHED_CSV.exists():
            df = pd.read_csv(ENRICHED_CSV)
            scanned = int(((df["is_scanned"] == True) | (df["char_count"] == 0)).sum())  # noqa: E712
            already_done = sum(
                1 for _, r in df.iterrows()
                if (PROCESSED_DIR / Path(r["filename"]).with_suffix(".txt").name).exists()
            )
            print(f"Moncloa docs     : {len(df)}")
            print(f"Already extracted: {already_done}")
            print(f"Scanned (OCR)    : {scanned}")
        if RTVE_CSV.exists():
            print(f"RTVE docs        : {len(pd.read_csv(RTVE_CSV))}")
        sys.exit(0)

    df_final = run(
        apply_ocr=args.ocr,
        max_extraction_workers=args.workers,
        force_reextract=args.force,
        save=not args.no_save,
    )

    print(f"\n{'='*50}")
    print(f"df_final shape   : {df_final.shape}")
    print(f"Columns          : {df_final.columns.tolist()}")
    print(f"Moncloa          : {(df_final.source == 'Moncloa').sum()} docs")
    print(f"RTVE             : {(df_final.source == 'RTVE').sum()} docs")
    print(f"With text        : {df_final.extracted_text.notna().sum()} Moncloa docs")
    print(f"With summary     : {df_final.rtve_summary.notna().sum()} RTVE docs")
