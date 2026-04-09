"""
Extractor: Text from 23-F document PDFs

Extracts text using pdfplumber (native PDFs) and detects scanned ones.
- Parallel processing with ThreadPoolExecutor (I/O + light CPU)
- Detects native vs scanned PDFs by character density
- Scanned ones are marked for subsequent OCR (ocr_extractor.py)
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
from tqdm import tqdm

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

# If avg_chars/page < threshold → probably scanned
SCANNED_THRESHOLD = 50

# Document type by keywords in filename
_TIPO_KEYWORDS: dict[str, str] = {
    "conversacion": "Telephone transcript",
    "telex": "Telex",
    "nota": "Intelligence note",
    "informe": "Report",
    "vista": "Oral hearing",
    "reservado": "Restricted",
    "secreto": "Secret",
}


# ---------------------------------------------------------------------------
# Extraction functions (executed in threads)
# ---------------------------------------------------------------------------

def extract_text_plumber(pdf_path: Path) -> tuple[list[dict], dict]:
    """
    Extracts text page by page with pdfplumber.

    Returns:
        (pages, metadata) where pages is a list of {page_num, text, char_count}.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            metadata: dict[str, Any] = {"pages": len(pdf.pages), "metadata": pdf.metadata}
            pages = [
                {"page_num": i + 1, "text": (page.extract_text() or ""), "char_count": len(page.extract_text() or "")}
                for i, page in enumerate(pdf.pages)
            ]
            return pages, metadata
    except Exception as e:
        return [], {"error": str(e)}


def is_scanned(pages: list[dict]) -> bool:
    """True if average chars/page is below the threshold."""
    if not pages:
        return True
    avg = sum(p["char_count"] for p in pages) / len(pages)
    return avg < SCANNED_THRESHOLD


def clean_text(text: str) -> str:
    """Normalizes spaces and line breaks."""
    if not text:
        return ""
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def extract_doc_info(filename: str) -> dict[str, str | None]:
    """Extracts number, date and type from the filename."""
    num_match = re.search(r"(?:23F[_\s]?|Documento[_\s]?)([\d]+)", filename, re.IGNORECASE)
    date_match = re.search(r"(\d{1,2}[-_]\d{1,2}[-_]\d{2,4})", filename)

    tipo = None
    for kw, label in _TIPO_KEYWORDS.items():
        if kw in filename.lower():
            tipo = label
            break

    return {
        "doc_num": num_match.group(1) if num_match else None,
        "date": date_match.group(1) if date_match else None,
        "doc_type": tipo,
    }


def _process_one(pdf_path: Path, txt_path: Path) -> dict:
    """Worker: extracts text from a PDF and saves it to txt_path."""
    pages, metadata = extract_text_plumber(pdf_path)

    if not pages and "error" in metadata:
        return {
            "pdf_path": str(pdf_path),
            "txt_path": str(txt_path),
            "rel_path": pdf_path.name,
            "success": False,
            "pages": 0,
            "char_count": 0,
            "is_scanned": True,
            "error": metadata["error"],
            **extract_doc_info(pdf_path.name),
        }

    scanned = is_scanned(pages)
    full_text = clean_text("\n\n".join(p["text"] for p in pages))

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(full_text, encoding="utf-8")

    return {
        "pdf_path": str(pdf_path),
        "txt_path": str(txt_path),
        "rel_path": pdf_path.name,
        "success": True,
        "pages": len(pages),
        "char_count": len(full_text),
        "is_scanned": scanned,
        "error": None,
        **extract_doc_info(pdf_path.name),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_all_pdfs(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = PROCESSED_DIR,
    max_workers: int = 4,
) -> pd.DataFrame | None:
    """
    Extracts text from all PDFs in raw_dir in parallel.

    Args:
        raw_dir:     Directory with original PDFs.
        output_dir:  Directory where .txt files are saved.
        max_workers: Concurrent threads (I/O-bound with pdfplumber).
    """
    print("=" * 60)
    print("EXTRACTOR - 23-F PDFs Text")
    print("=" * 60)

    pdf_files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdf_files:
        print(f"\nERROR: No PDFs found in {raw_dir}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTotal PDFs      : {len(pdf_files)}")
    print(f"Parallel threads: {max_workers}")
    print(f"Destination     : {output_dir}\n")

    # Prepare pairs (pdf_path, txt_path)
    tasks: list[tuple[Path, Path]] = [
        (p, output_dir / p.relative_to(raw_dir).with_suffix(".txt"))
        for p in pdf_files
    ]

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_process_one, pdf, txt): pdf for pdf, txt in tasks}
        for future in tqdm(as_completed(future_map), total=len(tasks), desc="Extracting"):
            results.append(future.result())

    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    ok = df["success"].sum()
    scanned = df["is_scanned"].sum()
    print(f"✓ Extracted       : {ok}/{len(df)}")
    print(f"⚠ Scanned         : {scanned} (pending OCR)")
    print(f"  Total chars     : {df['char_count'].sum():,}")

    df.to_csv(METADATA_DIR / "extraction_log.csv", index=False)
    print(f"\nLog: {METADATA_DIR / 'extraction_log.csv'}")

    return df


def verify_extraction(csv_path: Path | None = None) -> None:
    """Displays a summary of the extraction saved in CSV."""
    csv_path = csv_path or METADATA_DIR / "extraction_log.csv"
    df = pd.read_csv(csv_path)

    empty = df[(df["success"]) & (df["char_count"] < 100)]
    failed = df[~df["success"]]

    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    if not failed.empty:
        print(f"⚠ Failed          : {len(failed)}")
    if not empty.empty:
        print(f"⚠ Short texts     : {len(empty)} (<100 chars)")

    ok = df[df["success"]]
    if not ok.empty:
        print(f"✓ With text       : {len(ok)}")
        print(f"  Total pages     : {ok['pages'].sum():.0f}")
        print(f"  Total chars     : {ok['char_count'].sum():,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from 23-F PDFs")
    parser.add_argument("--workers", type=int, default=4, help="Parallel threads")
    parser.add_argument("--verify", action="store_true", help="Only verify existing extraction")
    args = parser.parse_args()

    if args.verify:
        verify_extraction()
    else:
        process_all_pdfs(max_workers=args.workers)
