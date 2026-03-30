"""
Build Document Corpus
Merges extracted texts from La Moncloa PDFs and RTVE summaries into a single
DataFrame/CSV for NLP and graph tasks. Includes data quality validation and 
optional LLM-based metadata extraction.

Output schema (document_corpus.csv):
  ... includes analysis_text which maps to the best text string for downstream ml ...
"""

import os
import re
import pandas as pd
from typing import Optional

from src.data_etl.data_cleaning import process_corpus_quality

MONCLOA_META = "data/metadata/documents_enriched.csv"
RTVE_META    = "data/metadata/rtve_documents.csv"
OUTPUT_CORPUS = "data/metadata/document_corpus.csv"

FINAL_COLUMNS = [
    "doc_id",
    "source",
    "title",
    "url",
    "ministry",
    "category",
    "filename",
    "date",
    "date_precision",
    "doc_type",
    "tags",
    "extracted_text",
    "extracted_text_length",
    "rtve_summary",
    "ocr_quality_score",
    "flag_illegible",
    "analysis_text"
]

def _clean_text(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

def load_moncloa() -> pd.DataFrame:
    if not os.path.exists(MONCLOA_META):
        print(f"Warning: {MONCLOA_META} not found. Skipping Moncloa.")
        return pd.DataFrame()

    df_m = pd.read_csv(MONCLOA_META)
    rows = []

    for idx, row in df_m.iterrows():
        txt_path = row.get("txt_path", "")
        extracted_text = ""
        if isinstance(txt_path, str) and txt_path:
            filepath = os.path.join("data", "processed", os.path.basename(txt_path))
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        extracted_text = f.read()
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

        extracted_text = _clean_text(extracted_text)

        rows.append({
            "doc_id":                 f"M{idx + 1:03d}",
            "source":                 "Moncloa",
            "title":                  row.get("name") or None,
            "url":                    row.get("url") or None,
            "ministry":               row.get("ministry") or None,
            "category":               row.get("category") or None,
            "filename":               row.get("filename") or None,
            "date":                   row.get("date") or None,
            "doc_type":               row.get("doc_type") or None,
            "tags":                   None,
            "extracted_text":         extracted_text or None,
            "extracted_text_length":  len(extracted_text),
            "rtve_summary":           None,
        })

    return pd.DataFrame(rows)


def load_rtve() -> pd.DataFrame:
    if not os.path.exists(RTVE_META):
        print(f"Warning: {RTVE_META} not found. Skipping RTVE.")
        return pd.DataFrame()

    df_r = pd.read_csv(RTVE_META)
    rows = []

    for idx, row in df_r.iterrows():
        rtve_summary = _clean_text(str(row.get("summary", "")))

        rows.append({
            "doc_id":                 f"R{idx + 1:03d}",
            "source":                 "RTVE",
            "title":                  row.get("name") or None,
            "url":                    row.get("link") or None,
            "ministry":               None,
            "category":               None,
            "filename":               None,
            "date":                   None,
            "doc_type":               None,
            "tags":                   row.get("tags") or None,
            "extracted_text":         None,
            "extracted_text_length":  0,
            "rtve_summary":           rtve_summary or None,
        })

    return pd.DataFrame(rows)


def apply_analysis_text(df: pd.DataFrame) -> pd.DataFrame:
    """Sets the canonical text metric based on available data and illegibility."""
    def get_canonical(row):
        if row["source"] == "RTVE":
            return row["rtve_summary"] if pd.notna(row["rtve_summary"]) else ""
        if row["source"] == "Moncloa":
            if row.get("flag_illegible", False):
                return "" # Drop noisy text from downstream NLP
            return row["extracted_text"] if pd.notna(row["extracted_text"]) else ""
        return ""
        
    df["analysis_text"] = df.apply(get_canonical, axis=1)
    return df

def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates date_precision and normalizes the date column."""
    def date_precision(date_str):
        if pd.isna(date_str) or str(date_str).lower() in ['nan', 'nat', 'none']:
            return 'unknown'
        s = str(date_str).strip()
        if len(s) == 10:   # 1981-02-23
            return 'day'
        elif len(s) == 7:  # 1981-02
            return 'month'
        else:
            return 'year'

    # Compute precision before parsing to datetime to preserve string length info
    df['date_precision'] = df['date'].astype(str).apply(date_precision)
    
    # Normalize dates
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True, errors='coerce')
    return df

def build_corpus(extract_metadata: bool = False, limit_metadata: Optional[int] = None) -> None:
    print("Loading Moncloa documents...")
    df_moncloa = load_moncloa()
    print(f"  {len(df_moncloa)} documents loaded.")

    print("Loading RTVE documents...")
    df_rtve = load_rtve()
    print(f"  {len(df_rtve)} documents loaded.")

    df_corpus = pd.concat([df_moncloa, df_rtve], ignore_index=True)
    
    # 1. OCR Quality Scanning
    print("\nScanning corpus for OCR quality...")
    df_corpus = process_corpus_quality(df_corpus, threshold=0.40)
    
    # 2. Assign canonical NLP text field
    df_corpus = apply_analysis_text(df_corpus)

    # 2.b Apply Simple Rules for doc_type inference
    print("\nApplying simple rules to infer doc_type...")
    from src.data_etl.doc_type_rules import fill_doc_types
    df_corpus = fill_doc_types(df_corpus)

    # 3. Optional LLM Extraction (Only on healthy documents)
    if extract_metadata:
        try:
            from src.data_etl.metadata_extractor import batch_extract_metadata
            print("\nStarting LLM Metadata Extraction...")
            df_corpus = batch_extract_metadata(df_corpus, limit=limit_metadata)
        except ImportError as e:
            print(f"Cannot run metadata extraction: {e}")

    # 4. Normalize Dates and add precision
    print("\nNormalizing dates and calculating precision...")
    df_corpus = normalize_dates(df_corpus)

    # Restrict columns and save
    missing_cols = [col for col in FINAL_COLUMNS if col not in df_corpus.columns]
    for col in missing_cols:
        df_corpus[col] = None
        
    df_corpus = df_corpus[FINAL_COLUMNS]
    
    os.makedirs(os.path.dirname(OUTPUT_CORPUS), exist_ok=True)
    df_corpus.to_csv(OUTPUT_CORPUS, index=False, encoding="utf-8")

    print(f"\nCorpus saved to {OUTPUT_CORPUS}")
    print(f"Total documents : {len(df_corpus)}")
    print(f"  Moncloa (legible) : len={len(df_corpus[(df_corpus['source'] == 'Moncloa') & (~df_corpus['flag_illegible'])])}")
    print(f"  Moncloa (illegible) : len={len(df_corpus[(df_corpus['source'] == 'Moncloa') & (df_corpus['flag_illegible'])])}")
    print(f"  RTVE          : {(df_corpus['source'] == 'RTVE').sum()}")


if __name__ == "__main__":
    build_corpus(extract_metadata=False)
