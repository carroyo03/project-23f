"""
Data Cleaning & OCR Quality Validation
Flags noisy/illegible OCR documents in the unified corpus.
"""

import pandas as pd
import re


PAGE_ONLY_PATTERN = re.compile(r"(?i)^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$")

def compute_ocr_quality(text: str) -> float:
    """
    Computes a heuristic score for OCR text quality.
    Score = percentage of words that are > 2 characters strictly alphabetical.
    Helps to detect "garbage" output from handwritten or highly degraded documents.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    # Extract all alphabetical words
    words = re.findall(r"\b[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]+\b", text)
    if not words:
        return 0.0
        
    short_words = sum(1 for w in words if len(w) <= 2)
    return 1.0 - (short_words / len(words))

def process_corpus_quality(df: pd.DataFrame, threshold: float = 0.45) -> pd.DataFrame:
    """
    Evaluates extracted text and sets a boolean 'flag_illegible' column.
    Applies to both Moncloa and RTVE documents.
    """
    df["ocr_quality_score"] = 0.0
    df["flag_illegible"] = False
    
    # Apply OCR quality scoring to both Moncloa and RTVE
    has_text_mask = (df["extracted_text"].notna()) & (df["extracted_text"] != "")
    
    # Fill scores for all documents with text
    df.loc[has_text_mask, "ocr_quality_score"] = df.loc[has_text_mask, "extracted_text"].apply(compute_ocr_quality)

    # Identify placeholder content such as "Page 3" or "Page 3 of 10".
    page_only_mask = df.loc[has_text_mask, "extracted_text"].str.fullmatch(PAGE_ONLY_PATTERN)
    
    # Flag illegible based on threshold (or if text is entirely empty)
    df.loc[has_text_mask, "flag_illegible"] = (
        (df.loc[has_text_mask, "ocr_quality_score"] < threshold) |
        (df.loc[has_text_mask, "extracted_text"].str.len() < 50) |
        page_only_mask
    )
    
    # Mark as illegible if no text at all
    df.loc[~has_text_mask, "flag_illegible"] = True
    
    print(f"Data cleaning completed: identified {df['flag_illegible'].sum()} potentially illegible documents.")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/metadata/document_corpus.csv")
    df = process_corpus_quality(df)
    print(df[df["flag_illegible"] is True][["doc_id", "title", "ocr_quality_score"]].head(10))
