"""
Data Cleaning & OCR Quality Validation
Flags noisy/illegible OCR documents in the unified corpus.
"""

import pandas as pd
import re

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
    """
    df["ocr_quality_score"] = 0.0
    df["flag_illegible"] = False
    
    # For Moncloa documents, compute OCR quality
    moncloa_mask = df["source"] == "Moncloa"
    
    # Fill scores
    df.loc[moncloa_mask, "ocr_quality_score"] = df.loc[moncloa_mask, "extracted_text"].apply(compute_ocr_quality)
    
    # Flag illegible based on threshold (or if text is entirely empty)
    df.loc[moncloa_mask, "flag_illegible"] = (df.loc[moncloa_mask, "ocr_quality_score"] < threshold) | (df.loc[moncloa_mask, "extracted_text"].isna()) | (df.loc[moncloa_mask, "extracted_text_length"] < 50)
    
    print(f"Data cleaning completed: identified {df['flag_illegible'].sum()} potentially illegible documents.")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/metadata/document_corpus.csv")
    df = process_corpus_quality(df)
    print(df[df["flag_illegible"] is True][["doc_id", "title", "ocr_quality_score"]].head(10))
