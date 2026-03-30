"""
Build Document Corpus
Merges extracted texts from La Moncloa PDFs and RTVE summaries into a single
DataFrame/CSV for NLP and graph tasks.

Output schema (document_corpus.csv):
  doc_id            - unique identifier (M001 = Moncloa, R001 = RTVE)
  source            - "Moncloa" | "RTVE"
  title             - document title / name
  url               - link to original document
  ministry          - ministry (Moncloa docs only)
  category          - document category (Moncloa docs only)
  filename          - PDF filename (Moncloa docs only)
  date              - document date extracted from filename (Moncloa docs only)
  doc_type          - document type (Moncloa docs only)
  tags              - thematic tags (RTVE docs only)
  extracted_text    - full text extracted from the Moncloa PDF
  extracted_text_length - character count of extracted_text
  rtve_summary      - full summary from the RTVE catalog
"""

import os
import re
import pandas as pd

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
    "doc_type",
    "tags",
    "extracted_text",
    "extracted_text_length",
    "rtve_summary",
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


def build_corpus() -> None:
    print("Loading Moncloa documents...")
    df_moncloa = load_moncloa()
    print(f"  {len(df_moncloa)} documents loaded.")

    print("Loading RTVE documents...")
    df_rtve = load_rtve()
    print(f"  {len(df_rtve)} documents loaded.")

    df_corpus = pd.concat([df_moncloa, df_rtve], ignore_index=True)
    df_corpus = df_corpus[FINAL_COLUMNS]

    os.makedirs(os.path.dirname(OUTPUT_CORPUS), exist_ok=True)
    df_corpus.to_csv(OUTPUT_CORPUS, index=False, encoding="utf-8")

    print(f"\nCorpus saved to {OUTPUT_CORPUS}")
    print(f"Total documents : {len(df_corpus)}")
    print(f"  Moncloa       : {(df_corpus['source'] == 'Moncloa').sum()}")
    print(f"  RTVE          : {(df_corpus['source'] == 'RTVE').sum()}")

    empty_moncloa = df_corpus[(df_corpus["source"] == "Moncloa") & (df_corpus["extracted_text_length"] == 0)]
    if not empty_moncloa.empty:
        print(f"  Warning: {len(empty_moncloa)} Moncloa docs with empty extracted text")

    empty_rtve = df_corpus[(df_corpus["source"] == "RTVE") & (df_corpus["rtve_summary"] == "")]
    if not empty_rtve.empty:
        print(f"  Warning: {len(empty_rtve)} RTVE docs with empty summary")


if __name__ == "__main__":
    build_corpus()
