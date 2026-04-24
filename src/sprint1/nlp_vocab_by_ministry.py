"""Basic NLP preprocessing and vocabulary frequencies by ministry.

Sprint 1 helper for Alberto's NLP kickoff tasks.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORPUS_CSV = ROOT / "data" / "metadata" / "document_corpus.csv"
OUT_DIR = ROOT / "outputs" / "sprint1"

STOPWORDS_ES = {
    "de",
    "la",
    "el",
    "en",
    "y",
    "a",
    "los",
    "del",
    "se",
    "las",
    "por",
    "un",
    "para",
    "con",
    "no",
    "una",
    "su",
    "al",
    "lo",
    "como",
    "mas",
    "pero",
    "sus",
    "le",
    "ya",
    "o",
    "este",
    "si",
    "porque",
    "esta",
    "entre",
    "cuando",
    "muy",
    "sin",
    "sobre",
    "tambien",
    "me",
    "hasta",
    "hay",
    "donde",
    "quien",
    "desde",
    "todo",
    "nos",
    "durante",
    "todos",
    "uno",
    "les",
    "ni",
    "contra",
    "otros",
    "que",
    "fue",
    "era",
    "ser",
    "han",
    "ha",
    "es",
    "son",
    "page",
    "pagina",
    "pag",
    "documento",
    "documentos",
}

TOKEN_RE = re.compile(r"[a-zA-Z\u00C0-\u017F]+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def tokenize(text: str, min_len: int) -> list[str]:
    norm = normalize_text(text)
    tokens = [t for t in TOKEN_RE.findall(norm) if len(t) >= min_len]
    return [t for t in tokens if t not in STOPWORDS_ES and not t.isdigit()]


def top_tokens(tokens: list[str], top_k: int) -> list[tuple[str, int]]:
    return Counter(tokens).most_common(top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute top vocabulary by ministry")
    parser.add_argument("--top-k", type=int, default=30, help="Top tokens per group")
    parser.add_argument("--min-len", type=int, default=3, help="Minimum token length")
    args = parser.parse_args()

    if not CORPUS_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {CORPUS_CSV}")

    df = pd.read_csv(CORPUS_CSV)

    text_col = "analysis_text" if "analysis_text" in df.columns else "extracted_text"
    moncloa = df[df["source"] == "Moncloa"].copy()
    if "flag_illegible" in moncloa.columns:
        moncloa = moncloa[~moncloa["flag_illegible"]].copy()
    moncloa = moncloa[moncloa[text_col].notna() & (moncloa[text_col].astype(str).str.strip() != "")].copy()

    if moncloa.empty:
        raise ValueError(f"No Moncloa {text_col} rows found")

    moncloa["tokens"] = moncloa[text_col].astype(str).apply(lambda t: tokenize(t, args.min_len))

    all_tokens = [tok for row in moncloa["tokens"] for tok in row]
    overall = pd.DataFrame(top_tokens(all_tokens, args.top_k), columns=["token", "count"])

    by_ministry_rows: list[dict] = []
    for ministry, group in moncloa.groupby("ministry", dropna=False):
        toks = [tok for row in group["tokens"] for tok in row]
        for rank, (token, count) in enumerate(top_tokens(toks, args.top_k), start=1):
            by_ministry_rows.append(
                {
                    "ministry": ministry if isinstance(ministry, str) else "Unknown",
                    "rank": rank,
                    "token": token,
                    "count": count,
                }
            )

    by_ministry = pd.DataFrame(by_ministry_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_path = OUT_DIR / "top_words_overall.csv"
    by_ministry_path = OUT_DIR / "top_words_by_ministry.csv"
    summary_path = OUT_DIR / "nlp_preprocess_summary.txt"

    overall.to_csv(overall_path, index=False, encoding="utf-8")
    by_ministry.to_csv(by_ministry_path, index=False, encoding="utf-8")

    summary = [
        f"Corpus rows (Moncloa with text): {len(moncloa)}",
        f"Unique ministries: {moncloa['ministry'].nunique(dropna=True)}",
        f"Total tokens (after cleaning): {len(all_tokens)}",
        f"Top-k: {args.top_k}",
        f"Min token length: {args.min_len}",
        f"Text column used: {text_col}",
        f"Output: {overall_path}",
        f"Output: {by_ministry_path}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("Saved:")
    print(f"- {overall_path}")
    print(f"- {by_ministry_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
