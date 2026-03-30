# ML Project 23-F — Declassified Documents

Analysis of the declassified documents relating to the February 23, 1981 attempted coup d'état in Spain.

---

## Project Structure

```text
├── data/
│   ├── raw/                        # Original PDFs (excluded from repo, ~279 MB)
│   ├── processed/                  # Text extracted from PDFs (.txt, excluded from repo)
│   └── metadata/
│       ├── moncloa_links.csv       # PDF links scraped from La Moncloa (155 docs)
│       ├── documents_enriched.csv  # Moncloa metadata + extraction results
│       ├── extraction_log.csv      # Per-PDF extraction log
│       ├── rtve_documents.csv      # RTVE 23-F catalog with summaries (167 docs)
│       └── document_corpus.csv     # Final corpus — main analytical dataset (322 docs)
├── src/
│   ├── moncloa_scraper.py          # Scrapes PDF links from La Moncloa
│   ├── download_pdfs.py            # Downloads PDFs with retries and concurrency
│   ├── pdf_extractor.py            # Native text extraction (pdfplumber)
│   ├── ocr_extractor.py            # Scanned PDF extraction (EasyOCR / GPU)
│   ├── rtve_scraper.py             # Scrapes metadata and summaries from RTVE
│   └── build_corpus.py             # Builds document_corpus.csv from all sources
├── notebooks/
│   └── 00_initial_exploration.ipynb
├── outputs/                        # Figures, Gephi exports
├── reports/                        # Project report
├── pipeline.py                     # End-to-end pipeline → df_final (main entry point)
├── main.py                         # Step-by-step CLI orchestrator
└── requirements.txt
```

---

## Data Sources

| Source | Documents | Content |
|--------|-----------|---------|
| **La Moncloa** | 155 | Declassified PDFs: phone transcripts, intelligence notes, police reports (Interior / Defensa / Exteriores) |
| **RTVE 23-F** | 167 | Judicial hearing records from the 23-F trial, with summaries from RTVE's archive platform |

Both sources cover the 23-F historical event but are **different document sets** — they are not joined but unified in the final corpus with dedicated columns for each content type.

---

## Final Corpus — `document_corpus.csv`

The main analytical dataset. 322 rows × 13 columns.

| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | str | Unique ID — `M001`…`M155` (Moncloa), `R001`…`R167` (RTVE) |
| `source` | str | `"Moncloa"` or `"RTVE"` |
| `title` | str | Document title / name (original Spanish) |
| `url` | str | Link to the original source |
| `ministry` | str | Ministry of origin (Moncloa only) |
| `category` | str | Document category (Moncloa only) |
| `filename` | str | PDF filename (Moncloa only) |
| `date` | str | Date extracted from filename (Moncloa only) |
| `doc_type` | str | Document type: Telephone transcript, Intelligence note, Report… (Moncloa only) |
| `tags` | str | Thematic tags (RTVE only) |
| `extracted_text` | str | Full text extracted from the PDF (Moncloa only — original Spanish) |
| `extracted_text_length` | int | Character count of `extracted_text` |
| `rtve_summary` | str | Summary from RTVE's archive platform (RTVE only — original Spanish) |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://www.github.com/carroyo03/project-23f.git
cd project-23f

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Scrape and download data (one-time)

```bash
python main.py --scrape     # Scrape PDF links from La Moncloa → moncloa_links.csv
python main.py --download   # Download 155 PDFs to data/raw/
python src/rtve_scraper.py  # Scrape RTVE 23-F catalog → rtve_documents.csv
```

### 3. Run the pipeline → get df_final

```bash
python pipeline.py          # Extract text + build corpus → document_corpus.csv
python pipeline.py --ocr    # Also run EasyOCR on scanned PDFs (GPU/MPS)
python pipeline.py --dry-run  # Show counts without processing
```

Or from a notebook / script:

```python
from pipeline import run

df_final = run()               # Uses cached .txt files (fast)
df_final = run(apply_ocr=True) # Include OCR for scanned PDFs
```

---

## Pipeline Reference

### `pipeline.py` — end-to-end (recommended)

```bash
python pipeline.py                   # Full run, saves document_corpus.csv
python pipeline.py --ocr             # Include OCR on scanned PDFs
python pipeline.py --force           # Re-extract even if .txt files exist
python pipeline.py --workers 8       # More extraction threads
python pipeline.py --dry-run         # Show status, no writes
python pipeline.py --no-save         # Run but do not overwrite CSV
```

### `main.py` — step-by-step control

```bash
python main.py --scrape              # Step 1: scrape La Moncloa links
python main.py --download            # Step 2: download PDFs
python main.py --extract             # Step 3: extract text (pdfplumber)
python main.py --build-corpus        # Step 4: build document_corpus.csv
python main.py --status              # Show current local data status
```

### Individual modules

```bash
python src/rtve_scraper.py                     # Scrape RTVE catalog
python src/ocr_extractor.py --dry-run          # Preview scanned PDFs pending OCR
python src/ocr_extractor.py --limit 5          # Test OCR on 5 docs
python src/ocr_extractor.py                    # OCR all scanned docs
python src/build_corpus.py                     # Rebuild document_corpus.csv only
```

---

## Metadata CSVs

| File | Columns | Generated by |
|------|---------|--------------|
| `moncloa_links.csv` | `name, url, ministry, category, filename` | `moncloa_scraper.py` |
| `documents_enriched.csv` | `name, url, ministry, category, filename, pdf_path, txt_path, rel_path, success, pages, char_count, is_scanned, doc_num, date, doc_type, error` | `pdf_extractor.py` |
| `extraction_log.csv` | `pdf_path, txt_path, rel_path, success, pages, char_count, is_scanned, doc_num, date, doc_type, error, filename` | `pdf_extractor.py` |
| `rtve_documents.csv` | `name, pages, size_kb, summary, tags, link` | `rtve_scraper.py` |
| `document_corpus.csv` | See schema above | `pipeline.py` / `build_corpus.py` |

---

## Notes

- All code, column names, and comments are in **English**. Document content (titles, extracted text, summaries) remains in **Spanish** — these are primary historical sources.
- `data/raw/`, `data/processed/`, and `data/metadata/` are excluded from the repository. Run the pipeline locally to regenerate them.
- Some Moncloa PDFs have no extractable text (fully scanned images). Run with `--ocr` to process them via EasyOCR.
- The `doc_type` field is inferred from filename keywords, not the document content. 116 out of 155 Moncloa docs have no detectable type.
