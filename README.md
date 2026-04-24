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
│       ├── download_log.csv        # Download results + local paths per URL
│       ├── documents_enriched.csv  # Moncloa metadata + extraction results (pipeline.py / OCR)
│       ├── extraction_log.csv      # Per-PDF extraction log
│       ├── rtve_documents.csv      # RTVE 23-F catalog with summaries (167 docs)
│       └── document_corpus.csv     # Final corpus — main analytical dataset (322 docs)
├── src/data_etl/
│   ├── moncloa_scraper.py          # Scrapes PDF links from La Moncloa
│   ├── download_pdfs.py            # Downloads PDFs with retries and concurrency
│   ├── pdf_extractor.py            # Native text extraction (pdfplumber)
│   ├── ocr_extractor.py            # Scanned PDF extraction (EasyOCR / GPU)
│   ├── rtve_scraper.py             # Scrapes metadata and summaries from RTVE
│   └── build_corpus.py             # Builds document_corpus.csv from all sources
├── notebooks/
│   └── 00_initial_exploration.ipynb
│   └── pipeline_entrega.ipynb
│   └── EDA_Sprint1_Alejandro.ipynb
│   └── sprint2_carlos_ner_graphs.ipynb
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

The main analytical dataset. 322 rows × 17 columns (155 Moncloa + 167 RTVE). 306 documents are fully usable for NLP after removing the 16 illegible.

### Known Limitations
* **Missing Dates**: 81 documents have no date (`date_precision = unknown`), usually because their content is too fragmentary or they lack clear dating headers.
* **Garbage OCR**: 16 Moncloa handwritten/garbled scanned documents have null `analysis_text` (`flag_illegible=True`) to prevent them from breaking NLP clustering and topic modeling downstream.
* **RTVE Text**: RTVE items lack raw `extracted_text`. Since they represent judicial summaries, their `rtve_summary` is mapped directly to the `analysis_text` column.
* **Short Documents**: The median length of `analysis_text` is ~303 characters, meaning many documents are extremely short memos, telexes, or very brief RTVE trial summaries.
* **Source Discrepancies**: `ministry` and `category` parameters are only available for the La Moncloa declassified dataset.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | str | Unique ID — `M001`…`M155` (Moncloa), `R001`…`R167` (RTVE) |
| `source` | str | `"Moncloa"` or `"RTVE"` |
| `title` | str | Document title / name (original Spanish) |
| `url` | str | Link to the original source |
| `ministry` | str | Ministry of origin (Moncloa only) |
| `category` | str | Document category (Moncloa only) |
| `filename` | str | PDF filename (Moncloa only) |
| `date` | str | LLM-Extracted Date (Normalized via pd.to_datetime, Pandas format) |
| `date_precision` | str | The precision level of the extracted date: `day`, `month`, `year`, or `unknown` |
| `doc_type` | str | Document type inference: Resumen de Juicio, Telephone transcript, Intelligence note...|
| `tags` | str | Thematic tags (RTVE only) |
| `extracted_text` | str | Full text extracted from the PDF (Moncloa only — original Spanish) |
| `extracted_text_length` | int | Character count of `extracted_text` |
| `rtve_summary` | str | Summary from RTVE's archive platform (RTVE only — original Spanish) |
| `ocr_quality_score` | float | Heuristic score measuring legibility |
| `flag_illegible` | bool | `True` for handwritten/garbage texts |
| `analysis_text` | str | Canonical unified text meant for NLP processing (omits garbage OCR) |

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

### 2. Configure Environment (Optional but Recommended)
For date extraction, Nvidia Build provides an OpenAI-compatible API to Llama 3.1.
```bash
cp .env.example .env
# Edit .env and paste your NVIDIA_API_KEY
```

### 3. Prepare the data (one-time)

```bash
python main.py --scrape     # Scrape PDF links from La Moncloa → moncloa_links.csv
python main.py --download   # Download 155 PDFs to data/raw/
python src/data_etl/rtve_scraper.py  # Scrape RTVE 23-F catalog → rtve_documents.csv
python pipeline.py          # Extract text, save documents_enriched.csv, and build document_corpus.csv
```

### 4. Verify you reached the expected 322 documents

```bash
python main.py --status
```

Expected:
- Downloaded PDFs: 155
- Extracted texts: 155
- Detected links: 155

And check the final corpus shape:

```bash
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv('data/metadata/document_corpus.csv')
print(df.shape)
print((df['source'] == 'Moncloa').sum(), (df['source'] == 'RTVE').sum())
PY
```

Expected:
- `(322, 13)`
- `155 167`

### 5. Sprint 1 (Alber) quick tasks

Generate the manual validation sample (10-15 rows suggested by the team plan):

```bash
python src/sprint1/manual_validation_sample.py --n 15
```

Output:
- `outputs/sprint1/manual_validation_sample.csv`

Start NLP preprocessing and vocabulary exploration by ministry:

```bash
python src/sprint1/nlp_vocab_by_ministry.py --top-k 30 --min-len 3
```

Outputs:
- `outputs/sprint1/top_words_overall.csv`
- `outputs/sprint1/top_words_by_ministry.csv`
- `outputs/sprint1/nlp_preprocess_summary.txt`

### 6. Sprint 2 (Carlos) — NER + grafo de co-ocurrencia

Pipeline end-to-end desde el notebook `notebooks/sprint2_carlos_ner_graphs.ipynb`:

1. **NER** con spaCy `es_core_news_lg` → `ner_extractor.run_ner_on_corpus`.
2. **Normalización** (whitelists 23-F + fuzzy) → `entity_normalizer.run_normalization` → `normalized_entities.csv` + `network_edges.csv`.
3. **Grafo** ponderado → `graph_builder.build_graph`.
4. **Métricas** (degree, betweenness, Louvain) → `graph_metrics.compute_metrics` + `top_brokers`.
5. **Export Gephi** con comunidades coloreadas → `gephi_exporter.export_gexf` → `outputs/sprint2/network.gexf`.

Outputs en `outputs/sprint2/`:
- `metrics.csv` — tabla por nodo (degree, betweenness, comunidad).
- `network.gexf` — para abrir en Gephi (color por comunidad, tamaño por betweenness).
- `network_overview.png` — figura de respaldo generada en el notebook.

```bash
jupyter notebook notebooks/sprint2_carlos_ner_graphs.ipynb
```

---

If you only want to inspect or rebuild the Moncloa extraction step, you can run:

```bash
python main.py --extract     # Extract native PDF text and write extraction_log.csv
python main.py --build-corpus  # Rebuild document_corpus.csv from documents_enriched.csv + RTVE
```

Or from a notebook / script:

```python
from pipeline import run

df_final = run()               # Uses cached .txt files (fast)
df_final = run(apply_ocr=True) # Include OCR for scanned PDFs
```

If you only need a quick status check, use `python main.py --status` or `python pipeline.py --dry-run`.

---

## Pipeline Reference

### `pipeline.py` — end-to-end preparation flow (recommended)

```bash
python pipeline.py                   # Extract text and build document_corpus.csv
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
python main.py --extract             # Step 3: extract text (writes extraction_log.csv)
python main.py --build-corpus        # Build document_corpus.csv from documents_enriched.csv + RTVE
python main.py --build-corpus --extract-metadata  # Build corpus + extract dates using Nvidia NIMs (Llama 3.1)
python main.py --status              # Show current local data status
```

### Individual modules

```bash
python src/data_etl/rtve_scraper.py                     # Scrape RTVE catalog
python src/data_etl/ocr_extractor.py --dry-run          # Preview scanned PDFs pending OCR
python src/data_etl/ocr_extractor.py --limit 5          # Test OCR on 5 docs
python src/data_etl/ocr_extractor.py                    # OCR all scanned docs
python src/data_etl/build_corpus.py                     # Rebuild document_corpus.csv only
python src/data_etl/metadata_extractor.py               # Test the LLM integration script directly
```

---

## Metadata CSVs

| File | Columns | Generated by |
|------|---------|--------------|
| `moncloa_links.csv` | `name, url, ministry, category, filename` | `moncloa_scraper.py` |
| `download_log.csv` | `status, url, local_path, size_kb, error` | `download_pdfs.py` |
| `documents_enriched.csv` | `name, url, ministry, category, filename, pdf_path, txt_path, rel_path, success, pages, char_count, is_scanned, doc_num, date, doc_type, error` | `pipeline.py` / `ocr_extractor.py` |
| `extraction_log.csv` | `pdf_path, txt_path, rel_path, success, pages, char_count, is_scanned, doc_num, date, doc_type, error, filename` | `pdf_extractor.py` |
| `rtve_documents.csv` | `name, pages, size_kb, summary, tags, link` | `rtve_scraper.py` |
| `document_corpus.csv` | See schema above | `pipeline.py` / `build_corpus.py` |

---

## Notes

- All code, column names, and comments are in **English**. Document content (titles, extracted text, summaries) remains in **Spanish** — these are primary historical sources.
- `data/raw/`, `data/processed/`, and `data/metadata/` are excluded from the repository. Run the pipeline locally to regenerate them.
- Some Moncloa PDFs have no extractable text (fully scanned images). Run with `--ocr` to process them via EasyOCR.
- The `doc_type` field is inferred from filename keywords, not the document content. 116 out of 155 Moncloa docs have no detectable type.
