# ML Project 23-F — Declassified Documents

Analysis of the declassified documents relating to the February 23, 1981 attempted coup d'état in Spain.


---

## Project Structure

```text
├── data/
│   ├── raw/                    # Original PDFs (Excluded from repo, ~279 MB)
│   ├── processed/              # Text extracted from PDFs (.txt - Excluded from repo)
│   └── metadata/               # CSVs with scraped data (Excluded from repo)
├── src/
│   ├── moncloa_scraper.py      # Link scraping from La Moncloa
│   ├── download_pdfs.py        # PDF downloading with retries and concurrency
│   ├── pdf_extractor.py        # Native text extraction (pdfplumber)
│   ├── ocr_extractor.py        # Text extraction via OCR (EasyOCR / GPU)
│   └── rtve_scraper.py         # Metadata scraping from RTVE 23-F search
├── notebooks/                  # One notebook per use case (Sprint 2+)
├── outputs/                    # Figures, Gephi exports
├── reports/                    # Project report
├── main.py                     # Orchestrator pipeline (entry point)
└── requirements.txt            # Python dependencies
```

---

## How to Obtain the Data (`.txt` and `.csv`)

To keep the repository clean and light, **all data files (PDFs, TXTs, and CSVs) are ignored in Git**. Every team member must generate the data locally by running the extraction pipeline.

### 1. Clone the repo and Setup

```bash
git clone https://www.github.com/carroyo03/project-23f.git
cd project-23f

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate the Data Files

Run the following commands in order to completely populate your local `data/` folder with the PDFs and their extracted `.txt` format:

```bash
# Step 1: Scrape the URLs and save to metadata CSVs
python main.py --scrape

# Step 2: Download the 153 PDFs to data/raw/
python main.py --download

# Step 3: Extract text from the native (non-scanned) PDFs to data/processed/
python main.py --extract

# Step 4: Extract text from the scanned PDFs using GPU/OCR to data/processed/
python src/ocr_extractor.py
```

If you just want to run the full automated pipeline (without OCR):
```bash
python main.py
```

---

## Pipeline Commands Reference

```bash
python main.py --scrape    # Scrape links from La Moncloa
python main.py --download  # Download PDFs
python main.py --extract   # Extract text (without OCR)
python main.py --status    # View current project status

# OCR for scanned PDFs (Sprint 1)
python src/ocr_extractor.py --dry-run  # See what would be processed
python src/ocr_extractor.py --limit 5  # Test with 5 PDFs
python src/ocr_extractor.py            # Process all scanned docs
```

---