"""
Step-by-step CLI orchestrator for the 23-F pipeline.

For the end-to-end pipeline (extraction + OCR + corpus build) use pipeline.py instead:
    python pipeline.py

This module provides granular step control:
    python main.py --scrape      # Scrape PDF links from La Moncloa
    python main.py --download    # Download PDFs to data/raw/
    python main.py --extract     # Extract text from native PDFs
    python main.py --build-corpus  # Build document_corpus.csv
    python main.py --status      # Show current local data status
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'data_etl'))


def run_scraper():
    """Executes La Moncloa scraping"""
    from moncloa_scraper import scrape_all, save_to_csv
    
    print("\n" + "="*60)
    print("STEP 1: LA MONCLOA SCRAPING")
    print("="*60)
    
    documents = scrape_all()
    
    if documents:
        save_to_csv(documents, 'data/metadata/links_moncloa.csv')
        print(f"\n✓ Scraping completed: {len(documents)} PDFs detected")
        return True
    else:
        print("\n✗ Scraping error")
        return False


def run_download():
    """Downloads the PDFs"""
    from download_pdfs import download_all
    
    print("\n" + "="*60)
    print("STEP 2: PDF DOWNLOAD")
    print("="*60)
    
    results = download_all(max_workers=3)
    
    if results is not None:
        success = (results['status'] == 'success').sum()
        print(f"\n✓ Download completed: {success}/{len(results)} PDFs")
        return True
    else:
        print("\n✗ Download error")
        return False


def run_extraction():
    """Extracts text from PDFs"""
    from pdf_extractor import process_all_pdfs, verify_extraction
    
    print("\n" + "="*60)
    print("STEP 3: TEXT EXTRACTION")
    print("="*60)
    
    df = process_all_pdfs(apply_ocr=False)
    
    if df is not None:
        success = df['success'].sum()
        print(f"\n✓ Extraction completed: {success}/{len(df)} PDFs")
        verify_extraction()
        return True
    else:
        print("\n✗ Extraction error")
        return False


def show_status():
    """Shows the current project status"""
    from pathlib import Path
    import pandas as pd
    
    print("\n" + "="*60)
    print("PROJECT STATUS")
    print("="*60)
    
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    metadata_dir = Path('data/metadata')
    
    # Count downloaded PDFs
    if raw_dir.exists():
        pdfs = list(raw_dir.rglob('*.pdf'))
        print(f"\nDownloaded PDFs: {len(pdfs)}")
    else:
        print("\nDownloaded PDFs: 0 (directory does not exist)")
    
    # Count extracted texts
    if processed_dir.exists():
        txts = list(processed_dir.rglob('*.txt'))
        print(f"Extracted texts: {len(txts)}")
    else:
        print("Extracted texts: 0 (directory does not exist)")
    
    # Detected links
    links_file = metadata_dir / 'moncloa_links.csv'
    if links_file.exists():
        df_links = pd.read_csv(links_file)
        print(f"Detected links: {len(df_links)}")
    else:
        print("Detected links: 0 (scraping has not been done)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='23-F Pipeline - Sprint 0')
    parser.add_argument('--scrape', action='store_true', help='Only scraping')
    parser.add_argument('--download', action='store_true', help='Only download')
    parser.add_argument('--extract', action='store_true', help='Only extraction')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--build-corpus', action='store_true', help='Build consolidated corpus')
    parser.add_argument('--extract-metadata', action='store_true', help='Use LLM to extract date and doc_type metadata (needs OPENAI_API_KEY)')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        print("23-F Pipeline - Sprint 0")
        print("="*40)
        print("Usage: python main.py [options]")
        print("")
        print("Options:")
        print("  --scrape    Execute only scraping")
        print("  --download  Execute only download")
        print("  --extract   Execute only extraction")
        print("  --status    Show current status")
        print("  (no args)   Execute complete pipeline")
        print("")
        show_status()
        sys.exit(0)
    
    if args.status:
        show_status()
        sys.exit(0)

    if args.build_corpus:
        from src.data_etl.build_corpus import build_corpus
        build_corpus(extract_metadata=args.extract_metadata)
        sys.exit(0)
    
    if args.scrape:
        success = run_scraper()
    elif args.download:
        success = run_download()
    elif args.extract:
        success = run_extraction()
    else:
        # Complete pipeline
        print("\n" + "="*60)
        print("COMPLETE PIPELINE - 23-F Documents")
        print("="*60)
        
        s1 = run_scraper()
        if not s1:
            print("\n✗ Pipeline stopped: error in scraping")
            sys.exit(1)
        
        s2 = run_download()
        if not s2:
            print("\n✗ Pipeline stopped: error in download")
            sys.exit(1)
        
        s3 = run_extraction()
        if not s3:
            print("\n✗ Pipeline stopped: error in extraction")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED")
        print("="*60)
        show_status()
