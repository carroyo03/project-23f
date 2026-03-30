"""
Downloader: 23-F Documents PDFs - La Moncloa

Downloads the PDFs previously detected by moncloa_scraper.py.
- HTTP/2 with niquests + Session per thread for connection pooling
- Rate limiting per thread (server-friendly)
- Retries with exponential backoff
- Integrity verification (Content-Length)
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import niquests
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
LINKS_FILE = DATA_DIR / "metadata" / "links_moncloa.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
    "Accept-Language": "es-ES,es;q=0.9",
}

RATE_LIMIT = 0.5   # seconds between requests per thread
MAX_RETRIES = 3
CHUNK_SIZE = 16_384

# Session per thread: reuses TCP/HTTP2 connection within each thread
_thread_local = threading.local()


def _session() -> niquests.Session:
    """Returns (or creates) the HTTP/2 Session for the current thread."""
    if not hasattr(_thread_local, "session"):
        s = niquests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return _thread_local.session


def _safe_name(url: str) -> str:
    return url.split("/")[-1].replace("%20", "_").replace(" ", "_")


def download_pdf(row: pd.Series, output_base: Path = RAW_DIR) -> dict:
    """
    Downloads a single PDF with retries.

    Returns dict with {status, url, local_path, size_kb, error}.
    """
    url: str = row["url"]
    filename = _safe_name(url)

    minister = str(row.get("ministerio", "General")).replace("/", "_").replace(" ", "_")
    categoria = str(row.get("categoria", "General")).replace("/", "_").replace(" ", "_")

    dest_dir = output_base / minister / categoria
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / filename

    if local_path.exists() and local_path.stat().st_size > 1_000:
        return {
            "status": "skipped_existed",
            "url": url,
            "local_path": str(local_path),
            "size_kb": local_path.stat().st_size / 1024,
            "error": None,
        }

    time.sleep(RATE_LIMIT)  # server-friendly, per thread

    session = _session()
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=60, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            if downloaded > 0 and (total_size == 0 or downloaded == total_size):
                return {
                    "status": "success",
                    "url": url,
                    "local_path": str(local_path),
                    "size_kb": local_path.stat().st_size / 1024,
                    "error": None,
                }
            else:
                local_path.unlink(missing_ok=True)
                return {
                    "status": "incomplete",
                    "url": url,
                    "local_path": str(local_path),
                    "size_kb": 0,
                    "error": f"Size does not match: {downloaded} vs {total_size}",
                }

        except niquests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
            else:
                local_path.unlink(missing_ok=True)
                return {
                    "status": "error",
                    "url": url,
                    "local_path": str(local_path),
                    "size_kb": 0,
                    "error": str(e),
                }

    return {"status": "error", "url": url, "local_path": str(local_path), "size_kb": 0, "error": "Max retries exceeded"}


def download_all(links_file: Path = LINKS_FILE, max_workers: int = 4) -> pd.DataFrame | None:
    """
    Downloads all PDFs from the links CSV in parallel.

    Args:
        links_file:  Path to the CSV with URL, ministry, and category columns.
        max_workers: Concurrent threads (I/O-bound → ThreadPoolExecutor).
    """
    print("=" * 60)
    print("DOWNLOADER - 23-F Documents La Moncloa")
    print("=" * 60)

    if not links_file.exists():
        print(f"ERROR: {links_file} not found")
        print("Run first: python main.py --scrape")
        return None

    df_links = pd.read_csv(links_file)
    print(f"\nTotal PDFs to download : {len(df_links)}")
    print(f"Destination directory  : {RAW_DIR}")
    print(f"Concurrent threads     : {max_workers}")
    print(f"Rate limit per thread  : {RATE_LIMIT}s\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_pdf, row): row["url"] for _, row in df_links.iterrows()}
        for future in tqdm(as_completed(future_to_url), total=len(future_to_url), desc="Downloading"):
            results.append(future.result())

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df_results["status"].value_counts().to_string())

    ok = df_results[df_results["status"] == "success"]
    if not ok.empty:
        print(f"\n✓ {len(ok)} PDFs  ({ok['size_kb'].sum() / 1024:.1f} MB)")

    errors = df_results[df_results["status"] == "error"]
    if not errors.empty:
        print(f"✗ {len(errors)} errors")
        for _, e in errors.head(5).iterrows():
            print(f"  {e['url'][:70]}: {e['error']}")

    log_path = DATA_DIR / "metadata" / "download_log.csv"
    df_results.to_csv(log_path, index=False)
    print(f"\nLog: {log_path}")

    return df_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download PDFs from La Moncloa")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent threads")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Seconds between requests per thread")
    args = parser.parse_args()

    RATE_LIMIT = args.rate_limit
    download_all(max_workers=args.workers)
