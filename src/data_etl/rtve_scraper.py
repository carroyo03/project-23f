"""
Scraper: Declassified 23-F Documents - RTVE
https://23fbuscador.rtve.es

Extracts: name, pages, size, full summary, tags.
Saves to data/metadata/rtve_23f.csv

Usage: python src/data_etl/rtve_scraper.py
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import niquests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://23fbuscador.rtve.es/"
OUTPUT_PATH = Path("data/metadata/rtve_documents.csv")
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9",
}

RATE_LIMIT = 0.3       # seconds between requests per thread
MAX_DETAIL_WORKERS = 8


def _request_with_retries(session: niquests.Session, url: str, params: dict | None = None, timeout: int = 30) -> niquests.Response:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except niquests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  ⚠️ Request failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise


def _get_page(session: niquests.Session, page_size: int = 200, page: int = 1) -> str:
    params = {"page_size": page_size, "page": page}
    r = _request_with_retries(session, BASE_URL, params=params, timeout=30)
    return r.text


def _parse_page(html: str) -> tuple[list[dict], str]:
    """Extracts table rows and pagination text."""
    soup = BeautifulSoup(html, "html.parser")

    filtros = soup.select_one(".filters-results")
    print(f"  → {filtros.get_text(strip=True) if filtros else '?'}")

    nav_pos = soup.select_one(".nav-position")
    pag_txt = nav_pos.get_text(strip=True) if nav_pos else "Page 1 of 1"
    print(f"  → {pag_txt}")

    rows: list[dict] = []
    for tr in soup.select("tbody tr"):
        tds = tr.select("td")
        if len(tds) < 4:
            continue

        link_el = tds[0].select_one("a")
        href = link_el["href"] if link_el else ""
        link = href.split("?")[0] if href else ""
        if link and not link.startswith("http"):
            link = "https://23fbuscador.rtve.es" + link

        tags = " | ".join(t.get_text(strip=True) for t in tds[4].select(".tag-chip")) if len(tds) > 4 else ""

        rows.append({
            "name": tds[0].get_text(strip=True),
            "pages": tds[1].get_text(strip=True),
            "size_kb": tds[2].get_text(strip=True),
            "summary_truncated": tds[3].get_text(separator=" ", strip=True),
            "tags": tags,
            "link": link,
        })
    return rows, pag_txt


def _scrape_listing(session: niquests.Session, page_size: int = 200) -> list[dict]:
    """Gets all pages from the listing."""
    all_rows: list[dict] = []
    page = 1
    while True:
        print(f"\n[Page {page}]")
        html = _get_page(session, page_size=page_size, page=page)
        rows, pag_txt = _parse_page(html)
        if not rows:
            break
        all_rows.extend(rows)
        print(f"  → Accumulated: {len(all_rows)}")
        try:
            parts = pag_txt.replace("Page", "").replace("of", "/").replace("Página", "").replace("de", "/").split("/")
            current, total = int(parts[0].strip()), int(parts[1].strip())
            if current >= total:
                break
            page += 1
            time.sleep(0.5)
        except Exception:
            break
    return all_rows


def _fetch_summary(session: niquests.Session, url: str) -> str:
    """Gets the full summary from a document's detail page."""
    if not url:
        return ""
    time.sleep(RATE_LIMIT)
    try:
        r = _request_with_retries(session, url, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")
        for sel in [".summary-cell", ".resumen", ".doc-summary", "[class*='summary']", "p.summary"]:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(separator=" ", strip=True)
                if text:
                    return text
        return ""
    except niquests.RequestException:
        return ""


def scrape_all(page_size: int = 200, max_detail_workers: int = MAX_DETAIL_WORKERS) -> pd.DataFrame:
    """
    Scrapes the RTVE search engine and fetches full summaries in parallel.

    Args:
        page_size:           Documents per page in the listing.
        max_detail_workers:  Threads for detail pages (I/O-bound).
    """
    print("=== Scraper 23F RTVE ===\n")

    with niquests.Session() as session:
        session.headers.update(HEADERS)

        rows = _scrape_listing(session, page_size=page_size)
        if not rows:
            print("❌ No data found.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        print(f"\n✅ Total documents in listing: {len(df)}")

        # Summaries in parallel with ThreadPoolExecutor
        print(f"\n📄 Fetching summaries ({len(df)} pages, {max_detail_workers} threads)...")
        summaries: list[str] = [""] * len(df)

        with ThreadPoolExecutor(max_workers=max_detail_workers) as executor:
            future_map = {
                executor.submit(_fetch_summary, session, row["link"]): i
                for i, (_, row) in enumerate(df.iterrows())
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Details"):
                idx = future_map[future]
                text = future.result()
                summaries[idx] = text or df.iloc[idx]["summary_truncated"]

        df["summary"] = summaries

    return df[["name", "pages", "size_kb", "summary", "tags", "link"]]


def main() -> None:
    df = scrape_all()
    if df.empty:
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n💾 Saved to: {OUTPUT_PATH}  ({len(df)} documents)")


if __name__ == "__main__":
    main()
