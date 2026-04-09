"""
Scraper: Declassified 23-F Documents - La Moncloa
https://www.lamoncloa.gob.es/consejodeministros/paginas/desclasificacion-documentos-23f.aspx

Extracts PDF links, categorizes them by ministry/source, and saves them to a CSV.
"""

import time
from pathlib import Path
from urllib.parse import urljoin

import niquests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://www.lamoncloa.gob.es/consejodeministros/paginas/desclasificacion-documentos-23f.aspx"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_MINISTERIOS = {
    "Interior": "Interior",
    "Defensa": "Defensa",
    "Exteriores": "Exteriores",
}

_CATEGORIAS_INTERIOR = {
    "Guardia Civil": "Guardia Civil",
    "Dirección General de la Policía": "Policia",
    "Otra documentación del Ministerio del Interior": "Interior_Otro",
}

_CATEGORIAS_DEFENSA = {
    "Centro Nacional de Inteligencia (CNI)": "CNI",
    "Archivo general e histórico del Ministerio de Defensa": "Defensa_Archivo",
}


def get_page(url: str = BASE_URL, max_retries: int = 3) -> str:
    """Downloads a page with retries and exponential backoff."""
    with niquests.Session() as session:
        session.headers.update(HEADERS)
        for attempt in range(max_retries):
            try:
                r = session.get(url, timeout=30)
                r.raise_for_status()
                r.encoding = "utf-8"
                return r.text
            except niquests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise RuntimeError(f"Could not access {url}: {e}") from e
    return ""


def parse_document_links(html: str) -> list[dict]:
    """
    Extracts all PDF links from La Moncloa's page.
    Categorizes by Ministry and document type.
    """
    soup = BeautifulSoup(html, "html.parser")
    documents: list[dict] = []

    current_ministerio: str | None = None
    current_categoria = "General"

    for header in soup.find_all(["h2", "h3", "h4"]):
        text = header.get_text(strip=True)

        if "Ministerio" in text:
            for key in _MINISTERIOS:
                if key in text:
                    current_ministerio = key
                    current_categoria = "General"
                    break

        elif current_ministerio and header.name == "h4":
            if "Guardia Civil" in text:
                current_categoria = "Guardia Civil"
            elif "Policía" in text or "Dirección General" in text:
                current_categoria = "Policia"
            elif "CNI" in text:
                current_categoria = "CNI"
            elif "Archivo" in text:
                current_categoria = "Archivo"
            else:
                current_categoria = text[:30]

        sibling = header.find_next_sibling()
        while sibling:
            if sibling.name in ["h2", "h3", "h4"]:
                break
            if hasattr(sibling, "find_all"):
                for link in sibling.find_all("a", href=True):
                    href: str = link["href"]
                    if ".pdf" in href.lower() and "boe" not in href.lower():
                        documents.append(
                            {
                                "name": link.get_text(strip=True) or href.split("/")[-1],
                                "url": urljoin(BASE_URL, href),
                                "ministry": current_ministerio or "Unknown",
                                "category": current_categoria,
                                "filename": href.split("/")[-1],
                            }
                        )
            sibling = sibling.find_next_sibling()

    # Deduplicate by URL
    seen: set[str] = set()
    unique: list[dict] = []
    for doc in documents:
        if doc["url"] not in seen:
            seen.add(doc["url"])
            unique.append(doc)

    return unique


def scrape_all() -> list[dict]:
    """Scrapes the main page and returns all found documents."""
    print(f"Accessing: {BASE_URL}")
    html = get_page(BASE_URL)
    documents = parse_document_links(html)

    print(f"\nTotal documents found: {len(documents)}")
    if documents:
        df = pd.DataFrame(documents)
        print("\nBy Ministry:\n", df["ministry"].value_counts().to_string())
        print("\nBy Category:\n", df["category"].value_counts().to_string())

    return documents


def save_to_csv(documents: list[dict], output_path: str | Path = "data/metadata/moncloa_links.csv") -> pd.DataFrame:
    """Saves the links to a CSV."""
    df = pd.DataFrame(documents)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved to: {output_path}")
    return df


if __name__ == "__main__":
    documents = scrape_all()
    if documents:
        save_to_csv(documents, "data/metadata/moncloa_links.csv")
        print(f"\nScraping completed! {len(documents)} PDFs detected.")
    else:
        print("\nERROR! No documents found. Verify the scraper.")
