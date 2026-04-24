"""
Metadata Extractor
==================
Uses an LLM via the NVIDIA NIM API (OpenAI-compatible) to infer structured
data from noisy OCR text extracted from the 23-F declassified documents.

**This is the module where the API call to NVIDIA models is made.**

The integration uses:
  - Endpoint : https://integrate.api.nvidia.com/v1  (OpenAI-compatible REST API)
  - Model    : meta/llama-3.1-8b-instruct  (served via NVIDIA NIM)
  - Auth     : NVIDIA_API_KEY environment variable (set in .env)

Entry points
------------
extract_metadata_llm(text)
    Sends a single document fragment to the NVIDIA NIM API and returns a
    DocumentMetadata object with the extracted date.

batch_extract_metadata(df)
    Runs extraction over the full corpus DataFrame (analysis_text column).

get_nvidia_client()
    Returns a configured OpenAI client pointing at the NVIDIA NIM endpoint.
    Use this function when you need to call NVIDIA models directly from other
    modules.
"""

import os
import time
import json
import logging
from typing import Optional
from dotenv import load_dotenv

import pandas as pd
from pydantic import BaseModel, Field

# Only try to import openai if installed (for local environments without it)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVIDIA NIM endpoint — all API calls to NVIDIA models go through this URL
# ---------------------------------------------------------------------------
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"


class DocumentMetadata(BaseModel):
    date: Optional[str] = Field(
        description="The date the document was produced or the event it describes, formatted as YYYY-MM-DD. If unknown, return null."
    )


def get_nvidia_client() -> "OpenAI":
    """
    Returns a configured OpenAI client that points to the NVIDIA NIM API endpoint.

    **This is the single place where the connection to NVIDIA models is established.**
    The client is OpenAI-compatible, so you can use the standard
    ``client.chat.completions.create(...)`` interface to call any model
    available on the NVIDIA NIM catalogue.

    Requires
    --------
    NVIDIA_API_KEY
        Set in the environment or in a ``.env`` file (see ``.env.example``).
        Obtain a free key at https://build.nvidia.com.

    Returns
    -------
    openai.OpenAI
        Client pre-configured with ``base_url=NVIDIA_API_BASE_URL``.

    Raises
    ------
    ImportError
        If the ``openai`` package is not installed.
    ValueError
        If ``NVIDIA_API_KEY`` is not set in the environment.
    """
    if not HAS_OPENAI:
        raise ImportError("OpenAI package is not installed.")

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        error_msg = (
            "Please set NVIDIA_API_KEY environment variable to use the freely "
            "provided NVIDIA Build models. You can create a .env file (see .env.example)."
        )
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # --- NVIDIA API call happens here ---
    return OpenAI(
        base_url=NVIDIA_API_BASE_URL,
        api_key=api_key,
    )


# Keep the private alias for backwards compatibility
_get_client = get_nvidia_client

def extract_metadata_llm(text: str) -> DocumentMetadata:
    """
    Sends *text* to the NVIDIA NIM API and extracts structured metadata.

    The actual HTTP request to the NVIDIA model is made inside this function
    via ``client.chat.completions.create()``.  The model used is
    ``NVIDIA_DEFAULT_MODEL`` (``meta/llama-3.1-8b-instruct``).

    Parameters
    ----------
    text : str
        Raw OCR / extracted text from a 23-F document.

    Returns
    -------
    DocumentMetadata
        Pydantic model with the extracted ``date`` field (YYYY-MM-DD) or
        ``None`` when the date cannot be determined.
    """
    if not text or len(text) < 50:
        return DocumentMetadata(date=None)

    # Truncate text context to save tokens and focus on the header where dates usually exist
    prompt_context = text[:2000]

    try:
        # --- NVIDIA NIM API call ---
        client = get_nvidia_client()
        response = client.chat.completions.create(
            model=NVIDIA_DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a specialized Spanish historian extracting event/creation dates "
                        "from 1980s declassified PDF OCR texts (mostly concerning the 23-F 1981 "
                        "coup in Spain). Output ONLY a valid JSON object with a single key 'date' "
                        "containing the date in YYYY-MM-DD format, or null if not found. "
                        "Do not include markdown formatting or extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Extract the date from the following text fragment:\n\n{prompt_context}",
                },
            ],
            temperature=0.0,
            max_tokens=100,
        )

        # Parse the JSON response manually since Pydantic strict Mode (.parse) might not be fully supported by this endpoint
        response_text = response.choices[0].message.content

        # Clean potential markdown code blocks if the model ignored instructions
        if response_text.startswith("```"):
            response_text = response_text.strip("`").replace("json", "").strip()

        data = json.loads(response_text)
        return DocumentMetadata(date=data.get("date"))
    except Exception as e:
        print(f"LLM API Error: {e}")
        return DocumentMetadata(date=None)

def batch_extract_metadata(df: pd.DataFrame, limit: int = None) -> pd.DataFrame:
    """
    Runs extraction over the 'analysis_text' column of the DataFrame.
    Skips if 'flag_illegible' is True.
    """
    if "flag_illegible" not in df.columns:
        print("Warning: flag_illegible column missing. Ensure data_cleaning.py is run first.")
        df["flag_illegible"] = False

    mask = (~df["flag_illegible"]) & (df["analysis_text"].notna()) & (df["analysis_text"] != "")
    target_indices = df[mask].index
    
    if limit is not None:
        target_indices = target_indices[:limit]
        
    print(f"Starting LLM extraction for {len(target_indices)} documents using NVIDIA API...")
    
    for count, idx in enumerate(target_indices):
        text = df.loc[idx, "analysis_text"]
        doc_id = df.loc[idx, "doc_id"]
        
        print(f"[{count+1}/{len(target_indices)}] Processing {doc_id}...")
        meta = extract_metadata_llm(text)
        
        # Only overwrite if LLM found something
        if meta.date:
            df.loc[idx, "date"] = meta.date
            
        # Respect Nvidia Build limits (40 req / sec)
        # Sequential processing is already slow enough, but adding a small sleep ensures we never burst
        time.sleep(0.05)
            
    print("Metadata extraction completed.")
    return df

if __name__ == "__main__":
    test_text = "Madrid, 24 de Febrero de 1981.\nNOTA INFORMATIVA DE LA 2ª SECCION.\nAsunto: Movimientos telúricos."
    try:
        res = extract_metadata_llm(test_text)
        print(f"Test Extraction: {res}")
    except ValueError:
        pass
