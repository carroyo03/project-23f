"""
Metadata Extractor
Uses an LLM via NVIDIA API (OpenAI compatible) to infer structured data from noisy text.
Expected structured output: Date (ISO format).
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

class DocumentMetadata(BaseModel):
    date: Optional[str] = Field(
        description="The date the document was produced or the event it describes, formatted as YYYY-MM-DD. If unknown, return null."
    )

def _get_client():
    if not HAS_OPENAI:
        raise ImportError("OpenAI package is not installed.")
    
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        error_msg = "Please set NVIDIA_API_KEY environment variable to use the freely provided Nvidia Build models. You can create a .env file."
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
        
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

def extract_metadata_llm(text: str) -> DocumentMetadata:
    """Passes text to an LLM to extract structured metadata using NVIDIA NIMs."""
    if not text or len(text) < 50:
        return DocumentMetadata(date=None)
        
    # Truncate text context to save tokens and focus on the header where dates usually exist
    prompt_context = text[:2000]
    
    try:
        client = _get_client()
        # Using typical Llama 3.1 8B Instruct model from Nvidia NIM, known for speed and JSON output.
        response = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized Spanish historian extracting event/creation dates from 1980s declassified PDF OCR texts (mostly concerning the 23-F 1981 coup in Spain). Output ONLY a valid JSON object with a single key 'date' containing the date in YYYY-MM-DD format, or null if not found. Do not include markdown formatting or extra text."
                },
                {
                    "role": "user",
                    "content": f"Extract the date from the following text fragment:\n\n{prompt_context}"
                }
            ],
            temperature=0.0,
            max_tokens=100
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
