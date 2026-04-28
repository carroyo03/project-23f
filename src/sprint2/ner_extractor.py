import spacy
import pandas as pd
from tqdm import tqdm
import re
import unicodedata

# Load model
nlp = spacy.load("es_core_news_lg")

def clean_ocr_text(text: str) -> str:
    """Clean OCR artifacts from Moncloa documents before spaCy processing.
    It should receive text with intact newlines to repair word wrapping."""
    # Remove page markers
    text = re.sub(r'---\s*Página\s*\d+\s*---', '', text)
    # Repair split words caused by line wrapping ("Valen-\ncia" -> "Valencia")
    text = re.sub(r'(\w)-\n\s*(\w)', r'\1\2', text)
    # Remove lines that are pure noise: loose characters, numbers, symbols
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Drop very short lines or lines made only of non-letters
        if len(stripped) < 4:
            continue
        if re.match(r'^[\W\d\s]{1,10}$', stripped):
            continue
        lines.append(stripped)
    return ' '.join(lines)

def select_text(row: pd.Series) -> str:
    """Select the correct text field based on document source."""
    if row["source"] == "RTVE":
        return row.get("rtve_summary", "")
    else:
        return row.get("extracted_text", "")

KNOWN_ACRONYMS = {"ETA", "PCE", "PSOE", "UCD", "ONU", "OTAN", "CESID", "SECED", "CIA", "FBI", "BVE"}

# Tokens that should never be treated as entities: greetings, fillers, interjections.
# They are compared accent-insensitively and lowercased, so accented forms match plain forms.
_INFORMAL_TOKENS = {
    # Greetings
    "hola", "buenas", "saludos",
    # Farewells
    "adios", "hasta", "chao", "chau", "ciao",
    # Fillers / confirmations
    "vale", "venga", "vamos", "claro", "exacto", "efectivamente",
    "bueno", "pues", "hombre", "mujer", "oye", "oiga", "mira", "anda", "vaya",
    # Interjections
    "cono", "joder", "hostia", "leche", "caray", "caramba", "coño",
    # Phone conversation tokens
    "digame", "diga", "alo", "halo",
    # Frequent colloquial verbs in transcripts
    "sabes", "entiendes", "comprendes", "escucha", "escuchame",
    "ocupate", "aprovecho", "estoy",
    # Other frequent false positives
    "ininteligible", "ininteligibie", "nada", "persona", "residencia",
}

def _no_accent(text: str) -> str:
    """Remove accents and lowercase text for robust comparisons."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    ).lower()

def normalize_entity_text(raw: str) -> str:
    """Normalize entity text by removing common OCR artifacts."""
    return raw.strip().rstrip(".,;:)")

def is_valid_entity(ent) -> bool:
    """Filter out noisy entities and OCR artifacts."""
    text = normalize_entity_text(ent.text)
    if len(text) < 3 or len(text) > 40:
        return False

    # OCR fragments: if it starts lowercase it is often the tail of a broken word
    if text[0].islower():
        return False

    # Proper names and organizations usually contain uppercase letters.
    if text.islower():
        return False

    tokens = text.split()

    # Too many words (>5) is usually a misclassified phrase.
    if len(tokens) > 5:
        return False

    # Single-token entities: require at least 4 chars (filters "Any", "Ray", "Val").
    if len(tokens) == 1 and len(text) < 4:
        return False

    # Reject multi-token entities containing very short tokens.
    if len(tokens) > 1 and any(len(t) <= 2 for t in tokens):
        return False

    # Reject entities with anomalous OCR symbols or odd punctuation.
    if re.search(r'[<>|?¿!¡+*#%&_\d""]', text):
        return False

    # Vowel ratio heuristic: Spanish tends toward ~40% vowels; OCR garbage tends lower.
    vowels = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        vowel_ratio = sum(1 for c in alpha_chars if c in vowels) / len(alpha_chars)
        if vowel_ratio < 0.20:
            return False

    # Reject invalid 3-consonant starts that often come from OCR garbage.
    invalid_clusters = re.compile(r'\b[bcdfghjklmnñpqrstvwxyz]{3}', re.IGNORECASE)
    if invalid_clusters.search(text):
        return False

    # If syntactic head is VERB/ADV/PRON/etc, it is likely a false positive.
    if ent.root.pos_ in ["VERB", "ADV", "PRON", "SCONJ", "INTJ", "DET", "ADP"]:
        return False

    # Reject if full text or any token is a greeting/farewell/filler/interjection.
    if _no_accent(text) in _INFORMAL_TOKENS:
        return False
    if any(_no_accent(t) in _INFORMAL_TOKENS for t in tokens):
        return False

    # Reject all-uppercase single-token entities unless they are known acronyms.
    if text.isupper() and len(tokens) == 1 and text not in KNOWN_ACRONYMS:
        return False

    # Reject all-uppercase multi-token phrases with stop tokens.
    # They are usually phrase fragments, not proper names.
    _CAPS_STOP_TOKENS = {"DEL", "DE", "LA", "LOS", "LAS", "EL", "UN", "UNA", "QUE", "CON", "POR", "TE", "SE"}
    if text.isupper() and len(tokens) > 1 and any(t in _CAPS_STOP_TOKENS for t in tokens):
        return False

    return True

def extract_entities(text: str, source: str = "Moncloa") -> dict:
    """Extract people and organizations from text."""
    if not isinstance(text, str):
        return {"people": [], "organizations": []}

    # For Moncloa docs: clean OCR first (before whitespace normalization)
    # so wrapping repair can use original newlines.
    if source == "Moncloa":
        text = clean_ocr_text(text)

    # Normalize remaining repeated spaces/tabs/newlines
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < 50:
        return {"people": [], "organizations": []}

    doc = nlp(text[:50000])  # truncate very long texts to avoid OOM

    people = list({normalize_entity_text(ent.text) for ent in doc.ents if ent.label_ == "PER" and is_valid_entity(ent)})
    organizations = list({normalize_entity_text(ent.text) for ent in doc.ents if ent.label_ == "ORG" and is_valid_entity(ent)})

    return {"people": people, "organizations": organizations}

def run_ner_on_corpus(corpus_csv: str, output_csv: str, min_ocr_quality: float = 0.60):
    """
    Apply NER over the full corpus and save results.

    - Moncloa: uses extracted_text (PDF OCR) with artifact cleanup.
               Only processes docs with ocr_quality_score >= min_ocr_quality.
    - RTVE:    uses rtve_summary. NOTE: summaries are truncated to ~303 chars,
               so NER coverage will be limited. They are still included so we do
               not lose entities present in that fragment.
    - Excludes documents marked as illegible and rows without useful text.
    """
    df = pd.read_csv(corpus_csv)

    # Exclude illegible documents
    df = df[~df["flag_illegible"].fillna(False)].copy()

    # For Moncloa, filter by minimum OCR quality
    mask_moncloa_low_quality = (
        (df["source"] == "Moncloa") &
        (df["ocr_quality_score"].fillna(0) < min_ocr_quality)
    )
    n_skipped = mask_moncloa_low_quality.sum()
    if n_skipped:
        print(f"[NER] Skipping {n_skipped} Moncloa docs with ocr_quality_score < {min_ocr_quality}")
    df = df[~mask_moncloa_low_quality].copy()

    # Select the right text field by source
    df["text_for_ner"] = df.apply(select_text, axis=1)

    # Exclude rows without useful text
    df = df[df["text_for_ner"].notna() & (df["text_for_ner"].astype(str).str.strip() != "nan")]

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ents = extract_entities(str(row["text_for_ner"]), source=row["source"])
        results.append({
            "doc_id": row["doc_id"],
            "source": row["source"],
            "people": "|".join(ents["people"]),
            "organizations": "|".join(ents["organizations"]),
            "n_people": len(ents["people"]),
            "n_orgs": len(ents["organizations"]),
        })

    df_ner = pd.DataFrame(results)
    df_ner.to_csv(output_csv, index=False)
    return df_ner
