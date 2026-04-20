import spacy
import pandas as pd
from tqdm import tqdm
import re
import unicodedata

# Cargar modelo
nlp = spacy.load("es_core_news_lg")

def clean_ocr_text(text: str) -> str:
    """Limpia artefactos OCR de documentos Moncloa antes de pasar a spaCy.
    Debe recibir el texto con newlines intactos para poder reparar word-wrap."""
    # Eliminar marcadores de página
    text = re.sub(r'---\s*Página\s*\d+\s*---', '', text)
    # Reparar palabras partidas por word-wrap ("Valen-\ncia" → "Valencia")
    text = re.sub(r'(\w)-\n\s*(\w)', r'\1\2', text)
    # Eliminar líneas con solo ruido: caracteres sueltos, números, símbolos
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Descartar líneas muy cortas o compuestas solo de no-letras
        if len(stripped) < 4:
            continue
        if re.match(r'^[\W\d\s]{1,10}$', stripped):
            continue
        lines.append(stripped)
    return ' '.join(lines)

def select_text(row: pd.Series) -> str:
    """Selecciona el campo de texto correcto según la fuente del documento."""
    if row["source"] == "RTVE":
        return row.get("rtve_summary", "")
    else:
        return row.get("extracted_text", "")

KNOWN_ACRONYMS = {"ETA", "PCE", "PSOE", "UCD", "ONU", "OTAN", "CESID", "SECED", "CIA", "FBI", "BVE"}

# Palabras que nunca son entidades: saludos, despedidas, muletillas, interjecciones.
# Se comparan sin tildes y en minúsculas, así "Sábes" y "Sabes" caen igual.
_INFORMAL_TOKENS = {
    # Saludos
    "hola", "buenas", "saludos",
    # Despedidas
    "adios", "hasta", "chao", "chau", "ciao",
    # Muletillas / confirmaciones
    "vale", "venga", "vamos", "claro", "exacto", "efectivamente",
    "bueno", "pues", "hombre", "mujer", "oye", "oiga", "mira", "anda", "vaya",
    # Interjecciones
    "cono", "joder", "hostia", "leche", "caray", "caramba", "coño",
    # Conversación telefónica
    "digame", "diga", "alo", "halo",
    # Verbos coloquiales frecuentes en transcripciones
    "sabes", "entiendes", "comprendes", "escucha", "escuchame",
    "ocupate", "aprovecho", "estoy",
    # Otros falsos positivos frecuentes
    "ininteligible", "ininteligibie", "nada", "persona", "residencia",
}

def _no_accent(text: str) -> str:
    """Elimina tildes y pasa a minúsculas para comparación robusta."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    ).lower()

def normalize_entity_text(raw: str) -> str:
    """Normaliza el texto de una entidad eliminando artefactos comunes de OCR."""
    return raw.strip().rstrip(".,;:)")

def is_valid_entity(ent) -> bool:
    """Filtra entidades con ruido u OCR defectuoso."""
    text = normalize_entity_text(ent.text)
    if len(text) < 3 or len(text) > 40:
        return False

    # Fragmentos OCR: si empieza en minúscula es la cola de una palabra cortada ("zas Ejército", "dias Civiles")
    if text[0].islower():
        return False

    # Nombres propios y organizaciones suelen tener mayúsculas. Rechazar todo minúsculas
    if text.islower():
        return False

    tokens = text.split()

    # Si son demasiadas palabras (más de 5), suele ser una frase mal clasificada por spacy
    if len(tokens) > 5:
        return False

    # Entidades de una sola palabra: mínimo 4 chars (filtra "Any", "Ray", "Val")
    if len(tokens) == 1 and len(text) < 4:
        return False

    # Rechazar entidades multi-palabra con tokens muy cortos (fragmentos OCR: "Ho odo", "Va cel")
    if len(tokens) > 1 and any(len(t) <= 2 for t in tokens):
        return False

    # Rechazar si contiene caracteres anómalos de OCR o puntuación rara
    if re.search(r'[<>|?¿!¡+*#%&_\d""]', text):
        return False

    # Ratio de vocales: el español tiene ~40% vocales; el garbage OCR suele tener muy pocas
    vowels = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        vowel_ratio = sum(1 for c in alpha_chars if c in vowels) / len(alpha_chars)
        if vowel_ratio < 0.20:
            return False

    # Rechazar clusters de consonantes inválidos en español al inicio de token (garbage OCR: "Lrá", "Mritn")
    invalid_clusters = re.compile(r'\b[bcdfghjklmnñpqrstvwxyz]{3}', re.IGNORECASE)
    if invalid_clusters.search(text):
        return False

    # Si la palabra principal sintáctica del fragmento es un VERBO, ADVERBIO o PRONOMBRE, es un falso positivo
    if ent.root.pos_ in ["VERB", "ADV", "PRON", "SCONJ", "INTJ", "DET", "ADP"]:
        return False

    # Rechazar si el texto completo o cualquier token es saludo/despedida/muletilla/interjección
    if _no_accent(text) in _INFORMAL_TOKENS:
        return False
    if any(_no_accent(t) in _INFORMAL_TOKENS for t in tokens):
        return False

    # Rechazar entidades que consisten solo en mayúsculas si no tienen al menos un espacio
    if text.isupper() and len(tokens) == 1 and text not in KNOWN_ACRONYMS:
        return False

    # Rechazar entidades todo-mayúsculas multi-palabra con artículos/preposiciones
    # ("ALUMNO DEL", "PAR TE DE") — son fragmentos de frases, no nombres propios
    _CAPS_STOP_TOKENS = {"DEL", "DE", "LA", "LOS", "LAS", "EL", "UN", "UNA", "QUE", "CON", "POR", "TE", "SE"}
    if text.isupper() and len(tokens) > 1 and any(t in _CAPS_STOP_TOKENS for t in tokens):
        return False

    return True

def extract_entities(text: str, source: str = "Moncloa") -> dict:
    """Extrae personas y organizaciones de un texto."""
    if not isinstance(text, str):
        return {"people": [], "organizations": []}

    # Para docs Moncloa: limpiar OCR primero (antes de normalizar espacios,
    # para que el word-wrap fix opere sobre newlines originales)
    if source == "Moncloa":
        text = clean_ocr_text(text)

    # Normalizar espacios múltiples/tabs/newlines restantes
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < 50:
        return {"people": [], "organizations": []}

    doc = nlp(text[:50000])  # truncar textos muy largos para evitar OOM

    people = list({normalize_entity_text(ent.text) for ent in doc.ents if ent.label_ == "PER" and is_valid_entity(ent)})
    organizations = list({normalize_entity_text(ent.text) for ent in doc.ents if ent.label_ == "ORG" and is_valid_entity(ent)})

    return {"people": people, "organizations": organizations}

def run_ner_on_corpus(corpus_csv: str, output_csv: str, min_ocr_quality: float = 0.60):
    """
    Aplica NER a todo el corpus y guarda resultados.

    - Moncloa: usa extracted_text (OCR de PDFs) con limpieza de artefactos.
               Solo procesa docs con ocr_quality_score >= min_ocr_quality.
    - RTVE:    usa rtve_summary. NOTA: los summaries están truncados a ~303 chars,
               por lo que el NER será escaso. Se incluyen igualmente para no perder
               entidades que aparezcan en ese fragmento.
    - Excluye documentos marcados como ilegibles y sin texto útil.
    """
    df = pd.read_csv(corpus_csv)

    # Excluir documentos ilegibles
    df = df[~df["flag_illegible"].fillna(False)].copy()

    # Para Moncloa, filtrar por calidad OCR mínima
    mask_moncloa_low_quality = (
        (df["source"] == "Moncloa") &
        (df["ocr_quality_score"].fillna(0) < min_ocr_quality)
    )
    n_skipped = mask_moncloa_low_quality.sum()
    if n_skipped:
        print(f"[NER] Saltando {n_skipped} docs Moncloa con ocr_quality_score < {min_ocr_quality}")
    df = df[~mask_moncloa_low_quality].copy()

    # Seleccionar el campo de texto correcto según la fuente
    df["text_for_ner"] = df.apply(select_text, axis=1)

    # Excluir filas sin texto útil
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
