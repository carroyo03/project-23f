import pandas as pd

def fill_doc_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infers missing doc_type fields based on simple keyword matching rules in
    filename and title for Moncloa documents, and automatically assigns 
    'Resumen de Juicio' for RTVE documents.
    """
    for idx, row in df.iterrows():
        # Only overwrite if it's missing or None
        if pd.notna(row.get("doc_type")) and str(row.get("doc_type")).strip() != "":
            continue
            
        if row["source"] == "RTVE":
            df.at[idx, "doc_type"] = "Resumen de Juicio (RTVE)"
        elif row["source"] == "Moncloa":
            # Check filename, title, or first few lines of text
            search_str = f"{str(row.get('filename', ''))} {str(row.get('title', ''))} {str(row.get('extracted_text', ''))[:500]}".lower()
            
            if "conversacion" in search_str or "telefónica" in search_str or "telefonica" in search_str:
                df.at[idx, "doc_type"] = "Telephone transcript"
            elif "nota informativa" in search_str or "nota" in search_str:
                df.at[idx, "doc_type"] = "Intelligence note"
            elif "telex" in search_str or "télex" in search_str:
                df.at[idx, "doc_type"] = "Telex"
            elif "informe" in search_str:
                df.at[idx, "doc_type"] = "Report"
            elif "oficio" in search_str:
                df.at[idx, "doc_type"] = "Report" # or Official letter
            elif "manuscrito" in search_str:
                df.at[idx, "doc_type"] = "Manuscrito"
            elif "reservado" in search_str:
                df.at[idx, "doc_type"] = "Restricted"
            elif "secreto" in search_str:
                df.at[idx, "doc_type"] = "Secret"
            else:
                df.at[idx, "doc_type"] = "Otro"
                
    return df
