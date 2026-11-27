# Ucitavanje CSV fajla (1. deo)
import re
import unicodedata
from pathlib import Path

import pandas as pd

#Najveca dozvoljena duzina imena kolone i vrednosti koje ce model videti
MAX_NAME_LEN = 64
MAX_VALUE_LEN = 40
#Vrednosti cistimo samo u kolonama sa malo razlicitih vrednosti (npr grupe)
MAX_DISTINCT_VALUES = 20



# Najveca dozvoljena duzina imena kolone i vrednosti koje ce model videti.
MAX_NAME_LEN = 64
MAX_VALUE_LEN = 40
# Vrednosti cistimo samo u kolonama sa malo razlicitih vrednosti (npr. grupe).
MAX_DISTINCT_VALUES = 20

#Ocisti tekst koji dolazi iz korisnikovog fajla.
def sanitize_text(text, max_len):
    text = str(text)
    # 1) Nevidljive i kontrolne znakove (tab, novi red...) zameni razmakom.
    cleaned = ""
    for ch in text:
        if unicodedata.category(ch).startswith("C"):
            cleaned += " "
        else:
            cleaned += ch

    # 2) Vise razmaka zaredom pretvori u jedan, i skini razmake sa krajeva.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 3) Ako je predugacko, skrati i dodaj "…" na kraj.
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1] + "…"

    return cleaned

#Ocisti jednu vrednost iz tabele, ali prazna polja ostavi prazna.
def _clean_value(value):
    
    if pd.isna(value):
        return value
    return sanitize_text(value, MAX_VALUE_LEN)


def load_csv(path):
    #Procita CSV fajl, ocisti imena i oznake, i vrati tabelu (DataFrame)
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Fajl ne postoji: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError("Fajl nema nijedan red.")

    # Ocisti imena kolona.
    df.columns = [sanitize_text(col, MAX_NAME_LEN) for col in df.columns]

    # Ocisti tekstualne vrednosti, ali samo u kolonama sa malo razlicitih vrednosti.
    for col in df.columns:
        is_text = pd.api.types.is_string_dtype(df[col])
        few_values = df[col].nunique() <= MAX_DISTINCT_VALUES
        if is_text and few_values:
            df[col] = df[col].apply(_clean_value)

    return df