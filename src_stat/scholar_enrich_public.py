"""
scholar_enrich_public.py
========================
Busca en Google Scholar a las personas identificadas como "public"
y rellena los campos disponibles en los CSVs enriquecidos.

Campos que se intentan rellenar desde Scholar:
  - Affiliation (Hospital)
  - Country                  (inferido de la afiliación)
  - Number of citations
  - h-index                  (columna nueva: scholar_hindex)
  - Specialty (ICU-Ane-Both) (inferido de los intereses del autor)
  - Year of qualification    (año de primera publicación como proxy)
  - scholar_profile_url      (columna nueva)
  - scholar_interests        (columna nueva)

Notas:
  - Date of birth NO está disponible en Google Scholar.
  - Los perfiles sin resultados se marcan como "not_found".
  - Los resultados se guardan en scholar_cache.json para no repetir búsquedas.
  - Se aplican delays aleatorios para evitar bloqueos de Google Scholar.

Salida:
  final_reports/csv_enriched/scholar_results.csv   ← resumen de lo encontrado
  final_reports/csv_enriched/<sesion>.csv           ← CSVs actualizados
"""

import json
import random
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
from scholarly import scholarly

# ── rutas ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
ENRICHED    = BASE / "final_reports" / "csv_enriched"
LOG_PATH    = ENRICHED / "match_log.csv"
PUB_PATH    = ENRICHED / "public_intros.csv"
CACHE_PATH  = ENRICHED / "scholar_cache.json"
RESULTS_OUT = ENRICHED / "scholar_results.csv"
EXCEL_PATH  = BASE / "esicm_talks_grouped_9_groups_new.xlsx"

# Nombres claramente no-persona (descartar, en minúsculas)
NOT_PEOPLE = {
    "artificial intelligence", "general ward",
}

# Prefijos de título a eliminar al inicio
TITLE_PREFIXES = re.compile(
    r"^(?:prof(?:essor)?|dr)\.?\s+", re.IGNORECASE
)

# Sufijos a eliminar al final (artefactos de transcripción)
TRAILING_ARTIFACTS = re.compile(
    r"\s+i'?m$|\s+i am$", re.IGNORECASE
)

# Nombres equivalentes (normalizar al primero)
NAME_ALIASES = {
    "Walter Vibriga": "Walter Vibringa",
}

# ── helpers ────────────────────────────────────────────────────────────────────
ICU_KEYWORDS = {
    "intensive care", "icu", "critical care", "intensivist",
    "anaesthesia", "anesthesia", "anaesthesiology", "anesthesiology",
    "resuscitation", "sepsis", "ventilation", "mechanical ventilation",
}

def infer_specialty(interests: list[str]) -> str:
    """Infiere Specialty (ICU-Ane-Both-Other) de los intereses del perfil."""
    joined = " ".join(interests).lower()
    has_icu = any(k in joined for k in
                  ["intensive care", "icu", "critical care", "intensivist",
                   "resuscitation", "sepsis", "ventilation"])
    has_ane = any(k in joined for k in
                  ["anaesthesia", "anesthesia", "anaesthesiology"])
    if has_icu and has_ane:
        return "Both"
    if has_icu:
        return "ICU"
    if has_ane:
        return "Ane"
    # Si no hay señal clara, devuelve el texto bruto
    return "Other"


COUNTRY_HINTS = {
    "uk": "UK", "united kingdom": "UK", "england": "UK", "scotland": "UK",
    "usa": "USA", "united states": "USA", "u.s.a": "USA",
    "germany": "Germany", "deutschland": "Germany",
    "france": "France",
    "spain": "Spain", "españa": "Spain",
    "italy": "Italy", "italia": "Italy",
    "netherlands": "Netherlands", "holland": "Netherlands",
    "belgium": "Belgium",
    "switzerland": "Switzerland",
    "sweden": "Sweden",
    "norway": "Norway",
    "denmark": "Denmark",
    "austria": "Austria",
    "portugal": "Portugal",
    "ireland": "Ireland",
    "australia": "Australia",
    "canada": "Canada",
    "brazil": "Brazil", "brasil": "Brazil",
    "china": "China",
    "japan": "Japan",
    "india": "India",
    "saudi arabia": "Saudi Arabia",
    "israel": "Israel",
    "turkey": "Turkey",
    "greece": "Greece",
    "poland": "Poland",
    "czech republic": "Czech Republic",
    "slovakia": "Slovakia",
}

def infer_country(affiliation: str) -> str:
    aff_lower = affiliation.lower()
    for hint, country in COUNTRY_HINTS.items():
        if hint in aff_lower:
            return country
    return ""


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def scholar_lookup(name: str, cache: dict) -> dict:
    """
    Busca el perfil de autor en Google Scholar.
    Devuelve un dict con los campos extraídos, o {'status': 'not_found'}.
    Usa caché para evitar repetir búsquedas.
    """
    if name in cache:
        return cache[name]

    print(f"  Buscando: {name!r} ...", end=" ", flush=True)
    result = {"status": "not_found", "name_searched": name}

    try:
        search_gen = scholarly.search_author(name)
        author = next(search_gen, None)

        if author is None:
            print("sin perfil")
        else:
            # Rellenar basics e indices (sin publicaciones, es más rápido)
            scholarly.fill(author, sections=["basics", "indices"])

            affiliation = author.get("affiliation", "")
            interests   = author.get("interests", [])
            citedby     = author.get("citedby", None)
            hindex      = author.get("hindex", None)
            url         = "https://scholar.google.com/citations?user=" + author.get("scholar_id", "")

            result = {
                "status":          "found",
                "scholar_name":    author.get("name", ""),
                "affiliation":     affiliation,
                "country":         infer_country(affiliation),
                "citedby":         citedby,
                "hindex":          hindex,
                "interests":       interests,
                "specialty":       infer_specialty(interests),
                "profile_url":     url,
                "name_searched":   name,
            }

            # Intentar obtener año de primera publicación
            try:
                scholarly.fill(author, sections=["counts"])
                cites_per_year = author.get("cites_per_year", {})
                if cites_per_year:
                    first_year = min(int(y) for y in cites_per_year.keys())
                    result["first_pub_year"] = first_year
            except Exception:
                pass

            print(f"✓  ({affiliation[:60]})")

    except StopIteration:
        print("sin resultados")
    except Exception as e:
        print(f"error: {e}")
        result["status"] = "error"
        result["error"]  = str(e)

    cache[name] = result
    save_cache(cache)

    # Delay aleatorio para no ser bloqueado
    time.sleep(random.uniform(6, 12))
    return result


# ── construir lista de personas públicas ──────────────────────────────────────
def clean_name(raw: str) -> str:
    """Elimina prefijos de título y artefactos de transcripción del nombre."""
    name = TRAILING_ARTIFACTS.sub("", raw).strip()
    name = TITLE_PREFIXES.sub("", name).strip()
    return NAME_ALIASES.get(name, name)


def build_public_list() -> pd.DataFrame:
    log = pd.read_csv(LOG_PATH)

    # 1. Del Excel (in_session=False)
    excel_pub = (
        log[log["in_session"] == False][["session", "speaker", "matched_person"]]
        .rename(columns={"matched_person": "name"})
        .assign(source="excel_public")
    )

    # 2. De public_intros (personas nuevas, no en Excel)
    pub_df  = pd.read_csv(PUB_PATH)
    new_pub = pub_df[pub_df["score"] < 65].copy()
    new_pub = new_pub[new_pub["said_name"].str.split().str.len() >= 2]

    # Limpiar nombres: quitar prefijos/sufijos y aplicar alias
    new_pub["name"] = new_pub["said_name"].apply(clean_name)

    # Descartar nombres que siguen siendo no-persona tras limpiar
    new_pub = new_pub[~new_pub["name"].str.lower().isin(NOT_PEOPLE)]
    # Mínimo 2 tokens, O 1 token si es nombre compuesto con guion (ej. Bessen-Boulis)
    new_pub = new_pub[
        (new_pub["name"].str.split().str.len() >= 2) |
        (new_pub["name"].str.contains("-"))
    ]

    new_pub = (
        new_pub[["session", "speaker", "name"]]
        .assign(source="new_public")
    )

    combined = pd.concat([excel_pub, new_pub], ignore_index=True)

    # Deduplicar: si el mismo nombre aparece en varias sesiones, buscamos Scholar
    # sólo una vez pero aplicamos a todas las sesiones
    return combined


# ── aplicar resultados a los CSVs enriquecidos ───────────────────────────────
def apply_to_csv(session: str, speaker: str, result: dict, source: str) -> None:
    csv_path = ENRICHED / f"{session}.csv"
    if not csv_path.exists():
        return

    df   = pd.read_csv(csv_path)
    mask = df["speaker"] == speaker

    if result["status"] != "found" or not mask.any():
        return

    def set_if_empty(col, value):
        """Sólo sobreescribe si la celda está vacía/NaN."""
        if col not in df.columns:
            df[col] = None
        is_empty = df.loc[mask, col].isna() | (df.loc[mask, col].astype(str).str.strip() == "")
        if is_empty.any() and value:
            df.loc[mask & is_empty, col] = str(value)

    # Campos que ya existen en el CSV
    if source == "new_public":
        # Para personas nuevas, rellenamos todos los campos disponibles
        set_if_empty("Affiliation (Hospital)", result.get("affiliation", ""))
        set_if_empty("Country",                result.get("country", ""))
        set_if_empty("Specialty (ICU-Ane-Both)", result.get("specialty", ""))
    # Actualizar citations siempre (Scholar es más reciente que Excel)
    if result.get("citedby") is not None:
        if "Number of citations" not in df.columns:
            df["Number of citations"] = None
        df.loc[mask, "Number of citations"] = result["citedby"]

    # Year of qualification (primera publicación como proxy)
    if result.get("first_pub_year"):
        set_if_empty("Year of qualification (specialty)",
                     result["first_pub_year"])

    # Columnas nuevas de Scholar
    for new_col, key in [
        ("scholar_hindex",      "hindex"),
        ("scholar_profile_url", "profile_url"),
        ("scholar_interests",   "interests"),
    ]:
        if new_col not in df.columns:
            df[new_col] = None
        val = result.get(key, "")
        if isinstance(val, list):
            val = "; ".join(val)
        df.loc[mask, new_col] = str(val) if val else None

    df.to_csv(csv_path, index=False)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    cache      = load_cache()
    public_df  = build_public_list()
    total      = len(public_df)

    print(f"Personas públicas a investigar: {total}")
    print(f"  Ya en caché: {sum(1 for n in public_df['name'] if n in cache)}")
    print()

    summary_rows = []

    for i, row in public_df.iterrows():
        name    = row["name"]
        session = row["session"]
        speaker = row["speaker"]
        source  = row["source"]

        print(f"[{i+1}/{total}] {name}")
        result = scholar_lookup(name, cache)
        apply_to_csv(session, speaker, result, source)

        summary_rows.append({
            "session":         session,
            "speaker":         speaker,
            "name":            name,
            "source":          source,
            "status":          result.get("status"),
            "scholar_name":    result.get("scholar_name", ""),
            "affiliation":     result.get("affiliation", ""),
            "country":         result.get("country", ""),
            "citedby":         result.get("citedby", ""),
            "hindex":          result.get("hindex", ""),
            "specialty":       result.get("specialty", ""),
            "first_pub_year":  result.get("first_pub_year", ""),
            "interests":       "; ".join(result.get("interests", [])),
            "profile_url":     result.get("profile_url", ""),
        })

    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv(RESULTS_OUT, index=False)

    found = results_df[results_df["status"] == "found"]
    print(f"""
{'='*60}
  COMPLETADO
  Total buscadas  : {total}
  Perfiles encontrados : {len(found)} / {total}
  Resultados en   : {RESULTS_OUT}
  Caché en        : {CACHE_PATH}
{'='*60}""")


if __name__ == "__main__":
    main()
