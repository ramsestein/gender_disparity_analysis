"""
c23_add_next_member_flag.py
============================
Añade la columna 'is_next_member' a cada CSV en final_reports/csv_enriched/,
indicando si el matched_person de cada fila pertenece a la lista de
"Next Members.xlsx".

Estrategia de matching
----------------------
  1. Construye una lista de nombres normalizados desde Next Members.xlsx
     (Name + Surname, en minúsculas sin tildes), junto con sus tokens.
  2. Para cada matched_person:
       a. Normalizar (minúsculas + sin tildes).
       b. Tokenizar.
       c. **Token-subset match**: todos los tokens de matched_person deben
          estar contenidos en los tokens del nombre de Next Members.
          Esto permite que falten apellidos, pero NO que haya palabras extra.
       d. Si no hay match por subset, se prueba fuzzy (token_set_ratio >= 85)
          para capturar errores ortográficos leves.

Output
------
  - Sobrescribe los CSVs en final_reports/csv_enriched/ con la nueva columna.
  - Guarda un log en final_reports/csv_enriched/next_member_log.csv
"""

import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).resolve().parent.parent
CSV_DIR       = BASE / "final_reports" / "csv_enriched"
NEXT_MEMBERS  = BASE / "Next Members.xlsx"
LOG_PATH      = CSV_DIR / "next_member_log.csv"

# Umbral fuzzy
THRESHOLD = 85


# ── helpers ────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Minúsculas + eliminar diacríticos (tildes, etc.)."""
    t = str(text).lower().strip()
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> set[str]:
    """Normaliza y divide en tokens."""
    return set(normalize(text).split())


def build_next_members_list(xlsx_path: Path) -> list[dict]:
    """
    Retorna una lista de dicts con:
      - raw: nombre original
      - norm: nombre normalizado completo
      - tokens: set de tokens normalizados
    """
    df = pd.read_excel(xlsx_path, sheet_name="Feuil1")
    members: list[dict] = []
    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        surname = str(row["Surname"]).strip()
        if name and surname and name != "nan" and surname != "nan":
            raw = f"{name} {surname}"
            members.append({
                "raw": raw,
                "norm": normalize(raw),
                "tokens": tokenize(raw),
            })
    return members


def is_next_member(person: object, members: list[dict]) -> bool:
    """
    Comprueba si 'person' está en Next Members.

    1. Normaliza y tokeniza matched_person.
    2. Para cada miembro:
       a. Token-subset: todos los tokens de person están en los tokens del miembro.
       b. Fuzzy: token_set_ratio >= THRESHOLD.
    """
    if pd.isna(person) or not str(person).strip():
        return False

    p_norm = normalize(str(person))
    if not p_norm:
        return False

    p_tokens = set(p_norm.split())

    for m in members:
        # Token-subset: todos los tokens de person deben estar en el miembro
        if p_tokens and p_tokens.issubset(m["tokens"]):
            return True

        # Fuzzy para errores ortográficos
        score = fuzz.token_set_ratio(p_norm, m["norm"])
        if score >= THRESHOLD:
            return True

    return False


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Cargando Next Members.xlsx ...")
    members = build_next_members_list(NEXT_MEMBERS)
    print(f"  → {len(members)} miembros cargados\n")

    csv_files = sorted(CSV_DIR.glob("*_report.csv"))
    if not csv_files:
        print("  No se encontraron CSVs en", CSV_DIR)
        return

    log_rows: list[dict] = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        n_orig = len(df)

        # Añadir columna
        df["is_next_member"] = df["matched_person"].apply(
            lambda p: is_next_member(p, members)
        )

        # Guardar
        df.to_csv(csv_path, index=False)

        n_true = df["is_next_member"].sum()
        n_false = n_orig - n_true
        print(f"  {csv_path.name}: {n_true} filas True, {n_false} filas False")

        # Log por sesión
        for speaker, grp in df.groupby("speaker"):
            person = grp["matched_person"].iloc[0]
            flag = grp["is_next_member"].iloc[0]
            log_rows.append({
                "session": csv_path.stem,
                "speaker": speaker,
                "matched_person": person if pd.notna(person) else "",
                "is_next_member": flag,
            })

    # Guardar log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_PATH, index=False)

    total_true = log_df["is_next_member"].sum()
    total_false = len(log_df) - total_true
    print(f"""
{'='*60}
  COMPLETADO
  Sesiones procesadas : {len(csv_files)}
  Next Members (total): {len(members)}
  Speakers en datos   : {len(log_df)}
    → is_next_member=True  : {total_true}
    → is_next_member=False : {total_false}
  Log guardado en     : {LOG_PATH}
{'='*60}""")


if __name__ == "__main__":
    main()
