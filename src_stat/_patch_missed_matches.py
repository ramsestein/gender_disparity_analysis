"""
_patch_missed_matches.py
========================
Corrige los 4 speakers que se presentaron con errores de transcripción
y no fueron captados por enrich_speakers.py.

Correcciones confirmadas:
  1. "20 The consensus..." SPEAKER_05  → Christian Jung
  2. "Video 3 emergency..." SPEAKER_02  → Jos Latour      (si no está ya)
  3. "20 The consensus..." SPEAKER_09  → Jean-Louis Teboul
  4. "Video 1 Therapeutic..." SPEAKER_00 → Richard Bourne
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE        = Path(__file__).resolve().parent.parent
ENRICHED    = BASE / "final_reports" / "csv_enriched"
EXCEL_PATH  = BASE / "esicm_talks_grouped_9_groups_new.xlsx"
LOG_PATH    = ENRICHED / "match_log.csv"

EXCEL_ADD_COLS = [
    "Group", "Talk name", "Country", "Affiliation (Hospital)",
    "Date of birth (DD/MM/YYYY)", "Year of qualification (specialty)",
    "Specialty (ICU-Ane-Both)", "Number of citations",
]

excel_df = pd.read_excel(EXCEL_PATH, sheet_name="Grouped participants")

# Correcciones: (csv_stem, speaker, excel_person_name)
CORRECTIONS = [
    ("20 The consensus guideline on shock and haemodynamic monitoring- Why should I follow it__report",
     "SPEAKER_05", "Christian Jung"),
    ("Video 3 emergency care workers-019_report",
     "SPEAKER_02", "Jos Latour"),
    ("20 The consensus guideline on shock and haemodynamic monitoring- Why should I follow it__report",
     "SPEAKER_09", "Jean-Louis Teboul"),
    ("Video 1 Therapeutic challanges-011_report",
     "SPEAKER_00", "Richard Bourne"),
]

log_df   = pd.read_csv(LOG_PATH)
new_logs = []

for stem, speaker, person_name in CORRECTIONS:
    csv_path = ENRICHED / f"{stem}.csv"
    if not csv_path.exists():
        print(f"  ⚠ No encontrado: {csv_path.name}")
        continue

    df = pd.read_csv(csv_path)

    # Verificar si el speaker ya tiene match
    already = df.loc[df["speaker"] == speaker, "matched_person"].dropna()
    if not already.empty:
        print(f"  ℹ {stem} | {speaker} ya está emparejado con '{already.iloc[0]}' — omitido")
        continue

    # Buscar la persona en Excel
    xlsx_key   = stem + ".xlsx"
    person_rows = excel_df[excel_df["Name of the person"] == person_name]
    if person_rows.empty:
        print(f"  ⚠ '{person_name}' no encontrado en Excel")
        continue

    # Preferir la fila donde oldNamesMapping == xlsx_key
    session_row = person_rows[person_rows["oldNamesMapping"] == xlsx_key]
    if not session_row.empty:
        row        = session_row.iloc[0]
        in_session = True
    else:
        row        = person_rows.iloc[0]
        in_session = False

    # Añadir columnas si no existen
    for col in EXCEL_ADD_COLS + ["Role", "matched_person", "match_confidence", "match_score"]:
        if col not in df.columns:
            df[col] = None

    mask = df["speaker"] == speaker
    for col in EXCEL_ADD_COLS:
        df.loc[mask, col] = row[col]

    df.loc[mask, "Role"]             = row["Role"] if in_session else "public"
    df.loc[mask, "matched_person"]   = person_name
    df.loc[mask, "match_confidence"] = "manual-patch"
    df.loc[mask, "match_score"]      = None

    df.to_csv(csv_path, index=False)

    role_assigned = row["Role"] if in_session else "public"
    print(f"  ✅ {stem} | {speaker} → {person_name} "
          f"({'en sesión' if in_session else 'public'}, role={role_assigned})")

    new_logs.append({
        "session":        stem,
        "speaker":        speaker,
        "matched_person": person_name,
        "score":          None,
        "confidence":     "manual-patch",
        "in_session":     in_session,
        "role_assigned":  role_assigned,
    })

# Actualizar match_log
if new_logs:
    updated_log = pd.concat([log_df, pd.DataFrame(new_logs)], ignore_index=True)
    updated_log.to_csv(LOG_PATH, index=False)
    print(f"\n  Log actualizado: {LOG_PATH}")
else:
    print("\n  No se añadieron nuevas entradas al log.")

print("\nHecho.")
