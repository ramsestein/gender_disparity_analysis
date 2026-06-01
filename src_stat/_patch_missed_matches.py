"""
_patch_missed_matches.py
========================
Corrige speakers que se presentaron con errores de transcripción
y no fueron captados por enrich_speakers.py.

Correcciones confirmadas (tanda 1):
  1. "20 The consensus..." SPEAKER_05  → Christian Jung
  2. "Video 3 emergency..." SPEAKER_02  → Jos Latour      (si no está ya)
  3. "20 The consensus..." SPEAKER_09  → Jean-Louis Teboul
  4. "Video 1 Therapeutic..." SPEAKER_00 → Richard Bourne

Correcciones confirmadas (tanda 2 — validadas por auto-presentación en transcripción):
  5. "13 Interactive session - Is armcuff..." SPEAKER_03 → Mohamed Alebsawy
     (se presenta como "Muhammad Al-Fsaoui from UK")
  6. "46_How_promote_inclusion_disability..." SPEAKER_09 → Margarita Borislavova
     (se presenta como "I'm Margarita...working in the ICU in France")
  7. "Video 3 emergency care workers-019..." SPEAKER_03 → Stephan Katzenschlager
     (se presenta como "Stefan Kacznerschleuer from Germany, Helbert")
  8. "53_Should_intensivist_be_in_ED..." SPEAKER_03 → Kevin Roedl
     (presentado por SPEAKER_07 como "Kevin...from Hamburg, Germany")

Correcciones confirmadas (tanda 3 — validadas por presentación de terceros en transcripción):
  9. "Video 11 When intubating..." SPEAKER_04 → Elena Sancho Ferrando
     (SPEAKER_06/05 la llaman "Elena" mientras ella presenta el caso)
  10. "32 Debate - Post-ICU..." SPEAKER_03 → Eleonora Balzani
      (SPEAKER_02 la llama "Dr. Balsani", ASR de Balzani)
  11. "32 Debate - Post-ICU..." SPEAKER_04 → Ana-Maria Ioan
      (se presenta como "Ana Maria Yuan [Ioan] from Madrid Spain")
  12. "Segunda tanda. Video 12 Interactive..." SPEAKER_01 → Luigi Zattera
      (presentado por Emilio Rodriguez-Ruiz como colega del comité NEXT)
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
    # Tanda 1
    ("20 The consensus guideline on shock and haemodynamic monitoring- Why should I follow it__report",
     "SPEAKER_05", "Christian Jung"),
    ("Video 3 emergency care workers-019_report",
     "SPEAKER_02", "Jos Latour"),
    ("20 The consensus guideline on shock and haemodynamic monitoring- Why should I follow it__report",
     "SPEAKER_09", "Jean-Louis Teboul"),
    ("Video 1 Therapeutic challanges-011_report",
     "SPEAKER_00", "Richard Bourne"),
    # Tanda 2 — validadas por auto-presentación en transcripción
    ("13 Interactive session - Is armcuff enough for monitoring extracorporeal support__report",
     "SPEAKER_03", "Mohamed Alebsawy"),
    ("46_How_promote_inclusion_disability_report",
     "SPEAKER_09", "Margarita Borislavova"),
    ("Video 3 emergency care workers-019_report",
     "SPEAKER_03", "Stephan Katzenschlager"),
    ("53_Should_intensivist_be_in_ED_report",
     "SPEAKER_03", "Kevin Roedl"),
    # Tanda 3 — validadas por presentación de terceros en transcripción
    # Elena Sancho Ferrando: SPEAKER_06 (=Gaetano Perchiazzi) y SPEAKER_05 la llaman "Elena"
    #   mientras presenta el caso clínico; ella es la ponente principal de Video 11
    ("Video 11 When intubating_report",
     "SPEAKER_04", "Elena Sancho Ferrando"),
    # Eleonora Balzani: SPEAKER_02 la llama "Dr. Balsani" (ASR de Balzani) en sesión 32
    ("32 Debate - Post-ICU outpatient clinic- Does it change my patient outcomes__report",
     "SPEAKER_03", "Eleonora Balzani"),
    # Ana-Maria Ioan: SPEAKER_04 se auto-presenta como "Ana Maria Yuan [Ioan] from Madrid Spain"
    #   y llama a Rita Fernández "my co-moderator" — SPEAKER_04 ES Ana-Maria Ioan
    ("32 Debate - Post-ICU outpatient clinic- Does it change my patient outcomes__report",
     "SPEAKER_04", "Ana-Maria Ioan"),
    # Luigi Zattera: SPEAKER_02 (=Emilio Rodriguez-Ruiz) lo presenta como
    #   "my colleague from the next committee, Luigi Sátera [Zattera]" en Video 12
    ("Segunda tanda. Video 12 Interactive_report",
     "SPEAKER_01", "Luigi Zattera"),
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
