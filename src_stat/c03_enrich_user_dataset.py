"""
c03_enrich_user_dataset.py
==========================
Enriquece el user_level_dataset.csv con las nuevas variables procedentes
de los CSVs enriquecidos (Role, Group, Specialty, citations, year_of_qual).

Genera variables dummy y numéricas listas para clustering.

Entrada:  final_reports/resultados/csv/user_level_dataset.csv
          final_reports/csv_enriched/*.csv
Salida:   final_reports/resultados/csv/user_level_dataset_enriched.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE     = Path(__file__).resolve().parent.parent
CSV_DIR  = BASE / "final_reports" / "resultados" / "csv"
ENRICHED = BASE / "final_reports" / "csv_enriched"

SKIP_FILES = {
    "match_log.csv", "public_intros.csv", "unmatched_intros.csv",
    "scholar_results.csv", "scholar_cache.json", "scholar_run.log",
}

# ── 1. Cargar dataset de usuarios existente ────────────────────────────────────
user_df = pd.read_csv(CSV_DIR / "user_level_dataset.csv")
# session viene con sufijo ".csv" en user_level
user_df["session_stem"] = user_df["session"].str.removesuffix(".csv")
print(f"Usuarios en dataset base: {len(user_df)}")

# ── 2. Extraer variables nuevas de cada CSV enriquecido ───────────────────────
new_rows = []
for csv_path in sorted(ENRICHED.glob("*.csv")):
    if csv_path.name in SKIP_FILES:
        continue
    df = pd.read_csv(csv_path)
    stem = csv_path.stem

    new_cols = ["Role", "Group", "Country", "Specialty (ICU-Ane-Both)",
                "Number of citations", "Year of qualification (specialty)",
                "Affiliation (Hospital)"]
    available = [c for c in new_cols if c in df.columns]
    if not available:
        continue

    for speaker, grp in df.groupby("speaker"):
        row = {"session_stem": stem, "speaker": speaker}
        for col in available:
            vals = grp[col].dropna()
            row[col] = vals.iloc[0] if not vals.empty else None
        new_rows.append(row)

new_data = pd.DataFrame(new_rows)
print(f"Filas con datos nuevos extraídos: {len(new_data)}")
print(f"Role asignados: {new_data['Role'].notna().sum()}")

# ── 3. Merge ──────────────────────────────────────────────────────────────────
merged = user_df.merge(
    new_data,
    left_on=["session_stem", "speaker"],
    right_on=["session_stem", "speaker"],
    how="left"
)
merged = merged.drop(columns=["session_stem"])
print(f"Usuarios tras merge: {len(merged)}")

# ── 4. Construir features dummy y numéricas ───────────────────────────────────

# --- Role → dummies ---
role_col = merged["Role"].fillna("unknown").str.strip().str.lower()
merged["role_known"]    = (role_col != "unknown").astype(int)
merged["is_moderator"]  = (role_col == "moderator").astype(int)
merged["is_speaker"]    = (role_col == "speaker").astype(int)
merged["is_public"]     = (role_col == "public").astype(int)

# --- Specialty → dummies ---
spec_col = merged["Specialty (ICU-Ane-Both)"].fillna("unknown").str.strip().str.upper()
merged["spec_known"] = (spec_col != "UNKNOWN").astype(int)
merged["is_ICU"]     = (spec_col == "ICU").astype(int)
merged["is_Ane"]     = (spec_col == "ANE").astype(int)
merged["is_Both"]    = (spec_col == "BOTH").astype(int)

# --- Citaciones → log1p ---
merged["Number of citations"] = pd.to_numeric(
    merged["Number of citations"], errors="coerce"
)
merged["log_citations"] = np.log1p(merged["Number of citations"].fillna(0))

# --- Años de carrera = 2025 - Year of qualification ---
merged["Year of qualification (specialty)"] = pd.to_numeric(
    merged["Year of qualification (specialty)"], errors="coerce"
)
merged["career_years"] = (
    (2025 - merged["Year of qualification (specialty)"])
    .clip(lower=0)
    .fillna(0)
)

# --- Group → dummies (9 grupos del Excel) ---
group_dummies = pd.get_dummies(
    merged["Group"].fillna("Unknown"), prefix="grp"
).astype(int)
merged = pd.concat([merged, group_dummies], axis=1)

# ── 5. Guardar ────────────────────────────────────────────────────────────────
out_path = CSV_DIR / "user_level_dataset_enriched.csv"
merged.to_csv(out_path, index=False)

# Resumen
new_feat = ["role_known", "is_moderator", "is_speaker", "is_public",
            "spec_known", "is_ICU", "is_Ane", "is_Both",
            "log_citations", "career_years"] + list(group_dummies.columns)

print(f"\n✅ Dataset enriquecido guardado: {out_path}")
print(f"   Filas: {len(merged)} | Columnas totales: {len(merged.columns)}")
print(f"   Nuevas features para clustering: {len(new_feat)}")
print(f"\n   Cobertura de variables nuevas:")
for col in ["role_known", "spec_known", "log_citations", "career_years"]:
    n = (merged[col] > 0).sum()
    print(f"     {col:25s}: {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")
print(f"\n   Role breakdown:")
print(merged["Role"].value_counts(dropna=False).to_string())
print(f"\n   Specialty breakdown:")
print(merged["Specialty (ICU-Ane-Both)"].value_counts(dropna=False).to_string())
print(f"\n   Group breakdown:")
print(merged["Group"].value_counts(dropna=False).to_string())
