"""
_public_intros.py
=================
Busca personas que se presentaron en el texto (con mayúsculas = nombre propio)
pero cuyo nombre NO está en el registro Excel de ESICM.
Trabaja sobre el texto ORIGINAL (case-sensitive) para evitar falsos positivos.
"""
import re, unicodedata
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz, process as rfp

BASE = Path(__file__).resolve().parent.parent
excel_df  = pd.read_excel(BASE / "esicm_talks_grouped_9_groups_new.xlsx",
                          sheet_name="Grouped participants")
all_names      = excel_df["Name of the person"].dropna().str.strip().tolist()

def norm(t):
    t = str(t).lower()
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")

all_names_norm = [norm(n) for n in all_names]

# Patrón case-sensitive: "I am/I'm/My name is" + 1-3 tokens con MAYÚSCULA inicial
INTRO = re.compile(
    r"\b(?:I am|I'm|My name is)\s+(?:Dr\.?\s+|Prof\.?\s+)?"
    r"([A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+){0,2})"
)

# Speakers ya emparejados por enrich_speakers.py
matched_log  = pd.read_csv(BASE / "final_reports/csv_enriched/match_log.csv")
matched_keys = set(zip(matched_log["session"], matched_log["speaker"]))

rows = []
for csv_path in sorted((BASE / "final_reports/csv_cleaned").glob("*.csv")):
    stem = csv_path.stem
    df   = pd.read_csv(csv_path)
    for speaker, grp in df.groupby("speaker"):
        if (stem, speaker) in matched_keys:
            continue
        text = " ".join(grp["text"].dropna().astype(str))
        seen = set()
        for m in INTRO.finditer(text):
            cand = m.group(1).strip()
            if cand in seen:
                continue
            seen.add(cand)
            cand_n = norm(cand)
            res    = rfp.extractOne(cand_n, all_names_norm,
                                    scorer=fuzz.token_sort_ratio)
            score  = res[1] if res else 0
            best   = all_names[all_names_norm.index(res[0])] if res else ""
            rows.append({"session": stem, "speaker": speaker,
                         "said_name": cand, "best_excel_match": best,
                         "score": round(score, 1)})

out = (pd.DataFrame(rows)
       .sort_values("score", ascending=False)
       .drop_duplicates(["session", "speaker"])
       .reset_index(drop=True))

public = out[out["score"] < 65].sort_values("said_name").reset_index(drop=True)
close  = out[out["score"] >= 65].sort_values("score", ascending=False).reset_index(drop=True)

# Guardar
OUT = BASE / "final_reports/csv_enriched/public_intros.csv"
public.to_csv(OUT, index=False)

print(f"Personas que se presentaron pero NO están en el registro: {len(public)}")
print()
print(public[["session", "speaker", "said_name", "best_excel_match", "score"]].to_string())
print()
if len(close):
    print(f"Posibles errores de transcripción (cerca del umbral, score >= 65): {len(close)}")
    print(close[["session", "speaker", "said_name", "best_excel_match", "score"]].to_string())
print()
print(f"Guardado en: {OUT}")
