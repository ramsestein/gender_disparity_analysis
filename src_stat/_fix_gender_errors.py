"""Corrige los 5 errores de género detectados por validación nombre vs. voz."""
import pandas as pd
from pathlib import Path

ENRICHED = Path(__file__).resolve().parent.parent / "final_reports" / "csv_enriched"

FIXES = [
    # (session_stem, speaker, wrong_gender, correct_gender)
    ("28 Can we trust our microbiological diagnostics__report", "SPEAKER_02", "female", "male"),
    ("2_report",                                               "SPEAKER_00", "male",   "female"),
    ("39_report",                                              "SPEAKER_06", "female", "male"),
    ("42_report",                                              "SPEAKER_01", "female", "male"),
    ("6 Immunomodulation in severe infections- Tool or toy_ _report", "SPEAKER_05", "female", "male"),
]

for stem, speaker, wrong, correct in FIXES:
    path = ENRICHED / f"{stem}.csv"
    if not path.exists():
        print(f"  NO ENCONTRADO: {path.name}")
        continue
    df = pd.read_csv(path)
    mask = df["speaker"] == speaker
    before = df.loc[mask, "gender"].unique()
    df.loc[mask, "gender"] = correct
    df.to_csv(path, index=False)
    print(f"  OK  {stem[:55]:55s} | {speaker} | {wrong} -> {correct}  (filas={mask.sum()})")
