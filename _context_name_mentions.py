"""
Para Eleonora, Elena y Luigi: mostrar el contexto de la mención
para identificar qué SPEAKER_XX es realmente cada persona.
"""
import pandas as pd

ENRICHED = r"final_reports/csv_enriched"

cases = [
    (r"final_reports/csv_enriched/32 Debate - Post-ICU outpatient clinic- Does it change my patient outcomes__report.csv",
     "SPEAKER_04", "Eleonora Balzani", "eleonora|balzani"),
    (r"final_reports/csv_enriched/Video 11 When intubating_report.csv",
     "SPEAKER_05", "Elena Sancho Ferrando", "elena"),
    (r"final_reports/csv_enriched/Segunda tanda. Video 12 Interactive_report.csv",
     "SPEAKER_01", "Luigi Zattera", "luigi"),
]

for fpath, mentioner_spk, person, pattern in cases:
    print("=" * 70)
    print(f"BUSCANDO: {person}  (mencionado por {mentioner_spk})")
    df = pd.read_csv(fpath)
    import re
    # Encontrar filas donde aparece el nombre
    mask = df["text"].str.contains(pattern, case=False, na=False, regex=True)
    idx_list = df[mask].index.tolist()
    for idx in idx_list:
        # Mostrar ventana ±5 filas
        start = max(0, idx - 3)
        end   = min(len(df), idx + 6)
        window = df.iloc[start:end][["turn_number","speaker","matched_person","text"]]
        print(f"\n  --- Contexto alrededor de fila {idx} ---")
        for _, row in window.iterrows():
            mp = f"[={row['matched_person']}]" if pd.notna(row['matched_person']) and str(row['matched_person']).strip() else ""
            flag = ">>>" if row.name == idx else "   "
            print(f"  {flag} t{int(row['turn_number'])} {row['speaker']}{mp}: {str(row['text'])[:100]}")
    print()
