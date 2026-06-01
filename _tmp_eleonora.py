import pandas as pd
f = r'final_reports/csv_enriched/32 Debate - Post-ICU outpatient clinic- Does it change my patient outcomes__report.csv'
df = pd.read_csv(f)
# Buscar menciones de Trento, Italy, Eleonora en el texto
keywords = ['trento', 'eleonora', 'balzani', 'italy', 'italian']
import re
for spk, grp in df.groupby('speaker'):
    for _, row in grp.iterrows():
        txt = str(row['text']).lower()
        if any(k in txt for k in keywords):
            print(f"{spk} t{int(row['turn_number'])}: {row['text'][:120]}")
# Tambien mostrar primeras 3 intervenciones de cada speaker sin match
print("\n--- Primeras intervenciones sin match ---")
unmatched = df[df['matched_person'].isna() | (df['matched_person'].astype(str).str.strip()=='')]
for spk, grp in unmatched.groupby('speaker'):
    for _, row in grp.head(3).iterrows():
        print(f"{spk} t{int(row['turn_number'])}: {str(row['text'])[:100]}")
    print()
