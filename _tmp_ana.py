import pandas as pd, re

f = r'final_reports/csv_enriched/32 Debate - Post-ICU outpatient clinic- Does it change my patient outcomes__report.csv'
df = pd.read_csv(f)

# SPEAKER_04 dijo "Ana Maria Yuan from Madrid Spain and my co-moderator [Rita?]"
# Buscar más pistas: Madrid, Spain, Ioan, Ana, Yuan
keywords = ['madrid', 'spain', 'yuan', 'ana', 'romanian', 'ion', 'anamar', 'ana-maria']

print("=== Menciones de Ana-Maria Ioan ===")
for _, row in df.iterrows():
    txt = str(row['text']).lower()
    if any(k in txt for k in keywords):
        print(f"  {row['speaker']} t{int(row['turn_number'])}: {row['text'][:120]}")

print("\n=== SPEAKER_04: intervenciones completas (moderador) ===")
sp4 = df[df['speaker'] == 'SPEAKER_04']
for _, row in sp4.head(8).iterrows():
    print(f"  t{int(row['turn_number'])}: {row['text'][:110]}")

print("\n=== ¿Algún speaker se autopresenta? (I'm / my name / I am from) ===")
selfintro = r"\bI(?:'m| am)\b|\bmy name\b"
for _, row in df.iterrows():
    if re.search(selfintro, str(row['text']), re.IGNORECASE):
        print(f"  {row['speaker']} t{int(row['turn_number'])}: {row['text'][:110]}")
