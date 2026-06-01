import pandas as pd, re

f = r'final_reports/csv_enriched/Segunda tanda. Video 12 Interactive_report.csv'
df = pd.read_csv(f)

print("=== Speakers y primeras intervenciones ===")
for spk, grp in df.groupby('speaker'):
    mp = list(grp['matched_person'].dropna().unique())
    first = grp['text'].dropna().iloc[0][:90] if len(grp['text'].dropna()) > 0 else ''
    print(f"{spk} ({len(grp)} rows) matched={mp}")
    print(f"  \"{first}\"")

print("\n=== Contexto alrededor de 'Luigi' ===")
mask = df['text'].str.contains('luigi|zattera', case=False, na=False)
for idx in df[mask].index:
    start, end = max(0, idx-4), min(len(df), idx+6)
    for _, row in df.iloc[start:end].iterrows():
        flag = ">>>" if row.name == idx else "   "
        mp = f"[={row['matched_person']}]" if pd.notna(row.get('matched_person','')) and str(row.get('matched_person','')).strip() else ""
        print(f"  {flag} t{int(row['turn_number'])} {row['speaker']}{mp}: {str(row['text'])[:100]}")

print("\n=== Self-introductions ===")
for _, row in df.iterrows():
    if re.search(r"\bI(?:'m| am)\b.*from|\bmy name\b", str(row['text']), re.I):
        print(f"  {row['speaker']} t{int(row['turn_number'])}: {row['text'][:110]}")
