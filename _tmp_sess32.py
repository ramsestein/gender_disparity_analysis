import pandas as pd
f = r'final_reports/csv_enriched/32 Debate - Post-ICU outpatient clinic- Does it change my patient outcomes__report.csv'
df = pd.read_csv(f)
for spk, grp in df.groupby('speaker'):
    mp = grp['matched_person'].dropna().unique()
    first_txt = grp['text'].dropna().iloc[0][:80] if len(grp['text'].dropna()) > 0 else ''
    print(f"{spk} ({len(grp)} rows) matched={list(mp)}: \"{first_txt}\"")
