import pandas as pd
f = r'final_reports/csv_enriched/Segunda tanda. Video 12 Interactive_report.csv'
df = pd.read_csv(f)
# Texto completo de SPEAKER_02 en t1
t1_spk2 = df[(df['speaker']=='SPEAKER_02') & (df['turn_number']==1)]
for _, row in t1_spk2.iterrows():
    print(row['text'])
