"""
Compara el comportamiento de los NEXT members identificados
vs el resto de moderadores del corpus.
"""
import os, glob, pandas as pd, numpy as np
from scipy import stats
from scipy.stats import norm

base = r'c:\Users\Ramsés\Desktop\Proyectos\gender_diaparity\final_reports\csv_enriched'
files = [f for f in glob.glob(os.path.join(base, '*.csv'))
         if 'match_log' not in f and 'next_member' not in f]

dfs = []
for fpath in files:
    try:
        df = pd.read_csv(fpath)
        dfs.append(df)
    except:
        pass

data = pd.concat(dfs, ignore_index=True)

# Normalizar booleano
data['is_next_member'] = data['is_next_member'].astype(str).str.lower().isin(['true', '1', 'yes'])

# Grupos
next_df = data[data['is_next_member'] == True]

role_col = 'Role' if 'Role' in data.columns else 'role'
mod_df = data[
    (data['is_next_member'] == False) &
    (data[role_col].astype(str).str.lower().str.contains('mod', na=False))
]

print(f"NEXT members:          {next_df['speaker'].nunique()} speakers únicos, {len(next_df)} intervenciones")
print(f"Moderadores no-NEXT:   {mod_df['speaker'].nunique()} speakers únicos, {len(mod_df)} intervenciones")
print(f"Género NEXT members:   {next_df['gender'].value_counts().to_dict()}")
print()

metrics = [
    ('duration',             'Duración media (s)'),
    ('interrupts_previous',  'Tasa de interrupciones emitidas'),
    ('interrupted_by_next',  'Tasa de interrupciones recibidas'),
    ('has_overlap',          'Tasa de solapamiento'),
    ('conflict_score',       'Conflict score'),
    ('assertiveness_score',  'Assertiveness score'),
]

print(f"{'Métrica':<30} {'NEXT mean':>10} {'Mod mean':>10} {'p-valor':>10} {'r (effect)':>12} {'sig':>5}")
print("-" * 80)

for col, label in metrics:
    if col not in data.columns:
        print(f"{label:<30}  [columna no encontrada]")
        continue
    a = pd.to_numeric(next_df[col], errors='coerce').dropna()
    b = pd.to_numeric(mod_df[col], errors='coerce').dropna()
    if len(a) < 5 or len(b) < 5:
        continue
    stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    z = norm.ppf(p / 2) if p > 0.001 else -3.09
    n = len(a) + len(b)
    r = abs(z) / np.sqrt(n) if n > 0 else np.nan
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f"{label:<30} {a.mean():>10.3f} {b.mean():>10.3f} {p:>10.4f} {r:>12.3f} {sig:>5}")

# Sentiment breakdown
if 'sentiment' in data.columns:
    print()
    print("Distribución de sentimiento:")
    s_next = next_df['sentiment'].value_counts(normalize=True).rename('NEXT')
    s_mod  = mod_df['sentiment'].value_counts(normalize=True).rename('Mod')
    print(pd.concat([s_next, s_mod], axis=1).round(3).to_string())
