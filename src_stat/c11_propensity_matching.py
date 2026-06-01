"""
c11_propensity_matching.py
===========================
Propensity Score Matching (PSM) para estimar el efecto causal puro del género.

Problema de confusión: los hombres y mujeres identificados difieren en rol,
especialidad y citaciones. PSM empareja cada mujer con el hombre más similar
en estas covariables, aislando el efecto del género.

Método:
  - Logistic regression: P(gender=male | role, specialty, log_citations)
  - Nearest-neighbor 1:1 matching sin reemplazo (caliper = 0.2*SD(logit))
  - Tras el matching: Mann-Whitney en cada variable de sesgo
  - Diagnóstico de balance: SMD antes/después

Muestra: 94 speakers identificados (45F / 49M)
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, logistic
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"

BIAS_VARS = [
    ("pct_is_mansplaining",    "Mansplaining"),
    ("pct_has_hedge",          "Hedge"),
    ("mean_lexical_diversity", "Div. Léxica"),
    ("mean_assertiveness_score","Asertividad"),
    ("pct_has_agreement",      "% Acuerdo"),
    ("pct_has_disagreement",   "% Desacuerdo"),
    ("pct_has_courtesy",       "% Cortesía"),
    ("n_interventions",        "N Interv."),
    ("total_duration",         "Dur. Total"),
]

df_all = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
df = df_all[df_all["Role"].notna()].copy()
bias_cols  = [v for v, _ in BIAS_VARS if v in df.columns]
bias_names = {v: n for v, n in BIAS_VARS if v in df.columns}

print("="*60)
print("  C11 — PROPENSITY SCORE MATCHING")
print("="*60)
print(f"  Speakers identificados: {len(df)}  (F={( df['gender']=='female').sum()}, M={(df['gender']=='male').sum()})")

# ── Preparar covariables ───────────────────────────────────────────────────────
# Role dummies
df["is_moderator"] = (df["Role"] == "Moderator").astype(float)
df["is_speaker"]   = (df["Role"] == "Speaker").astype(float)
# Specialty
spec = df["Specialty (ICU-Ane-Both)"].fillna("unknown").str.upper()
df["is_ICU"]  = (spec == "ICU").astype(float)
df["is_BOTH"] = (spec == "BOTH").astype(float)
df["is_ANE"]  = (spec == "ANE").astype(float)
# Citations
df["log_cit"] = df["log_citations"].fillna(0)
# Outcome: binary gender
df["is_male"] = (df["gender"] == "male").astype(int)

COVARS = ["is_moderator","is_speaker","is_ICU","is_BOTH","is_ANE","log_cit"]
df_ps = df[COVARS + ["is_male"] + bias_cols].dropna(subset=COVARS)

# ── Propensity Score ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(df_ps[COVARS])
y = df_ps["is_male"].values

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X, y)
df_ps = df_ps.copy()
df_ps["ps"]    = lr.predict_proba(X)[:,1]
df_ps["logit"] = np.log(df_ps["ps"] / (1 - df_ps["ps"] + 1e-10))

print(f"\n  Propensity score: mean_F={df_ps[df_ps['is_male']==0]['ps'].mean():.3f}  "
      f"mean_M={df_ps[df_ps['is_male']==1]['ps'].mean():.3f}")

# ── Nearest-Neighbor Matching 1:1 ─────────────────────────────────────────────
males   = df_ps[df_ps["is_male"] == 1].copy()
females = df_ps[df_ps["is_male"] == 0].copy()

caliper = 0.2 * df_ps["logit"].std()
print(f"  Caliper (0.2 * SD_logit) = {caliper:.4f}")

nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(males[["logit"]].values)
distances, indices = nn.kneighbors(females[["logit"]].values)

matched_pairs = []
used_males = set()
for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
    if dist <= caliper and idx not in used_males:
        matched_pairs.append((females.index[i], males.index[idx]))
        used_males.add(idx)

print(f"  Pares emparejados: {len(matched_pairs)} (de {len(females)}F y {len(males)}M)")

female_idx = [p[0] for p in matched_pairs]
male_idx   = [p[1] for p in matched_pairs]
df_matched = pd.concat([df_ps.loc[female_idx], df_ps.loc[male_idx]])

# ── Balance: SMD antes y después del matching ─────────────────────────────────
def smd(a, b):
    pool = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return abs(a.mean() - b.mean()) / pool if pool > 0 else 0

print(f"\n  BALANCE (SMD) — antes / después del matching:")
print(f"  {'Covariable':<20} {'SMD_antes':>10} {'SMD_después':>12}")
for cov in COVARS:
    b_f = df_ps[df_ps["is_male"]==0][cov]
    b_m = df_ps[df_ps["is_male"]==1][cov]
    a_f = df_matched[df_matched["is_male"]==0][cov]
    a_m = df_matched[df_matched["is_male"]==1][cov]
    s_before = smd(b_f, b_m)
    s_after  = smd(a_f, a_m)
    flag = "OK" if s_after < 0.1 else "WARN"
    print(f"  {cov:<20} {s_before:>10.3f} {s_after:>12.3f}  {flag}")

# ── Efecto de género en muestra emparejada ────────────────────────────────────
print(f"\n  EFECTOS DE GÉNERO EN MUESTRA EMPAREJADA (Mann-Whitney):")
print(f"  {'Variable':<30} {'d_matched':>10} {'p_matched':>10} {'sig':>5}")

matched_results = []
for col in bias_cols:
    fm = df_matched[df_matched["is_male"]==0][col].dropna()
    mm = df_matched[df_matched["is_male"]==1][col].dropna()
    if len(fm) < 3 or len(mm) < 3:
        continue
    pool = np.sqrt((fm.std()**2 + mm.std()**2)/2)
    d    = (fm.mean() - mm.mean()) / pool if pool > 0 else 0
    try: _, p = mannwhitneyu(fm, mm, alternative="two-sided")
    except: p = 1.0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    matched_results.append({"variable": col, "d_matched": round(d,3),
                             "p_matched": round(p,4), "sig": sig,
                             "n_female": len(fm), "n_male": len(mm)})
    print(f"  {col:<30} {d:>+10.3f} {p:>10.4f} {sig:>5}")

res_df = pd.DataFrame(matched_results)
res_df.to_csv(CSV_DIR / "c11_psm_results.csv", index=False)

# ── Gráfico: PS distribution + matched effect sizes ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Propensity Score Matching — Efecto causal del género", fontsize=12, fontweight="bold")

# PS distribution
ax = axes[0]
ax.hist(df_ps[df_ps["is_male"]==0]["ps"], bins=15, alpha=0.6, color="#e74c3c", label="Mujeres (orig)")
ax.hist(df_ps[df_ps["is_male"]==1]["ps"], bins=15, alpha=0.6, color="#3498db", label="Hombres (orig)")
ax.hist(df_matched[df_matched["is_male"]==0]["ps"], bins=15, alpha=0.3, color="#e74c3c",
        label="Mujeres (matched)", linestyle="--", edgecolor="darkred", linewidth=1.5)
ax.hist(df_matched[df_matched["is_male"]==1]["ps"], bins=15, alpha=0.3, color="#3498db",
        label="Hombres (matched)", linestyle="--", edgecolor="darkblue", linewidth=1.5)
ax.set_xlabel("Propensity Score", fontsize=10)
ax.set_ylabel("Frecuencia", fontsize=10)
ax.set_title("Distribución de PS antes/después del matching", fontsize=10)
ax.legend(fontsize=8)

# Effect sizes
ax = axes[1]
if len(res_df):
    colors = ["#e74c3c" if d < 0 else "#3498db" for d in res_df["d_matched"]]
    bars = ax.barh(res_df["variable"].map(bias_names), res_df["d_matched"],
                   color=colors, alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    for bar, row in zip(bars, res_df.itertuples()):
        if row.sig:
            ax.text(row.d_matched + 0.005 * np.sign(row.d_matched),
                    bar.get_y() + bar.get_height()/2,
                    row.sig, va="center", fontsize=10, fontweight="bold",
                    color="black")
    ax.set_xlabel("Cohen's d (F vs M) — muestra emparejada\n(- = hombres > mujeres)", fontsize=10)
    ax.set_title(f"Efectos tras PSM (n={len(matched_pairs)} pares)", fontsize=10)

plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(FIG_DIR / "c11_psm_results.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\n  Figura: {FIG_DIR}/c11_psm_results.png")
print(f"  CSV:    {CSV_DIR}/c11_psm_results.csv")
sig_count = (res_df["sig"] != "").sum()
print(f"\n  Hallazgos significativos tras PSM: {sig_count}/{len(res_df)}")
