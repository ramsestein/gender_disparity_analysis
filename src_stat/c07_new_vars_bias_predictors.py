"""
c07_new_vars_bias_predictors.py
================================
Analiza si las nuevas variables (país, citaciones, especialidad, rol, grupo)
predicen o moderan el sesgo de género en las variables de comunicación.

Preguntas:
  1. ¿Las citaciones correlacionan con sesgo? ¿Igual en hombres y mujeres?
  2. ¿El país/región de origen se asocia con más o menos sesgo?
  3. ¿La especialidad (ICU vs Ane) modera las diferencias de género?
  4. ¿El rol (Moderator vs Speaker vs public) modera las diferencias?

Muestra: solo los 94 speakers identificados (con datos enriquecidos).

Salida:
  csv/c07_citations_bias_corr.csv
  csv/c07_country_gender_gap.csv
  csv/c07_specialty_gender_gap.csv
  graficos/c07_*.png
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from scipy.stats import chi2_contingency

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos" / "c07_predictors"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Variables de sesgo (outcomes) ─────────────────────────────────────────────
BIAS_VARS = [
    ("pct_is_mansplaining",       "Mansplaining"),
    ("pct_has_hedge",             "Hedge"),
    ("mean_lexical_diversity",    "Div. Léxica"),
    ("mean_assertiveness_score",  "Asertividad"),
    ("pct_has_agreement",         "% Acuerdo"),
    ("pct_has_disagreement",      "% Desacuerdo"),
    ("pct_has_courtesy",          "% Cortesía"),
    ("n_interventions",           "N Interv."),
    ("total_duration",            "Dur. Total"),
]

# ── Cargar datos ──────────────────────────────────────────────────────────────
df_all = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
# Solo identificados
df = df_all[df_all["Role"].notna()].copy()
print(f"Speakers identificados: {len(df)}  (F={( df['gender']=='female').sum()}, M={(df['gender']=='male').sum()})")

bias_cols  = [v for v, _ in BIAS_VARS if v in df.columns]
bias_names = {v: n for v, n in BIAS_VARS if v in df.columns}

COUNTRY_REGION = {
    "Netherlands":"Europa", "Belgium":"Europa", "France":"Europa",
    "UK":"Europa", "United Kingdom":"Europa", "Germany":"Europa",
    "Switzerland":"Europa", "Spain":"Europa", "Italy":"Europa",
    "Sweden":"Europa", "Denmark":"Europa", "Norway":"Europa",
    "Finland":"Europa", "Austria":"Europa", "Portugal":"Europa",
    "Ireland":"Europa", "Greece":"Europa", "Poland":"Europa",
    "Czech Republic":"Europa", "Hungary":"Europa", "Romania":"Europa",
    "Israel":"M. Oriente/Asia", "Turkey":"M. Oriente/Asia",
    "Japan":"M. Oriente/Asia", "China":"M. Oriente/Asia",
    "Australia":"Oceanía/América", "USA":"Oceanía/América",
    "United States":"Oceanía/América", "Canada":"Oceanía/América",
    "Brazil":"Oceanía/América", "Argentina":"Oceanía/América",
}
df["region"] = df["Country"].map(COUNTRY_REGION).fillna("Otros/Desconocido")

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    pool = np.sqrt(((na-1)*a.std()**2 + (nb-1)*b.std()**2) / (na+nb-2))
    return (a.mean() - b.mean()) / pool if pool > 0 else 0

def mw_p(a, b):
    if len(a) < 3 or len(b) < 3: return np.nan
    try: return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except: return np.nan

# ══════════════════════════════════════════════════════════════════════════════
# 1. CITACIONES × VARIABLES DE SESGO  (Spearman, por género)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  1. CITACIONES vs SESGO (Spearman por género)")
print("="*60)

cit_results = []
df_cit = df[df["log_citations"] > 0]   # solo los que tienen citaciones

for col in bias_cols:
    for g in ["female", "male"]:
        sub = df_cit[df_cit["gender"] == g][[col, "log_citations"]].dropna()
        if len(sub) < 5:
            continue
        r, p = spearmanr(sub["log_citations"], sub[col])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        cit_results.append({"variable": col, "gender": g,
                             "n": len(sub), "rho": round(r, 3),
                             "p_value": round(p, 4), "sig": sig})
        if sig:
            print(f"  {sig:>3} {g:6s} | {col:<30} rho={r:+.3f}  p={p:.4f}")

cit_df = pd.DataFrame(cit_results)
cit_df.to_csv(CSV_DIR / "c07_citations_bias_corr.csv", index=False)

# Test de diferencia de correlaciones entre géneros (Fisher's z)
print("\n  Test: ¿difiere la correlacion por género? (Fisher z)")
from scipy.special import ndtr

def fisher_z_diff(r1, n1, r2, n2):
    z1 = np.arctanh(np.clip(r1, -0.999, 0.999))
    z2 = np.arctanh(np.clip(r2, -0.999, 0.999))
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z  = (z1 - z2) / se
    p  = 2 * (1 - ndtr(abs(z)))
    return z, p

for col in bias_cols:
    r_f = cit_df[(cit_df["variable"] == col) & (cit_df["gender"] == "female")]
    r_m = cit_df[(cit_df["variable"] == col) & (cit_df["gender"] == "male")]
    if len(r_f) and len(r_m):
        z, p = fisher_z_diff(r_f["rho"].values[0], r_f["n"].values[0],
                             r_m["rho"].values[0], r_m["n"].values[0])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            print(f"  {sig:>3} {col:<30} rho_F={r_f['rho'].values[0]:+.3f}  "
                  f"rho_M={r_m['rho'].values[0]:+.3f}  z={z:+.2f}  p={p:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PAÍS/REGIÓN × SESGO DE GÉNERO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  2. REGIÓN × SESGO (gender gap por región)")
print("="*60)

regions_with_both = []
for reg in df["region"].unique():
    sub = df[df["region"] == reg]
    if sub["gender"].nunique() == 2:
        regions_with_both.append(reg)

print(f"  Regiones con ambos géneros: {regions_with_both}")

country_rows = []
for reg in sorted(regions_with_both):
    sub = df[df["region"] == reg]
    nf  = (sub["gender"] == "female").sum()
    nm  = (sub["gender"] == "male").sum()
    for col in bias_cols:
        f_vals = sub[sub["gender"] == "female"][col].dropna()
        m_vals = sub[sub["gender"] == "male"][col].dropna()
        d = cohens_d(f_vals, m_vals)
        p = mw_p(f_vals, m_vals)
        country_rows.append({"region": reg, "variable": col,
                              "cohens_d_FvsM": round(d, 3), "p": round(p, 4) if not np.isnan(p) else np.nan,
                              "n_female": nf, "n_male": nm,
                              "mean_female": round(f_vals.mean(), 4) if len(f_vals) else np.nan,
                              "mean_male":   round(m_vals.mean(), 4) if len(m_vals) else np.nan})

country_df = pd.DataFrame(country_rows)
country_df.to_csv(CSV_DIR / "c07_country_gender_gap.csv", index=False)

pivot_d = country_df.pivot_table(index="variable", columns="region",
                                  values="cohens_d_FvsM", aggfunc="first")
print(f"\n  Cohen's d (F vs M) por región y variable (+ = F>M):")
print(pivot_d.round(3).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 3. ESPECIALIDAD × SESGO DE GÉNERO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  3. ESPECIALIDAD × SESGO (ICU vs Ane/Both)")
print("="*60)

spec_col = "Specialty (ICU-Ane-Both)"
df["spec_broad"] = df[spec_col].fillna("unknown").str.upper()
df["spec_broad"] = df["spec_broad"].replace({"OTHER": "Other", "UNK": "unknown"})

spec_rows = []
for spec in df["spec_broad"].unique():
    if spec in ("unknown", "OTHER"): continue
    sub = df[df["spec_broad"] == spec]
    nf = (sub["gender"] == "female").sum()
    nm = (sub["gender"] == "male").sum()
    for col in bias_cols:
        f_vals = sub[sub["gender"] == "female"][col].dropna()
        m_vals = sub[sub["gender"] == "male"][col].dropna()
        d = cohens_d(f_vals, m_vals)
        p = mw_p(f_vals, m_vals)
        spec_rows.append({"specialty": spec, "variable": col,
                           "cohens_d_FvsM": round(d, 3),
                           "p": round(p, 4) if not np.isnan(p) else np.nan,
                           "n_female": nf, "n_male": nm})

spec_df = pd.DataFrame(spec_rows)
spec_df.to_csv(CSV_DIR / "c07_specialty_gender_gap.csv", index=False)

print(f"\n  Cohen's d (F vs M) por especialidad:")
pivot_s = spec_df.pivot_table(index="variable", columns="specialty",
                               values="cohens_d_FvsM", aggfunc="first")
print(pivot_s.round(3).to_string())

# También: ¿la especialidad per se predice comportamiento (sin cruzar con género)?
print(f"\n  Kruskal-Wallis: especialidad ~ variable de sesgo (sin distincion genero)")
for col in bias_cols:
    groups = [df[df["spec_broad"] == s][col].dropna()
              for s in df["spec_broad"].unique() if s not in ("unknown","OTHER")
              and len(df[df["spec_broad"] == s][col].dropna()) >= 3]
    if len(groups) >= 2:
        try:
            stat, p = kruskal(*groups)
            if p < 0.05:
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                print(f"  {sig:>3} {col:<30} H={stat:.2f}  p={p:.4f}")
        except: pass

# ══════════════════════════════════════════════════════════════════════════════
# 4. ROL × SESGO DE GÉNERO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4. ROL × SESGO (Moderator vs Speaker vs public)")
print("="*60)

role_rows = []
for role in ["Moderator", "Speaker", "public"]:
    sub = df[df["Role"] == role]
    nf = (sub["gender"] == "female").sum()
    nm = (sub["gender"] == "male").sum()
    for col in bias_cols:
        f_vals = sub[sub["gender"] == "female"][col].dropna()
        m_vals = sub[sub["gender"] == "male"][col].dropna()
        d = cohens_d(f_vals, m_vals)
        p = mw_p(f_vals, m_vals)
        role_rows.append({"role": role, "variable": col,
                          "cohens_d_FvsM": round(d, 3),
                          "p": round(p, 4) if not np.isnan(p) else np.nan,
                          "n_female": nf, "n_male": nm,
                          "mean_female": round(f_vals.mean(), 4) if len(f_vals) else np.nan,
                          "mean_male":   round(m_vals.mean(), 4) if len(m_vals) else np.nan})

role_df = pd.DataFrame(role_rows)
role_df.to_csv(CSV_DIR / "c07_role_gender_gap.csv", index=False)

print(f"\n  Cohen's d (F vs M) por rol:")
pivot_r = role_df.pivot_table(index="variable", columns="role",
                               values="cohens_d_FvsM", aggfunc="first")
print(pivot_r.round(3).to_string())

sig_role = role_df[role_df["p"].notna() & (role_df["p"] < 0.05)]
if len(sig_role):
    print(f"\n  Significativo (p<0.05) por rol:")
    for _, r in sig_role.iterrows():
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*"
        print(f"  {sig:>3} {r['role']:<12} | {r['variable']:<30} "
              f"d={r['cohens_d_FvsM']:+.3f}  p={r['p']:.4f}  "
              f"n_F={r['n_female']}  n_M={r['n_male']}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

# --- 5a. Citaciones vs mansplaining + hedge (scatter por género) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Citaciones (log) vs variables de sesgo", fontsize=13, fontweight="bold")

GCOLORS = {"female": "#e74c3c", "male": "#3498db"}
for ax, (col, label) in zip(axes, [("pct_is_mansplaining", "Mansplaining"),
                                     ("pct_has_hedge", "% Hedge")]):
    for g, gc in GCOLORS.items():
        sub = df_cit[df_cit["gender"] == g][[col, "log_citations"]].dropna()
        ax.scatter(sub["log_citations"], sub[col], c=gc, label=g.capitalize(),
                   alpha=0.7, s=60, edgecolors="white", lw=0.5)
        if len(sub) >= 5:
            m, b = np.polyfit(sub["log_citations"], sub[col], 1)
            xs = np.linspace(sub["log_citations"].min(), sub["log_citations"].max(), 100)
            ax.plot(xs, m*xs+b, color=gc, linewidth=2, linestyle="--")
    r_df = cit_df[cit_df["variable"] == col]
    note = "  ".join([f"rho_{row['gender'][0].upper()}={row['rho']:+.2f}{'*' if row['sig'] else ''}"
                      for _, row in r_df.iterrows()])
    ax.set_xlabel("log(citaciones + 1)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f"{label}\n{note}", fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "c07_01_citations_scatter.png", dpi=200, bbox_inches="tight")
plt.close()

# --- 5b. Cohen's d por rol × variable (heatmap) ---
cols_plot  = [c for c in bias_cols if c in pivot_r.index]
roles_plot = pivot_r.columns.tolist()
val_matrix = pivot_r.loc[cols_plot, roles_plot].values.astype(float)

fig, ax = plt.subplots(figsize=(max(8, len(roles_plot)*2.5), max(5, len(cols_plot)*0.6)))
vmax = max(0.5, float(np.nanmax(np.abs(val_matrix))))
im = ax.imshow(val_matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(roles_plot))); ax.set_xticklabels(roles_plot, fontsize=12, fontweight="bold")
ax.set_yticks(range(len(cols_plot)));  ax.set_yticklabels([bias_names.get(c, c) for c in cols_plot], fontsize=10)
ax.set_title("Cohen's d (F vs M) por Rol\n(+ = Mujeres > Hombres)", fontsize=12, fontweight="bold")

p_matrix = role_df.pivot_table(index="variable", columns="role", values="p", aggfunc="first")
for i, col in enumerate(cols_plot):
    for j, role in enumerate(roles_plot):
        d_val = val_matrix[i, j]
        if np.isnan(d_val): continue
        p_val = p_matrix.loc[col, role] if col in p_matrix.index and role in p_matrix.columns else 1
        sig   = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        color = "white" if abs(d_val) > vmax*0.6 else "black"
        ax.text(j, i, f"{d_val:+.2f}{sig}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)
plt.colorbar(im, ax=ax, label="Cohen's d")
plt.tight_layout()
fig.savefig(FIG_DIR / "c07_02_role_gender_gap.png", dpi=200, bbox_inches="tight")
plt.close()

# --- 5c. Cohen's d por especialidad × variable (heatmap) ---
if len(pivot_s.columns) >= 2:
    specs_plot = pivot_s.columns.tolist()
    val_s = pivot_s.loc[[c for c in bias_cols if c in pivot_s.index], specs_plot].values.astype(float)
    rows_s = [c for c in bias_cols if c in pivot_s.index]

    fig, ax = plt.subplots(figsize=(max(8, len(specs_plot)*2.5), max(5, len(rows_s)*0.6)))
    vmax_s = max(0.5, float(np.nanmax(np.abs(val_s))))
    im = ax.imshow(val_s, cmap="RdBu_r", aspect="auto", vmin=-vmax_s, vmax=vmax_s)
    ax.set_xticks(range(len(specs_plot))); ax.set_xticklabels(specs_plot, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(rows_s)));     ax.set_yticklabels([bias_names.get(c, c) for c in rows_s], fontsize=10)
    ax.set_title("Cohen's d (F vs M) por Especialidad\n(+ = Mujeres > Hombres)", fontsize=12, fontweight="bold")

    p_s = spec_df.pivot_table(index="variable", columns="specialty", values="p", aggfunc="first")
    for i, col in enumerate(rows_s):
        for j, sp in enumerate(specs_plot):
            d_val = val_s[i, j]
            if np.isnan(d_val): continue
            p_val = p_s.loc[col, sp] if col in p_s.index and sp in p_s.columns else 1
            sig   = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            color = "white" if abs(d_val) > vmax_s*0.6 else "black"
            ax.text(j, i, f"{d_val:+.2f}{sig}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, label="Cohen's d")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "c07_03_specialty_gender_gap.png", dpi=200, bbox_inches="tight")
    plt.close()

# --- 5d. Citaciones: rho por género + IC (barras) ---
if len(cit_df):
    fig, ax = plt.subplots(figsize=(14, 5))
    pivot_cit = cit_df.pivot_table(index="variable", columns="gender", values="rho", aggfunc="first")
    x = np.arange(len(pivot_cit))
    width = 0.35
    for i, (g, color) in enumerate([("female","#e74c3c"), ("male","#3498db")]):
        if g in pivot_cit.columns:
            rhos = pivot_cit[g].values
            bars = ax.bar(x + (i-0.5)*width, rhos, width, label=g.capitalize(),
                          color=color, alpha=0.8, edgecolor="white")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([bias_names.get(v, v) for v in pivot_cit.index],
                                          rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Spearman rho (citaciones ~ variable)", fontsize=11)
    ax.set_title("Correlación citaciones vs sesgo, por género", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "c07_04_citations_rho_bygender.png", dpi=200, bbox_inches="tight")
    plt.close()

print(f"\n  Graficos guardados en {FIG_DIR}/")
print(f"  CSVs guardados en {CSV_DIR}/")
