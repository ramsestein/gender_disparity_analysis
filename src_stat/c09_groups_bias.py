"""
c09_groups_bias.py
==================
Analiza la distribución de género y el sesgo de comunicación
por área temática (9 grupos ESICM).

Estrategia:
  - El grupo pertenece a la sesión (no al speaker individual).
  - Se mapea sesión → grupo y se asigna a TODOS los speakers de esa sesión.
  - Esto extiende el análisis a los 558 no identificados también.

Preguntas:
  1. ¿Cómo se distribuye el género por área temática?
  2. ¿Qué áreas muestran mayor/menor sesgo de género?
  3. ¿Algunas áreas son más equitativas que otras?
  4. ¿El grupo per se predice el estilo comunicativo?

Salida:
  csv/c09_group_gender_distribution.csv
  csv/c09_group_bias_gap.csv
  graficos/c09_*.png
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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
    ("mean_duration",             "Dur. Media"),
]

# ── Cargar y mapear grupos ────────────────────────────────────────────────────
df_all = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")

# Mapa sesión → grupo (a partir de los speakers identificados)
sess_group = (df_all[df_all["Group"].notna()]
              [["session", "Group"]].drop_duplicates("session")
              .set_index("session")["Group"])

df_all["group"] = df_all["session"].map(sess_group)
df = df_all[df_all["group"].notna()].copy()

n_total    = len(df)
n_sessions = df["session"].nunique()
print(f"Speakers en sesiones con grupo: {n_total}  ({n_sessions} sesiones)")
print(f"  Mujeres: {(df['gender']=='female').sum()}  ({(df['gender']=='female').mean()*100:.1f}%)")
print(f"  Hombres: {(df['gender']=='male').sum()}  ({(df['gender']=='male').mean()*100:.1f}%)")

bias_cols  = [v for v, _ in BIAS_VARS if v in df.columns]
bias_names = {v: n for v, n in BIAS_VARS if v in df.columns}

GROUPS_SHORT = {
    "Education / Professional development":      "Education",
    "Sepsis / Infection":                        "Sepsis",
    "Ethics / End-of-life / Communication":      "Ethics/EoL",
    "Respiratory":                               "Respiratory",
    "Neuro":                                     "Neuro",
    "Haemodynamics":                             "Haemodynamics",
    "Perioperative / Emergency / Major incidents":"Periop/Emerg",
    "AI / Digital health":                       "AI/Digital",
    "Renal / Metabolic / Nutrition":             "Renal/Metab",
}
df["group_short"] = df["group"].map(GROUPS_SHORT).fillna(df["group"])

groups = df["group_short"].unique()
groups_sorted = (df.groupby("group_short")["gender"]
                  .count().sort_values(ascending=False).index.tolist())

# ── Helpers ───────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    pool = np.sqrt(((na-1)*a.std()**2 + (nb-1)*b.std()**2) / (na+nb-2))
    return (a.mean() - b.mean()) / pool if pool > 0 else 0.0

def mw_p(a, b):
    if len(a) < 3 or len(b) < 3: return np.nan
    try: return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except: return np.nan

# ══════════════════════════════════════════════════════════════════════════════
# 1. DISTRIBUCIÓN DE GÉNERO POR GRUPO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  1. DISTRIBUCIÓN DE GÉNERO POR GRUPO")
print("="*60)

dist_rows = []
for grp in groups_sorted:
    sub = df[df["group_short"] == grp]
    nf  = (sub["gender"] == "female").sum()
    nm  = (sub["gender"] == "male").sum()
    n   = len(sub)
    pf  = nf / n * 100
    dist_rows.append({"group": grp, "n_total": n, "n_female": nf, "n_male": nm,
                      "pct_female": round(pf, 1),
                      "n_sessions": sub["session"].nunique()})
    print(f"  {grp:<20} n={n:3d}  F={nf:2d} ({pf:4.1f}%)  M={nm:2d}  ses={sub['session'].nunique()}")

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(CSV_DIR / "c09_group_gender_distribution.csv", index=False)

# Chi-cuadrado global: ¿difiere la proporción de género entre grupos?
ct = pd.crosstab(df["group_short"], df["gender"])
chi2, p_chi, dof, _ = chi2_contingency(ct)
print(f"\n  Chi2 global género×grupo: χ²={chi2:.2f}  df={dof}  p={p_chi:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SESGO DE GÉNERO POR GRUPO (Cohen's d y p-valor)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  2. SESGO DE GÉNERO POR GRUPO (Cohen's d, F vs M)")
print("="*60)

gap_rows = []
for grp in groups_sorted:
    sub = df[df["group_short"] == grp]
    nf  = (sub["gender"] == "female").sum()
    nm  = (sub["gender"] == "male").sum()
    for col in bias_cols:
        fv = sub[sub["gender"] == "female"][col].dropna()
        mv = sub[sub["gender"] == "male"][col].dropna()
        d  = cohens_d(fv, mv)
        p  = mw_p(fv, mv)
        sig = ("***" if not np.isnan(p) and p < 0.001 else
               "**"  if not np.isnan(p) and p < 0.01  else
               "*"   if not np.isnan(p) and p < 0.05  else "")
        gap_rows.append({"group": grp, "variable": col,
                         "cohens_d": round(d, 3) if not np.isnan(d) else np.nan,
                         "p_value": round(p, 4) if not np.isnan(p) else np.nan,
                         "sig": sig,
                         "n_female": nf, "n_male": nm,
                         "mean_female": round(fv.mean(), 4) if len(fv) else np.nan,
                         "mean_male":   round(mv.mean(), 4) if len(mv) else np.nan})

gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv(CSV_DIR / "c09_group_bias_gap.csv", index=False)

# Solo los significativos
sig_df = gap_df[gap_df["sig"] != ""]
print(f"\n  Efectos significativos (p<0.05):")
print(f"  {'Sig':>3} {'Grupo':<20} {'Variable':<30} {'d':>7}  {'p':>8}")
for _, r in sig_df.sort_values(["group","p_value"]).iterrows():
    print(f"  {r['sig']:>3} {r['group']:<20} {r['variable']:<30} {r['cohens_d']:>7.3f}  {r['p_value']:>8.4f}  "
          f"n_F={r['n_female']}  n_M={r['n_male']}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. KRUSKAL-WALLIS: ¿difiere el nivel GLOBAL (sin género) entre grupos?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  3. KRUSKAL-WALLIS: diferencias entre grupos (sin distinción género)")
print("="*60)

for col in bias_cols:
    grp_lists = [df[df["group_short"] == g][col].dropna()
                 for g in groups_sorted
                 if len(df[df["group_short"] == g][col].dropna()) >= 3]
    if len(grp_lists) >= 2:
        try:
            H, p = kruskal(*grp_lists)
            if p < 0.1:
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "."
                print(f"  {sig:>3} {col:<35} H={H:.2f}  p={p:.4f}")
        except: pass

# ══════════════════════════════════════════════════════════════════════════════
# 4. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

# --- 4a. Distribución de género por grupo (barras apiladas) ---
fig, ax = plt.subplots(figsize=(12, 5))
x     = np.arange(len(dist_df))
bar_f = ax.bar(x, dist_df["pct_female"], color="#e74c3c", alpha=0.85, label="Mujeres")
bar_m = ax.bar(x, 100 - dist_df["pct_female"], bottom=dist_df["pct_female"],
               color="#3498db", alpha=0.85, label="Hombres")
ax.axhline(50, color="black", lw=1.5, linestyle="--", label="50%")
ax.set_xticks(x)
ax.set_xticklabels(dist_df["group"], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("% de speakers", fontsize=11)
ax.set_title("Distribución de género por área temática", fontsize=13, fontweight="bold")
ax.set_ylim(0, 100)

for i, row in dist_df.iterrows():
    ax.text(i, row["pct_female"]/2, f"{row['pct_female']:.0f}%",
            ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.text(i, row["pct_female"] + (100-row["pct_female"])/2,
            f"n={row['n_total']}", ha="center", va="center", color="white", fontsize=8)

ax.legend(loc="upper right")
plt.tight_layout()
fig.savefig(FIG_DIR / "c09_01_gender_distribution_by_group.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"\n  Guardado: c09_01_gender_distribution_by_group.png")

# --- 4b. Heatmap Cohen's d por grupo × variable ---
pivot_d = gap_df.pivot_table(index="variable", columns="group",
                              values="cohens_d", aggfunc="first")
pivot_p = gap_df.pivot_table(index="variable", columns="group",
                              values="p_value", aggfunc="first")

cols_plot = [c for c in bias_cols if c in pivot_d.index]
grps_plot = [g for g in groups_sorted if g in pivot_d.columns]
val_mat   = pivot_d.loc[cols_plot, grps_plot].values.astype(float)

vmax = max(0.5, float(np.nanmax(np.abs(val_mat))))
fig, ax = plt.subplots(figsize=(max(12, len(grps_plot)*1.5), max(6, len(cols_plot)*0.7)))
im = ax.imshow(val_mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(grps_plot)))
ax.set_xticklabels(grps_plot, rotation=35, ha="right", fontsize=9, fontweight="bold")
ax.set_yticks(range(len(cols_plot)))
ax.set_yticklabels([bias_names.get(c, c) for c in cols_plot], fontsize=10)
ax.set_title("Cohen's d (F vs M) por Área Temática\n(+ = Mujeres > Hombres  |  − = Hombres > Mujeres)",
             fontsize=12, fontweight="bold")

for i, col in enumerate(cols_plot):
    for j, grp in enumerate(grps_plot):
        d_val = val_mat[i, j]
        if np.isnan(d_val): continue
        p_val = pivot_p.loc[col, grp] if (col in pivot_p.index and grp in pivot_p.columns) else 1.0
        if pd.isna(p_val): p_val = 1.0
        sig   = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        color = "white" if abs(d_val) > vmax * 0.6 else "black"
        ax.text(j, i, f"{d_val:+.2f}{sig}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=color)

plt.colorbar(im, ax=ax, label="Cohen's d")
plt.tight_layout()
fig.savefig(FIG_DIR / "c09_02_group_bias_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Guardado: c09_02_group_bias_heatmap.png")

# --- 4c. Mansplaining y Hedge por grupo (barras agrupadas) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Mansplaining y Hedge por área temática y género", fontsize=13, fontweight="bold")

GCOLORS = {"female": "#e74c3c", "male": "#3498db"}
for ax, (col, label) in zip(axes, [("pct_is_mansplaining", "% Mansplaining"),
                                     ("pct_has_hedge",       "% Hedge")]):
    means_f, means_m, labels = [], [], []
    for grp in groups_sorted:
        sub = df[df["group_short"] == grp]
        fv  = sub[sub["gender"] == "female"][col].dropna()
        mv  = sub[sub["gender"] == "male"][col].dropna()
        means_f.append(fv.mean() if len(fv) else np.nan)
        means_m.append(mv.mean() if len(mv) else np.nan)
        labels.append(grp)

    x     = np.arange(len(labels))
    width = 0.38
    b_f   = ax.bar(x - width/2, means_f, width, color="#e74c3c", alpha=0.85,
                   label="Mujeres", edgecolor="white")
    b_m   = ax.bar(x + width/2, means_m, width, color="#3498db", alpha=0.85,
                   label="Hombres", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # Marcar diferencias significativas
    for i, grp in enumerate(groups_sorted):
        sub = df[df["group_short"] == grp]
        fv  = sub[sub["gender"] == "female"][col].dropna()
        mv  = sub[sub["gender"] == "male"][col].dropna()
        p   = mw_p(fv, mv)
        if not np.isnan(p) and p < 0.05:
            ymax = max(means_f[i] or 0, means_m[i] or 0)
            sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            ax.text(i, ymax * 1.08, sig, ha="center", va="bottom", fontsize=11,
                    color="#2c3e50", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "c09_03_mansplaining_hedge_by_group.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Guardado: c09_03_mansplaining_hedge_by_group.png")

# --- 4d. Ranking de equidad por grupo (|d| medio sobre todas las variables) ---
group_mean_abs_d = (gap_df.groupby("group")["cohens_d"]
                    .apply(lambda x: x.abs().mean())
                    .sort_values())

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#27ae60" if v < 0.3 else "#e67e22" if v < 0.6 else "#e74c3c"
          for v in group_mean_abs_d.values]
bars = ax.barh(group_mean_abs_d.index, group_mean_abs_d.values,
               color=colors, alpha=0.85, edgecolor="white")
ax.axvline(0.2, color="green",  lw=1.5, linestyle="--", label="Bajo (0.2)")
ax.axvline(0.5, color="orange", lw=1.5, linestyle="--", label="Medio (0.5)")
for bar, val in zip(bars, group_mean_abs_d.values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("|Cohen's d| medio (todas las variables de sesgo)", fontsize=11)
ax.set_title("Ranking de equidad de género por área temática\n(menor = más equitativo)",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(FIG_DIR / "c09_04_equity_ranking.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Guardado: c09_04_equity_ranking.png")

# ── Resumen final ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESUMEN — RANKING EQUIDAD (|d| medio, menor=mejor)")
print(f"{'='*60}")
for grp, val in group_mean_abs_d.items():
    level = "EQUITATIVO" if val < 0.3 else "MODERADO" if val < 0.6 else "ALTO SESGO"
    print(f"  {level:12s} | {grp:<22} |d|_medio={val:.3f}")

n_sig_total = (gap_df["sig"] != "").sum()
print(f"\n  Total efectos significativos (p<0.05): {n_sig_total} / {len(gap_df)}")
print(f"  CSVs: {CSV_DIR}/c09_*.csv")
print(f"  Figuras: {FIG_DIR}/c09_*.png")
