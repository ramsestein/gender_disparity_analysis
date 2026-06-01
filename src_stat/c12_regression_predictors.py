"""
c12_regression_predictors.py
=============================
Regresion OLS multiple con betas estandarizados para identificar
que variables predicen mejor:
  1. Mansplaining (pct_is_mansplaining)
  2. Ser interrumpida (pct_interrupted_by_next, solo mujeres)

Todos los predictores se z-normalizan antes del ajuste para que los
coeficientes sean directamente comparables (beta estandarizado).
La magnitud absoluta del beta indica el peso/importancia de cada variable.

Metodos:
  A) OLS  (statsmodels) → coefs + p-valores + R2
  B) Ridge (sklearn)    → coefs regularizados, robusto a multicolinealidad
  C) Lasso (sklearn)    → seleccion automatica de variables (coefs=0 si irrelevantes)

Graficos:
  - Bar chart horizontal de betas OLS ordenados (+ = aumenta outcome, - = reduce)
  - Comparacion OLS vs Ridge vs Lasso
  - VIF para diagnotico de multicolinealidad

Salida:
  csv/c12_ols_mansplaining.csv
  csv/c12_ols_interrupted_women.csv
  graficos/c12_*.png
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"

# ── Cargar datos ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")

# ── Preparar variables ────────────────────────────────────────────────────────
df["gender_bin"]    = (df["gender"] == "male").astype(float)
df["is_moderator"]  = (df["Role"] == "Moderator").astype(float)  if "Role" in df.columns else 0.0
df["is_speaker"]    = (df["Role"] == "Speaker").astype(float)    if "Role" in df.columns else 0.0
df["is_public"]     = (df["Role"] == "public").astype(float)     if "Role" in df.columns else 0.0

spec = df.get("Specialty (ICU-Ane-Both)", pd.Series("", index=df.index)).fillna("").str.upper()
df["is_ICU"]  = (spec == "ICU").astype(float)
df["is_BOTH"] = (spec == "BOTH").astype(float)
df["is_ANE"]  = (spec == "ANE").astype(float)

df["log_cit"]    = df.get("log_citations", pd.Series(0, index=df.index)).fillna(0)
df["cluster_bin"] = df["cluster"].astype(float) if "cluster" in df.columns else 0.0

# Group dummies (ya vienen en el CSV del paso c03)
grp_cols = [c for c in df.columns if c.startswith("grp_")]

# Todos los predictores posibles
CONTINUOUS = [
    "n_interventions", "mean_duration", "std_duration", "total_duration",
    "mean_wpm", "mean_lexical_diversity", "mean_latency_s", "mean_overlap_duration",
    "mean_echoing_score", "mean_assertiveness_score", "mean_conflict_score",
    "pct_has_hedge", "pct_has_agreement", "pct_has_disagreement",
    "pct_has_courtesy", "pct_has_apology", "pct_is_question",
    "pct_has_vulnerability", "pct_is_backchannel", "pct_interrupts_previous",
    "min_turn_number", "is_top3_speaker", "log_cit",
]
BINARY = ["gender_bin", "is_moderator", "is_speaker", "is_public",
          "is_ICU", "is_BOTH", "is_ANE", "cluster_bin"] + grp_cols

# Deduplicar por si acaso
ALL_PREDICTORS = list(dict.fromkeys([c for c in CONTINUOUS + BINARY if c in df.columns]))

PRETTY = {
    "gender_bin": "GENERO (hombre)",
    "is_moderator": "Rol: Moderador",
    "is_speaker": "Rol: Speaker",
    "is_public": "Rol: Publico",
    "is_ICU": "Especialidad: ICU",
    "is_BOTH": "Especialidad: Both",
    "is_ANE": "Especialidad: ANE",
    "cluster_bin": "Cluster (1=audiencia)",
    "log_cit": "Log(citaciones)",
    "n_interventions": "N intervenciones",
    "mean_duration": "Duracion media (s)",
    "std_duration": "SD duracion",
    "total_duration": "Duracion total (s)",
    "mean_wpm": "Palabras/minuto",
    "mean_lexical_diversity": "Diversidad lexica",
    "mean_latency_s": "Latencia media (s)",
    "mean_overlap_duration": "Solapamiento",
    "mean_echoing_score": "Echoing",
    "mean_assertiveness_score": "Asertividad",
    "mean_conflict_score": "Conflicto",
    "pct_has_hedge": "% Hedge",
    "pct_has_agreement": "% Acuerdo",
    "pct_has_disagreement": "% Desacuerdo",
    "pct_has_courtesy": "% Cortesia",
    "pct_has_apology": "% Disculpa",
    "pct_is_question": "% Preguntas",
    "pct_has_vulnerability": "% Vulnerabilidad",
    "pct_is_backchannel": "% Backchannel",
    "pct_interrupts_previous": "% Interrumpe",
    "min_turn_number": "Turno inicio",
    "is_top3_speaker": "Top-3 hablante",
}

def label(c):
    for k, v in PRETTY.items():
        if c == k: return v
    return c.replace("grp_", "Grupo: ").replace("_", " ")

# ── Helper: VIF ───────────────────────────────────────────────────────────────
def compute_vif(X_df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    vif["variable"] = X_df.columns
    vif["VIF"] = [variance_inflation_factor(X_df.values, i)
                  for i in range(X_df.shape[1])]
    return vif.sort_values("VIF", ascending=False)

# ── Helper: ajustar y reportar regresion ─────────────────────────────────────
def fit_and_report(df_model, outcome, predictors, label_str):
    sub = df_model[[outcome] + predictors].dropna()
    y   = sub[outcome].values
    X   = sub[predictors].values

    # Estandarizar
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)
    y_std  = (y - y.mean()) / (y.std() + 1e-10)

    print(f"\n{'='*65}")
    print(f"  OUTCOME: {label_str}  (n={len(sub)})")
    print(f"{'='*65}")

    # A) OLS estandarizado
    X_sm   = sm.add_constant(X_std)
    model  = sm.OLS(y_std, X_sm).fit()
    betas  = model.params[1:]   # excluir constante
    pvals  = model.pvalues[1:]
    r2     = model.rsquared
    r2_adj = model.rsquared_adj

    print(f"\n  OLS: R2={r2:.3f}  R2_adj={r2_adj:.3f}")
    print(f"\n  {'Variable':<35} {'Beta':>8} {'p':>8} {'Sig':>4}")
    ols_rows = []
    for i, pred in enumerate(predictors):
        sig = "***" if pvals[i]<0.001 else "**" if pvals[i]<0.01 else "*" if pvals[i]<0.05 else ""
        if sig or abs(betas[i]) > 0.05:
            print(f"  {label(pred):<35} {betas[i]:>+8.4f} {pvals[i]:>8.4f} {sig:>4}")
        ols_rows.append({"variable": pred, "label": label(pred),
                         "beta_OLS": round(float(betas[i]),4),
                         "p_OLS":    round(float(pvals[i]),4),
                         "sig_OLS":  sig})

    # B) Ridge
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    ridge.fit(X_std, y_std)
    beta_ridge = ridge.coef_
    r2_ridge   = ridge.score(X_std, y_std)
    print(f"\n  Ridge: alpha={ridge.alpha_:.4f}  R2={r2_ridge:.3f}")

    # C) Lasso
    lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
    lasso.fit(X_std, y_std)
    beta_lasso = lasso.coef_
    r2_lasso   = lasso.score(X_std, y_std)
    n_nonzero  = (beta_lasso != 0).sum()
    print(f"  Lasso: alpha={lasso.alpha_:.4f}  R2={r2_lasso:.3f}  vars_seleccionadas={n_nonzero}/{len(predictors)}")

    # Combinar resultados
    res = pd.DataFrame(ols_rows)
    res["beta_Ridge"] = [round(float(b), 4) for b in beta_ridge]
    res["beta_Lasso"] = [round(float(b), 4) for b in beta_lasso]
    res["R2_OLS"]   = round(r2, 3)
    res["R2_Ridge"] = round(r2_ridge, 3)
    res["R2_Lasso"] = round(r2_lasso, 3)
    res = res.sort_values("beta_OLS", key=abs, ascending=False)

    # Top influencers
    top_pos = res[res["beta_OLS"] > 0].head(5)
    top_neg = res[res["beta_OLS"] < 0].head(5)
    print(f"\n  TOP 5 que AUMENTAN {label_str}:")
    for _, r in top_pos.iterrows():
        print(f"    {r['label']:<35} beta={r['beta_OLS']:+.4f}  (Ridge={r['beta_Ridge']:+.4f}  Lasso={r['beta_Lasso']:+.4f})")
    print(f"\n  TOP 5 que REDUCEN {label_str}:")
    for _, r in top_neg.iterrows():
        print(f"    {r['label']:<35} beta={r['beta_OLS']:+.4f}  (Ridge={r['beta_Ridge']:+.4f}  Lasso={r['beta_Lasso']:+.4f})")

    return res, r2, r2_ridge, r2_lasso

# ══════════════════════════════════════════════════════════════════════════════
# MODELO 1: MANSPLAINING  (toda la muestra)
# ══════════════════════════════════════════════════════════════════════════════
predictors_1 = [p for p in ALL_PREDICTORS if p != "pct_is_mansplaining"]
res1, r2_1, r2r_1, r2l_1 = fit_and_report(
    df, "pct_is_mansplaining", predictors_1, "Mansplaining")
res1.to_csv(CSV_DIR / "c12_ols_mansplaining.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# MODELO 2: SER INTERRUMPIDA (solo mujeres)
# ══════════════════════════════════════════════════════════════════════════════
df_women = df[df["gender"] == "female"].copy()
predictors_2 = [p for p in ALL_PREDICTORS
                if p not in ("pct_interrupted_by_next", "gender_bin")
                and p in df_women.columns]
res2, r2_2, r2r_2, r2l_2 = fit_and_report(
    df_women, "pct_interrupted_by_next", predictors_2, "Ser interrumpida (mujeres)")
res2.to_csv(CSV_DIR / "c12_ols_interrupted_women.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# GRAFICOS
# ══════════════════════════════════════════════════════════════════════════════

def plot_betas(res_df, title, fname, top_n=20, r2_vals=None):
    """Bar chart horizontal de betas OLS, Ridge y Lasso ordenados por |beta_OLS|."""
    top = res_df.head(top_n).copy()
    top = top.sort_values("beta_OLS")

    fig, ax = plt.subplots(figsize=(10, max(6, len(top)*0.38)))
    y_pos = np.arange(len(top))
    width = 0.28

    bars_ols   = ax.barh(y_pos + width,   top["beta_OLS"],   width, label="OLS",   color="#3498db", alpha=0.85)
    bars_ridge = ax.barh(y_pos,           top["beta_Ridge"], width, label="Ridge",  color="#e67e22", alpha=0.75)
    bars_lasso = ax.barh(y_pos - width,   top["beta_Lasso"], width, label="Lasso",  color="#27ae60", alpha=0.75)

    ax.axvline(0, color="black", lw=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["label"], fontsize=9)
    ax.set_xlabel("Beta estandarizado", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    # Marcar significativos OLS
    for i, (bar, row) in enumerate(zip(bars_ols, top.itertuples())):
        if row.sig_OLS:
            ax.text(bar.get_width() + 0.002 * np.sign(bar.get_width()),
                    bar.get_y() + bar.get_height()/2,
                    row.sig_OLS, va="center", ha="left" if bar.get_width()>0 else "right",
                    fontsize=8, fontweight="bold", color="#2c3e50")

    if r2_vals:
        ax.set_xlabel(
            f"Beta estandarizado\n(R² OLS={r2_vals[0]:.3f}  Ridge={r2_vals[1]:.3f}  Lasso={r2_vals[2]:.3f})",
            fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {fname}")

plot_betas(res1, "Predictores del Mansplaining (beta estandarizado)\n+ = aumenta  |  - = reduce",
           "c12_01_mansplaining_betas.png", top_n=20, r2_vals=(r2_1, r2r_1, r2l_1))

plot_betas(res2, "Predictores de ser interrumpida (mujeres, beta estandarizado)\n+ = aumenta prob. interrupcion  |  - = reduce",
           "c12_02_interrupted_women_betas.png", top_n=20, r2_vals=(r2_2, r2r_2, r2l_2))

# Grafico comparativo: top 10 de cada outcome superpuestos
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Regresion multivariable: predictores del sesgo de genero\n(betas OLS estandarizados, top 15 por magnitud)",
             fontsize=12, fontweight="bold")

for ax, res, title, color in [
    (axes[0], res1, "MANSPLAINING", "#e74c3c"),
    (axes[1], res2, "SER INTERRUMPIDA (mujeres)", "#9b59b6"),
]:
    top = res.head(15).sort_values("beta_OLS")
    colors = [color if b > 0 else "#7f8c8d" for b in top["beta_OLS"]]
    bars = ax.barh(top["label"], top["beta_OLS"], color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Beta estandarizado", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for bar, row in zip(bars, top.itertuples()):
        if row.sig_OLS:
            ax.text(bar.get_width() + 0.002 * np.sign(bar.get_width()),
                    bar.get_y() + bar.get_height()/2,
                    row.sig_OLS, va="center", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "c12_03_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Guardado: c12_03_comparison.png")

print(f"\n  CSVs: {CSV_DIR}/c12_*.csv")
