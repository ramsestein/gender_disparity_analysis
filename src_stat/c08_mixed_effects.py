"""
c08_mixed_effects.py
====================
Modelos mixtos (LME) con sesión como efecto aleatorio.
Corrige la pseudoreplicación: speakers de la misma sesión no son independientes.

Modelos por variable de sesgo:
  M0 (nulo):  Y ~ 1 + (1|session)  → estima ICC
  M1:         Y ~ gender + (1|session)
  M2:         Y ~ gender + is_moderator + is_speaker + is_ICU + log_citations + (1|session)

Reporta: ICC, coeficiente de género (M1 y M2), LRT M0 vs M1, R² marginal/condicional.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.formula.api as smf

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

df = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
df["gender_bin"] = (df["gender"] == "male").astype(int)   # 1=male, 0=female
# Asegurar que tenemos las columnas necesarias
for col in ["is_moderator","is_speaker","is_public","is_ICU","log_citations"]:
    if col not in df.columns:
        df[col] = 0

bias_cols  = [v for v, _ in BIAS_VARS if v in df.columns]
bias_names = {v: n for v, n in BIAS_VARS if v in df.columns}

print("="*60)
print("  C08 — MODELOS MIXTOS CON SESIÓN COMO EFECTO ALEATORIO")
print("="*60)
print(f"  N speakers: {len(df)}  |  N sesiones: {df['session'].nunique()}")

# ── ICC helper ────────────────────────────────────────────────────────────────
def icc_from_model(md):
    """ICC = var_session / (var_session + var_residual)"""
    try:
        var_re  = float(md.cov_re.iloc[0,0])
        var_res = float(md.scale)
        return var_re / (var_re + var_res)
    except: return np.nan

# ── R² marginal/condicional (Nakagawa & Schielzeth) ──────────────────────────
def r2_mixed(md, df_used, outcome):
    try:
        fe_pred  = md.fittedvalues - md.resid   # fixed + random
        y        = df_used[outcome]
        var_fix  = np.var(md.predict(exog=md.model.exog))
        var_res  = float(md.scale)
        var_re   = max(0.0, float(md.cov_re.iloc[0,0]))
        denom    = var_fix + var_re + var_res
        if denom < 1e-12: return np.nan, np.nan
        r2_marg  = var_fix / denom
        r2_cond  = (var_fix + var_re) / denom
        return round(r2_marg, 3), round(r2_cond, 3)
    except: return np.nan, np.nan

results = []
print(f"\n  {'Variable':<28} {'ICC':>5} {'b_gender(M1)':>13} {'p_M1':>8} {'b_gender(M2)':>13} {'p_M2':>8} {'R2marg':>7}")
print(f"  {'-'*90}")

for col in bias_cols:
    sub = df[["session", col, "gender_bin",
              "is_moderator","is_speaker","is_ICU","log_citations"]].dropna()
    if len(sub) < 20 or sub["session"].nunique() < 5:
        continue

    try:
        # M0: null model
        m0 = smf.mixedlm(f"{col} ~ 1", sub, groups=sub["session"]).fit(reml=True, method="lbfgs")
        icc = icc_from_model(m0)

        # M1: gender only
        m1 = smf.mixedlm(f"{col} ~ gender_bin", sub, groups=sub["session"]).fit(reml=False, method="lbfgs")
        b_g1 = m1.params.get("gender_bin", np.nan)
        p_g1 = m1.pvalues.get("gender_bin", np.nan)

        # M2: gender + covariates
        m2 = smf.mixedlm(
            f"{col} ~ gender_bin + is_moderator + is_speaker + is_ICU + log_citations",
            sub, groups=sub["session"]
        ).fit(reml=False, method="lbfgs")
        b_g2 = m2.params.get("gender_bin", np.nan)
        p_g2 = m2.pvalues.get("gender_bin", np.nan)

        r2m, r2c = r2_mixed(m1, sub, col)

        sig1 = "***" if p_g1 < 0.001 else "**" if p_g1 < 0.01 else "*" if p_g1 < 0.05 else ""
        sig2 = "***" if p_g2 < 0.001 else "**" if p_g2 < 0.01 else "*" if p_g2 < 0.05 else ""

        results.append({"variable": col, "icc": round(icc, 3),
                        "beta_gender_M1": round(b_g1, 4), "p_M1": round(p_g1, 4), "sig_M1": sig1,
                        "beta_gender_M2": round(b_g2, 4), "p_M2": round(p_g2, 4), "sig_M2": sig2,
                        "r2_marginal": r2m, "r2_conditional": r2c})

        print(f"  {col:<28} {icc:>5.3f} "
              f"{b_g1:>+10.4f}{sig1:>3}  {p_g1:>8.4f} "
              f"{b_g2:>+10.4f}{sig2:>3}  {p_g2:>8.4f} "
              f"{r2m:>7.3f}")
    except Exception as e:
        print(f"  {col:<28} ERROR: {e}")

res_df = pd.DataFrame(results)
res_df.to_csv(CSV_DIR / "c08_mixed_effects_results.csv", index=False)

# ── Gráfico: ICC y β género por variable ─────────────────────────────────────
if len(res_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Modelos Mixtos — Efecto de Género con Sesión como Efecto Aleatorio",
                 fontsize=12, fontweight="bold")

    # ICC
    ax = axes[0]
    colors = ["#e74c3c" if v > 0.1 else "#3498db" for v in res_df["icc"]]
    ax.barh(res_df["variable"].map(bias_names), res_df["icc"], color=colors, alpha=0.8)
    ax.axvline(0.1, color="red", lw=1.5, linestyle="--", label="ICC=0.10")
    ax.set_xlabel("ICC (var. entre sesiones)", fontsize=10)
    ax.set_title("ICC por variable\n(rojo = dependencia sesión significativa)", fontsize=10)
    ax.legend()

    # β género (M1 y M2)
    ax = axes[1]
    x  = np.arange(len(res_df))
    ax.bar(x - 0.2, res_df["beta_gender_M1"], 0.35, label="M1 (solo género)",
           color="#9b59b6", alpha=0.8)
    ax.bar(x + 0.2, res_df["beta_gender_M2"], 0.35, label="M2 (+ covariables)",
           color="#1abc9c", alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(res_df["variable"].map(bias_names), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("β género (hombre vs mujer)", fontsize=10)
    ax.set_title("Coeficiente género en M1 y M2\n(+ = hombres mayor)", fontsize=10)
    ax.legend()

    # Marcar significativos
    for i, row in res_df.iterrows():
        if row["sig_M1"]:
            ax.text(i - 0.2, row["beta_gender_M1"] + 0.001 * np.sign(row["beta_gender_M1"]),
                    row["sig_M1"], ha="center", va="bottom", fontsize=9, color="#9b59b6")
        if row["sig_M2"]:
            ax.text(i + 0.2, row["beta_gender_M2"] + 0.001 * np.sign(row["beta_gender_M2"]),
                    row["sig_M2"], ha="center", va="bottom", fontsize=9, color="#1abc9c")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "c08_mixed_effects.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafico: {FIG_DIR}/c08_mixed_effects.png")

print(f"  CSV:     {CSV_DIR}/c08_mixed_effects_results.csv")

# Resumen
print(f"\n  RESUMEN:")
sig_m1 = res_df[res_df["sig_M1"] != ""]
sig_m2 = res_df[res_df["sig_M2"] != ""]
print(f"  Significativos en M1 (solo genero):        {len(sig_m1)}/{len(res_df)}")
print(f"  Significativos en M2 (+ covariables):      {len(sig_m2)}/{len(res_df)}")
lost = set(sig_m1["variable"]) - set(sig_m2["variable"])
new  = set(sig_m2["variable"]) - set(sig_m1["variable"])
print(f"  Pierden sig al controlar covariables: {lost}")
print(f"  Ganan sig al controlar covariables:   {new}")
print(f"  ICC medio: {res_df['icc'].mean():.3f}  (max={res_df['icc'].max():.3f})")
