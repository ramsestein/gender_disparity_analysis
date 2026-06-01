"""
c06_gender_corrected_bias.py
============================
Corrige los análisis de sesgo de género para tener en cuenta el error
de clasificación de género, estimado en 5.7% global.

Método dual:
  A) Corrección de Cohen's d (Rogan-Gladen):
       kappa = Se + Sp - 1  =  0.974 + 0.920 - 1  = 0.894
       d_corrected = d_obs / kappa

  B) Corrección Monte Carlo de p-valores:
       1000 simulaciones: voltear aleatoriamente géneros según tasas de error
          P(F->M | classified F) = 1 - Se = 0.026
          P(M->F | classified M) = 1 - Sp = 0.080
       p_adj = percentil 75 de la distribución de p-valores simulados
       (conservador: el 25% de simulaciones dan un p peor que éste)

Los 94 speakers ya validados y los 5 errores ya corregidos NO se vuelven
a voltear. El volteo aplica solo a los 558 speakers NO validados.

Parámetros (de c05_gender_name_validation.py):
  Sensibilidad (female): Se = 0.974
  Especificidad (male):  Sp = 0.920
  kappa                    = 0.894

Entrada:  user_level_enriched_with_clusters.csv
          c04_gender_bias_by_cluster.csv
Salida:   c06_gender_bias_corrected.csv
          c06_correction_report.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"

# ── Parámetros de clasificación ───────────────────────────────────────────────
SE   = 0.974   # P(classified female | true female)
SP   = 0.920   # P(classified male   | true male)
KAPPA = SE + SP - 1   # = 0.894
ERR_F = 1 - SE         # P(female → misclassified as male)   = 0.026
ERR_M = 1 - SP         # P(male   → misclassified as female) = 0.080

N_SIM = 1000
P_PERCENTILE = 75   # conservador: el 75° percentil de la distribución de p

print("=" * 60)
print("  C06 — CORRECCIÓN POR ERROR DE CLASIFICACIÓN DE GÉNERO")
print("=" * 60)
print(f"  Se (female) = {SE:.3f}  |  Sp (male) = {SP:.3f}")
print(f"  kappa = Se + Sp - 1 = {KAPPA:.4f}")
print(f"  Error F->M = {ERR_F:.3f}  |  Error M->F = {ERR_M:.3f}")
print(f"  Simulaciones Monte Carlo: {N_SIM}  |  Percentil p: {P_PERCENTILE}")

# ── Cargar datos ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
orig = pd.read_csv(CSV_DIR / "c04_gender_bias_by_cluster.csv")

# Speakers validados (ya corregidos) — no se vuelven a voltear en simulación
validated = pd.read_csv(CSV_DIR / "c05_gender_name_validation.csv")
validated_keys = set(zip(validated["session"].str.removesuffix(".csv"),
                         validated["speaker"]))

# Marcar cuáles tienen género validado
df["session_stem"] = df["session"].str.removesuffix(".csv")
df["is_validated"] = df.apply(
    lambda r: (r["session_stem"], r["speaker"]) in validated_keys, axis=1
)
n_validated = df["is_validated"].sum()
n_unvalidated = (~df["is_validated"]).sum()
print(f"\n  Speakers validados (no se vuelven a voltear): {n_validated}")
print(f"  Speakers NO validados (sujetos a volteo):     {n_unvalidated}")

# Variables de sesgos a testar (las mismas que en c04)
bias_vars = [
    'n_interventions', 'mean_duration', 'total_duration', 'mean_wpm',
    'mean_lexical_diversity', 'mean_conflict_score', 'mean_assertiveness_score',
    'mean_echoing_score', 'pct_interrupts_previous', 'pct_interrupted_by_next',
    'pct_interruption_success', 'pct_has_hedge', 'pct_has_disagreement',
    'pct_has_agreement', 'pct_has_courtesy', 'pct_has_apology',
    'pct_is_question', 'pct_has_vulnerability', 'pct_is_mansplaining',
    'pct_is_backchannel', 'mean_overlap_duration', 'std_duration',
    'min_turn_number', 'is_top3_speaker'
]
bias_vars = [v for v in bias_vars if v in df.columns]
clusters  = sorted(df["cluster"].unique())

# ── Función auxiliar: voltear géneros en no-validados ─────────────────────────
rng = np.random.default_rng(42)

def flip_genders(df_in):
    """Devuelve copia con géneros aleatoriamente volteados en no-validados."""
    g = df_in["gender"].copy()
    mask = ~df_in["is_validated"]
    # voltear F→M
    f_mask = mask & (g == "female")
    flip_f = rng.random(f_mask.sum()) < ERR_F
    g[f_mask] = np.where(flip_f, "male", "female")
    # voltear M→F
    m_mask = mask & (g == "male")
    flip_m = rng.random(m_mask.sum()) < ERR_M
    g[m_mask] = np.where(flip_m, "female", "male")
    df_out = df_in.copy()
    df_out["gender"] = g
    return df_out

# ── Simulación Monte Carlo ────────────────────────────────────────────────────
print(f"\n  Ejecutando {N_SIM} simulaciones...", end="", flush=True)
sim_pvals = {c: {v: [] for v in bias_vars} for c in clusters}

for i in range(N_SIM):
    df_sim = flip_genders(df)
    for c in clusters:
        cdata   = df_sim[df_sim["cluster"] == c]
        females = cdata[cdata["gender"] == "female"]
        males   = cdata[cdata["gender"] == "male"]
        for var in bias_vars:
            fv = females[var].dropna()
            mv = males[var].dropna()
            if len(fv) < 3 or len(mv) < 3:
                sim_pvals[c][var].append(1.0)
                continue
            try:
                _, p = mannwhitneyu(fv, mv, alternative="two-sided")
            except Exception:
                p = 1.0
            sim_pvals[c][var].append(p)
    if (i + 1) % 200 == 0:
        print(f" {i+1}", end="", flush=True)
print(" listo.")

# ── Construir tabla de resultados corregidos ──────────────────────────────────
results = []
for _, row in orig.iterrows():
    c   = row["cluster"]
    var = row["variable"]
    d_obs = row["cohens_d"]
    p_obs = row["p_value"]

    # A) Corrección Rogan-Gladen del Cohen's d
    d_corr = d_obs / KAPPA

    # B) p-valor ajustado por Monte Carlo (percentil 75)
    sim_p_list = sim_pvals.get(c, {}).get(var, [])
    p_adj = float(np.percentile(sim_p_list, P_PERCENTILE)) if sim_p_list else p_obs

    sig_obs  = ("***" if p_obs  < 0.001 else "**" if p_obs  < 0.01
                else "*" if p_obs  < 0.05 else "")
    sig_adj  = ("***" if p_adj  < 0.001 else "**" if p_adj  < 0.01
                else "*" if p_adj  < 0.05 else "")

    results.append({
        "cluster":     c,
        "variable":    var,
        "cohens_d_obs":   round(d_obs, 4),
        "cohens_d_corr":  round(d_corr, 4),
        "kappa":          round(KAPPA, 4),
        "p_obs":          round(p_obs, 6),
        "p_adj_mc":       round(p_adj, 6),
        "sig_obs":        sig_obs,
        "sig_adj":        sig_adj,
        "robust":         sig_adj != "",    # sigue siendo sig tras corrección
        "n_female":       row["n_female"],
        "n_male":         row["n_male"],
    })

res_df = pd.DataFrame(results)
res_df.to_csv(CSV_DIR / "c06_gender_bias_corrected.csv", index=False)

# ── Imprimir resumen ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULTADOS CORREGIDOS POR ERROR DE CLASIFICACION")
print(f"{'='*60}")

for c in clusters:
    cdf = res_df[res_df["cluster"] == c].copy()
    print(f"\n  --- CLUSTER {c} ---")
    print(f"  {'Variable':<32} {'d_obs':>7} {'d_corr':>7}  {'p_obs':>8} {'p_adj':>8}  {'Sig_obs':>7} {'Sig_adj':>7} {'Robusta':>7}")
    for _, r in cdf.sort_values("p_adj_mc").iterrows():
        rob = "SI" if r["robust"] else "NO"
        print(f"  {r['variable']:<32} {r['cohens_d_obs']:>7.3f} {r['cohens_d_corr']:>7.3f}  "
              f"{r['p_obs']:>8.4f} {r['p_adj_mc']:>8.4f}  "
              f"{r['sig_obs']:>7} {r['sig_adj']:>7} {rob:>7}")

sig_obs_total = (res_df["sig_obs"] != "").sum()
sig_adj_total = (res_df["sig_adj"] != "").sum()
robust_total  = res_df["robust"].sum()
lost          = sig_obs_total - robust_total

print(f"\n  {'='*55}")
print(f"  RESUMEN GLOBAL (ambos clusters)")
print(f"  {'='*55}")
print(f"  kappa (factor de correccion):   {KAPPA:.4f}")
print(f"  Hallasgos originalmente sig:    {sig_obs_total}")
print(f"  Hallazgos sig tras correccion:  {sig_adj_total}")
print(f"  Hallazgos robustos:             {robust_total}")
print(f"  Hallazgos que pierden sig:      {lost}")

# Mostrar los que pierden significación
if lost > 0:
    print(f"\n  Hallazgos que dejan de ser sig tras correccion:")
    lost_df = res_df[(res_df["sig_obs"] != "") & (~res_df["robust"])]
    for _, r in lost_df.iterrows():
        print(f"    Cluster {r['cluster']} | {r['variable']:<32} "
              f"d_corr={r['cohens_d_corr']:+.3f} p_adj={r['p_adj_mc']:.4f}")

print(f"\n  Guardado en: {CSV_DIR}/c06_gender_bias_corrected.csv")
