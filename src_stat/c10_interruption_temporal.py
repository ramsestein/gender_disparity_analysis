"""
c10_interruption_temporal.py
=============================
1. RED DE INTERRUPCIONES DIRIGIDA: quién interrumpe a quién por género
   - Matriz de transición F->F, F->M, M->F, M->M
   - Chi-cuadrado: ¿las interrupciones son aleatorias respecto al género?
   - Odds ratio de interrumpir al género opuesto

2. ANÁLISIS TEMPORAL INTRA-SESIÓN (phase_quartile)
   - ¿Cambia el sesgo de género a lo largo de la sesión?
   - Variables clave por cuartil de turno
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"

# ── Cargar todos los CSVs enriquecidos ────────────────────────────────────────
files = sorted(glob.glob(str(BASE / "final_reports" / "csv_enriched" / "*.csv")))
dfs   = []
for f in files:
    try:
        d = pd.read_csv(f, low_memory=False)
        d["session"] = Path(f).stem
        dfs.append(d)
    except: pass

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all[df_all["gender"].isin(["female","male"])].copy()
df_all["turn_number"] = pd.to_numeric(df_all["turn_number"], errors="coerce")
print(f"Intervenciones totales (F+M): {len(df_all)}")
print(f"Sesiones: {df_all['session'].nunique()}")
print(f"  F: {(df_all['gender']=='female').sum()}  M: {(df_all['gender']=='male').sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. RED DE INTERRUPCIONES DIRIGIDA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  1. RED DE INTERRUPCIONES DIRIGIDA (quien interrumpe a quien)")
print("="*60)

# Para cada intervención t que interrumpe la anterior (t-1):
# género del interruptor (t) → género del interrumpido (t-1)
df_sorted = df_all.sort_values(["session","turn_number"]).copy()
df_sorted["prev_gender"] = df_sorted.groupby("session")["gender"].shift(1)
df_sorted["next_gender"] = df_sorted.groupby("session")["gender"].shift(-1)

# Quien interrumpe: intervención t con interrupts_previous=True → género de t interrumpe género de t-1
interrupters = df_sorted[df_sorted["interrupts_previous"] == True].copy()
interrupters = interrupters[interrupters["prev_gender"].notna()]

print(f"\n  Total interrupciones detectadas: {len(interrupters)}")

# Matriz: interruptor_gender × interrumpido_gender
ct = pd.crosstab(interrupters["gender"], interrupters["prev_gender"])
ct.index.name   = "Interruptor"
ct.columns.name = "Interrumpido"
print(f"\n  Matriz de interrupciones (filas=interruptor, cols=interrumpido):")
print(ct.to_string())

if ct.shape == (2,2):
    chi2, p_chi, dof, exp = chi2_contingency(ct)
    print(f"\n  Chi2 = {chi2:.3f}  df={dof}  p={p_chi:.4f}")

    # Odds ratio: P(M interrumpe F) / P(M interrumpe M)  vs  P(F interrumpe M) / P(F interrumpe F)
    a = ct.loc["male","female"]    # M interrumpe F
    b = ct.loc["male","male"]      # M interrumpe M
    c = ct.loc["female","female"]  # F interrumpe F
    d = ct.loc["female","male"]    # F interrumpe M
    or_mf = (a/b) / (d/c) if (b>0 and c>0 and d>0) else np.nan
    print(f"\n  Odds Ratio (M->F vs M->M frente a F->F vs F->M): {or_mf:.3f}")
    print(f"  (OR>1 = hombres interrumpen desproporcionadamente a mujeres)")

    # Porcentajes
    row_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print(f"\n  Porcentaje de a quien interrumpen:")
    print(row_pct.round(1).to_string())

    # Test Fisher para cross-gender interruptions
    od_ratio, p_fisher = fisher_exact([[a,b],[d,c]])
    print(f"\n  Fisher exact (M->F vs M->M): OR={od_ratio:.3f}  p={p_fisher:.4f}")

# También: interrupciones exitosas por género (que toman el turno)
intr_success = df_sorted[df_sorted["interruption_success"] == True].copy()
intr_success = intr_success[intr_success["prev_gender"].notna()]
if len(intr_success):
    ct_s = pd.crosstab(intr_success["gender"], intr_success["prev_gender"])
    print(f"\n  Interrupciones EXITOSAS (toman el turno):")
    print(ct_s.to_string())

# Guardar
interr_rows = []
for g_intr in ["female","male"]:
    for g_prev in ["female","male"]:
        n = ((interrupters["gender"]==g_intr) & (interrupters["prev_gender"]==g_prev)).sum()
        interr_rows.append({"interruptor":g_intr, "interrumpido":g_prev, "n":n})
pd.DataFrame(interr_rows).to_csv(CSV_DIR / "c10_interruption_matrix.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 2. ANÁLISIS TEMPORAL (cuartil de turno intra-sesión)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  2. ANÁLISIS TEMPORAL (cuartil de turno)")
print("="*60)

# Calcular cuartil de turno dentro de cada sesión
df_sorted["turn_pct"] = df_sorted.groupby("session")["turn_number"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
)
df_sorted["quartile"] = pd.cut(df_sorted["turn_pct"],
                                bins=[0, 0.25, 0.5, 0.75, 1.001],
                                labels=["Q1 (inicio)", "Q2", "Q3", "Q4 (final)"],
                                include_lowest=True)

TEMP_VARS = ["is_mansplaining","has_hedge","lexical_diversity",
             "interrupts_previous","interruption_success","is_question"]
TEMP_VARS = [v for v in TEMP_VARS if v in df_sorted.columns]

temp_rows = []
for q in ["Q1 (inicio)","Q2","Q3","Q4 (final)"]:
    sub = df_sorted[df_sorted["quartile"] == q]
    for v in TEMP_VARS:
        for g in ["female","male"]:
            vals = sub[sub["gender"]==g][v].dropna()
            temp_rows.append({"quartile":q, "variable":v, "gender":g,
                               "mean":round(float(vals.mean()),4) if len(vals) else np.nan,
                               "n":len(vals)})

temp_df = pd.DataFrame(temp_rows)
temp_df.to_csv(CSV_DIR / "c10_temporal_by_quartile.csv", index=False)

# Mostrar mansplaining por cuartil y género
print(f"\n  Mansplaining por cuartil y género:")
if "is_mansplaining" in TEMP_VARS:
    piv = temp_df[temp_df["variable"]=="is_mansplaining"].pivot_table(
        index="quartile", columns="gender", values="mean")
    print(piv.round(4).to_string())
    print("\n  Gap (M-F) por cuartil:")
    if "male" in piv.columns and "female" in piv.columns:
        gap = piv["male"] - piv["female"]
        print(gap.round(4).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 3. MASA CRÍTICA: ¿% mujeres en sesión → reduce el sesgo?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  3. MASA CRÍTICA: % mujeres en sesión vs sesgo")
print("="*60)

sess_stats = []
for sess, grp in df_all.groupby("session"):
    pf = (grp["gender"] == "female").mean()
    nf = (grp["gender"] == "female").sum()
    nm = (grp["gender"] == "male").sum()
    n  = len(grp)
    if n < 5 or nf < 2 or nm < 2:
        continue
    row = {"session": sess, "pct_female": pf, "n": n, "n_female": nf, "n_male": nm}
    for v in ["is_mansplaining","has_hedge","lexical_diversity","interrupts_previous"]:
        if v in grp.columns:
            fv = grp[grp["gender"]=="female"][v].dropna()
            mv = grp[grp["gender"]=="male"][v].dropna()
            if len(fv) >= 2 and len(mv) >= 2:
                row[f"gap_{v}"] = float(mv.mean() - fv.mean())   # M-F: positivo = hombres más
    sess_stats.append(row)

sess_df = pd.DataFrame(sess_stats)
sess_df.to_csv(CSV_DIR / "c10_session_critical_mass.csv", index=False)
print(f"  Sesiones analizables: {len(sess_df)}")

from scipy.stats import spearmanr
gap_vars = [c for c in sess_df.columns if c.startswith("gap_")]
print(f"\n  Spearman(% mujeres, gap M-F) — negativo = más mujeres = menor sesgo:")
for gv in gap_vars:
    sub = sess_df[[gv, "pct_female"]].dropna()
    if len(sub) >= 5:
        r, p = spearmanr(sub["pct_female"], sub[gv])
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "."if p<0.1 else ""
        print(f"  {sig:>3} {gv:<35} rho={r:+.3f}  p={p:.4f}")

# Efecto umbral 30%
sess_df["above_30pct"] = sess_df["pct_female"] >= 0.30
print(f"\n  Sesiones >= 30% mujeres: {sess_df['above_30pct'].sum()} / {len(sess_df)}")
for gv in gap_vars:
    lo = sess_df[~sess_df["above_30pct"]][gv].dropna()
    hi = sess_df[sess_df["above_30pct"]][gv].dropna()
    if len(lo)>=3 and len(hi)>=3:
        _, p = mannwhitneyu(hi, lo, alternative="less")   # H1: más mujeres → menor gap
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        print(f"  {sig:>3} {gv:<35} gap<30%={lo.mean():+.4f}  gap>={30}%={hi.mean():+.4f}  p={p:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

# --- 4a. Matriz de interrupciones (heatmap) ---
if "ct" in dir() and ct.shape==(2,2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Red de Interrupciones por Género", fontsize=12, fontweight="bold")

    for ax, data, title in [
        (axes[0], ct, "Todas las interrupciones"),
        (axes[1], row_pct, "% por género del interruptor"),
    ]:
        im = ax.imshow(data.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks([0,1]); ax.set_xticklabels(["interrumpido: F","interrumpido: M"], fontsize=10)
        ax.set_yticks([0,1]); ax.set_yticklabels(["interruptor: F","interruptor: M"], fontsize=10)
        ax.set_title(title, fontsize=10)
        for i in range(2):
            for j in range(2):
                val = data.values[i,j]
                ax.text(j, i, f"{val:.1f}{'%' if title.startswith('%') else ''}",
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if val > data.values.max()*0.6 else "black")
        plt.colorbar(im, ax=ax)
    plt.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(FIG_DIR / "c10_01_interruption_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

# --- 4b. Temporal: mansplaining y hedge por cuartil ---
if "is_mansplaining" in TEMP_VARS and "has_hedge" in TEMP_VARS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Evolución temporal del sesgo intra-sesión (cuartiles de turno)",
                 fontsize=12, fontweight="bold")
    GCOLORS = {"female":"#e74c3c","male":"#3498db"}
    for ax, var, label in [(axes[0],"is_mansplaining","Mansplaining"),
                           (axes[1],"has_hedge","Hedge")]:
        sub = temp_df[temp_df["variable"]==var]
        for g, gc in GCOLORS.items():
            sg = sub[sub["gender"]==g].sort_values("quartile")
            ax.plot(range(len(sg)), sg["mean"], marker="o", color=gc, label=g.capitalize(),
                    linewidth=2, markersize=7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Q1\n(inicio)","Q2","Q3","Q4\n(final)"], fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(FIG_DIR / "c10_02_temporal_bias.png", dpi=200, bbox_inches="tight")
    plt.close()

# --- 4c. Masa crítica: scatter % mujeres vs gap mansplaining ---
if "gap_is_mansplaining" in sess_df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if v >= 0.30 else "#3498db" for v in sess_df["pct_female"]]
    ax.scatter(sess_df["pct_female"]*100, sess_df["gap_is_mansplaining"],
               c=colors, alpha=0.7, s=70, edgecolors="white")
    from scipy.stats import linregress
    sl = sess_df[["pct_female","gap_is_mansplaining"]].dropna()
    m, b, r, p, _ = linregress(sl["pct_female"]*100, sl["gap_is_mansplaining"])
    xs = np.linspace(0, 100, 100)
    ax.plot(xs, m*xs+b, "k--", linewidth=1.5, label=f"r={r:.2f}  p={p:.3f}")
    ax.axvline(30, color="orange", lw=1.5, linestyle=":", label="Umbral 30%")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("% Mujeres en sesión", fontsize=11)
    ax.set_ylabel("Gap mansplaining (M-F)", fontsize=11)
    ax.set_title("Masa crítica: % mujeres en sesión vs gap de mansplaining\n(rojo=≥30% mujeres)",
                 fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "c10_03_critical_mass.png", dpi=200, bbox_inches="tight")
    plt.close()

print(f"\n  Figuras guardadas en {FIG_DIR}/c10_*.png")
print(f"  CSVs guardados en {CSV_DIR}/c10_*.csv")
