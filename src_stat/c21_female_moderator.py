"""c21_female_moderator.py — Efecto de la moderadora femenina sobre el sesgo de genero"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"

df = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
df["is_male"]   = df["gender"] == "male"
df["is_female"] = df["gender"] == "female"

if "Role" not in df.columns:
    print("Columna Role no disponible.")
    import sys; sys.exit(0)

df["is_moderator"] = df["Role"].str.lower().str.contains("moderator", na=False)

# ── Identificar tipo de moderacion por sesion ─────────────────────────────────
def session_mod_type(grp):
    mods = grp[grp["is_moderator"]]
    if len(mods) == 0: return "none"
    has_f = (mods["is_female"]).any()
    has_m = (mods["is_male"]).any()
    if has_f and has_m: return "mixed"
    if has_f: return "female_only"
    return "male_only"

sess_type = df.groupby("session").apply(session_mod_type).reset_index()
sess_type.columns = ["session", "mod_type"]
df = df.merge(sess_type, on="session", how="left")

print("Distribucion de tipo de moderacion por sesion:")
print(df.groupby("session")["mod_type"].first().value_counts().to_string())

OUTCOMES = {
    "pct_is_mansplaining": "Mansplaining",
    "pct_interrupted_by_next": "Ser interrumpido",
    "pct_has_hedge": "Hedge",
    "mean_lexical_diversity": "Div. lexica",
    "pct_has_disagreement": "Desacuerdo",
}

def cohen_d(a,b):
    na,nb=len(a),len(b)
    if na<2 or nb<2: return np.nan
    s=np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(a)-np.mean(b))/(s+1e-10)

# ── A) Diferencia de genero DENTRO de sesiones con mod.F vs mod.M ─────────────
print(f"\n{'='*65}")
print("  A) Gap de genero segun tipo de moderacion de la sesion")
print(f"{'='*65}")

results = []
for mod in ["female_only","male_only","mixed","none"]:
    sub = df[df["mod_type"] == mod]
    n_sess = sub["session"].nunique()
    if n_sess < 3: continue
    m_sub = sub[sub["is_male"]]
    f_sub = sub[sub["is_female"]]
    row = {"mod_type": mod, "n_sessions": n_sess, "n_M": len(m_sub), "n_F": len(f_sub)}
    print(f"\n  [{mod}]  n_sesiones={n_sess}  n_M={len(m_sub)}  n_F={len(f_sub)}")
    for out, label in OUTCOMES.items():
        a = m_sub[out].dropna().values
        b = f_sub[out].dropna().values
        if len(a)<4 or len(b)<4: continue
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d    = cohen_d(a, b)
        sig  = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        row[f"d_{out}"] = round(d,3)
        row[f"p_{out}"] = round(p,4)
        if sig or abs(d)>0.2:
            print(f"    {label:<22} d={d:+.3f} p={p:.4f} {sig}")
    results.append(row)

rdf = pd.DataFrame(results)
rdf.to_csv(CSV_DIR / "c21_female_moderator.csv", index=False)

# ── B) Comparacion directa: mod.F vs mod.M en mansplaining gap ───────────────
print(f"\n{'='*65}")
print("  B) Comparacion mod.Femenina vs mod.Masculina — Mansplaining gap de sesion")
print(f"{'='*65}")

session_gaps = []
for sess, grp in df.groupby("session"):
    mod_t = grp["mod_type"].iloc[0]
    m = grp[grp["is_male"]]
    f = grp[grp["is_female"]]
    if len(m)<2 or len(f)<2: continue
    session_gaps.append({
        "session": sess,
        "mod_type": mod_t,
        "mansplain_gap": m["pct_is_mansplaining"].mean() - f["pct_is_mansplaining"].mean(),
        "interrupt_gap": m["pct_interrupted_by_next"].mean() - f["pct_interrupted_by_next"].mean(),
        "pct_female": len(f)/(len(f)+len(m)),
    })

sgdf = pd.DataFrame(session_gaps)
for g1, g2, label in [
    ("female_only","male_only","ModF vs ModM"),
    ("female_only","none","ModF vs Sin-mod"),
]:
    a = sgdf[sgdf["mod_type"]==g1]["mansplain_gap"].values
    b = sgdf[sgdf["mod_type"]==g2]["mansplain_gap"].values
    if len(a)<3 or len(b)<3:
        print(f"  {label}: n insuficiente ({len(a)} vs {len(b)})")
        continue
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d    = cohen_d(a, b)
    print(f"  {label}: mean_F={a.mean():+.4f}  mean_M={b.mean():+.4f}  d={d:+.3f}  p={p:.4f}")

# ── C) Efecto de la moderadora sobre la interrupcion de mujeres ───────────────
print(f"\n  Interrupciones de mujeres segun tipo moderacion:")
for mod in ["female_only","male_only"]:
    sub = df[(df["mod_type"]==mod) & df["is_female"]]
    v   = sub["pct_interrupted_by_next"].dropna().values
    print(f"  {mod}: mean={v.mean():.4f}  n={len(v)}")
if "female_only" in sgdf["mod_type"].values and "male_only" in sgdf["mod_type"].values:
    a = df[(df["mod_type"]=="female_only") & df["is_female"]]["pct_interrupted_by_next"].dropna().values
    b = df[(df["mod_type"]=="male_only")   & df["is_female"]]["pct_interrupted_by_next"].dropna().values
    if len(a)>5 and len(b)>5:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d    = cohen_d(a, b)
        print(f"  Test ModF vs ModM en interrupciones de mujeres: d={d:+.3f}  p={p:.4f}")

# ── Grafico ───────────────────────────────────────────────────────────────────
d_cols = [c for c in rdf.columns if c.startswith("d_")]
if not rdf.empty and d_cols:
    fig, ax = plt.subplots(figsize=(10, 5))
    x   = np.arange(len(d_cols))
    w   = 0.18
    mod_types = rdf["mod_type"].tolist()
    colors_map = {"female_only":"#e74c3c","male_only":"#3498db","mixed":"#9b59b6","none":"#7f8c8d"}
    for i, row in rdf.iterrows():
        vals = [row.get(c, np.nan) for c in d_cols]
        offset = (i - len(rdf)/2) * w
        bars = ax.bar(x + offset, vals, w,
                      label=row["mod_type"],
                      color=colors_map.get(row["mod_type"],"gray"), alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("d_pct_","").replace("d_mean_","").replace("_"," ")
                        for c in d_cols], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Cohen's d (M-F)")
    ax.set_title("Gap de genero (Cohen's d) segun tipo de moderacion\n(>0: hombres mayor valor; <0: mujeres mayor valor)",
                 fontsize=10, fontweight="bold")
    ax.legend(title="Tipo moderacion"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "c21_female_moderator.png", dpi=180, bbox_inches="tight")
    plt.close()

# Boxplot mansplaining gap por tipo moderacion
if len(sgdf) > 0:
    mod_order = [m for m in ["female_only","male_only","mixed","none"] if m in sgdf["mod_type"].values]
    fig, ax = plt.subplots(figsize=(8,5))
    data  = [sgdf[sgdf["mod_type"]==m]["mansplain_gap"].values for m in mod_order]
    bp    = ax.boxplot(data, labels=mod_order, patch_artist=True, notch=False)
    cmap  = {"female_only":"#e74c3c","male_only":"#3498db","mixed":"#9b59b6","none":"#7f8c8d"}
    for box, m in zip(bp["boxes"], mod_order):
        box.set_facecolor(cmap.get(m,"gray")); box.set_alpha(0.7)
    ax.axhline(0, color="black", ls="--", lw=1.5)
    ax.set_ylabel("Gap mansplaining (M-F, por sesion)")
    ax.set_title("Mansplaining gap por tipo de moderacion", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "c21_moderator_boxplot.png", dpi=180, bbox_inches="tight")
    plt.close()

print("\nDone c21_female_moderator.py")
