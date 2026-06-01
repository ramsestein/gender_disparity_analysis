"""c15_interactions.py — Efectos de interaccion Gender x Role, Gender x Group, Gender x Specialty"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"

df = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
df["gender_bin"] = (df["gender"] == "male").astype(float)
df["is_male"]    = df["gender"] == "male"
spec = df.get("Specialty (ICU-Ane-Both)", pd.Series("", index=df.index)).fillna("").str.upper()
df["specialty"] = spec.where(spec.isin(["ICU","ANE","BOTH"]), "Unknown")
df["role"]      = df.get("Role", pd.Series("Unknown", index=df.index)).fillna("Unknown")
df["group"]     = df.get("Group", pd.Series("Unknown", index=df.index)).fillna("Unknown")

OUTCOMES = ["pct_is_mansplaining", "pct_interrupted_by_next",
            "pct_has_hedge", "mean_lexical_diversity", "pct_has_disagreement"]

def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    s = np.sqrt(((na-1)*np.var(a,ddof=1) + (nb-1)*np.var(b,ddof=1)) / (na+nb-2))
    return (np.mean(a)-np.mean(b)) / (s+1e-10)

rows = []

# ── 1. Gender × Role ──────────────────────────────────────────────────────────
print("\n=== Gender x Role ===")
for role in ["Moderator", "Speaker", "public"]:
    sub = df[df["role"] == role]
    m   = sub[sub["is_male"]]
    f   = sub[~sub["is_male"]]
    for out in OUTCOMES:
        a = m[out].dropna().values
        b = f[out].dropna().values
        if len(a)<5 or len(b)<5: continue
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = cohen_d(a, b)
        sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        if sig or abs(d)>.2:
            print(f"  {role:<12} {out:<30} d={d:+.3f} p={p:.4f} {sig}")
        rows.append({"moderator": "Gender x Role", "group": role, "outcome": out,
                     "n_M": len(a), "n_F": len(b), "cohen_d": round(d,3), "p": round(p,4), "sig": sig})

# ── 2. Gender × Specialty ─────────────────────────────────────────────────────
print("\n=== Gender x Specialty ===")
for spec_g in ["ICU","ANE","BOTH"]:
    sub = df[df["specialty"] == spec_g]
    m   = sub[sub["is_male"]]
    f   = sub[~sub["is_male"]]
    for out in OUTCOMES:
        a = m[out].dropna().values
        b = f[out].dropna().values
        if len(a)<5 or len(b)<5: continue
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = cohen_d(a, b)
        sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        if sig or abs(d)>.25:
            print(f"  {spec_g:<6} {out:<30} d={d:+.3f} p={p:.4f} {sig}")
        rows.append({"moderator": "Gender x Specialty", "group": spec_g, "outcome": out,
                     "n_M": len(a), "n_F": len(b), "cohen_d": round(d,3), "p": round(p,4), "sig": sig})

# ── 3. Gender × Group ─────────────────────────────────────────────────────────
print("\n=== Gender x Group ===")
for grp in df["group"].unique():
    if grp == "Unknown": continue
    sub = df[df["group"] == grp]
    m   = sub[sub["is_male"]]
    f   = sub[~sub["is_male"]]
    for out in ["pct_is_mansplaining","pct_interrupted_by_next"]:
        a = m[out].dropna().values
        b = f[out].dropna().values
        if len(a)<4 or len(b)<4: continue
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = cohen_d(a, b)
        sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        if sig or abs(d)>.3:
            print(f"  {grp[:30]:<32} {out:<30} d={d:+.3f} p={p:.4f} {sig}")
        rows.append({"moderator": "Gender x Group", "group": grp, "outcome": out,
                     "n_M": len(a), "n_F": len(b), "cohen_d": round(d,3), "p": round(p,4), "sig": sig})

pd.DataFrame(rows).to_csv(CSV_DIR / "c15_interactions.csv", index=False)

# ── Heatmap: d por grupo x outcome (solo mansplaining + interruption) ─────────
for inter_type, mod_col in [("Gender x Role","role"), ("Gender x Group","group")]:
    sub_rows = [r for r in rows if r["moderator"]==f"Gender x {mod_col.capitalize()}"]
    if not sub_rows: sub_rows = [r for r in rows if r["moderator"]==inter_type]
    if not sub_rows: continue
    rdf   = pd.DataFrame(sub_rows)
    pivot = rdf.pivot_table(index="group", columns="outcome", values="cohen_d", aggfunc="mean")
    pivot = pivot[[c for c in pivot.columns if c in OUTCOMES]]
    if pivot.empty: continue
    fig, ax = plt.subplots(figsize=(max(7, len(pivot.columns)*1.8), max(4, len(pivot)*0.55)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("pct_","").replace("has_","").replace("_"," ")[:18]
                        for c in pivot.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(g)[:25] for g in pivot.index], fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(v)>.4 else "black")
    plt.colorbar(im, ax=ax, label="Cohen's d (M-F)")
    ax.set_title(f"Interaccion {inter_type} — Cohen's d por outcome", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = f"c15_heatmap_{mod_col}.png"
    fig.savefig(FIG_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {fname}")

print("Done c15_interactions.py")
