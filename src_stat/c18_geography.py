"""c18_geography.py — Sesgo de genero por region / pais"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"

df = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
df["is_male"] = df["gender"] == "male"

# Intentar usar Country o Region
country_col = None
for c in ["Country", "country", "Region", "region"]:
    if c in df.columns:
        country_col = c; break

if country_col is None:
    print("No se encontro columna de pais/region. Intentando inferir desde enriched CSVs...")
    enrich_dir = BASE / "final_reports" / "csv_enriched"
    if enrich_dir.exists():
        parts = []
        for f in sorted(enrich_dir.glob("*.csv")):
            try:
                tmp = pd.read_csv(f, encoding="utf-8-sig")
                for col in ["Country","country","Region","region"]:
                    if col in tmp.columns:
                        parts.append(tmp[["speaker", col]].rename(columns={col:"country"}))
                        break
            except Exception:
                pass
        if parts:
            cdf = pd.concat(parts).drop_duplicates("speaker")
            df  = df.merge(cdf, on="speaker", how="left")
            country_col = "country"

if country_col is None:
    print("Columna de pais no disponible. Saltando c18.")
    import sys; sys.exit(0)

df["region"] = df[country_col].fillna("Unknown")
# Simplificar a macro-regiones si hay demasiados paises
vc = df["region"].value_counts()
print(f"Paises/regiones disponibles ({len(vc)} unicos):")
print(vc.head(15).to_string())

REGION_MAP = {
    "Spain":"S.Europe","Italy":"S.Europe","France":"S.Europe","Portugal":"S.Europe",
    "Greece":"S.Europe","Belgium":"W.Europe","Netherlands":"W.Europe","Switzerland":"W.Europe",
    "Germany":"W.Europe","Austria":"W.Europe","UK":"Anglo","Ireland":"Anglo",
    "USA":"Anglo","Canada":"Anglo","Australia":"Anglo",
    "Sweden":"N.Europe","Norway":"N.Europe","Denmark":"N.Europe","Finland":"N.Europe",
    "Poland":"E.Europe","Czech Republic":"E.Europe","Hungary":"E.Europe","Turkey":"E.Europe",
    "Israel":"Middle East","Saudi Arabia":"Middle East",
    "China":"Asia","Japan":"Asia","India":"Asia","South Korea":"Asia",
    "Brazil":"Latin America","Argentina":"Latin America","Colombia":"Latin America",
}
df["macro_region"] = df["region"].map(REGION_MAP).fillna(df["region"])

OUTCOMES = ["pct_is_mansplaining","pct_interrupted_by_next","pct_has_hedge","mean_lexical_diversity"]

def cohen_d(a,b):
    na,nb=len(a),len(b)
    if na<2 or nb<2: return np.nan
    s = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(a)-np.mean(b))/(s+1e-10)

rows = []
print("\n=== Gap de genero por macro-region ===")
for reg in sorted(df["macro_region"].unique()):
    sub = df[df["macro_region"] == reg]
    m   = sub[sub["is_male"]]
    f   = sub[~sub["is_male"]]
    if len(m)<4 or len(f)<4: continue
    for out in OUTCOMES:
        a = m[out].dropna().values
        b = f[out].dropna().values
        if len(a)<3 or len(b)<3: continue
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d    = cohen_d(a, b)
        sig  = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        rows.append({"region":reg,"outcome":out,"n_M":len(a),"n_F":len(b),
                     "cohen_d":round(d,3),"p":round(p,4),"sig":sig})
    print(f"  {reg:<15} n_M={len(m):3d} n_F={len(f):3d}", end="  ")
    d_ms = cohen_d(m["pct_is_mansplaining"].dropna().values, f["pct_is_mansplaining"].dropna().values)
    print(f"mansplain_d={d_ms:+.3f}")

rdf = pd.DataFrame(rows)
rdf.to_csv(CSV_DIR / "c18_geography_bias.csv", index=False)

if rdf.empty:
    print("No hay datos suficientes por region. Fin.")
    import sys; sys.exit(0)

# Heatmap
pivot = rdf.pivot_table(index="region", columns="outcome", values="cohen_d", aggfunc="mean").fillna(0)
fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*2), max(4, len(pivot)*0.5)))
im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([c.replace("pct_","").replace("has_","").replace("_"," ")[:18]
                    for c in pivot.columns], rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i,j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(v)>.45 else "black")
plt.colorbar(im, ax=ax, label="Cohen's d (M-F)")
ax.set_title("Sesgo de genero por region geografica — Cohen's d", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "c18_geography_heatmap.png", dpi=180, bbox_inches="tight")
plt.close()
print("Done c18_geography.py")
