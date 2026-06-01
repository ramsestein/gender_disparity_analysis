"""c20_power_analysis.py — Analisis de potencia estadistica para los efectos observados"""
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

# ── Calcular efectos observados ───────────────────────────────────────────────
OUTCOMES = {
    "pct_is_mansplaining": "Mansplaining",
    "pct_interrupted_by_next": "Interrumpido",
    "pct_has_hedge": "Hedge",
    "mean_lexical_diversity": "Div. lexica",
    "pct_has_disagreement": "Desacuerdo",
    "mean_overlap_duration": "Solapamiento",
}

def cohen_d(a,b):
    na,nb=len(a),len(b)
    if na<2 or nb<2: return np.nan
    s=np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return abs(np.mean(a)-np.mean(b))/(s+1e-10)

def power_ttest(d, n1, n2, alpha=0.05):
    """Potencia estadistica para Mann-Whitney aproximado por t-test de dos muestras."""
    se = np.sqrt(1/n1 + 1/n2)
    nc = d / se
    t_crit = stats.t.ppf(1-alpha/2, df=n1+n2-2)
    power  = 1 - stats.t.cdf(t_crit, df=n1+n2-2, loc=nc) + stats.t.cdf(-t_crit, df=n1+n2-2, loc=nc)
    return power

def n_for_power(d, power=0.80, alpha=0.05):
    """N total necesario para detectar efecto d con potencia dada."""
    if d < 1e-4: return np.inf
    for n_per in range(5, 5000):
        if power_ttest(d, n_per, n_per, alpha) >= power:
            return 2 * n_per
    return np.inf

m_all = df[df["gender"]=="male"]
f_all = df[df["gender"]=="female"]
n_m, n_f = len(m_all), len(f_all)

rows = []
print(f"N total: {len(df)}  N_M={n_m}  N_F={n_f}")
print(f"\n{'Outcome':<25} {'d_obs':>7} {'Power_obs':>10} {'N_80%':>8} {'N_90%':>8} {'N_95%':>8}")
print("-"*65)
for out, label in OUTCOMES.items():
    a = m_all[out].dropna().values
    b = f_all[out].dropna().values
    d = cohen_d(a, b)
    if np.isnan(d): continue
    pow_obs = power_ttest(d, n_m, n_f)
    n80  = n_for_power(d, 0.80)
    n90  = n_for_power(d, 0.90)
    n95  = n_for_power(d, 0.95)
    fmt = lambda n: ">5000" if not np.isfinite(n) else str(int(n))
    print(f"  {label:<23} {d:>+7.3f} {pow_obs:>10.3f} {fmt(n80):>8} {fmt(n90):>8} {fmt(n95):>8}")
    safe = lambda n: -1 if not np.isfinite(n) else int(n)
    rows.append({"outcome":out,"label":label,"cohen_d":round(d,3),
                 "power_obs":round(pow_obs,3),"n_80":safe(n80),"n_90":safe(n90),"n_95":safe(n95),
                 "n_M":n_m,"n_F":n_f})

rdf = pd.DataFrame(rows)
rdf.to_csv(CSV_DIR / "c20_power_analysis.csv", index=False)

# ── Curvas de potencia ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Analisis de potencia estadistica", fontsize=12, fontweight="bold")

# Panel 1: potencia vs N para cada efecto observado
n_range = np.arange(10, 800, 10)
colors  = plt.cm.tab10(np.linspace(0, 1, len(rdf)))
for i, row in rdf.iterrows():
    powers = [power_ttest(row["cohen_d"], n//2, n//2) for n in n_range]
    axes[0].plot(n_range, powers, label=f"{row['label']} (d={row['cohen_d']:.2f})",
                 color=colors[i], lw=2)
axes[0].axhline(0.80, color="gray", ls="--", lw=1.5, label="80% potencia")
axes[0].axhline(0.90, color="gray", ls=":", lw=1.5, label="90% potencia")
axes[0].axvline(n_m+n_f, color="black", ls="-.", lw=1.5, label=f"N actual={n_m+n_f}")
axes[0].set_xlabel("N total (n_M = n_F = N/2)"); axes[0].set_ylabel("Potencia estadistica")
axes[0].set_title("Curvas de potencia por outcome"); axes[0].legend(fontsize=7)
axes[0].set_ylim(0,1); axes[0].grid(True, alpha=0.3)

# Panel 2: barras N requerido para 80% potencia
rdf_s = rdf.sort_values("n_80")
colors_bar = ["#27ae60" if row["n_80"] <= n_m+n_f else "#e74c3c"
              for _, row in rdf_s.iterrows()]
bars = axes[1].barh(rdf_s["label"], rdf_s["n_80"], color=colors_bar, alpha=0.85)
axes[1].axvline(n_m+n_f, color="black", ls="--", lw=2, label=f"N actual = {n_m+n_f}")
axes[1].set_xlabel("N total necesario para 80% potencia")
axes[1].set_title("N requerido vs N actual\n(verde = ya alcanzado)")
axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="x")
for bar, row in zip(bars, rdf_s.itertuples()):
    axes[1].text(bar.get_width()+3, bar.get_y()+bar.get_height()/2,
                 str(row.n_80), va="center", fontsize=9)

plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(FIG_DIR / "c20_power_analysis.png", dpi=180, bbox_inches="tight")
plt.close()
print("\nDone c20_power_analysis.py")
