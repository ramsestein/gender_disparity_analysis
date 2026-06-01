"""c16_session_bias.py — Que caracteristicas de una sesion predicen mayor sesgo de genero?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"

df = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
df["is_male"]     = df["gender"] == "male"
df["is_female"]   = df["gender"] == "female"
df["is_moderator"]= (df.get("Role","") == "Moderator").astype(float) if "Role" in df.columns else 0.0

# ── Agregar a nivel de sesion ─────────────────────────────────────────────────
def session_bias(grp):
    m = grp[grp["is_male"]]
    f = grp[grp["is_female"]]
    n_m, n_f = len(m), len(f)
    if n_m < 2 or n_f < 2:
        return None
    r = {
        "n_speakers":   len(grp),
        "n_male":       n_m,
        "n_female":     n_f,
        "pct_female":   n_f / len(grp),
        "mansplain_gap": m["pct_is_mansplaining"].mean() - f["pct_is_mansplaining"].mean(),
        "interrupt_gap": m["pct_interrupted_by_next"].mean() - f["pct_interrupted_by_next"].mean(),
        "hedge_gap":     f["pct_has_hedge"].mean() - m["pct_has_hedge"].mean(),
        "lexdiv_gap":    m["mean_lexical_diversity"].mean() - f["mean_lexical_diversity"].mean(),
        "has_female_mod": int((grp["is_moderator"].astype(bool) & grp["is_female"]).any()),
        "has_male_mod":   int((grp["is_moderator"].astype(bool) & grp["is_male"]).any()),
        "mean_log_cit":  grp.get("log_citations", pd.Series(0, index=grp.index)).mean()
                         if "log_citations" in grp.columns else 0,
    }
    # Grupo tematico de sesion (el mas comun)
    if "Group" in grp.columns:
        r["group"] = grp["Group"].mode()[0] if not grp["Group"].mode().empty else "Unknown"
    return r

sess = []
for s, grp in df.groupby("session"):
    r = session_bias(grp)
    if r:
        r["session"] = s
        sess.append(r)

sdf = pd.DataFrame(sess).dropna()
print(f"Sesiones con ambos generos: {len(sdf)}")
print(f"Gap mansplaining: media={sdf['mansplain_gap'].mean():.4f}  SD={sdf['mansplain_gap'].std():.4f}")
print(f"Gap interrupt:    media={sdf['interrupt_gap'].mean():.4f}  SD={sdf['interrupt_gap'].std():.4f}")

# ── Score global de sesgo de sesion ──────────────────────────────────────────
sdf["bias_score"] = (
    (sdf["mansplain_gap"] - sdf["mansplain_gap"].mean()) / (sdf["mansplain_gap"].std()+1e-9) +
    (sdf["interrupt_gap"] - sdf["interrupt_gap"].mean()) / (sdf["interrupt_gap"].std()+1e-9) +
    (sdf["lexdiv_gap"]    - sdf["lexdiv_gap"].mean())    / (sdf["lexdiv_gap"].std()+1e-9)
) / 3.0

print(f"\nSesiones mas sesgadas (bias_score):")
top5 = sdf.sort_values("bias_score", ascending=False).head(5)
for _, r in top5.iterrows():
    sname = str(r['session'])[:30].encode('ascii','replace').decode()
    print(f"  {sname:<32} score={r['bias_score']:+.3f}  pct_F={r['pct_female']:.2f}  mod_F={r['has_female_mod']}")

# ── Correlaciones con bias_score ─────────────────────────────────────────────
print(f"\nCorrelaciones con bias_score global:")
predictors = ["pct_female","n_speakers","has_female_mod","has_male_mod","mean_log_cit"]
for pred in predictors:
    if pred in sdf.columns:
        rho, p = stats.spearmanr(sdf[pred], sdf["bias_score"], nan_policy="omit")
        sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        print(f"  {pred:<22}  rho={rho:+.3f}  p={p:.4f} {sig}")

# ── Moderadora femenina: t-test ───────────────────────────────────────────────
print(f"\n=== Efecto de Moderadora Femenina ===")
has_fmod  = sdf[sdf["has_female_mod"] == 1]
no_fmod   = sdf[sdf["has_female_mod"] == 0]
print(f"Sesiones con mod.F: {len(has_fmod)}  sin mod.F: {len(no_fmod)}")
for col, label in [("mansplain_gap","Gap Mansplaining"),("interrupt_gap","Gap Interrupcion"),
                    ("bias_score","Score sesgo global")]:
    a, b = has_fmod[col].values, no_fmod[col].values
    if len(a)>3 and len(b)>3:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d_val = (a.mean()-b.mean()) / (np.std(np.concatenate([a,b]))+1e-9)
        sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        print(f"  {label:<25}  ModF={a.mean():+.4f}  NoModF={b.mean():+.4f}  d={d_val:+.3f}  p={p:.4f} {sig}")

# ── OLS: predictores del bias_score ──────────────────────────────────────────
pred_cols = [c for c in ["pct_female","n_speakers","has_female_mod","mean_log_cit"] if c in sdf.columns]
Xreg = sdf[pred_cols].fillna(0)
yreg = sdf["bias_score"].values
scaler = StandardScaler()
Xreg_std = scaler.fit_transform(Xreg)
Xreg_sm  = sm.add_constant(Xreg_std)
ols = sm.OLS(yreg, Xreg_sm).fit()
print(f"\nOLS bias_score ~ session_features  R2={ols.rsquared:.3f}")
for i, col in enumerate(pred_cols):
    b, p = ols.params[i+1], ols.pvalues[i+1]
    sig = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
    print(f"  {col:<22}  beta={b:+.4f}  p={p:.4f} {sig}")

# ── Graficos ──────────────────────────────────────────────────────────────────
# Bias score vs % mujeres
fig, axes = plt.subplots(1,2,figsize=(12,5))
fig.suptitle("Caracteristicas de sesion vs Sesgo de genero", fontsize=12, fontweight="bold")

axes[0].scatter(sdf["pct_female"], sdf["bias_score"], alpha=0.6, color="#e74c3c")
m_fit, b_fit = np.polyfit(sdf["pct_female"], sdf["bias_score"], 1)
x_line = np.linspace(0,1,50)
axes[0].plot(x_line, m_fit*x_line+b_fit, "k--", lw=2)
rho, p = stats.spearmanr(sdf["pct_female"], sdf["bias_score"])
axes[0].set_xlabel("% Mujeres en sesion"); axes[0].set_ylabel("Bias score global")
axes[0].set_title(f"Masa critica  rho={rho:+.3f}  p={p:.4f}")
axes[0].grid(True, alpha=0.3)

# Moderadora F vs no-F
bp = axes[1].boxplot([no_fmod["bias_score"].values, has_fmod["bias_score"].values],
                      labels=["Sin mod. femenina","Con mod. femenina"],
                      patch_artist=True, notch=False)
bp["boxes"][0].set_facecolor("#3498db"); bp["boxes"][1].set_facecolor("#e74c3c")
axes[1].set_ylabel("Bias score global")
axes[1].set_title("Efecto de moderadora femenina")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(FIG_DIR / "c16_session_bias.png", dpi=180, bbox_inches="tight")
plt.close()

# Group-level bias heatmap
if "group" in sdf.columns:
    grp_bias = sdf.groupby("group")[["mansplain_gap","interrupt_gap","hedge_gap","lexdiv_gap"]].mean()
    fig, ax = plt.subplots(figsize=(9, max(3, len(grp_bias)*0.5)))
    im = ax.imshow(grp_bias.values.T, cmap="RdBu_r", vmin=-0.05, vmax=0.05, aspect="auto")
    ax.set_xticks(range(len(grp_bias.index)))
    ax.set_xticklabels([g[:20] for g in grp_bias.index], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(grp_bias.columns)))
    ax.set_yticklabels(["Mansplaining gap","Interrupcion gap","Hedge gap","Div.lexica gap"], fontsize=9)
    for i in range(len(grp_bias.columns)):
        for j in range(len(grp_bias.index)):
            v = grp_bias.values[j,i]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v)>.03 else "black")
    plt.colorbar(im, ax=ax)
    ax.set_title("Sesgo medio por grupo tematico", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "c16_group_session_bias.png", dpi=180, bbox_inches="tight")
    plt.close()

sdf.to_csv(CSV_DIR / "c16_session_bias.csv", index=False)
print("\nDone c16_session_bias.py")
