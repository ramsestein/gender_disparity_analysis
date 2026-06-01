"""c19_bayesian.py — Estimacion bayesiana de efectos de genero via MCMC (Metropolis-Hastings)
Sin dependencias externas pesadas: implementacion propia con numpy."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"

df = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")

OUTCOMES = {
    "pct_is_mansplaining": "Mansplaining (M>F)",
    "pct_interrupted_by_next": "Ser interrumpido",
    "pct_has_hedge": "Hedge (F>M)",
    "mean_lexical_diversity": "Diversidad lexica (M>F)",
    "pct_has_disagreement": "Desacuerdo (F>M)",
}

def bayesian_ttest_mcmc(x_m, x_f, n_samples=20000, prior_mu=0.0, prior_sigma=0.5, rng=None):
    """
    Modelo bayesiano simple para diferencia de medias:
      delta ~ Normal(prior_mu, prior_sigma)   <- prior sobre el tamano del efecto
      sigma ~ HalfNormal(0.3)
      y_m ~ Normal(mu_f + delta*sigma_pool, sigma_pool)
      y_f ~ Normal(mu_f, sigma_pool)
    Aproximacion por grid sobre delta, integrando sigma_pool analiticament
    (Metropolis-Hastings sobre delta).
    """
    if rng is None: rng = np.random.default_rng(42)
    n_m, n_f   = len(x_m), len(x_f)
    pool_std   = np.std(np.concatenate([x_m, x_f]), ddof=1)
    if pool_std < 1e-9: return None

    # Normalizar
    xm_n = (x_m - np.mean(x_f)) / pool_std
    xf_n = (x_f - np.mean(x_f)) / pool_std

    # Log-posterior(delta) proportional to:
    # sum log N(xm_i | delta, 1) + sum log N(xf_i | 0, 1) + log N(delta | prior_mu, prior_sigma)
    def log_post(delta):
        ll_m  = np.sum(stats.norm.logpdf(xm_n, loc=delta, scale=1.0))
        ll_f  = np.sum(stats.norm.logpdf(xf_n, loc=0.0,   scale=1.0))
        log_p = stats.norm.logpdf(delta, loc=prior_mu, scale=prior_sigma)
        return ll_m + ll_f + log_p

    # Metropolis-Hastings
    samples = np.empty(n_samples)
    current = 0.0
    lp_cur  = log_post(current)
    prop_sd = 0.08
    n_accept = 0
    for i in range(n_samples):
        proposal = current + rng.normal(0, prop_sd)
        lp_prop  = log_post(proposal)
        if np.log(rng.uniform()) < (lp_prop - lp_cur):
            current  = proposal
            lp_cur   = lp_prop
            n_accept += 1
        samples[i] = current

    samples = samples[n_samples//4:]  # burn-in 25%
    return samples * pool_std  # back to original scale

rows = []
fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(4*len(OUTCOMES), 4))
fig.suptitle("Estimacion bayesiana: distribucion posterior de diferencia de medias (M - F)",
             fontsize=11, fontweight="bold")

for ax, (out, label) in zip(axes, OUTCOMES.items()):
    m_vals = df[df["gender"]=="male"][out].dropna().values
    f_vals = df[df["gender"]=="female"][out].dropna().values
    if len(m_vals)<10 or len(f_vals)<10:
        ax.set_title(label); ax.text(0.5,0.5,"n insuf.",transform=ax.transAxes,ha="center"); continue

    samples = bayesian_ttest_mcmc(m_vals, f_vals)
    if samples is None:
        ax.set_title(label); continue

    post_mean  = np.mean(samples)
    hdi_lo     = np.percentile(samples, 2.5)
    hdi_hi     = np.percentile(samples, 97.5)
    p_pos      = np.mean(samples > 0)
    p_neg      = 1 - p_pos

    print(f"\n[{label}]")
    print(f"  Posterior mean={post_mean:+.4f}  HDI95=[{hdi_lo:+.4f}, {hdi_hi:+.4f}]")
    print(f"  P(M>F)={p_pos:.3f}  P(F>M)={p_neg:.3f}")

    rows.append({"outcome": out, "label": label,
                 "post_mean": round(post_mean,4),
                 "hdi_lo": round(hdi_lo,4), "hdi_hi": round(hdi_hi,4),
                 "p_M_gt_F": round(p_pos,3), "p_F_gt_M": round(p_neg,3),
                 "n_M": len(m_vals), "n_F": len(f_vals)})

    color = "#e74c3c" if post_mean > 0 else "#3498db"
    ax.hist(samples, bins=60, density=True, color=color, alpha=0.7)
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.axvline(post_mean, color=color, lw=2)
    ax.axvspan(hdi_lo, hdi_hi, alpha=0.15, color=color)
    ax.set_title(f"{label[:22]}\nE={post_mean:+.3f} [HDI95: {hdi_lo:+.3f},{hdi_hi:+.3f}]",
                 fontsize=8, fontweight="bold")
    ax.set_xlabel("Diferencia M-F"); ax.grid(True, alpha=0.3)
    # Anotacion de probabilidad
    side = p_pos if post_mean>0 else p_neg
    ax.text(0.97, 0.95, f"P(dir.)={side:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, fontweight="bold")

plt.tight_layout(rect=[0,0,1,0.93])
fig.savefig(FIG_DIR / "c19_bayesian_posteriors.png", dpi=180, bbox_inches="tight")
plt.close()

rdf = pd.DataFrame(rows)
rdf.to_csv(CSV_DIR / "c19_bayesian_results.csv", index=False)

print(f"\n=== RESUMEN BAYESIANO ===")
print(f"{'Outcome':<35} {'E(delta)':>10} {'HDI95_lo':>10} {'HDI95_hi':>10} {'P(dir.)':>8}")
for _, r in rdf.iterrows():
    p_dir = max(r["p_M_gt_F"], r["p_F_gt_M"])
    print(f"{r['label']:<35} {r['post_mean']:>+10.4f} {r['hdi_lo']:>+10.4f} {r['hdi_hi']:>+10.4f} {p_dir:>8.3f}")

print("\nDone c19_bayesian.py")
