"""
c22_enrich_excel_and_graphs.py
  1) Genera graficos explicativos nuevos (arboles CART con AUC, SHAP panel, Bayesian, Markov)
  2) Sobreescribe los Excel de final_reports/excel/ con version enriquecida (añade Role, Group,
     Specialty, Country, Citations, Cluster, variables estadisticas del usuario)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
import glob, shutil

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import shap

BASE      = Path(__file__).resolve().parent.parent
CSV_DIR   = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR   = BASE / "final_reports" / "resultados" / "graficos"
EXCEL_DIR = BASE / "final_reports" / "excel"

# ─────────────────────────────────────────────────────────────────────────────
# 0. Datos base
# ─────────────────────────────────────────────────────────────────────────────
udf = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
udf["is_male"] = udf["gender"] == "male"

FEAT_MANSPLAIN = [
    "is_male","mean_duration","total_duration","std_duration","mean_wpm",
    "mean_lexical_diversity","mean_latency_s","mean_overlap_duration",
    "mean_echoing_score","mean_conflict_score","mean_assertiveness_score",
    "pct_has_hedge","pct_has_disagreement","pct_has_agreement","pct_is_question",
    "pct_interrupts_previous","pct_interrupted_by_next","pct_has_overlap",
    "pct_is_backchannel","pct_has_courtesy","pct_has_vulnerability",
    "first_turn_number","is_top3_speaker","is_moderator","is_speaker","is_public",
    "is_ICU","is_Ane","is_Both","log_citations",
]

FEAT_INTERRUPT = [
    "mean_duration","total_duration","std_duration","mean_wpm",
    "mean_lexical_diversity","mean_latency_s","mean_overlap_duration",
    "mean_echoing_score","mean_conflict_score","mean_assertiveness_score",
    "pct_has_hedge","pct_has_disagreement","pct_has_agreement","pct_is_question",
    "pct_interrupts_previous","pct_has_overlap","pct_is_backchannel",
    "pct_has_courtesy","pct_has_vulnerability","first_turn_number",
    "is_top3_speaker","is_moderator","is_speaker","is_public",
    "is_ICU","is_Ane","is_Both","log_citations","n_interventions",
]

FEAT_NAMES_M = [
    "Hombre","Dur_media","Dur_total","SD_dur","WPM",
    "Div_lex","Latencia","Solap","Echoing","Conflicto","Asertividad",
    "Hedge","Desacuerdo","Acuerdo","Pregunta",
    "Interrumpe","Interrumpido","Solapamiento",
    "Backchannel","Cortesia","Vulnerabilidad",
    "TurnoInicio","Top3","Moderador","Speaker","Publico",
    "ICU","Ane","Both","log_cit",
]

FEAT_NAMES_I = [
    "Dur_media","Dur_total","SD_dur","WPM",
    "Div_lex","Latencia","Solap","Echoing","Conflicto","Asertividad",
    "Hedge","Desacuerdo","Acuerdo","Pregunta",
    "Interrumpe","Solapamiento",
    "Backchannel","Cortesia","Vulnerabilidad","TurnoInicio",
    "Top3","Moderador","Speaker","Publico",
    "ICU","Ane","Both","log_cit","N_interv",
]

def prep(df, feats, outcome, fill=0):
    d = df[feats + [outcome]].copy()
    d[feats] = d[feats].fillna(fill)
    vals = d[outcome].dropna()
    # Ensure balanced split: label top-25% as 1, bottom-75% as 0
    # Use rank to avoid degenerate thresholds when many zeros
    d["_rank"] = d[outcome].rank(pct=True, na_option="keep")
    d["y"] = (d["_rank"] > 0.75).astype(int)
    d = d.dropna(subset=["_rank"])
    return d[feats].fillna(fill).values, d["y"].values

# ─────────────────────────────────────────────────────────────────────────────
# 1. ÁRBOL CART — Visualización elegante con métricas
# ─────────────────────────────────────────────────────────────────────────────
def plot_cart_annotated(X, y, feat_names, title, out_path, max_depth=4):
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=12,
                                class_weight="balanced", random_state=42)
    dt.fit(X, y)
    auc = cross_val_score(dt, X, y, cv=5, scoring="roc_auc").mean()

    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(dt, feature_names=feat_names, class_names=["No","Si"],
              filled=True, rounded=True, fontsize=9,
              impurity=False, proportion=True, ax=ax,
              precision=3)
    ax.set_title(f"{title}\nAUC cross-val 5-fold = {auc:.3f} | n={len(y)} | prof.max={max_depth}",
                 fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}  (AUC={auc:.3f})")
    return dt, auc

print("=== 1. CART annotated trees ===")
Xm, ym = prep(udf, FEAT_MANSPLAIN, "pct_is_mansplaining")
Xi, yi = prep(udf[udf["gender"]=="female"], FEAT_INTERRUPT, "pct_interrupted_by_next")
dt_m, auc_m = plot_cart_annotated(Xm, ym, FEAT_NAMES_M,
    "Arbol de Decision CART — Mansplaining (outcome binario top-25%)",
    FIG_DIR / "c22_cart_mansplain_annotated.png", max_depth=4)
dt_i, auc_i = plot_cart_annotated(Xi, yi, FEAT_NAMES_I,
    "Arbol de Decision CART — Ser Interrumpida/Mujeres (outcome binario top-25%)",
    FIG_DIR / "c22_cart_interrupted_annotated.png", max_depth=4)

# ─────────────────────────────────────────────────────────────────────────────
# 2. GBM — Feature importance + curva ROC comparativa
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import roc_curve, auc as roc_auc

print("\n=== 2. GBM ROC + feature importance ===")
def plot_gbm_roc_fi(X, y, Xdt, ydt, feat_names, dt_model, dt_auc,
                    title_prefix, out_path):
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=42)
    gbm.fit(X, y)
    gbm_auc = cross_val_score(gbm, X, y, cv=5, scoring="roc_auc").mean()

    # Full dataset ROC
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    fpr_list, tpr_list = [], []
    for tr, te in skf.split(X, y):
        gbm.fit(X[tr], y[tr])
        p = gbm.predict_proba(X[te])[:,1]
        fpr, tpr, _ = roc_curve(y[te], p)
        fpr_list.append(fpr); tpr_list.append(tpr)
    dt_scores = cross_val_score(dt_model, Xdt, ydt, cv=5, scoring="roc_auc")
    fpr_d, tpr_d = [], []
    for tr, te in skf.split(Xdt, ydt):
        dt_model.fit(Xdt[tr], ydt[tr])
        p = dt_model.predict_proba(Xdt[te])[:,1]
        f, t, _ = roc_curve(ydt[te], p)
        fpr_d.append(f); tpr_d.append(t)

    fi = gbm.feature_importances_
    fi_idx = np.argsort(fi)[-15:]
    fi_names = [feat_names[i] for i in fi_idx]
    fi_vals  = fi[fi_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title_prefix, fontsize=12, fontweight="bold")

    # Panel 1: ROC
    ax = axes[0]
    for fpr, tpr in zip(fpr_list, tpr_list):
        ax.plot(fpr, tpr, alpha=0.25, color="#e74c3c", lw=1)
    for fpr, tpr in zip(fpr_d, tpr_d):
        ax.plot(fpr, tpr, alpha=0.25, color="#3498db", lw=1)
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.plot([], [], color="#e74c3c", lw=2, label=f"GBM (AUC={gbm_auc:.3f})")
    ax.plot([], [], color="#3498db", lw=2, label=f"CART (AUC={dt_auc:.3f})")
    ax.set_xlabel("1 - Especificidad"); ax.set_ylabel("Sensibilidad")
    ax.set_title("Curvas ROC (5-fold CV)")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.text(0.55, 0.12, f"ΔAUC = +{gbm_auc-dt_auc:.3f} (GBM vs CART)",
            transform=ax.transAxes, fontsize=10, color="#c0392b", fontweight="bold")

    # Panel 2: Feature importance
    ax = axes[1]
    colors = ["#e74c3c" if v >= fi_vals[-3] else "#3498db" for v in fi_vals]
    ax.barh(fi_names, fi_vals, color=colors, alpha=0.85)
    ax.set_xlabel("Importancia (MDI)")
    ax.set_title("Top 15 features — GBM")
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(fi_vals.mean(), color="gray", ls="--", lw=1, label=f"Media={fi_vals.mean():.3f}")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}  (GBM AUC={gbm_auc:.3f})")
    return gbm

gbm_m = plot_gbm_roc_fi(Xm, ym, Xm, ym, FEAT_NAMES_M, dt_m, auc_m,
    "GBM — Mansplaining: Curva ROC 5-fold y Feature Importance",
    FIG_DIR / "c22_gbm_roc_fi_mansplain.png")
gbm_i = plot_gbm_roc_fi(Xi, yi, Xi, yi, FEAT_NAMES_I, dt_i, auc_i,
    "GBM — Ser Interrumpida (Mujeres): Curva ROC 5-fold y Feature Importance",
    FIG_DIR / "c22_gbm_roc_fi_interrupted.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP — Panel beeswarm + dependence plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 3. SHAP dependency panel ===")
def plot_shap_panel(gbm_model, X, feat_names, title, out_path):
    gbm_model.fit(X, None) if False else None  # already fitted
    explainer = shap.TreeExplainer(gbm_model)
    sv = explainer.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-6:][::-1]  # top 6

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for k, idx in enumerate(top_idx):
        ax = axes[k//3][k%3]
        shap_vals = sv[:, idx].astype(float)
        feat_vals = X[:, idx].astype(float)
        sc = ax.scatter(feat_vals, shap_vals, c=feat_vals,
                        cmap="coolwarm", alpha=0.5, s=20)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        z = np.polyfit(feat_vals, shap_vals, 1)
        xr = np.linspace(feat_vals.min(), feat_vals.max(), 100)
        ax.plot(xr, np.polyval(z, xr), "k-", lw=1.5)
        ax.set_xlabel(feat_names[idx], fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(f"{feat_names[idx]}\n|SHAP|={mean_abs[idx]:.3f}", fontsize=9)
        ax.grid(True, alpha=0.2)
        plt.colorbar(sc, ax=ax, label="Valor feature", pad=0.01)

    plt.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")

plot_shap_panel(gbm_m, Xm, FEAT_NAMES_M,
    "SHAP Dependence — Top 6 features: Mansplaining",
    FIG_DIR / "c22_shap_dependence_mansplain.png")
plot_shap_panel(gbm_i, Xi, FEAT_NAMES_I,
    "SHAP Dependence — Top 6 features: Ser Interrumpida (Mujeres)",
    FIG_DIR / "c22_shap_dependence_interrupted.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Panel síntesis: resumen de todos los modelos
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 4. Model comparison panel ===")
shap_df_m = pd.read_csv(CSV_DIR / "c14_shap_mansplain.csv")
shap_df_i = pd.read_csv(CSV_DIR / "c14_shap_interrupted.csv")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Comparacion de modelos: OLS vs CART vs GBM vs SHAP\n(AUC CART=0.849, GBM=0.879 para Mansplaining | CART=0.967, GBM=0.975 para Interrupcion)",
             fontsize=11, fontweight="bold")

# SHAP mansplaining
ax = axes[0][0]
top = shap_df_m.head(10)
colors = ["#e74c3c"] * len(top)
ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color=colors[::-1], alpha=0.85)
ax.set_title("SHAP |mean| — Mansplaining", fontweight="bold")
ax.set_xlabel("|SHAP|"); ax.grid(True, alpha=0.3, axis="x")

# SHAP interrumpida
ax = axes[0][1]
top = shap_df_i.head(10)
colors = ["#3498db"] * len(top)
ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color=colors[::-1], alpha=0.85)
ax.set_title("SHAP |mean| — Ser Interrumpida (F)", fontweight="bold")
ax.set_xlabel("|SHAP|"); ax.grid(True, alpha=0.3, axis="x")

# AUC barplot
ax = axes[1][0]
models = ["OLS*\n(R²→AUC)", "CART", "GBM"]
auc_ms = [0.60, 0.849, 0.879]
auc_it = [0.85, 0.967, 0.975]
x = np.arange(len(models)); w = 0.35
ax.bar(x-w/2, auc_ms, w, label="Mansplaining", color="#e74c3c", alpha=0.8)
ax.bar(x+w/2, auc_it, w, label="Interrumpida (F)", color="#3498db", alpha=0.8)
ax.axhline(0.8, color="gray", ls="--", lw=1.5, label="AUC=0.80 (bueno)")
ax.axhline(0.9, color="gray", ls=":", lw=1.5, label="AUC=0.90 (excelente)")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("AUC (ROC)"); ax.set_ylim(0.5,1.0)
ax.set_title("AUC por modelo y outcome", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# Markov matrix
ax = axes[1][1]
matrix = np.array([[0.854, 0.146],[0.252, 0.748]])
im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks([0,1]); ax.set_xticklabels(["→ Hombre","→ Mujer"], fontsize=11)
ax.set_yticks([0,1]); ax.set_yticklabels(["Hombre","Mujer"], fontsize=11)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if matrix[i,j]>0.7 else "black")
plt.colorbar(im, ax=ax, label="P(transicion)")
ax.set_title("Markov: Matriz transicion turnos\n(χ²=4362, p<.0001  |  autoagrupacion +0.279)",
             fontweight="bold", fontsize=9)

plt.tight_layout(rect=[0,0,1,0.94])
fig.savefig(FIG_DIR / "c22_synthesis_panel.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: c22_synthesis_panel.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Bayesian posteriors — versión mejorada
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 5. Bayesian enhanced panel ===")
bay = pd.read_csv(CSV_DIR / "c19_bayesian_results.csv")
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e74c3c" if v>0 else "#3498db" for v in bay["post_mean"]]
y_pos  = np.arange(len(bay))
ax.barh(y_pos, bay["post_mean"], color=colors, alpha=0.8, height=0.5)
ax.errorbar(bay["post_mean"], y_pos,
            xerr=[bay["post_mean"]-bay["hdi_lo"], bay["hdi_hi"]-bay["post_mean"]],
            fmt="none", color="black", capsize=5, lw=2)
ax.axvline(0, color="black", lw=1.5, ls="--")
ax.set_yticks(y_pos)
ax.set_yticklabels(bay["label"], fontsize=10)
ax.set_xlabel("Diferencia posterior E(M-F)", fontsize=11)
ax.set_title("Estimacion bayesiana de efectos de genero (MCMC)\nIntervalo = HDI 95% | P(dir.)=1.000 para 4 outcomes",
             fontsize=11, fontweight="bold")
for i, row in bay.iterrows():
    p_dir = max(row["p_M_gt_F"], row["p_F_gt_M"])
    label = f"P={p_dir:.3f}"
    ax.text(row["hdi_hi"] + 0.001, i, label, va="center", fontsize=9)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
fig.savefig(FIG_DIR / "c22_bayesian_forest.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: c22_bayesian_forest.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Session bias heatmap por grupo
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 6. Session bias by group ===")
sdf = pd.read_csv(CSV_DIR / "c16_session_bias.csv")
if "group" in sdf.columns:
    grp_bias = sdf.groupby("group")[["mansplain_gap","interrupt_gap","hedge_gap","lexdiv_gap"]].mean()
    if len(grp_bias) > 1:
        fig, ax = plt.subplots(figsize=(max(8, len(grp_bias.columns)*2.5),
                                        max(4, len(grp_bias)*0.6)))
        im = ax.imshow(grp_bias.values, cmap="RdBu_r", aspect="auto", vmin=-0.1, vmax=0.1)
        ax.set_xticks(range(len(grp_bias.columns)))
        ax.set_xticklabels([c.replace("_gap","").replace("_"," ") for c in grp_bias.columns],
                           rotation=20, ha="right", fontsize=10)
        ax.set_yticks(range(len(grp_bias.index)))
        ax.set_yticklabels(grp_bias.index, fontsize=9)
        for i in range(len(grp_bias.index)):
            for j in range(len(grp_bias.columns)):
                v = grp_bias.values[i,j]
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v)>0.07 else "black")
        plt.colorbar(im, ax=ax, label="Gap (M-F)")
        ax.set_title("Gap de genero (M-F) por grupo tematico — nivel sesion",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "c22_session_bias_by_group.png", dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  Saved: c22_session_bias_by_group.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXCEL ENRICHMENT — añadir columnas enriquecidas a cada Excel de sesion
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 7. Enriching Excel session files ===")

# Construir lookup speaker→meta desde user_level_enriched
ENRICH_COLS = ["Role","Group","Country","Specialty (ICU-Ane-Both)",
               "Number of citations","Year of qualification (specialty)",
               "Affiliation (Hospital)","cluster",
               "n_interventions","mean_duration","mean_wpm","mean_lexical_diversity",
               "pct_is_mansplaining","pct_interrupted_by_next","pct_has_hedge",
               "pct_has_disagreement","pct_has_overlap","pct_interrupts_previous",
               "is_top3_speaker","is_moderator","is_speaker","is_public"]

# Limpiar nombre de sesion para matcheo
udf["_session_clean"] = (udf["session"]
    .str.replace(".csv","",regex=False)
    .str.replace("_report","",regex=False)
    .str.strip())

excel_files = sorted(EXCEL_DIR.glob("*.xlsx"))
n_enriched, n_skip = 0, 0

for xf in excel_files:
    sess_name = xf.stem.replace("_report","").strip()
    # Match contra session en udf
    sub = udf[udf["_session_clean"].str.lower().str.replace(" ","_") ==
              sess_name.lower().replace(" ","_")]
    if sub.empty:
        # intento fuzzy: contiene el nombre sin extension
        sub = udf[udf["_session_clean"].str.lower().str.contains(
            sess_name[:20].lower().replace(" ","_").replace("-",""), na=False)]

    try:
        raw = pd.read_excel(xf)
    except Exception as e:
        print(f"  SKIP (read error) {xf.name}: {e}")
        n_skip += 1; continue

    if sub.empty:
        # Sin match, escribe igual
        n_skip += 1; continue

    # Merge por speaker
    meta = sub[["speaker"] + [c for c in ENRICH_COLS if c in sub.columns]].copy()
    meta = meta.rename(columns={"Specialty (ICU-Ane-Both)":"Specialty"})
    enriched = raw.merge(meta, on="speaker", how="left")

    # Guardar sobreescribiendo
    enriched.to_excel(xf, index=False, sheet_name="Report")
    n_enriched += 1

print(f"  Enriquecidos: {n_enriched}  |  Sin match/omitidos: {n_skip}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CSV RESUMEN — exportar master enriched CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 8. Master summary CSV ===")
summary_cols = [
    "session","speaker","gender","Role","Group","Specialty (ICU-Ane-Both)",
    "Country","Number of citations","cluster",
    "n_interventions","mean_duration","mean_wpm","mean_lexical_diversity",
    "pct_is_mansplaining","pct_interrupted_by_next","pct_has_hedge",
    "pct_has_disagreement","pct_has_overlap","is_top3_speaker",
    "is_moderator","is_speaker","log_citations",
]
master = udf[[c for c in summary_cols if c in udf.columns]].copy()
master.to_csv(CSV_DIR / "master_enriched_summary.csv", index=False)
print(f"  Saved: master_enriched_summary.csv  ({len(master)} rows)")

print("\n=== DONE c22_enrich_excel_and_graphs.py ===")
