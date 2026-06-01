"""c14_shap.py — SHAP values para GBM (mansplaining + interrumpida)"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import shap

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"

df = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
df["gender_bin"]   = (df["gender"] == "male").astype(float)
df["is_moderator"] = (df["Role"] == "Moderator").astype(float) if "Role" in df.columns else 0.0
df["is_speaker"]   = (df["Role"] == "Speaker").astype(float)   if "Role" in df.columns else 0.0
df["is_public"]    = (df["Role"] == "public").astype(float)    if "Role" in df.columns else 0.0
spec = df.get("Specialty (ICU-Ane-Both)", pd.Series("", index=df.index)).fillna("").str.upper()
df["is_ICU"]  = (spec == "ICU").astype(float)
df["is_BOTH"] = (spec == "BOTH").astype(float)
df["is_ANE"]  = (spec == "ANE").astype(float)
df["log_cit"]     = df.get("log_citations", pd.Series(0, index=df.index)).fillna(0)
df["cluster_bin"] = df.get("cluster", pd.Series(0, index=df.index)).astype(float)
grp_cols = [c for c in df.columns if c.startswith("grp_")]

CONTINUOUS = ["n_interventions","mean_duration","std_duration","total_duration",
    "mean_wpm","mean_lexical_diversity","mean_latency_s","mean_overlap_duration",
    "mean_echoing_score","mean_assertiveness_score","mean_conflict_score",
    "pct_has_hedge","pct_has_agreement","pct_has_disagreement","pct_has_courtesy",
    "pct_has_apology","pct_is_question","pct_has_vulnerability","pct_is_backchannel",
    "pct_interrupts_previous","min_turn_number","is_top3_speaker","log_cit"]
BINARY = ["gender_bin","is_moderator","is_speaker","is_public","is_ICU","is_BOTH","is_ANE","cluster_bin"] + grp_cols
FEAT_NAMES_RAW = list(dict.fromkeys([c for c in CONTINUOUS + BINARY if c in df.columns]))
PRETTY = {"gender_bin":"Genero(H)","is_moderator":"Moderador","is_speaker":"Speaker",
    "is_public":"Publico","is_ICU":"ICU","is_BOTH":"Both","is_ANE":"ANE",
    "cluster_bin":"Cluster","log_cit":"Log(cit)","n_interventions":"N_interv",
    "mean_duration":"Dur_media","std_duration":"SD_dur","total_duration":"Dur_total",
    "mean_wpm":"WPM","mean_lexical_diversity":"Div_lex","mean_latency_s":"Latencia",
    "mean_overlap_duration":"Solap","mean_echoing_score":"Echoing",
    "mean_assertiveness_score":"Asert","mean_conflict_score":"Conflicto",
    "pct_has_hedge":"Hedge","pct_has_agreement":"Acuerdo","pct_has_disagreement":"Desacuerdo",
    "pct_has_courtesy":"Cortesia","pct_has_apology":"Disculpa","pct_is_question":"Preguntas",
    "pct_has_vulnerability":"Vulnerab","pct_is_backchannel":"Backchannel",
    "pct_interrupts_previous":"Interrumpe","min_turn_number":"TurnoInicio","is_top3_speaker":"Top3"}
def lbl(c): return PRETTY.get(c, c.replace("grp_","G:").replace("_"," ")[:20])
feat_names = [lbl(c) for c in FEAT_NAMES_RAW]

def binarize(s, q=0.75): return (s > s.quantile(q)).astype(int)

results = []
for outcome, feat_raw, fn_list, label, df_model, fname in [
    ("pct_is_mansplaining",    FEAT_NAMES_RAW,                              feat_names,
     "Mansplaining",           df,                                           "mansplain"),
    ("pct_interrupted_by_next",[c for c in FEAT_NAMES_RAW if c!="gender_bin"], [lbl(c) for c in FEAT_NAMES_RAW if c!="gender_bin"],
     "Interrumpida(F)",        df[df["gender"]=="female"].copy(),            "interrupted"),
]:
    sub  = df_model[[outcome]+feat_raw].dropna()
    X    = sub[feat_raw].values
    y    = binarize(sub[outcome]).values
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
    gbm.fit(X_std, y)

    # SHAP TreeExplainer
    explainer   = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(X_std)   # shape (n, p) for class-1

    # ── Global importance ──
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df   = pd.DataFrame({"feature": fn_list, "mean_abs_shap": mean_abs})
    imp_df   = imp_df.sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(CSV_DIR / f"c14_shap_{fname}.csv", index=False)

    print(f"\n[{label}]  Top 10 SHAP:")
    for _, r in imp_df.head(10).iterrows():
        print(f"  {r['feature']:<28}  SHAP={r['mean_abs_shap']:.4f}")

    # ── Beeswarm (summary plot) ──
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_std, feature_names=fn_list,
                      max_display=15, show=False, plot_type="dot")
    plt.title(f"SHAP Beeswarm — {label}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"c14_01_shap_beeswarm_{fname}.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ── Bar global importance ──
    fig, ax = plt.subplots(figsize=(8, 6))
    top = imp_df.head(15).sort_values("mean_abs_shap")
    ax.barh(top["feature"], top["mean_abs_shap"], color="#3498db", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"SHAP Global Importance — {label}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"c14_02_shap_bar_{fname}.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ── Waterfall para el caso de mayor probabilidad de outcome ──
    proba   = gbm.predict_proba(X_std)[:, 1]
    top_idx = np.argmax(proba)
    expl_single = shap.Explanation(
        values     = shap_values[top_idx],
        base_values= explainer.expected_value,
        data       = X_std[top_idx],
        feature_names = fn_list
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(expl_single, max_display=12, show=False)
    plt.title(f"SHAP Waterfall — caso extremo {label} (prob={proba[top_idx]:.3f})",
              fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"c14_03_shap_waterfall_{fname}.png", dpi=180, bbox_inches="tight")
    plt.close()

    results.append({"label": label, "imp_df": imp_df})
    print(f"  Guardado: c14_*_{fname}.png")

print("\nDone c14_shap.py")
