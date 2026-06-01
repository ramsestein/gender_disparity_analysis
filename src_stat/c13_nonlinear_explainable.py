"""
c13_nonlinear_explainable.py
=============================
Sistemas NO-LINEALES EXPLICABLES para predecir:
  1. Mansplaining alto (top quartil)
  2. Ser interrumpida frecuentemente (top quartil, solo mujeres)

Metodos:
  A) Decision Tree CART (max_depth=4)
     - Reglas IF-THEN explicitas en texto
     - Feature importance

  B) Gradient Boosting (GBM)
     - Feature importance (MDI)
     - Partial Dependence para top 3 variables

  C) MILP: Best-Subset L1 Regression con PuLP
     Formulacion:
       min  (1/n) * sum_i e_i           <- error L1 medio
       s.t. e_i >= y_i - (X*beta + b)  para todo i
            e_i >= -(y_i - (X*beta + b)) para todo i
            e_i >= 0
            -M*z_j <= beta_j <= M*z_j   (big-M: beta_j=0 si z_j=0)
            sum_j z_j <= K              (a lo sumo K features)
            z_j in {0,1}
     Resuelve la combinacion optima de K variables que minimizan el error L1.

  D) Comparacion de consistencia: que variables aparecen en todos los metodos

Salida:
  csv/c13_cart_importance.csv
  csv/c13_gbm_importance.csv
  csv/c13_milp_selected_features.csv
  graficos/c13_*.png
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pulp

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos"

# ── Cargar datos ──────────────────────────────────────────────────────────────
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

CONTINUOUS = [
    "n_interventions", "mean_duration", "std_duration", "total_duration",
    "mean_wpm", "mean_lexical_diversity", "mean_latency_s", "mean_overlap_duration",
    "mean_echoing_score", "mean_assertiveness_score", "mean_conflict_score",
    "pct_has_hedge", "pct_has_agreement", "pct_has_disagreement",
    "pct_has_courtesy", "pct_has_apology", "pct_is_question",
    "pct_has_vulnerability", "pct_is_backchannel", "pct_interrupts_previous",
    "min_turn_number", "is_top3_speaker", "log_cit",
]
BINARY = ["gender_bin", "is_moderator", "is_speaker", "is_public",
          "is_ICU", "is_BOTH", "is_ANE", "cluster_bin"] + grp_cols

FEAT_NAMES_RAW = list(dict.fromkeys([c for c in CONTINUOUS + BINARY if c in df.columns]))
PRETTY = {
    "gender_bin": "Genero (H)", "is_moderator": "Rol:Moderador",
    "is_speaker": "Rol:Speaker", "is_public": "Rol:Publico",
    "is_ICU": "ICU", "is_BOTH": "Both", "is_ANE": "ANE",
    "cluster_bin": "Cluster:audiencia", "log_cit": "Log(citaciones)",
    "n_interventions": "N_interv", "mean_duration": "Dur_media",
    "std_duration": "SD_dur", "total_duration": "Dur_total",
    "mean_wpm": "WPM", "mean_lexical_diversity": "Div_lex",
    "mean_latency_s": "Latencia", "mean_overlap_duration": "Solap",
    "mean_echoing_score": "Echoing", "mean_assertiveness_score": "Asert",
    "mean_conflict_score": "Conflicto", "pct_has_hedge": "Hedge",
    "pct_has_agreement": "Acuerdo", "pct_has_disagreement": "Desacuerdo",
    "pct_has_courtesy": "Cortesia", "pct_has_apology": "Disculpa",
    "pct_is_question": "Preguntas", "pct_has_vulnerability": "Vulnerab",
    "pct_is_backchannel": "Backchannel", "pct_interrupts_previous": "Interrumpe",
    "min_turn_number": "TurnoInicio", "is_top3_speaker": "Top3",
}
def lbl(c): return PRETTY.get(c, c.replace("grp_","G:").replace("_"," ")[:20])

# ── Preparar outcomes binarios ────────────────────────────────────────────────
def binarize(series, q=0.75):
    """Top quartil = 1 (alto), resto = 0 (bajo/normal)"""
    thr = series.quantile(q)
    return (series > thr).astype(int)

# Outcome 1: mansplaining alto (n=652)
sub1 = df[FEAT_NAMES_RAW + ["pct_is_mansplaining"]].dropna()
X1   = sub1[FEAT_NAMES_RAW].values
y1   = binarize(sub1["pct_is_mansplaining"])
feat_names = [lbl(c) for c in FEAT_NAMES_RAW]

# Outcome 2: ser interrumpida (solo mujeres, n≈254)
df_w = df[df["gender"] == "female"].copy()
feat2 = [c for c in FEAT_NAMES_RAW if c != "gender_bin"]
sub2 = df_w[feat2 + ["pct_interrupted_by_next"]].dropna()
X2   = sub2[feat2].values
y2   = binarize(sub2["pct_interrupted_by_next"])
feat_names2 = [lbl(c) for c in feat2]

print(f"Outcome 1: Mansplaining alto  n={len(sub1)}  positivos={y1.sum()} ({y1.mean()*100:.1f}%)")
print(f"Outcome 2: Interrumpida (F)   n={len(sub2)}  positivos={y2.sum()} ({y2.mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# FUNCION MILP: Best-Subset L1 Regression con PuLP
# ══════════════════════════════════════════════════════════════════════════════
def milp_best_subset(X_std, y_bin, feature_names, K=6, M=5.0, label="", max_n=300):
    """
    MILP: min L1 error con a lo sumo K features activas.
    Devuelve (features_seleccionadas, betas, MAE_MILP).
    """
    y = y_bin.astype(float)
    # Submuestro BALANCEADO: igual n+ y n- (critico para clases muy desequilibradas)
    rng = np.random.default_rng(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_each  = min(len(pos_idx), len(neg_idx), max_n // 2)
    pos_sel = rng.choice(pos_idx, n_each, replace=False)
    neg_sel = rng.choice(neg_idx, n_each, replace=False)
    idx     = np.concatenate([pos_sel, neg_sel])
    X_s, y_s = X_std[idx], y[idx]
    n, p = X_s.shape

    print(f"\n  MILP {label}: n={n}  p={p}  K={K}  M={M}")
    prob = pulp.LpProblem(f"BestSubset_{label}", pulp.LpMinimize)

    # Variables: error L1, coeficientes beta, seleccion binaria z, intercepto b
    e = [pulp.LpVariable(f"e_{i}", lowBound=0) for i in range(n)]
    beta = [pulp.LpVariable(f"b_{j}", lowBound=-M, upBound=M) for j in range(p)]
    z    = [pulp.LpVariable(f"z_{j}", cat="Binary") for j in range(p)]
    b0   = pulp.LpVariable("b0", lowBound=-2, upBound=2)

    # Objetivo: minimizar error L1 medio
    prob += pulp.lpSum(e) / n

    # Restricciones de error L1
    for i in range(n):
        pred = pulp.lpSum(beta[j] * float(X_s[i, j]) for j in range(p)) + b0
        prob += e[i] >= y_s[i] - pred
        prob += e[i] >= -(y_s[i] - pred)

    # Big-M: beta_j = 0 si z_j = 0
    for j in range(p):
        prob += beta[j] <= M * z[j]
        prob += beta[j] >= -M * z[j]

    # A lo sumo K features
    prob += pulp.lpSum(z) <= K

    # Resolver
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=90)
    status = prob.solve(solver)
    print(f"  Status: {pulp.LpStatus[prob.status]}  MAE={pulp.value(prob.objective):.4f}")

    selected, betas_out = [], []
    for j in range(p):
        zv = pulp.value(z[j])
        bv = pulp.value(beta[j])
        if zv is not None and zv > 0.5:
            selected.append((feature_names[j], round(bv, 4)))
            betas_out.append(bv)

    intercept = pulp.value(b0)
    mae = pulp.value(prob.objective)
    return selected, intercept, mae

# ══════════════════════════════════════════════════════════════════════════════
# A) DECISION TREE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  A) DECISION TREE (CART, max_depth=4)")
print("="*65)

results_cart = []
for X, y, fn, label, fname_suf in [
    (X1, y1, feat_names,  "Mansplaining",   "mansplain"),
    (X2, y2, feat_names2, "Interrumpida(F)", "interrupted"),
]:
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    cart = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15,
                                  class_weight="balanced", random_state=42)
    cart.fit(X_std, y)
    cv_acc = cross_val_score(cart, X_std, y, cv=5, scoring="roc_auc").mean()
    print(f"\n  [{label}]  AUC-CV={cv_acc:.3f}")

    # Reglas IF-THEN
    rules = export_text(cart, feature_names=fn, max_depth=4, spacing=2)
    print(f"\n  Reglas (top ramas):\n{rules[:1500]}")

    # Feature importance
    imp = pd.Series(cart.feature_importances_, index=fn).sort_values(ascending=False)
    top_imp = imp[imp > 0].head(10)
    print(f"\n  Top 10 features por importancia Gini:")
    for feat, val in top_imp.items():
        print(f"    {feat:<28} {val:.4f}")

    results_cart.append({"label": label, "imp": imp, "cv_auc": cv_acc,
                          "fn": fn, "cart": cart, "X_std": X_std, "y": y})

    # Guardar
    imp.reset_index().rename(columns={"index":"feature",0:"importance_gini"}).to_csv(
        CSV_DIR / f"c13_cart_{fname_suf}.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# B) GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  B) GRADIENT BOOSTING (GBM)")
print("="*65)

results_gbm = []
for X, y, fn, label, fname_suf in [
    (X1, y1, feat_names,  "Mansplaining",   "mansplain"),
    (X2, y2, feat_names2, "Interrumpida(F)", "interrupted"),
]:
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
    gbm.fit(X_std, y)
    cv_auc = cross_val_score(gbm, X_std, y, cv=5, scoring="roc_auc").mean()
    print(f"\n  [{label}]  AUC-CV={cv_auc:.3f}")

    imp = pd.Series(gbm.feature_importances_, index=fn).sort_values(ascending=False)
    top = imp.head(10)
    print(f"  Top 10 features GBM:")
    for f, v in top.items():
        print(f"    {f:<28} {v:.4f}")

    results_gbm.append({"label": label, "imp": imp, "cv_auc": cv_auc,
                         "fn": fn, "gbm": gbm, "X_std": X_std, "y": y})
    imp.reset_index().rename(columns={"index":"feature",0:"importance_gbm"}).to_csv(
        CSV_DIR / f"c13_gbm_{fname_suf}.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# C) MILP: Best-Subset L1 Regression
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  C) MILP — Best Subset L1 Regression (K=6 features)")
print("="*65)

milp_results = []
scaler1 = StandardScaler(); X1_std = scaler1.fit_transform(X1)
scaler2 = StandardScaler(); X2_std = scaler2.fit_transform(X2)

for X_std, y, fn, label in [
    (X1_std, y1.values, feat_names,  "Mansplaining"),
    (X2_std, y2.values, feat_names2, "Interrumpida(F)"),
]:
    sel, intercept, mae = milp_best_subset(X_std, y, fn, K=6, label=label)
    print(f"\n  [{label}] Intercept={intercept:.4f}  MAE={mae:.4f}")
    print(f"  Features MILP seleccionadas (K={len(sel)}):")
    for feat, beta in sorted(sel, key=lambda x: abs(x[1]), reverse=True):
        direction = "AUMENTA" if beta > 0 else "REDUCE"
        print(f"    {direction:7s}  {feat:<28}  beta={beta:+.4f}")
    milp_results.append({"label": label, "selected": sel, "intercept": intercept, "mae": mae})

# Guardar MILP
milp_rows = []
for r in milp_results:
    for feat, beta in r["selected"]:
        milp_rows.append({"outcome": r["label"], "feature": feat, "beta": beta,
                          "direction": "AUMENTA" if beta > 0 else "REDUCE"})
pd.DataFrame(milp_rows).to_csv(CSV_DIR / "c13_milp_selected.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# D) CONSISTENCIA: features que aparecen en CART + GBM + MILP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  D) CONSISTENCIA ENTRE METODOS")
print("="*65)

for i, label in enumerate(["Mansplaining", "Interrumpida(F)"]):
    cart_top = set(results_cart[i]["imp"].head(8).index)
    gbm_top  = set(results_gbm[i]["imp"].head(8).index)
    milp_sel = set(f for f, b in milp_results[i]["selected"])
    consensus = cart_top & gbm_top & milp_sel
    union2    = (cart_top & gbm_top) | (cart_top & milp_sel) | (gbm_top & milp_sel)
    print(f"\n  [{label}]")
    print(f"    Consenso total (CART+GBM+MILP): {sorted(consensus)}")
    print(f"    Consenso 2/3 metodos:           {sorted(union2)}")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFICOS
# ══════════════════════════════════════════════════════════════════════════════

# --- Decision Tree plot ---
for rc, fname_suf, label in [
    (results_cart[0], "mansplain",   "Mansplaining"),
    (results_cart[1], "interrupted", "Interrumpida (mujeres)"),
]:
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(rc["cart"], feature_names=rc["fn"], class_names=["Bajo","Alto"],
              filled=True, rounded=True, fontsize=8, ax=ax, max_depth=3,
              impurity=False, proportion=True)
    ax.set_title(f"Decision Tree — {label}  (AUC-CV={rc['cv_auc']:.3f})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"c13_01_tree_{fname_suf}.png", dpi=180, bbox_inches="tight")
    plt.close()

# --- Feature importance comparison (CART vs GBM) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Feature Importance: CART vs GBM (top 15 por importancia)",
             fontsize=12, fontweight="bold")

for ax, (rc, rg), label in zip(axes,
    [(results_cart[0], results_gbm[0]), (results_cart[1], results_gbm[1])],
    ["Mansplaining", "Interrumpida (mujeres)"]):

    fn = rc["fn"]
    combined = pd.DataFrame({
        "CART": rc["imp"],
        "GBM":  rg["imp"],
    }).fillna(0)
    combined["max_imp"] = combined.max(axis=1)
    top = combined.sort_values("max_imp", ascending=True).tail(15)

    x    = np.arange(len(top))
    width = 0.38
    ax.barh(x - width/2, top["CART"], width, label="CART",   color="#3498db", alpha=0.8)
    ax.barh(x + width/2, top["GBM"],  width, label="GBM",    color="#e67e22", alpha=0.8)
    ax.set_yticks(x); ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Importancia (Gini / MDI)", fontsize=10)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(FIG_DIR / "c13_02_importance_cart_gbm.png", dpi=200, bbox_inches="tight")
plt.close()

# --- MILP: seleccion optima ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MILP — Variables optimas K=6 (Best-Subset L1)\nbeta>0 AUMENTA outcome  |  beta<0 REDUCE outcome",
             fontsize=11, fontweight="bold")

for ax, r, color_pos, color_neg in zip(axes, milp_results,
    ["#e74c3c","#9b59b6"], ["#7f8c8d","#2ecc71"]):
    if not r["selected"]: continue
    sel_sorted = sorted(r["selected"], key=lambda x: x[1])
    feats  = [s[0] for s in sel_sorted]
    betas  = [s[1] for s in sel_sorted]
    colors = [color_pos if b > 0 else color_neg for b in betas]
    bars   = ax.barh(feats, betas, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=1.2)
    for bar, b in zip(bars, betas):
        ax.text(b + 0.02*np.sign(b), bar.get_y() + bar.get_height()/2,
                f"{b:+.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Coeficiente MILP (L1 estandarizado)", fontsize=10)
    ax.set_title(f"{r['label']}  MAE={r['mae']:.4f}", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout(rect=[0,0,1,0.93])
fig.savefig(FIG_DIR / "c13_03_milp_selection.png", dpi=200, bbox_inches="tight")
plt.close()

# --- Partial Dependence: top 3 GBM para mansplaining ---
rc_gbm = results_gbm[0]
top3_idx = rc_gbm["imp"].argsort()[-3:][::-1].values.tolist()  # indices top 3
try:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Partial Dependence Plot — GBM Mansplaining (top 3 features)",
                 fontsize=11, fontweight="bold")
    disp = PartialDependenceDisplay.from_estimator(
        rc_gbm["gbm"], rc_gbm["X_std"], features=top3_idx,
        feature_names=rc_gbm["fn"], ax=ax, n_cols=3,
        pd_line_kw={"color": "#e74c3c", "linewidth": 2.5})
    plt.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(FIG_DIR / "c13_04_pdp_gbm_mansplain.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Guardado: c13_04_pdp_gbm_mansplain.png")
except Exception as e:
    print(f"  PDP: {e}")

print(f"\n  Figuras: {FIG_DIR}/c13_*.png")
print(f"  CSVs:    {CSV_DIR}/c13_*.csv")
print("\n  AUC-CV resumen:")
for rc, rg in [(results_cart[0], results_gbm[0]), (results_cart[1], results_gbm[1])]:
    print(f"    {rc['label']}: CART={rc['cv_auc']:.3f}  GBM={rg['cv_auc']:.3f}")
