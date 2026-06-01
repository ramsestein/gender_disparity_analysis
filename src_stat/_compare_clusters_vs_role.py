"""
_compare_clusters_vs_role.py
=============================
Compara el clustering no supervisado previo (Cluster 0 = Moderadores/Panelistas,
Cluster 1 = Público/Audiencia) con el nuevo campo 'Role' obtenido del Excel
(Speaker, Moderator, public / NaN = no identificado).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

BASE      = Path(__file__).resolve().parent.parent
CSV_DIR   = BASE / "final_reports" / "resultados" / "csv"
ENRICHED  = BASE / "final_reports" / "csv_enriched"

# ── cargar clustering previo ──────────────────────────────────────────────────
clusters = pd.read_csv(CSV_DIR / "user_level_with_clusters.csv")
# session viene con sufijo ".csv" → quitarlo para que coincida con csv_enriched stems
clusters["session"] = clusters["session"].str.removesuffix(".csv")
clusters = clusters[["session", "speaker", "cluster_kmeans", "gender"]].copy()

# ── cargar el Role de los CSVs enriquecidos ───────────────────────────────────
role_rows = []
for csv_path in sorted(ENRICHED.glob("*.csv")):
    if csv_path.name in ("match_log.csv", "public_intros.csv",
                          "scholar_results.csv", "unmatched_intros.csv",
                          "scholar_cache.json", "scholar_run.log"):
        continue
    df = pd.read_csv(csv_path)
    if "Role" not in df.columns:
        continue
    stem = csv_path.stem
    for speaker, grp in df.groupby("speaker"):
        role_val = grp["Role"].dropna().iloc[0] if grp["Role"].notna().any() else None
        role_rows.append({"session": stem, "speaker": speaker, "Role": role_val})

roles = pd.DataFrame(role_rows)

# ── merge ─────────────────────────────────────────────────────────────────────
merged = clusters.merge(roles, on=["session", "speaker"], how="left")
total  = len(merged)

print(f"Usuarios totales en clustering: {total}")
print(f"Con Role asignado: {merged['Role'].notna().sum()} ({merged['Role'].notna().mean()*100:.1f}%)")
print(f"Sin Role (no identificados): {merged['Role'].isna().sum()}")
print()
print("Distribución de Role:")
print(merged["Role"].value_counts(dropna=False).to_string())
print()
print("Distribución de cluster_kmeans:")
print(merged["cluster_kmeans"].value_counts().to_string())

# ── mapear Role a binario (0=panelista, 1=publico) ───────────────────────────
# Moderator/Speaker -> 0 (Panelistas)  |  public -> 1 (Público)  |  NaN -> NaN
def role_to_binary(r):
    if pd.isna(r):
        return np.nan
    r = str(r).strip().lower()
    if r in ("moderator", "speaker"):
        return 0
    if r == "public":
        return 1
    return np.nan

merged["role_binary"] = merged["Role"].apply(role_to_binary)

# ── sólo sobre los que tienen role conocido ───────────────────────────────────
known = merged[merged["role_binary"].notna()].copy()
known["role_binary"] = known["role_binary"].astype(int)

print(f"\n{'='*60}")
print(f"Análisis sobre {len(known)} usuarios con Role conocido")
print(f"{'='*60}")

# Tabla de contingencia
ct = pd.crosstab(
    known["cluster_kmeans"],
    known["role_binary"],
    rownames=["Cluster (0=Panelistas,1=Público)"],
    colnames=["Role (0=Panel,1=Public)"]
)
ct.columns = ["Panel (Role)", "Public (Role)"]
ct.index   = ["Cluster 0 (Moderadores)", "Cluster 1 (Público)"]
print("\nTabla de contingencia:")
print(ct.to_string())

# Porcentajes por cluster
ct_pct = pd.crosstab(
    known["cluster_kmeans"],
    known["role_binary"],
    normalize="index"
) * 100
ct_pct.columns = ["Panel (Role) %", "Public (Role) %"]
ct_pct.index   = ["Cluster 0 (Moderadores)", "Cluster 1 (Público)"]
print("\nPorcentajes (por fila):")
print(ct_pct.round(1).to_string())

# Métricas de acuerdo
y_true = known["role_binary"].values
y_pred = known["cluster_kmeans"].values

# Probar los dos mapeos posibles (cluster 0 = panel ó cluster 0 = publico)
acc_direct  = (y_true == y_pred).mean()
acc_inverted = (y_true == (1 - y_pred)).mean()

if acc_direct >= acc_inverted:
    print(f"\nMejor alineación: Cluster 0 → Panel, Cluster 1 → Público")
    y_pred_aligned = y_pred
    acc = acc_direct
else:
    print(f"\nMejor alineación: Cluster 0 → Público, Cluster 1 → Panel (inversión)")
    y_pred_aligned = 1 - y_pred
    acc = acc_inverted

kappa = cohen_kappa_score(y_true, y_pred_aligned)
print(f"\nAcuerdo global (accuracy):  {acc*100:.1f}%")
print(f"Cohen's kappa:              {kappa:.4f}  ", end="")
if kappa >= 0.8:
    print("(muy bueno)")
elif kappa >= 0.6:
    print("(bueno)")
elif kappa >= 0.4:
    print("(moderado)")
else:
    print("(bajo)")

print("\nClassification report:")
print(classification_report(y_true, y_pred_aligned,
                             target_names=["Panel", "Público"],
                             zero_division=0))

# ── casos sin role (mayoría de usuarios) ─────────────────────────────────────
unknown = merged[merged["role_binary"].isna()]
print(f"\n{'='*60}")
print(f"Usuarios SIN role identificado: {len(unknown)}")
print(f"Distribución en clusters:")
print(unknown["cluster_kmeans"].value_counts().to_string())
print(f"\n  → El {unknown[unknown['cluster_kmeans']==1].shape[0]/len(unknown)*100:.1f}% de los no identificados")
print(f"     están en Cluster 1 (el esperado para 'público')")
