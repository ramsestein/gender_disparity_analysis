"""
c05_gender_name_validation.py
==============================
Valida la asignación de género por voz comparándola con el género inferido
del nombre propio de cada speaker identificado (matched_person).

- Fuente de género asignado : columna 'gender' en csv_enriched (por voz/diarización)
- Fuente de género por nombre: gender_guesser (primer nombre de matched_person)
- Se ignoran: unknowns (sin matched_person) y nombres con género ambiguo/desconocido

Salida:
  final_reports/resultados/csv/c05_gender_name_validation.csv
  final_reports/resultados/graficos/c05_gender_confusion.png
"""

import unicodedata
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import gender_guesser.detector as gd

BASE     = Path(__file__).resolve().parent.parent
ENRICHED = BASE / "final_reports" / "csv_enriched"
LOG_PATH = ENRICHED / "match_log.csv"
OUT_CSV  = BASE / "final_reports" / "resultados" / "csv" / "c05_gender_name_validation.csv"
OUT_FIG  = BASE / "final_reports" / "resultados" / "graficos" / "c05_gender_confusion.png"

SKIP_FILES = {
    "match_log.csv", "public_intros.csv", "unmatched_intros.csv",
    "scholar_results.csv",
}

detector = gd.Detector(case_sensitive=False)

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def first_name(full: str) -> str:
    """Extrae el primer nombre (ignora títulos Dr./Prof.)."""
    parts = full.strip().split()
    # Quitar prefijos de título
    while parts and parts[0].lower().rstrip('.') in ('dr', 'prof', 'professor', 'mr', 'mrs', 'ms'):
        parts = parts[1:]
    return parts[0] if parts else ""

def infer_gender_from_name(name: str) -> str:
    """Devuelve 'male', 'female' o '' (ambiguo/desconocido)."""
    fn = first_name(name)
    if not fn:
        return ""
    # Intentar con acentos y sin acentos
    for candidate in [fn, strip_accents(fn)]:
        result = detector.get_gender(candidate)
        if result in ("male", "female"):
            return result
        if result == "mostly_male":
            return "male"
        if result == "mostly_female":
            return "female"
    return ""   # andy / unknown → ignorar

# ── Cargar datos ──────────────────────────────────────────────────────────────
log = pd.read_csv(LOG_PATH)

# Obtener género asignado por voz de cada (session, speaker)
rows = []
for csv_path in sorted(ENRICHED.glob("*.csv")):
    if csv_path.name in SKIP_FILES:
        continue
    df = pd.read_csv(csv_path)
    if "matched_person" not in df.columns or "gender" not in df.columns:
        continue
    stem = csv_path.stem
    for speaker, grp in df.groupby("speaker"):
        mp = grp["matched_person"].dropna()
        if mp.empty:
            continue
        person_name   = str(mp.iloc[0]).strip()
        voice_gender  = grp["gender"].dropna().iloc[0] if grp["gender"].notna().any() else None
        role          = grp["Role"].dropna().iloc[0] if "Role" in grp.columns and grp["Role"].notna().any() else None
        if voice_gender in (None, "unknown", ""):
            continue
        rows.append({
            "session":       stem,
            "speaker":       speaker,
            "matched_person": person_name,
            "voice_gender":  voice_gender,
            "role":          role,
        })

data = pd.DataFrame(rows)
print(f"Speakers identificados con género por voz: {len(data)}")

# ── Inferir género por nombre ─────────────────────────────────────────────────
data["name_gender"] = data["matched_person"].apply(infer_gender_from_name)

# Descartar ambiguos / desconocidos
data_known = data[data["name_gender"] != ""].copy()
data_ambig = data[data["name_gender"] == ""].copy()

print(f"Con género claro por nombre: {len(data_known)}")
print(f"Nombre ambiguo/desconocido (ignorados): {len(data_ambig)}")
if len(data_ambig):
    print(f"  Ejemplos ambiguos: {data_ambig['matched_person'].unique()[:10].tolist()}")

# ── Comparación ───────────────────────────────────────────────────────────────
data_known["match"] = data_known["voice_gender"] == data_known["name_gender"]
n_total   = len(data_known)
n_correct = data_known["match"].sum()
n_wrong   = (~data_known["match"]).sum()
accuracy  = n_correct / n_total * 100

print(f"\n{'='*55}")
print(f"  RESULTADOS DE VALIDACIÓN")
print(f"{'='*55}")
print(f"  Total comparables : {n_total}")
print(f"  Correctos         : {n_correct} ({accuracy:.1f}%)")
print(f"  Incorrectos       : {n_wrong}  ({100-accuracy:.1f}%)")

# ── Casos incorrectos ─────────────────────────────────────────────────────────
wrong = data_known[~data_known["match"]].copy()
if len(wrong):
    print(f"\n  Discordancias (voz vs nombre):")
    print(wrong[["session","speaker","matched_person",
                 "voice_gender","name_gender","role"]].to_string(index=False))

# ── Tabla de confusión ────────────────────────────────────────────────────────
ct = pd.crosstab(
    data_known["name_gender"],
    data_known["voice_gender"],
    rownames=["Nombre (real)"],
    colnames=["Voz (asignado)"]
)
print(f"\n  Tabla de confusion (filas=nombre, columnas=voz):")
print(ct.to_string())

# ── Precisión por género ──────────────────────────────────────────────────────
for g in ["female", "male"]:
    sub  = data_known[data_known["name_gender"] == g]
    acc  = (sub["voice_gender"] == g).mean() * 100
    print(f"  Precision {g:6s}: {acc:.1f}%  (n={len(sub)})")

# ── Guardar CSV ───────────────────────────────────────────────────────────────
data_known.to_csv(OUT_CSV, index=False)
print(f"\n  Guardado en: {OUT_CSV}")

# ── Gráfico: confusion matrix visual ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Validación de género: nombre vs. voz/diarización", fontsize=14, fontweight='bold')

# Heatmap
ax = axes[0]
ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
labels  = ct_norm.columns.tolist()
im = ax.imshow(ct_norm.values, cmap='Blues', vmin=0, vmax=100)
ax.set_xticks(range(len(ct_norm.columns)))
ax.set_xticklabels([f"Voz: {c}" for c in ct_norm.columns], fontsize=11)
ax.set_yticks(range(len(ct_norm.index)))
ax.set_yticklabels([f"Nombre: {r}" for r in ct_norm.index], fontsize=11)
ax.set_title("Matriz de confusión (% por fila)", fontsize=12, fontweight='bold')
for i in range(len(ct_norm.index)):
    for j in range(len(ct_norm.columns)):
        n   = ct.values[i, j]
        pct = ct_norm.values[i, j]
        ax.text(j, i, f"{pct:.1f}%\n(n={n})",
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white' if pct > 60 else 'black')
plt.colorbar(im, ax=ax, label='% (normalizado por fila)')

# Barras: correcto vs incorrecto por rol
ax = axes[1]
role_acc = data_known.groupby("role")["match"].agg(["sum", "count"])
role_acc["wrong"] = role_acc["count"] - role_acc["sum"]
role_acc["pct_correct"] = role_acc["sum"] / role_acc["count"] * 100
role_acc = role_acc.sort_values("pct_correct", ascending=True)

colors_bar = ['#2ecc71' if p >= 90 else '#f39c12' if p >= 75 else '#e74c3c'
              for p in role_acc["pct_correct"]]
bars = ax.barh(range(len(role_acc)), role_acc["pct_correct"],
               color=colors_bar, alpha=0.85)
for i, (_, row) in enumerate(role_acc.iterrows()):
    ax.text(row["pct_correct"] + 0.5, i,
            f'{row["pct_correct"]:.1f}%  (n={int(row["count"])})',
            va='center', fontsize=10)

ax.set_yticks(range(len(role_acc)))
ax.set_yticklabels(role_acc.index.fillna("unknown"), fontsize=11)
ax.set_xlabel("% asignación correcta", fontsize=11)
ax.set_title("Precisión por Role", fontsize=12, fontweight='bold')
ax.set_xlim(0, 115)
ax.axvline(x=accuracy, color='#e74c3c', linestyle='--', linewidth=1.5,
           label=f'Media global ({accuracy:.1f}%)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_FIG, dpi=200, bbox_inches='tight')
plt.close()
print(f"  Grafico guardado en: {OUT_FIG}")
