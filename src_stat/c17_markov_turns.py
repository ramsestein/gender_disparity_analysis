"""c17_markov_turns.py — Cadenas de Markov en secuencias de turnos por genero"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import glob

BASE      = Path(__file__).resolve().parent.parent
CSV_DIR   = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR   = BASE / "final_reports" / "resultados" / "graficos"
CLEAN_DIR = BASE / "final_reports" / "csv_cleaned"

# Cargar gender map desde user_level
udf = pd.read_csv(CSV_DIR / "user_level_enriched_with_clusters.csv")
gender_map = {}
for _, row in udf.iterrows():
    gender_map[(str(row["session"]).replace(".csv",""), str(row["speaker"]))] = row["gender"]

# Construir secuencias de genero por sesion
transitions = {"MM":0, "MF":0, "FM":0, "FF":0}
session_trans = []

csv_files = sorted(glob.glob(str(CLEAN_DIR / "*.csv")))
if not csv_files:
    CLEAN_DIR2 = BASE / "final_reports" / "csv_cleaned"
    csv_files  = sorted(glob.glob(str(CLEAN_DIR2 / "*.csv")))

for fpath in csv_files:
    sess_name = Path(fpath).stem
    try:
        raw = pd.read_csv(fpath)
    except Exception:
        continue
    if "speaker" not in raw.columns: continue
    raw = raw.sort_values("turn_number" if "turn_number" in raw.columns else raw.columns[0])
    genders = []
    for spk in raw["speaker"].astype(str):
        g = gender_map.get((sess_name, spk))
        if g is None:
            g = gender_map.get((sess_name.replace(".csv",""), spk))
        genders.append(g)
    raw["_g"] = genders
    seq = [g for g in genders if g in ("male","female")]
    local = {"MM":0,"MF":0,"FM":0,"FF":0}
    for i in range(len(seq)-1):
        a = "M" if seq[i]=="male" else "F"
        b = "M" if seq[i+1]=="male" else "F"
        key = a+b
        transitions[key] += 1
        local[key] += 1
    total_loc = sum(local.values())
    if total_loc > 0:
        session_trans.append({
            "session": sess_name,
            "MM": local["MM"], "MF": local["MF"],
            "FM": local["FM"], "FF": local["FF"],
            "n_trans": total_loc,
            "pMF": local["MF"]/total_loc,
            "pFM": local["FM"]/total_loc,
        })

total = sum(transitions.values())
print(f"Total transiciones: {total}")
print(f"Matriz observada:")
print(f"  M->M={transitions['MM']}  M->F={transitions['MF']}")
print(f"  F->M={transitions['FM']}  F->F={transitions['FF']}")

# Frecuencias relativas
pM = (transitions["MM"] + transitions["MF"]) / (total+1e-9)
pF = (transitions["FM"] + transitions["FF"]) / (total+1e-9)
print(f"\nBase rates: P(inicio=M)={pM:.3f}  P(inicio=F)={pF:.3f}")

# Probabilidades de transicion condicional
pMM_cond = transitions["MM"] / (transitions["MM"]+transitions["MF"]+1e-9)
pMF_cond = transitions["MF"] / (transitions["MM"]+transitions["MF"]+1e-9)
pFM_cond = transitions["FM"] / (transitions["FM"]+transitions["FF"]+1e-9)
pFF_cond = transitions["FF"] / (transitions["FM"]+transitions["FF"]+1e-9)
print(f"\nTransicion condicional:")
print(f"  P(M|M)={pMM_cond:.3f}  P(F|M)={pMF_cond:.3f}")
print(f"  P(M|F)={pFM_cond:.3f}  P(F|F)={pFF_cond:.3f}")

# Test chi-cuadrado vs hipotesis nula (transicion aleatoria basada en base rates)
obs = np.array([transitions["MM"], transitions["MF"],
                transitions["FM"], transitions["FF"]])
exp = np.array([total*pM*pM, total*pM*pF,
                total*pF*pM, total*pF*pF])
chi2, p_chi = stats.chisquare(obs, f_exp=exp)
print(f"\nChi2 vs aleatorio: chi2={chi2:.2f}  p={p_chi:.6f}")

# Sesgo de auto-seguimiento: P(mismo genero) vs esperado
p_same_obs = (transitions["MM"]+transitions["FF"]) / total
p_same_exp = pM**2 + pF**2
print(f"P(mismo genero) obs={p_same_obs:.3f}  esperado={p_same_exp:.3f}")
print(f"Exceso de auto-seguimiento: {p_same_obs - p_same_exp:+.3f}")

# Guardar
sdf = pd.DataFrame(session_trans)
sdf.to_csv(CSV_DIR / "c17_markov_transitions.csv", index=False)

# Grafico
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Cadenas de Markov — Secuencias de genero en turnos", fontsize=12, fontweight="bold")

# Matriz de transicion
matrix = np.array([[pMM_cond, pMF_cond],[pFM_cond, pFF_cond]])
im = axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=1)
axes[0].set_xticks([0,1]); axes[0].set_xticklabels(["->Hombre","->Mujer"], fontsize=11)
axes[0].set_yticks([0,1]); axes[0].set_yticklabels(["Hombre","Mujer"], fontsize=11)
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if matrix[i,j]>0.6 else "black")
plt.colorbar(im, ax=axes[0], label="P(transicion)")
axes[0].set_title(f"Matriz de transicion\n(chi2={chi2:.1f}, p={p_chi:.4f})", fontsize=10)

# Distribucion de pMF y pFM por sesion
if len(sdf) > 0:
    axes[1].hist(sdf["pMF"], bins=15, alpha=0.7, label="P(H->M por sesion)", color="#3498db")
    axes[1].hist(sdf["pFM"], bins=15, alpha=0.7, label="P(M->H por sesion)", color="#e74c3c")
    axes[1].axvline(pMF_cond, color="#3498db", lw=2, ls="--")
    axes[1].axvline(pFM_cond, color="#e74c3c", lw=2, ls="--")
    axes[1].set_xlabel("Probabilidad de transicion inter-genero")
    axes[1].set_ylabel("N sesiones")
    axes[1].set_title("Distribucion por sesion")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(FIG_DIR / "c17_markov.png", dpi=180, bbox_inches="tight")
plt.close()
print("Done c17_markov_turns.py")
