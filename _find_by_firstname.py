"""
Para cada uno de los 10 participantes no identificados,
busca en los speakers no emparejados de su sesión si
alguien menciona su nombre (nombre, apellido o ambos).
Busca en TODO el texto de la sesión, no solo self-intros.
"""
import os, glob, re, pandas as pd

EXCEL = r"esicm_talks_grouped_9_groups_new.xlsx"
ENRICHED = r"final_reports/csv_enriched"

TARGETS = [
    "Ana-Maria Ioan",
    "Thilo von Groote",
    "Brigitta Fazzini",
    "Eleonora Balzani",
    "Elena Sancho Ferrando",
    "Kristine Koekkoek",
    "Luigi Zattera",
    "Rhona Sloss",
    "Irene Steinberg",
    "Patryk Mysior",
]

xl = pd.read_excel(EXCEL, sheet_name="Grouped participants")

def get_sessions(person_name):
    rows = xl[xl["Name of the person"].str.lower().str.strip() == person_name.lower().strip()]
    sessions = []
    for _, row in rows.iterrows():
        raw = str(row.get("oldNamesMapping", ""))
        for s in raw.split(";"):
            s = s.strip()
            if s and s != "nan":
                sessions.append(s)
    return sessions

def name_tokens(full_name):
    """Retorna tokens significativos del nombre (>= 3 letras, excluye partículas)"""
    stop = {"von","de","del","van","la","el","the","of","da"}
    return [t for t in full_name.lower().split() if len(t) >= 3 and t not in stop]

print("=" * 80)
for person in TARGETS:
    sessions = get_sessions(person)
    tokens = name_tokens(person)
    first_name = person.split()[0].lower()
    last_name = person.split()[-1].lower()

    if not sessions:
        print(f"\n{person}")
        print(f"  ⚠ Sin sesión en Excel")
        continue

    found_any = False
    for stem in sessions:
        # Buscar archivo CSV
        pattern = os.path.join(ENRICHED, f"*{stem[:30]}*")
        matches = glob.glob(pattern)
        if not matches:
            # fallback: buscar por stem parcial
            all_files = glob.glob(os.path.join(ENRICHED, "*.csv"))
            matches = [f for f in all_files if stem[:20].lower() in os.path.basename(f).lower()]
        if not matches:
            continue

        fpath = matches[0]
        df = pd.read_csv(fpath)

        # Speakers sin emparejar
        unmatched = df[df["matched_person"].isna() | (df["matched_person"].astype(str).str.strip() == "")]
        if unmatched.empty:
            continue

        # Buscar menciones de nombre/apellido en todo el texto de cada speaker
        for spk, grp in unmatched.groupby("speaker"):
            all_text = " ".join(grp["text"].dropna().astype(str)).lower()
            hits = [t for t in tokens if re.search(r'\b' + re.escape(t) + r'\b', all_text)]
            if len(hits) >= 2 or (len(hits) == 1 and (first_name in all_text or last_name in all_text)):
                # Mostrar las frases donde aparece
                matched_lines = []
                for txt in grp["text"].dropna().astype(str):
                    if any(re.search(r'\b' + re.escape(t) + r'\b', txt.lower()) for t in tokens):
                        matched_lines.append(txt.strip())
                if not found_any:
                    print(f"\n{person}  [tokens: {tokens}]")
                    found_any = True
                print(f"  → {spk} en {os.path.basename(fpath)[:55]}")
                for line in matched_lines[:5]:
                    print(f"      \"{line[:100]}\"")

    if not found_any:
        print(f"\n{person}  [tokens: {tokens}]")
        print(f"  ✗ No encontrado en sesiones: {sessions}")

print("\n" + "=" * 80)
