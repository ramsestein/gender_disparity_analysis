"""
enrich_speakers.py
==================
Connects anonymous SPEAKER_XX in each CSV to real persons in
esicm_talks_grouped_9_groups_new.xlsx by searching transcribed text
for self-introduction patterns ("I am / I'm / My name is [Name]").

Matching strategy
-----------------
  1. Extract the name candidate from intro patterns (up to 3 tokens).
  2. Fuzzy-match the candidate against all Excel names.
       - 2+ word candidate : token_sort_ratio >= 65
       - 1-word  candidate : WRatio          >= 85  AND person expected in session
  3. Greedy assignment (highest score first, no Excel person used twice per session).

Columns added to every row of a matched speaker
-------------------------------------------------
  From Excel (always): Group, Talk name, Country, Affiliation (Hospital),
      Date of birth (DD/MM/YYYY), Year of qualification (specialty),
      Specialty (ICU-Ane-Both), Number of citations
  Role:
      - Excel Role  → if person's oldNamesMapping  == this CSV stem + '.xlsx'
      - 'public'    → otherwise
  matched_person, match_confidence, match_score  (diagnostic columns)

Output: final_reports/csv_enriched/<same_filename>.csv
        final_reports/csv_enriched/match_log.csv
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# ── paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
CSV_DIR    = BASE / "final_reports" / "csv_cleaned"
OUT_DIR    = BASE / "final_reports" / "csv_enriched"
EXCEL_PATH = BASE / "esicm_talks_grouped_9_groups_new.xlsx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns copied from Excel (all except Role)
EXCEL_ADD_COLS = [
    "Group", "Talk name", "Country", "Affiliation (Hospital)",
    "Date of birth (DD/MM/YYYY)", "Year of qualification (specialty)",
    "Specialty (ICU-Ane-Both)", "Number of citations",
]

# Fuzzy-match thresholds
THRESH_MULTI = 65   # 2+ word extracted name  →  token_sort_ratio
THRESH_SINGLE = 85  # 1-word extracted name   →  WRatio  (+ must be in session)

# Words that are never part of a name (filter extracted candidates)
STOP_FIRST = {
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    # pronouns
    "i", "it", "its", "he", "she", "we", "they",
    # auxiliaries / copulas
    "is", "be", "am", "are", "was", "were", "been",
    # prepositions / conjunctions (common after names)
    "from", "in", "at", "of", "to", "for", "with", "by", "on",
    "as", "and", "or", "but", "also",
    # adverbs / filler
    "not", "one", "just", "so", "very", "well", "ok", "now",
    "here", "really", "actually", "basically", "going", "happy",
    "pleased", "glad", "honored",
}

# Regex patterns for self-introduction detection
_NAME_TOKEN = r"[a-z][a-z'\-]+"
_TITLE      = r"(?:dr\.?\s+|prof(?:\.|essor)?\s+)?"
_INTRO_RE   = re.compile(
    r"\bi(?:'m| am)\s+" + _TITLE + r"(" + _NAME_TOKEN + r"(?:\s+" + _NAME_TOKEN + r"){0,2})"
    r"|"
    r"\bmy name is\s+" + _TITLE + r"(" + _NAME_TOKEN + r"(?:\s+" + _NAME_TOKEN + r"){0,2})",
    re.IGNORECASE,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase + strip diacritics."""
    t = str(text).lower()
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def clean_candidate(raw: str) -> str | None:
    """
    Remove leading/trailing stop words and verb forms (-ing suffix).
    Returns cleaned candidate or None if nothing usable remains.
    """
    parts = raw.split()
    # Drop trailing tokens that are stop words or verb gerunds
    while parts and (parts[-1] in STOP_FIRST or parts[-1].endswith("ing")):
        parts.pop()
    # Drop leading stop words / gerunds / single chars
    while parts and (parts[0] in STOP_FIRST or parts[0].endswith("ing") or len(parts[0]) < 2):
        parts.pop(0)
    # Drop any internal single-char tokens
    parts = [p for p in parts if len(p) >= 2]
    return " ".join(parts) if parts else None


def extract_intro_candidates(text_norm: str) -> list[str]:
    """Extract self-introduction name candidates from normalised text."""
    candidates = []
    for m in _INTRO_RE.finditer(text_norm):
        raw = (m.group(1) or m.group(2) or "").strip()
        cleaned = clean_candidate(raw)
        if cleaned:
            candidates.append(cleaned)
    return candidates


def score_candidate(candidate: str, excel_name_norm: str, in_session: bool) -> float:
    """
    Returns a numeric match score (0 = no match):
      - 2+ word candidate : token_sort_ratio if >= THRESH_MULTI, else 0
      - 1-word  candidate : WRatio if >= THRESH_SINGLE AND in_session, else 0
    """
    words = candidate.split()
    if len(words) >= 2:
        s = fuzz.token_sort_ratio(candidate, excel_name_norm)
        return float(s) if s >= THRESH_MULTI else 0.0
    else:  # single word
        if not in_session:
            return 0.0
        s = fuzz.WRatio(candidate, excel_name_norm)
        return float(s) if s >= THRESH_SINGLE else 0.0


# ── per-session enrichment ─────────────────────────────────────────────────────

def enrich_session(csv_path: Path, excel_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    df       = pd.read_csv(csv_path)
    stem     = csv_path.stem                     # e.g. "39_report"
    xlsx_key = stem + ".xlsx"                    # for oldNamesMapping comparison

    # Build person index:  name -> {rows, session_row, name_norm}
    persons: dict[str, dict] = {}
    for _, row in excel_df.iterrows():
        name = str(row["Name of the person"]).strip()
        if not name or name == "nan":
            continue
        if name not in persons:
            persons[name] = {
                "rows": [],
                "session_row": None,
                "name_norm": normalize(name),
            }
        persons[name]["rows"].append(row)
        if str(row["oldNamesMapping"]).strip() == xlsx_key:
            persons[name]["session_row"] = row

    # Build per-speaker combined text
    speaker_texts: dict[str, str] = (
        df.groupby("speaker")["text"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .to_dict()
    )

    # Score every (speaker, person) pair
    # all_candidates: list of (score, speaker, person_name)
    all_candidates: list[tuple[float, str, str]] = []

    for speaker, text in speaker_texts.items():
        text_norm = normalize(text)
        candidates = extract_intro_candidates(text_norm)
        if not candidates:
            continue

        for person_name, info in persons.items():
            in_session = info["session_row"] is not None
            best_for_pair = 0.0
            for cand in candidates:
                s = score_candidate(cand, info["name_norm"], in_session)
                if s > best_for_pair:
                    best_for_pair = s
            if best_for_pair > 0:
                # Slight boost for session-expected persons (tiebreaker only)
                eff = best_for_pair + (0.1 if in_session else 0.0)
                all_candidates.append((eff, speaker, person_name))

    # Greedy assignment: highest score first, each speaker and each person once
    all_candidates.sort(key=lambda x: -x[0])
    assignment: dict[str, tuple[str, float, dict]] = {}  # speaker -> (name, score, info)
    used_names: set[str] = set()

    for eff_score, speaker, person_name in all_candidates:
        if speaker in assignment or person_name in used_names:
            continue
        assignment[speaker] = (person_name, eff_score, persons[person_name])
        used_names.add(person_name)

    # Enrich DataFrame – use None (object dtype) to avoid FutureWarning
    for col in EXCEL_ADD_COLS:
        df[col] = None
    df["Role"]             = None
    df["matched_person"]   = None
    df["match_confidence"] = None
    df["match_score"]      = None

    log_rows: list[dict] = []

    for speaker, (name, eff_score, info) in assignment.items():
        mask = df["speaker"] == speaker
        in_session = info["session_row"] is not None
        row = info["session_row"] if in_session else info["rows"][0]

        for col in EXCEL_ADD_COLS:
            df.loc[mask, col] = row[col]

        df.loc[mask, "Role"]             = row["Role"] if in_session else "public"
        df.loc[mask, "matched_person"]   = name
        df.loc[mask, "match_score"]      = round(eff_score, 1)

        # Confidence label
        base_score = eff_score - (0.1 if in_session else 0.0)
        if base_score >= 85:
            conf = "high"
        elif base_score >= 70:
            conf = "moderate"
        else:
            conf = "low"
        df.loc[mask, "match_confidence"] = conf

        log_rows.append({
            "session":        stem,
            "speaker":        speaker,
            "matched_person": name,
            "score":          round(base_score, 1),
            "confidence":     conf,
            "in_session":     in_session,
            "role_assigned":  row["Role"] if in_session else "public",
        })

    return df, log_rows


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    excel_df  = pd.read_excel(EXCEL_PATH, sheet_name="Grouped participants")
    csv_files = sorted(CSV_DIR.glob("*.csv"))

    print(f"Procesando {len(csv_files)} sesiones...\n")
    all_logs: list[dict] = []

    for csv_path in csv_files:
        enriched, logs = enrich_session(csv_path, excel_df)
        out_path = OUT_DIR / csv_path.name
        enriched.to_csv(out_path, index=False)
        all_logs.extend(logs)

        n_speakers = enriched["speaker"].nunique()
        n_matched  = enriched.groupby("speaker")["matched_person"].first().notna().sum()
        flag = "" if n_matched > 0 else "  (no matches)"
        print(f"  {csv_path.name}: {n_matched}/{n_speakers} speakers matched{flag}")

    # Save match log
    log_path = OUT_DIR / "match_log.csv"
    pd.DataFrame(all_logs).to_csv(log_path, index=False)

    total_matched = len(all_logs)
    in_session    = sum(1 for r in all_logs if r["in_session"])
    public        = total_matched - in_session

    print(f"""
{'='*60}
  COMPLETADO
  Sesiones procesadas : {len(csv_files)}
  Matches totales     : {total_matched}
    → en sesión (Role oficial) : {in_session}
    → fuera de sesión (public) : {public}
  Log guardado en     : {log_path}
  CSVs enriquecidos   : {OUT_DIR}
{'='*60}""")


if __name__ == "__main__":
    main()
