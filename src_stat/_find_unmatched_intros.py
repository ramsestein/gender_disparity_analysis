"""
_find_unmatched_intros.py
=========================
Finds speakers who said their name ("I am / I'm / My name is [Name]")
but whose name could NOT be matched to anyone in the ESICM Excel registry.
These are likely audience members / public participants.

Output: final_reports/csv_enriched/unmatched_intros.csv
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process as rfprocess

BASE       = Path(__file__).resolve().parent.parent
CSV_DIR    = BASE / "final_reports" / "csv_cleaned"
EXCEL_PATH = BASE / "esicm_talks_grouped_9_groups_new.xlsx"
OUT_PATH   = BASE / "final_reports" / "csv_enriched" / "unmatched_intros.csv"

# ── same helpers as enrich_speakers.py ────────────────────────────────────────
STOP_FIRST = {
    "a", "an", "the", "this", "that", "these", "those",
    "i", "it", "its", "he", "she", "we", "they",
    "is", "be", "am", "are", "was", "were", "been",
    "from", "in", "at", "of", "to", "for", "with", "by", "on",
    "as", "and", "or", "but", "also",
    "not", "one", "just", "so", "very", "well", "ok", "now",
    "here", "really", "actually", "basically", "going", "happy",
    "pleased", "glad", "honored",
}

_NAME_TOKEN = r"[a-z][a-z'\-]+"
_TITLE      = r"(?:dr\.?\s+|prof(?:\.|essor)?\s+)?"
_INTRO_RE   = re.compile(
    r"\bi(?:'m| am)\s+" + _TITLE + r"(" + _NAME_TOKEN + r"(?:\s+" + _NAME_TOKEN + r"){0,2})"
    r"|"
    r"\bmy name is\s+" + _TITLE + r"(" + _NAME_TOKEN + r"(?:\s+" + _NAME_TOKEN + r"){0,2})",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    t = str(text).lower()
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def clean_candidate(raw: str) -> str | None:
    parts = raw.split()
    while parts and (parts[-1] in STOP_FIRST or parts[-1].endswith("ing")):
        parts.pop()
    while parts and (parts[0] in STOP_FIRST or parts[0].endswith("ing") or len(parts[0]) < 2):
        parts.pop(0)
    parts = [p for p in parts if len(p) >= 2]
    return " ".join(parts) if parts else None


def extract_intro_candidates(text_norm: str) -> list[str]:
    seen, result = set(), []
    for m in _INTRO_RE.finditer(text_norm):
        raw = (m.group(1) or m.group(2) or "").strip()
        cleaned = clean_candidate(raw)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


# ── load data ─────────────────────────────────────────────────────────────────
excel_df   = pd.read_excel(EXCEL_PATH, sheet_name="Grouped participants")
all_names  = excel_df["Name of the person"].dropna().str.strip().tolist()
all_names_norm = [normalize(n) for n in all_names]

# Already-matched speakers from the enrichment run
matched_df = pd.read_csv(BASE / "final_reports" / "csv_enriched" / "match_log.csv")
matched_keys = set(zip(matched_df["session"], matched_df["speaker"]))

# ── scan every CSV ─────────────────────────────────────────────────────────────
rows = []

for csv_path in sorted(CSV_DIR.glob("*.csv")):
    stem = csv_path.stem
    df   = pd.read_csv(csv_path)

    speaker_texts = (
        df.groupby("speaker")["text"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .to_dict()
    )

    for speaker, text in speaker_texts.items():
        # Skip already matched
        if (stem, speaker) in matched_keys:
            continue

        text_norm  = normalize(text)
        candidates = extract_intro_candidates(text_norm)
        if not candidates:
            continue

        for cand in candidates:
            # Best fuzzy match in Excel (no threshold — we want to see everything)
            result = rfprocess.extractOne(
                cand, all_names_norm,
                scorer=fuzz.token_sort_ratio
            )
            if result:
                best_score = result[1]
                best_name  = all_names[all_names_norm.index(result[0])]
            else:
                best_score = 0
                best_name  = ""

            rows.append({
                "session":           stem,
                "speaker":           speaker,
                "extracted_name":    cand,
                "best_excel_match":  best_name,
                "best_score":        round(best_score, 1),
                "likely_public":     best_score < 65,
            })

out_df = pd.DataFrame(rows)

# Keep only the best candidate per (session, speaker)
out_df = (
    out_df.sort_values("best_score", ascending=False)
    .drop_duplicates(subset=["session", "speaker"])
    .sort_values(["likely_public", "session", "speaker"], ascending=[False, True, True])
    .reset_index(drop=True)
)

out_df.to_csv(OUT_PATH, index=False)

public  = out_df[out_df["likely_public"]]
close   = out_df[~out_df["likely_public"]]

print(f"Speakers with self-intro NOT in the registry: {len(public)}")
print(f"Speakers with self-intro CLOSE to registry (may be missed matches): {len(close)}")
print(f"\nSaved to: {OUT_PATH}")
print()
print("=== Likely PUBLIC (no Excel match) ===")
print(public[["session","speaker","extracted_name","best_excel_match","best_score"]].to_string())
print()
if len(close):
    print("=== CLOSE to registry (score >= 65, below match threshold) ===")
    print(close[["session","speaker","extracted_name","best_excel_match","best_score"]].to_string())
