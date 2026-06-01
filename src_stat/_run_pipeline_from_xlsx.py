"""
_run_pipeline_from_xlsx.py
==========================
Re-ejecuta todos los análisis que dependen de esicm_talks_grouped_9_groups_new.xlsx.

Orden:
  Fase 1 – Enriquecimiento de CSVs (csv_enriched/)
    1. enrich_speakers.py
    2. _patch_missed_matches.py
    3. _fix_gender_errors.py

  Fase 2 – Dataset de usuarios (c01–c22)
    c01 → c22  en orden numérico

  Fase 3 – Análisis de publicación (p01–p23)
    p01 → p23  en orden numérico
"""

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

SCRIPTS = [
    # ── Fase 1: enriquecimiento ──────────────────────────────────────────────
    "enrich_speakers.py",
    "_patch_missed_matches.py",
    "_fix_gender_errors.py",
    # ── Fase 2: cadena c0x ──────────────────────────────────────────────────
    "c01_build_user_dataset.py",
    "c02_cluster_analysis.py",
    "c03_enrich_user_dataset.py",
    "c04_cluster_enriched.py",
    "c05_gender_name_validation.py",
    "c06_gender_corrected_bias.py",
    "c07_new_vars_bias_predictors.py",
    "c08_mixed_effects.py",
    "c09_groups_bias.py",
    "c10_interruption_temporal.py",
    "c11_propensity_matching.py",
    "c12_regression_predictors.py",
    "c13_nonlinear_explainable.py",
    "c14_shap.py",
    "c15_interactions.py",
    "c16_session_bias.py",
    "c17_markov_turns.py",
    "c18_geography.py",
    "c19_bayesian.py",
    "c20_power_analysis.py",
    "c21_female_moderator.py",
    "c22_enrich_excel_and_graphs.py",
    # ── Fase 3: análisis de publicación ─────────────────────────────────────
    "p01_effect_sizes.py",
    "p02_fdr_correction.py",
    "p03_mixed_models.py",
    "p04_question_response.py",
    "p05_appropriation.py",
    "p06_power_positions.py",
    "p07_predictive_model.py",
    "p08_icc_reporting.py",
    "p09_assumptions.py",
    "p10_partial_corrs.py",
    "p11_confidence_intervals.py",
    "p12_master_table.py",
    "p13_climate_analysis.py",
    "p14_backlash_analysis.py",
    "p15_explaining_pattern.py",
    "p16_sticky_floor.py",
    "p17_segregation.py",
    "p18_temporal_evolution.py",
    "p19_qualitative_cases.py",
    "p20_subgroups.py",
    "p21_power_analysis.py",
    "p22_sensitivity.py",
    "p23_plots.py",
]

failed = []

for script in SCRIPTS:
    path = BASE / script
    if not path.exists():
        print(f"[SKIP]  {script}  (no encontrado)")
        continue

    print(f"\n{'='*60}")
    print(f"  >> {script}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=False,   # imprime en tiempo real
        text=True,
    )

    if result.returncode != 0:
        print(f"[FALLO] {script}  (código {result.returncode})")
        failed.append(script)
    else:
        print(f"[OK]    {script}")

print("\n" + "="*60)
if failed:
    print(f"COMPLETADO CON {len(failed)} FALLO(S):")
    for s in failed:
        print(f"  - {s}")
else:
    print("PIPELINE COMPLETADO SIN ERRORES")
print("="*60)
