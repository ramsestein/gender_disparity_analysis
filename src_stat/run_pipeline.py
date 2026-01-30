import os
import subprocess
from pathlib import Path

scripts_order = [
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
    "p23_plots.py"
]

def run_all():
    print("START Starting Modular Publication Analysis Pipeline...")
    # Dynamically find this script's directory
    base_dir = Path(__file__).resolve().parent
    
    for script in scripts_order:
        script_path = base_dir / script
        if script_path.exists():
            print(f"--- Running {script} ---")
            # We run using the same python interpreter
            result = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FAILED Error in {script}:")
                # Print stderr only if helpful, or first 500 chars
                print(result.stderr[:500])
            else:
                out = result.stdout.strip()
                if out: print(out)
        else:
            print(f"WARNING Warning: {script} not found in {base_dir}")
            
    print("DONE All scripts executed.")

if __name__ == "__main__":
    run_all()
