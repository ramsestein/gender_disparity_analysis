import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pub_utils import load_data, OUTPUT_DIR

def analyze_backlash():
    df = load_data()
    df['gender_bin'] = (df['gender'] == 'male').astype(int)
    
    try:
        model = smf.mixedlm("interrupted_by_next ~ gender_bin * assertiveness_score", 
                            df, groups=df["session"]).fit()
        
        b_assert = model.params.get('assertiveness_score', 0)
        b_inter = model.params.get('gender_bin:assertiveness_score', 0)
        
        summary = (
            f"Slope for Females (Backlash): {b_assert:.4f}\n"
            f"Slope for Males (Backlash): {b_assert + b_inter:.4f}\n"
            f"Interaction term: {b_inter:.4f}\n"
        )
        with open(OUTPUT_DIR / 'p14_backlash_interpretation.txt', 'w') as f:
            f.write(summary)
            
    except Exception as e:
        print(f"Error in Backlash: {e}")

    print("[OK] P14 completed: Backlash analyzed.")

if __name__ == "__main__":
    analyze_backlash()
