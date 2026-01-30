import pandas as pd
import numpy as np
from pub_utils import load_data, OUTPUT_DIR, NUMERIC_VARS
import statsmodels.formula.api as smf

def analyze_icc():
    df = load_data()
    results = []
    
    print("Calculating ICC for all numeric variables...")
    for var in NUMERIC_VARS:
        try:
            model = smf.mixedlm(f"{var} ~ 1", df, groups=df["session"]).fit()
            var_resid = model.scale
            var_random = model.cov_re.iloc[0, 0]
            icc = var_random / (var_random + var_resid) if (var_random + var_resid) > 0 else 0
            
            results.append({
                'variable': var,
                'icc': icc,
                'variance_between_sessions': var_random,
                'variance_within_sessions': var_resid,
                'interpretation': (
                    "large session effect" if icc >= 0.15 else 
                    "moderate session effect" if icc >= 0.10 else 
                    "small session effect" if icc >= 0.05 else "negligible"
                )
            })
        except Exception as e:
            print(f"Error in {var}: {e}")
        
    icc_df = pd.DataFrame(results)
    icc_df.to_csv(OUTPUT_DIR / 'p08_icc_summary.csv', index=False)
    print("[OK] P08 completed: ICC analyzed.")

if __name__ == "__main__":
    analyze_icc()
