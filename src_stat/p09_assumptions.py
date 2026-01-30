import pandas as pd
import numpy as np
from pub_utils import load_data, OUTPUT_DIR, NUMERIC_VARS
from scipy.stats import shapiro, levene, mannwhitneyu, ttest_ind

def verify_assumptions():
    df = load_data()
    results = []
    
    print(" Checking statistical assumptions for numeric variables...")
    for var in NUMERIC_VARS:
        m = df[df['gender']=='male'][var].dropna()
        f = df[df['gender']=='female'][var].dropna()
        
        # Normality
        p_norm_m = shapiro(m)[1] if len(m) > 3 else 0
        p_norm_f = shapiro(f)[1] if len(f) > 3 else 0
        norm_met = (p_norm_m > 0.05) and (p_norm_f > 0.05)
        
        # Variance
        p_levene = levene(m, f)[1] if (len(m) > 1 and len(f) > 1) else 0
        var_met = (p_levene > 0.05)
        
        recommended = "t-test" if (norm_met and var_met) else "Mann-Whitney U"
        
        # Actual execution
        if recommended == "t-test":
            p_val = ttest_ind(m, f, equal_var=var_met).pvalue
        else:
            p_val = mannwhitneyu(m, f).pvalue
            
        results.append({
            'variable': var,
            'shapiro_p_male': p_norm_m,
            'shapiro_p_female': p_norm_f,
            'levene_p': p_levene,
            'normal_assumption_met': norm_met,
            'variance_assumption_met': var_met,
            'n_male': len(m),
            'n_female': len(f),
            'recommended_test': recommended,
            'test_actually_used': recommended,
            'final_p_value': p_val
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / 'p09_assumptions_check.csv', index=False)
    print("[OK] P09 completed: Assumptions checked.")

if __name__ == "__main__":
    verify_assumptions()
