import pandas as pd
import numpy as np
from pub_utils import load_data, OUTPUT_DIR, NUMERIC_VARS, CATEGORICAL_VARS
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import ttest_ind

def calculate_all_cis():
    df = load_data()
    
    # 11.1 Numeric
    num_results = []
    for var in NUMERIC_VARS:
        m = df[df['gender']=='male'][var].dropna()
        f = df[df['gender']=='female'][var].dropna()
        
        diff = m.mean() - f.mean()
        # CI for difference of means (Welch)
        n1, n2 = len(m), len(f)
        v1, v2 = m.var(), f.var()
        se = np.sqrt(v1/n1 + v2/n2)
        ci_low = diff - 1.96 * se
        ci_high = diff + 1.96 * se
        
        num_results.append({
            'variable': var,
            'mean_male': m.mean(),
            'mean_female': f.mean(),
            'difference': diff,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'ci_crosses_zero': ci_low < 0 < ci_high
        })
        
    # 11.2 Categorical
    cat_results = []
    for var in CATEGORICAL_VARS:
        m_vals = df[df['gender']=='male'][var]
        f_vals = df[df['gender']=='female'][var]
        
        n_m, k_m = len(m_vals), m_vals.sum()
        n_f, k_f = len(f_vals), f_vals.sum()
        
        rate_m = k_m / n_m
        rate_f = k_f / n_f
        
        ci_m_low, ci_m_high = proportion_confint(k_m, n_m, method='wilson')
        ci_f_low, ci_f_high = proportion_confint(k_f, n_f, method='wilson')
        
        diff = rate_m - rate_f
        se_diff = np.sqrt((rate_m*(1-rate_m)/n_m) + (rate_f*(1-rate_f)/n_f))
        ci_diff_low = diff - 1.96 * se_diff
        ci_diff_high = diff + 1.96 * se_diff
        
        cat_results.append({
            'variable': var,
            'rate_male': rate_m,
            'rate_male_ci_lower': ci_m_low,
            'rate_male_ci_upper': ci_m_high,
            'rate_female': rate_f,
            'rate_female_ci_lower': ci_f_low,
            'rate_female_ci_upper': ci_f_high,
            'difference': diff,
            'ci_diff_lower': ci_diff_low,
            'ci_diff_high': ci_diff_high
        })
        
    pd.DataFrame(num_results).to_csv(OUTPUT_DIR / 'p11_confidence_intervals_numeric.csv', index=False)
    pd.DataFrame(cat_results).to_csv(OUTPUT_DIR / 'p11_confidence_intervals_categorical.csv', index=False)
    print("[OK] P11 completed: Confidence intervals calculated.")

if __name__ == "__main__":
    calculate_all_cis()
