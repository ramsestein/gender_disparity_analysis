import pandas as pd
import numpy as np
from pub_utils import load_data, OUTPUT_DIR, cramers_v_calc
from scipy.stats import chi2_contingency
from statsmodels.stats.power import GofChisquarePower

def analyze_explaining_pattern():
    df = load_data()
    
    # Define pattern: interruption by one gender to another with disagreement/correction
    # Already computed in raw data as 'Explaining Pattern' or similar, 
    # but let's re-verify or use the available metric.
    if 'Explaining Pattern' not in df.columns:
        # Fallback to general interruption/disagreement logic
        df['explaining'] = (df['interrupts_previous']) & (df['has_disagreement'])
    else:
        df['explaining'] = df['Explaining Pattern']
        
    ct = pd.crosstab(df['gender'], df['explaining'])
    v, magnitude, chi2, p = cramers_v_calc(ct)
    
    # 15.2 Power analysis
    # What effect size (w) can we detect with 80% power?
    power_analysis = GofChisquarePower()
    n_total = len(df)
    min_w = power_analysis.solve_power(n_bins=2, nobs=n_total, alpha=0.05, power=0.8)
    
    # Save CSV
    pd.DataFrame({
        'variable': ['explaining_pattern'],
        'chi2': [chi2],
        'p_value': [p],
        'cramers_v': [v],
        'magnitude': [magnitude],
        'min_detectable_w': [min_w],
        'n_total': [n_total]
    }).to_csv(OUTPUT_DIR / 'p15_explaining_pattern_test.csv', index=False)
    
    print("[OK] P15 completed: Explaining pattern analyzed.")

if __name__ == "__main__":
    analyze_explaining_pattern()
