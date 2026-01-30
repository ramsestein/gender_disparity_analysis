import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pub_utils import load_data, OUTPUT_DIR

def analyze_climate():
    df = load_data()
    if df.empty: return
    
    # Filter for relevant climate types
    df_clean = df[df['climate_type'].isin(['hostile', 'calm'])].copy()
    
    # Check if we have both genders and both climate types
    if len(df_clean['gender'].unique()) < 2 or len(df_clean['climate_type'].unique()) < 2:
        print("[WARN] Skipping P13: Insufficient variance in gender or climate_type.")
        return
        
    try:
        model = ols('interrupted_by_next ~ C(gender) * C(climate_type)', data=df_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_table.to_csv(OUTPUT_DIR / 'p13_climate_anova.csv')
        
        # Mean stats
        stats = df_clean.groupby(['gender', 'climate_type'])['interrupted_by_next'].mean().unstack()
        stats.to_csv(OUTPUT_DIR / 'p13_climate_interaction_means.csv')
        
        print("[OK] P13 completed: Climate interaction analyzed.")
    except Exception as e:
        print(f"Error in P13: {e}")

if __name__ == "__main__":
    analyze_climate()
