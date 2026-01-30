import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pub_utils import load_data, OUTPUT_DIR

def analyze_temporal_evolution():
    df = load_data()
    if df.empty: return
    
    df['gender_bin'] = (df['gender'] == 'male').astype(int)
    
    # Map phase_quartile to numeric 1-4
    q_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    # Convert to string first to avoid Categorical mapping issues
    df['q_num'] = df['phase_quartile'].astype(str).map(q_map).fillna(0).astype(int)
    
    metrics = ['duration', 'wpm', 'conflict_score', 'interrupts_previous']
    results = []
    interactions = []
    
    for m in metrics:
        if m not in df.columns: continue
        
        # Trends by gender
        for g in ['male', 'female']:
            g_df = df[df['gender'] == g]
            if len(g_df) > 5:
                try:
                    model = smf.ols(f"{m} ~ q_num", data=g_df).fit()
                    results.append({
                        'metric': m,
                        'gender': g,
                        'slope': model.params['q_num'],
                        'p_value': model.pvalues['q_num'],
                        'r_squared': model.rsquared
                    })
                except: continue
            
        # Interaction test
        if len(df) > 10:
            try:
                model_int = smf.ols(f"{m} ~ q_num * gender_bin", data=df).fit()
                if 'q_num:gender_bin' in model_int.pvalues:
                    interactions.append({
                        'metric': m,
                        'interaction_p_value': model_int.pvalues['q_num:gender_bin']
                    })
            except: continue
        
    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'p18_temporal_trends_regression.csv', index=False)
    pd.DataFrame(interactions).to_csv(OUTPUT_DIR / 'p18_temporal_interaction_tests.csv', index=False)
    
    print("[OK] P18 completed: Temporal trends analyzed.")

if __name__ == "__main__":
    analyze_temporal_evolution()
