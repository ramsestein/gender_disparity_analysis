import pandas as pd
import numpy as np
from pub_utils import load_data, OUTPUT_DIR, cramers_v_calc
from scipy.stats import chi2_contingency, ttest_ind

def analyze_segregation():
    df = load_data()
    
    transitions = []
    for i in range(len(df)-1):
        prev = df.iloc[i]
        curr = df.iloc[i+1]
        transitions.append({
            'prev_gender': prev['gender'],
            'curr_gender': curr['gender'],
            'is_cross': prev['gender'] != curr['gender'],
            'curr_duration': curr['duration'],
            'curr_conflict': curr['conflict_score'],
            'curr_echoing': curr['echoing_score']
        })
        
    t_df = pd.DataFrame(transitions)
    
    # 17.1 Matrix Test
    ct = pd.crosstab(t_df['prev_gender'], t_df['curr_gender'])
    v, magnitude, chi2, p = cramers_v_calc(ct)
    
    # 17.2 Cross-gender characteristics
    cross_stats = t_df.groupby('is_cross')[['curr_duration', 'curr_conflict', 'curr_echoing']].mean().reset_index()
    
    # T-tests for cross vs intra
    cross_tests = []
    for var in ['curr_duration', 'curr_conflict', 'curr_echoing']:
        p_val = ttest_ind(t_df[t_df['is_cross']==True][var], t_df[t_df['is_cross']==False][var]).pvalue
        cross_tests.append({'metric': var, 'p_value': p_val})
        
    # Save CSVs
    pd.DataFrame({'Metric': ['chi2', 'p_value', 'cramers_v'], 'Value': [chi2, p, v]}).to_csv(OUTPUT_DIR / 'p17_transition_matrix_test.csv', index=False)
    cross_stats.to_csv(OUTPUT_DIR / 'p17_cross_gender_transitions_characteristics.csv', index=False)
    
    print("[OK] P17 completed: Segregation analyzed.")

if __name__ == "__main__":
    analyze_segregation()
