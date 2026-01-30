import pandas as pd
import numpy as np
from pub_utils import load_data, OUTPUT_DIR, cohens_d
from scipy.stats import ttest_ind, mannwhitneyu

def analyze_sticky_floor():
    df = load_data()
    
    # Identify first intervention of each speaker in each session
    df['first_turn_idx'] = df.groupby(['session', 'speaker']).cumcount()
    first_turns = df[df['first_turn_idx'] == 0].copy()
    
    # 16.1 Statistics
    # How many turns into the session does each gender first speak? (Approx by index)
    stats = first_turns.groupby('gender')['turn_number'].agg(['mean', 'median', 'std']).reset_index()
    
    # compare
    m = first_turns[first_turns['gender']=='male']['turn_number']
    f = first_turns[first_turns['gender']=='female']['turn_number']
    p_val = mannwhitneyu(m, f).pvalue
    
    # 16.3 Characteristics of first interventions
    char_results = []
    for var in ['duration', 'word_count']:
        m_vals = first_turns[first_turns['gender']=='male'][var]
        f_vals = first_turns[first_turns['gender']=='female'][var]
        char_results.append({
            'metric': var,
            'male_mean': m_vals.mean(),
            'female_mean': f_vals.mean(),
            'p_value': ttest_ind(m_vals, f_vals).pvalue
        })
        
    # Save CSVs
    stats.to_csv(OUTPUT_DIR / 'p16_sticky_floor_stats.csv', index=False)
    pd.DataFrame(char_results).to_csv(OUTPUT_DIR / 'p16_first_intervention_characteristics.csv', index=False)
    
    with open(OUTPUT_DIR / 'p16_sticky_floor_test_results.txt', 'w') as f_out:
        f_out.write(f"Mann-Whitney U p-value for first turn entry: {p_val:.4f}\n")
        f_out.write(f"Hedges' g: {cohens_d(m, f)[1]:.4f}")

    print("[OK] P16 completed: Sticky floor analyzed.")

if __name__ == "__main__":
    analyze_sticky_floor()
