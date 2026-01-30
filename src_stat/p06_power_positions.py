import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pub_utils import load_data, OUTPUT_DIR, FIG_DIR, cohens_d
from scipy.stats import binomtest, ttest_ind

def analyze_power_positions():
    df = load_data()
    if df.empty: return
    
    sessions = df['session'].unique()
    first_speakers = []
    last_speakers = []
    
    for s in sessions:
        s_df = df[df['session'] == s].sort_values('start_time')
        if not s_df.empty:
            first_speakers.append(s_df.iloc[0])
            last_speakers.append(s_df.iloc[-1])
        
    f_df = pd.DataFrame(first_speakers)
    l_df = pd.DataFrame(last_speakers)
    
    if f_df.empty or l_df.empty:
        print("[WARN] No boundary speakers identified.")
        return

    # 6.1 Distribution
    f_dist = f_df['gender'].value_counts(normalize=True).to_dict()
    l_dist = l_df['gender'].value_counts(normalize=True).to_dict()
    
    # Binomial Tests
    f_p = binomtest((f_df['gender'] == 'male').sum(), len(f_df), p=0.5).pvalue
    l_p = binomtest((l_df['gender'] == 'male').sum(), len(l_df), p=0.5).pvalue
    
    # 6.2 Characteristics
    metric_cols = ['duration', 'word_count', 'conflict_score', 'assertiveness_score']
    boundary_chars = []
    for m in metric_cols:
        if m in f_df.columns:
            m_avg = f_df[f_df['gender']=='male'][m].mean() if 'male' in f_df['gender'].values else 0
            f_avg = f_df[f_df['gender']=='female'][m].mean() if 'female' in f_df['gender'].values else 0
            p_val = ttest_ind(f_df[f_df['gender']=='male'][m], f_df[f_df['gender']=='female'][m], nan_policy='omit')[1] if ('male' in f_df['gender'].values and 'female' in f_df['gender'].values) else 1.0
            boundary_chars.append({'position': 'First', 'metric': m, 'male': m_avg, 'female': f_avg, 'p_value': p_val})
        
        if m in l_df.columns:
            m_avg = l_df[l_df['gender']=='male'][m].mean() if 'male' in l_df['gender'].values else 0
            f_avg = l_df[l_df['gender']=='female'][m].mean() if 'female' in l_df['gender'].values else 0
            p_val = ttest_ind(l_df[l_df['gender']=='male'][m], l_df[l_df['gender']=='female'][m], nan_policy='omit')[1] if ('male' in l_df['gender'].values and 'female' in l_df['gender'].values) else 1.0
            boundary_chars.append({'position': 'Last', 'metric': m, 'male': m_avg, 'female': f_avg, 'p_value': p_val})

    # Save CSVs
    pd.DataFrame({'Position': ['First', 'Last'], 
                  'Male_Pct': [f_dist.get('male', 0)*100, l_dist.get('male', 0)*100], 
                  'p_value': [f_p, l_p]}).to_csv(OUTPUT_DIR / 'p06_boundary_speakers_distribution.csv', index=False)
    pd.DataFrame(boundary_chars).to_csv(OUTPUT_DIR / 'p06_boundary_characteristics.csv', index=False)
    
    # Plots
    plot_data = pd.DataFrame([
        {'Position': 'First', 'Gender': 'Male', 'Percentage': f_dist.get('male', 0)*100},
        {'Position': 'First', 'Gender': 'Female', 'Percentage': f_dist.get('female', 0)*100},
        {'Position': 'Last', 'Gender': 'Male', 'Percentage': l_dist.get('male', 0)*100},
        {'Position': 'Last', 'Gender': 'Female', 'Percentage': l_dist.get('female', 0)*100}
    ])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x='Position', y='Percentage', hue='Gender')
    plt.axhline(50, color='red', linestyle='--', alpha=0.5)
    plt.title('Speaker Distribution at Debate Boundaries')
    plt.savefig(FIG_DIR / 'p06_barplot_first_last_speakers.png', dpi=300)
    plt.close()

    print("[OK] P06 completed: Power positions analyzed.")

if __name__ == "__main__":
    analyze_power_positions()
