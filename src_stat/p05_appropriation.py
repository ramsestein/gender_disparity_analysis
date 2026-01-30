import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pub_utils import load_data, OUTPUT_DIR, FIG_DIR, cramers_v_calc

def analyze_appropriation():
    df = load_data()
    building_pairs = []
    for i in range(len(df)-1):
        prev = df.iloc[i]
        curr = df.iloc[i+1]
        appropriation = (curr['echoing_score'] > 0.3) and (not curr['has_attribution'])
        amplification = (curr['echoing_score'] > 0.3) and (curr['has_attribution'])
        building_pairs.append({
            'prev_gender': prev['gender'],
            'curr_gender': curr['gender'],
            'echoing': curr['echoing_score'],
            'has_attribution': curr['has_attribution'],
            'appropriation': appropriation,
            'amplification': amplification
        })
    b_df = pd.DataFrame(building_pairs)
    app_matrix = b_df.groupby(['prev_gender', 'curr_gender'])['appropriation'].mean().unstack().fillna(0) * 100
    amp_matrix = b_df.groupby(['prev_gender', 'curr_gender'])['amplification'].mean().unstack().fillna(0) * 100
    
    try:
        m_to_f = app_matrix.loc['female', 'male'] if ('female' in app_matrix.index and 'male' in app_matrix.columns) else 0
        f_to_m = app_matrix.loc['male', 'female'] if ('male' in app_matrix.index and 'female' in app_matrix.columns) else 0
        asymmetry_score = m_to_f - f_to_m
    except:
        asymmetry_score = 0
    
    app_matrix.to_csv(OUTPUT_DIR / 'p05_appropriation_matrix.csv')
    amp_matrix.to_csv(OUTPUT_DIR / 'p05_amplification_matrix.csv')
    
    with open(OUTPUT_DIR / 'p05_appropriation_asymmetry_score.txt', 'w') as f:
        f.write(f"Asymmetry Score (M->F - F->M): {asymmetry_score:.2f}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(app_matrix, annot=True, fmt='.1f', cmap='Reds')
    plt.title('Appropriation Rate Matrix (%)')
    plt.savefig(FIG_DIR / 'p05_heatmap_appropriation.png', dpi=300)
    plt.close()
    print("[OK] P05 completed: Appropriation analyzed.")

if __name__ == "__main__":
    analyze_appropriation()
