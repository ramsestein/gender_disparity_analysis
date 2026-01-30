import pandas as pd
from pub_utils import load_data, OUTPUT_DIR

def analyze_sensitivity():
    df = load_data()
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    # Sensitivity for Appropriation Rate (M->F)
    # prev=female, curr=male
    m_to_f_df = []
    for i in range(len(df)-1):
        if df.iloc[i]['gender'] == 'female' and df.iloc[i+1]['gender'] == 'male':
            m_to_f_df.append(df.iloc[i+1])
    m_to_f_df = pd.DataFrame(m_to_f_df)
    
    for t in thresholds:
        rate = (m_to_f_df['echoing_score'] > t).mean() * 100
        results.append({
            'metric': 'appropriation_M->F',
            'threshold': t,
            'result': rate
        })
        
    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'p22_sensitivity_analysis.csv', index=False)
    print("[OK] P22 completed: Sensitivity analysis performed.")

if __name__ == "__main__":
    analyze_sensitivity()
