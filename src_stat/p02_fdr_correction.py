import pandas as pd
from pub_utils import load_data, OUTPUT_DIR, NUMERIC_VARS, CATEGORICAL_VARS
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.stats.multitest import multipletests

def run_fdr():
    df = load_data()
    all_p = []
    
    # Numeric Naive
    for var in NUMERIC_VARS:
        m = df[df['gender']=='male'][var]
        f = df[df['gender']=='female'][var]
        p = ttest_ind(m, f, equal_var=False).pvalue
        all_p.append({'variable': var, 'p_original': p})
        
    # Categorical Naive
    for var in CATEGORICAL_VARS:
        ct = pd.crosstab(df['gender'], df[var])
        p = chi2_contingency(ct)[1]
        all_p.append({'variable': var, 'p_original': p})
        
    p_df = pd.DataFrame(all_p)
    rejected, corrected, _, _ = multipletests(p_df['p_original'], method='fdr_bh')
    
    p_df['p_fdr_corrected'] = corrected
    p_df['significant_original'] = p_df['p_original'] < 0.05
    p_df['significant_after_fdr'] = rejected
    p_df['survived_correction'] = p_df['significant_original'] & p_df['significant_after_fdr']
    
    # Global Metrics
    orig_sig = p_df['significant_original'].sum()
    fdr_sig = p_df['significant_after_fdr'].sum()
    survival_rate = (fdr_sig / orig_sig * 100) if orig_sig > 0 else 0
    
    summary_text = f"Original significant: {orig_sig}/{len(p_df)}\nAfter FDR: {fdr_sig}/{len(p_df)}\nSurvival rate: {survival_rate:.1f}%"
    with open(OUTPUT_DIR / 'p02_fdr_global_summary.txt', 'w') as f:
        f.write(summary_text)
        
    p_df.to_csv(OUTPUT_DIR / 'p02_fdr_correction_summary.csv', index=False)
    print("[OK] P02 completed: fdr_correction_summary.csv generated.")

if __name__ == "__main__":
    run_fdr()
