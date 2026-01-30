import pandas as pd
import numpy as np
from pub_utils import OUTPUT_DIR

def consolidate_master():
    print(" Consolidating Master Statistical Results...")
    
    try:
        es_df = pd.read_csv(OUTPUT_DIR / 'p01_effect_sizes_table.csv')
        fdr_df = pd.read_csv(OUTPUT_DIR / 'p02_fdr_correction_summary.csv')
        mixed_df = pd.read_csv(OUTPUT_DIR / 'p03_mixed_models_summary.csv')
        ci_num = pd.read_csv(OUTPUT_DIR / 'p11_confidence_intervals_numeric.csv')
        ci_cat = pd.read_csv(OUTPUT_DIR / 'p11_confidence_intervals_categorical.csv')
    except Exception as e:
        print(f"Missing some CSV dependency: {e}")
        return

    # Merge ES and FDR
    master = pd.merge(es_df, fdr_df, on='variable', how='left')
    
    # Merge Mixed Models results
    master = pd.merge(master, mixed_df[['variable', 'p_mixed_model', 'coefficient_mixed', 'icc']], on='variable', how='left')
    
    # Merge Confidence Intervals (numeric and cat pooled)
    cis_pool = pd.concat([
        ci_num[['variable', 'ci_lower', 'ci_upper', 'ci_crosses_zero']],
        ci_cat[['variable', 'ci_diff_lower', 'ci_diff_high']].rename(columns={'ci_diff_lower':'ci_lower', 'ci_diff_high':'ci_upper'})
    ], ignore_index=True)
    
    master = pd.merge(master, cis_pool, on='variable', how='left')
    
    # Final Directionality and Clean up
    def get_direction(row):
        # We need a reference. Let's use difference = male - female
        diff = row.get('difference', 0)
        if pd.isna(diff): return "N/A"
        if row.get('significant_after_fdr', False):
            return "male_higher" if diff > 0 else "female_higher"
        return "no_difference"
        
    master['direction'] = master.apply(get_direction, axis=1)
    
    master.to_csv(OUTPUT_DIR / 'MASTER_STATISTICAL_RESULTS.csv', index=False)
    print(" MASTER_STATISTICAL_RESULTS.csv generated successfully.")

if __name__ == "__main__":
    consolidate_master()
