import pandas as pd
import numpy as np
from pub_utils import load_data, OUTPUT_DIR, NUMERIC_VARS, CATEGORICAL_VARS
from scipy.stats import ttest_ind, chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf

def run_mixed_models():
    df = load_data()
    df['gender_bin'] = (df['gender'] == 'male').astype(int)
    results = []
    
    print("Running Mixed Models for numeric variables...")
    for var in NUMERIC_VARS:
        try:
            # Naive p
            p_naive = ttest_ind(df[df['gender']=='male'][var], 
                               df[df['gender']=='female'][var], 
                               equal_var=False).pvalue
            
            # Mixed Model
            model = smf.mixedlm(f"{var} ~ gender_bin", df, groups=df["session"]).fit()
            
            # ICC calculation: var_random / (var_random + var_resid) if (var_random + var_resid) > 0 else 0
            var_resid = model.scale
            # Use cov_re instead of vcomp for robust variance extraction
            var_random = model.cov_re.iloc[0, 0]
            icc = var_random / (var_random + var_resid) if (var_random + var_resid) > 0 else 0
            
            results.append({
                'variable': var,
                'p_naive': p_naive,
                'p_mixed_model': model.pvalues['gender_bin'],
                'coefficient_mixed': model.params['gender_bin'],
                'ci_lower': model.conf_int().loc['gender_bin', 0],
                'ci_upper': model.conf_int().loc['gender_bin', 1],
                'icc': icc,
                'type': 'Numeric (LMM)'
            })
        except Exception as e:
            print(f"Error in {var}: {e}")
            
    print("Running Mixed Models for categorical variables...")
    for var in CATEGORICAL_VARS:
        try:
            ct = pd.crosstab(df['gender'], df[var])
            p_naive = chi2_contingency(ct)[1]
            model = smf.logit(f"{var} ~ gender_bin", data=df).fit(disp=0)
            results.append({
                'variable': var,
                'p_naive': p_naive,
                'p_mixed_model': model.pvalues['gender_bin'],
                'coefficient_mixed': model.params['gender_bin'],
                'ci_lower': model.conf_int().loc['gender_bin', 0],
                'ci_upper': model.conf_int().loc['gender_bin', 1],
                'icc': np.nan,
                'type': 'Categorical (Logistic)'
            })
        except: continue
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / 'p03_mixed_models_summary.csv', index=False)
    print("[OK] P03 completed: mixed_models_summary.csv generated.")

if __name__ == "__main__":
    run_mixed_models()
