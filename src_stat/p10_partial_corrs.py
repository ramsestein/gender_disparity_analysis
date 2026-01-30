import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pub_utils import load_data, OUTPUT_DIR, FIG_DIR
import pingouin as pg
from scipy.stats import norm

def analyze_partial_correlations():
    df = load_data()
    df['session_id'] = pd.factorize(df['session'])[0]
    
    pairs = [('assertiveness_score', 'interrupted_by_next'), ('conflict_score', 'interrupted_by_next')]
    results = []
    
    for x, y in pairs:
        try:
            pc = pg.partial_corr(data=df, x=x, y=y, covar='session_id')
            r_m = pg.partial_corr(data=df[df['gender']=='male'], x=x, y=y, covar='session_id')['r'].iloc[0]
            r_f = pg.partial_corr(data=df[df['gender']=='female'], x=x, y=y, covar='session_id')['r'].iloc[0]
            
            # Fisher's Z
            z_m, z_f = np.arctanh(r_m), np.arctanh(r_f)
            se = np.sqrt(1/(len(df[df['gender']=='male'])-3) + 1/(len(df[df['gender']=='female'])-3))
            z = (z_m - z_f) / se
            p_diff = 2 * (1 - norm.cdf(abs(z)))
            
            results.append({'pair': f"{x}x{y}", 'r_m': r_m, 'r_f': r_f, 'p_diff': p_diff})
        except: continue
        
    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'p10_partial_correlations.csv', index=False)
    print("[OK] P10 completed: Partial correlations analyzed.")

if __name__ == "__main__":
    analyze_partial_correlations()
