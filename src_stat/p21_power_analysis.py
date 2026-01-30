import pandas as pd
import numpy as np
from pub_utils import load_data, OUTPUT_DIR, cohens_d
from statsmodels.stats.power import TTestIndPower

def analyze_power():
    df = load_data()
    n_total = len(df)
    
    # Priority variables (near significance)
    targets = ['interrupts_previous', 'interrupted_by_next']
    
    analysis = TTestIndPower()
    results = []
    
    for var in targets:
        m = df[df['gender']=='male'][var]
        f = df[df['gender']=='female'][var]
        d, g = cohens_d(m, f)
        
        # Power achieved with current N and d
        power = analysis.solve_power(effect_size=abs(g), nobs1=len(m), 
                                     ratio=len(f)/len(m), alpha=0.05)
        
        # Min detectable effect size for power=0.8
        min_d = analysis.solve_power(nobs1=len(m), ratio=len(f)/len(m), 
                                     alpha=0.05, power=0.8)
        
        results.append({
            'variable': var,
            'observed_hedges_g': g,
            'power_achieved': power,
            'minimum_detectable_g': min_d,
            'interpretation': "Poor power" if power < 0.8 else "Sufficient power"
        })
        
    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'p21_power_analysis.csv', index=False)
    print("[OK] P21 completed: Power analysis performed.")

if __name__ == "__main__":
    analyze_power()
