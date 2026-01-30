import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, levene, normaltest, chi2_contingency, mannwhitneyu, binomtest, norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Dynamically determine the base path (go up two levels from src_stat/publication)
BASE_PATH = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = BASE_PATH / "final_reports" / "csv_cleaned"
OUTPUT_DIR = BASE_PATH / "final_reports" / "resultados"
FIG_DIR = OUTPUT_DIR / "graficos"

NUMERIC_VARS = ['duration', 'word_count', 'lexical_diversity', 'wpm', 'latency_s', 
                'overlap_duration', 'echoing_score', 'conflict_score', 'assertiveness_score']
CATEGORICAL_VARS = ['has_overlap', 'interrupts_previous', 'interrupted_by_next', 
                   'interruption_success', 'has_hedge', 'has_disagreement', 
                   'has_agreement', 'has_apology', 'has_courtesy', 'is_question']

def load_data():
    csv_files = glob.glob(str(INPUT_DIR / "*.csv"))
    all_data = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df = df[df['gender'].isin(['male', 'female'])].copy()
            df['session'] = os.path.basename(f)
            df = df.sort_values('start_time')
            
            # --- Missing Metrics Calculation ---
            
            # 1. Latency (start[i] - end[i-1])
            df['latency_s'] = (df['start_time'] - df['end_time'].shift(1)).fillna(0)
            
            # 2. Proxy for num_imperatives (if missing)
            if 'num_imperatives' not in df.columns:
                df['num_imperatives'] = 0 # Placeholder if NLP was not run for this
            
            # 3. Indices
            df['conflict_score'] = (df['has_disagreement'].astype(int) * 2 + 
                                     df['num_imperatives'] + 
                                     df['interrupts_previous'].astype(int) - 
                                     df['has_agreement'].astype(int) - 
                                     df['has_courtesy'].astype(int))
            
            df['assertiveness_score'] = (df['num_imperatives'] * 2 + 
                                          df['has_disagreement'].astype(int) + 
                                          (1 - df['has_hedge'].astype(int)) + 
                                          (1 - df['has_apology'].astype(int)))
            
            # 4. Temporal Phases
            total_dur = df['end_time'].max() if not df.empty else 1
            df['session_phase'] = df['start_time'] / total_dur
            df['phase_quartile'] = pd.cut(df['session_phase'], bins=[0, 0.25, 0.5, 0.75, 1.05], labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            # 5. Climate
            # Density of interruptions in rolling window of 5
            df['interruption_climate'] = df['interrupted_by_next'].rolling(window=5, min_periods=1).mean()
            df['climate_type'] = df['interruption_climate'].apply(lambda x: 'hostile' if x > 0.3 else ('calm' if x < 0.1 else 'neutral'))
            
            # Ensure boolean-like columns are numeric
            bool_cols = CATEGORICAL_VARS + ['has_vulnerability', 'has_title', 'has_attribution', 'is_backchannel']
            for col in bool_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
            all_data.append(df)
        except Exception as e:
            print(f"Error in {f}: {e}")
            
    if not all_data: return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2: return 0, 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0: return 0, 0
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    correction_factor = 1 - (3 / (4*(n1+n2-2) - 1))
    hedges_g = d * correction_factor
    return d, hedges_g

def cramers_v_calc(contingency_table):
    try:
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        if n == 0 or min_dim == 0: return 0, "negligible", chi2, p
        v = np.sqrt(chi2 / (n * min_dim))
        if min_dim == 1:
            if v < 0.1: magnitude = "negligible"
            elif v < 0.3: magnitude = "small"
            elif v < 0.5: magnitude = "medium"
            else: magnitude = "large"
        else:
            if v < 0.07: magnitude = "negligible"
            elif v < 0.21: magnitude = "small"
            elif v < 0.35: magnitude = "medium"
            else: magnitude = "large"
        return v, magnitude, chi2, p
    except:
        return 0, "N/A", 0, 1.0

def get_magnitude(val, type='hedges_g'):
    val = abs(val)
    if type == 'hedges_g':
        if val < 0.2: return "negligible"
        if val < 0.5: return "small"
        if val < 0.8: return "medium"
        return "large"
    return "N/A"
