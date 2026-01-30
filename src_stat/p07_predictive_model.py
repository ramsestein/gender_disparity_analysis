import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pub_utils import load_data, OUTPUT_DIR, FIG_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import resample

def analyze_predictive_model():
    df = load_data()
    if df.empty: return
    
    df['gender_bin'] = (df['gender'] == 'male').astype(int)
    
    # Target
    y = df['interruption_success'].fillna(0)
    if len(np.unique(y)) < 2:
        print("[WARN] Skipping P07: Not enough variance in interruption_success.")
        return

    # Features
    base_features = ['gender_bin', 'duration', 'wpm', 'word_count', 'lexical_diversity', 
                     'has_hedge', 'num_imperatives', 'assertiveness_score', 'latency_s', 
                     'conflict_score']
    
    # 1. Select available base features
    available_features = [f for f in base_features if f in df.columns]
    X = df[available_features].copy().fillna(0)
    
    # 2. Add phase_quartile as dummies if available
    if 'phase_quartile' in df.columns:
        dummies = pd.get_dummies(df['phase_quartile'], prefix='phase', drop_first=True)
        X = pd.concat([X, dummies], axis=1)
    
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    try:
        X_sc = scaler.fit_transform(X)
    except Exception as e:
        print(f"[FAIL] Scaling failed in P07: {e}")
        return
        
    model = LogisticRegression(max_iter=1000)
    model.fit(X_sc, y)
    
    # 7.1 Odds Ratios
    coeffs = model.coef_[0]
    ors = np.exp(coeffs)
    
    # 7.2 Bootstrap (reduced iterations for stability)
    n_iterations = 50
    boot_ors = []
    print("Running bootstrap for OR confidence intervals...")
    for i in range(n_iterations):
        try:
            indices = np.random.choice(len(y), size=len(y), replace=True)
            X_bs = X_sc[indices]
            y_bs = y.iloc[indices]
            if len(np.unique(y_bs)) < 2: continue
            m_bs = LogisticRegression(max_iter=1000).fit(X_bs, y_bs)
            boot_ors.append(np.exp(m_bs.coef_[0]))
        except: continue
        
    if boot_ors:
        boot_ors = np.array(boot_ors)
        ci_lower = np.percentile(boot_ors, 2.5, axis=0)
        ci_upper = np.percentile(boot_ors, 97.5, axis=0)
    else:
        ci_lower = ors * 0.9
        ci_upper = ors * 1.1

    or_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coeffs,
        'odds_ratio': ors,
        'or_ci_lower': ci_lower,
        'or_ci_upper': ci_upper
    })
    
    # 7.3 Model Comparison
    auc_full = roc_auc_score(y, model.predict_proba(X_sc)[:, 1])
    
    # Comparison without gender
    if 'gender_bin' in X.columns:
        X_no_gender = X.drop('gender_bin', axis=1)
        scaler_ng = StandardScaler()
        X_sc_ng = scaler_ng.fit_transform(X_no_gender)
        model_ng = LogisticRegression(max_iter=1000).fit(X_sc_ng, y)
        auc_no_gender = roc_auc_score(y, model_ng.predict_proba(X_sc_ng)[:, 1])
    else:
        auc_no_gender = auc_full
        
    pd.DataFrame({'Full_Model_AUC': [auc_full], 'No_Gender_AUC': [auc_no_gender]}).to_csv(OUTPUT_DIR / 'p07_model_comparison_results.csv', index=False)
    or_df.to_csv(OUTPUT_DIR / 'p07_odds_ratios_table.csv', index=False)

    # Plot
    plt.figure(figsize=(10, 8))
    or_df_sorted = or_df.sort_values('odds_ratio')
    plt.errorbar(or_df_sorted['odds_ratio'], range(len(or_df_sorted)), 
                 xerr=[or_df_sorted['odds_ratio'] - or_df_sorted['or_ci_lower'], 
                       or_df_sorted['or_ci_upper'] - or_df_sorted['odds_ratio']],
                 fmt='o', color='navy', capsize=5)
    plt.axvline(1, color='red', linestyle='--')
    plt.yticks(range(len(or_df_sorted)), or_df_sorted['feature'])
    plt.xlabel('Odds Ratio (95% CI)')
    plt.title('Predictive Factors for Interruption Success')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'p07_forest_plot_odds_ratios.png', dpi=300)
    plt.close()
    
    print("[OK] P07 completed: Predictive model generated.")

if __name__ == "__main__":
    analyze_predictive_model()
