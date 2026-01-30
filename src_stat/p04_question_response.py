import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pub_utils import load_data, OUTPUT_DIR, FIG_DIR, cohens_d, get_magnitude
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

def analyze_qa():
    df = load_data()
    df['is_question'] = df['is_question'].astype(bool)
    
    qa_pairs = []
    for i in range(len(df)-1):
        if df.iloc[i]['is_question']:
            q = df.iloc[i]
            r = df.iloc[i+1]
            qa_pairs.append({
                'session': q['session'],
                'questioner_gender': q['gender'],
                'responder_gender': r['gender'],
                'response_duration': r['duration'],
                'response_word_count': r['word_count'],
                'question_word_count': q['word_count'],
                'echoing_score': r['echoing_score'],
                'ignored': r['echoing_score'] < 0.1
            })
    
    qa_df = pd.DataFrame(qa_pairs)
    if qa_df.empty:
        print("Empty Q&A Data")
        return

    # 4.1 Response Duration
    stats_dur = qa_df.groupby('questioner_gender')['response_duration'].agg(['mean', 'median', 'std']).reset_index()
    
    # Mixed Model for duration
    try:
        qa_df['q_gender_bin'] = (qa_df['questioner_gender'] == 'male').astype(int)
        model = smf.mixedlm("response_duration ~ q_gender_bin", qa_df, groups=qa_df["session"]).fit()
        stats_dur['p_mixed'] = model.pvalues['q_gender_bin']
        stats_dur['coefficient'] = model.params['q_gender_bin']
    except: pass
    
    # 4.2 Matrix
    ct = pd.crosstab(qa_df['questioner_gender'], qa_df['responder_gender'])
    ct_pct = pd.crosstab(qa_df['questioner_gender'], qa_df['responder_gender'], normalize='index') * 100
    
    # 4.3 Ignored Rate
    ignored_stats = qa_df.groupby('questioner_gender')['ignored'].mean() * 100
    
    # 4.4 Quality Ratio
    qa_df['quality_ratio'] = qa_df['response_word_count'] / qa_df['question_word_count'].replace(0, 1)
    quality_stats = qa_df.groupby('questioner_gender')['quality_ratio'].mean()

    # Save CSVs
    stats_dur.to_csv(OUTPUT_DIR / 'p04_question_response_stats.csv', index=False)
    ct.to_csv(OUTPUT_DIR / 'p04_response_matrix_counts.csv')
    qa_df.to_csv(OUTPUT_DIR / 'p04_qa_raw_pairs.csv', index=False)
    
    # Plots
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=qa_df, x='questioner_gender', y='response_duration')
    plt.title('Response Duration by Questioner Gender')
    plt.savefig(FIG_DIR / 'p04_violin_plot_response_duration.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(ct_pct, annot=True, fmt='.1f', cmap='Blues')
    plt.title('Who responds to whom? (%)')
    plt.savefig(FIG_DIR / 'p04_heatmap_response_matrix.png', dpi=300)
    plt.close()

    print("[OK] P04 completed: Question-Response dynamics analyzed.")

if __name__ == "__main__":
    analyze_qa()
