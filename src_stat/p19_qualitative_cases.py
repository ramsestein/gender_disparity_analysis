import pandas as pd
from pub_utils import load_data, OUTPUT_DIR

def extract_qualitative_cases():
    df = load_data()
    
    # 19.1 Appropriation (High echoing, no attribution)
    app_cases = df[(df['echoing_score'] > 0.4) & (df['has_attribution'] == 0)].sort_values('echoing_score', ascending=False).head(10)
    
    # 19.2 Ignored questions
    # Identify questions followed by low echo response
    ignored_questions = []
    for i in range(len(df)-1):
        if df.iloc[i]['is_question'] and df.iloc[i+1]['echoing_score'] < 0.05:
            ignored_questions.append({
                'session': df.iloc[i]['session'],
                'question_speaker': df.iloc[i]['speaker'],
                'question_text': df.iloc[i]['text'] if 'text' in df.columns else "",
                'response_echo': df.iloc[i+1]['echoing_score']
            })
    ign_df = pd.DataFrame(ignored_questions).head(10)
    
    # 19.3 Backlash (High assertiveness followed by quick interruption)
    backlash_cases = df[(df['assertiveness_score'] > 0.7) & (df['interrupted_by_next'] == 1)].head(10)
    
    # Save CSVs
    app_cases.to_csv(OUTPUT_DIR / 'p19_appropriation_extreme_cases.csv', index=False)
    ign_df.to_csv(OUTPUT_DIR / 'p19_ignored_questions_extreme_cases.csv', index=False)
    backlash_cases.to_csv(OUTPUT_DIR / 'p19_backlash_extreme_cases.csv', index=False)
    
    print("[OK] P19 completed: Qualitative cases extracted.")

if __name__ == "__main__":
    extract_qualitative_cases()
