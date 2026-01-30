import pandas as pd
from pub_utils import load_data, OUTPUT_DIR

def analyze_subgroups():
    df = load_data()
    if df.empty or 'conflict_score' not in df.columns: 
        print("Skipping P20: Data empty or missing conflict_score")
        return
    
    try:
        session_conflict = df.groupby('session')['conflict_score'].mean()
        df['conflict_level'] = pd.qcut(session_conflict.reindex(df['session']).values, 3, labels=['low', 'med', 'high'], duplicates='drop')
        
        sub_conflict = df.groupby(['conflict_level', 'gender'])['duration'].mean().unstack().fillna(0)
        sub_conflict.to_csv(OUTPUT_DIR / 'p20_subgroup_analysis_conflict.csv')
        print("[OK] P20 completed: Subgroups analyzed.")
    except Exception as e:
        print(f"Error in P20: {e}")

if __name__ == "__main__":
    analyze_subgroups()
