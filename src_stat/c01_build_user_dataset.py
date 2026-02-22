"""
c01_build_user_dataset.py
=========================
Construye un dataset a nivel de usuario único.
Cada SPEAKER_XX en cada sesión se trata como un usuario diferente.
Se agregan todas las métricas de intervención a nivel de usuario.

Salida: final_reports/resultados/csv/user_level_dataset.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from pathlib import Path
from src_stat.core.utils import load_data, NUMERIC_VARS, CATEGORICAL_VARS

BASE_PATH = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_PATH / "final_reports" / "resultados" / "csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_user_dataset():
    """Carga datos de intervenciones y agrega a nivel de usuario único."""
    print("=" * 60)
    print("  PASO 1: Construcción del dataset de usuarios únicos")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Cargar datos completos (todas las sesiones)
    # ------------------------------------------------------------------
    df = load_data()
    print(f"\n  Intervenciones cargadas: {len(df):,}")
    print(f"  Sesiones: {df['session'].nunique()}")

    # ------------------------------------------------------------------
    # 2. Crear user_id único = session + speaker
    # ------------------------------------------------------------------
    df['user_id'] = df['session'].astype(str) + "__" + df['speaker'].astype(str)
    n_users = df['user_id'].nunique()
    print(f"  Usuarios únicos detectados: {n_users}")

    # ------------------------------------------------------------------
    # 3. Definir columnas booleanas y numéricas para agregar
    # ------------------------------------------------------------------
    bool_cols = [
        'has_overlap', 'interrupts_previous', 'interrupted_by_next',
        'interruption_success', 'has_hedge', 'has_disagreement',
        'has_agreement', 'has_apology', 'has_courtesy', 'is_question',
        'has_vulnerability', 'has_title', 'has_attribution',
        'is_backchannel', 'is_mansplaining'
    ]
    # Filtrar solo las que existen
    bool_cols = [c for c in bool_cols if c in df.columns]

    numeric_cols = [
        'duration', 'word_count', 'wpm', 'lexical_diversity',
        'latency_s', 'overlap_duration', 'echoing_score',
        'conflict_score', 'assertiveness_score', 'num_imperatives'
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Asegurar tipos numéricos
    for c in bool_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # ------------------------------------------------------------------
    # 4a. Calcular si el usuario está entre los 3 primeros hablantes de la sesión
    # ------------------------------------------------------------------
    # Para cada sesión, identificar los 3 primeros speakers distintos por start_time
    def get_top3_speakers(grp):
        first_appearances = grp.drop_duplicates(subset='speaker', keep='first').nsmallest(3, 'start_time')
        return set(first_appearances['speaker'])

    top3_by_session = df.groupby('session').apply(get_top3_speakers)
    df['is_top3_speaker'] = df.apply(
        lambda row: 1 if row['speaker'] in top3_by_session.get(row['session'], set()) else 0, axis=1
    )

    # Calcular el turno mínimo de cada usuario (cuándo entra por primera vez)
    first_turns = df.groupby('user_id')['turn_number'].min().reset_index()
    first_turns.columns = ['user_id', 'first_turn_number']

    # ------------------------------------------------------------------
    # 4. Construir agregaciones
    # ------------------------------------------------------------------
    agg_dict = {}

    # --- Género (moda) ---
    agg_dict['gender'] = ('gender', 'first')  # Consistente dentro de cada user

    # --- Número de intervenciones ---
    agg_dict['n_interventions'] = ('intervention_id', 'count')

    # --- Numéricas: media y std ---
    for col in numeric_cols:
        agg_dict[f'mean_{col}'] = (col, 'mean')
    # Estadísticas extra para algunas variables clave
    agg_dict['total_duration'] = ('duration', 'sum')
    agg_dict['std_duration'] = ('duration', 'std')
    agg_dict['total_word_count'] = ('word_count', 'sum')
    agg_dict['max_conflict_score'] = ('conflict_score', 'max')
    agg_dict['max_assertiveness_score'] = ('assertiveness_score', 'max')

    # --- Primera intervención ---
    agg_dict['is_top3_speaker'] = ('is_top3_speaker', 'max')  # 1 si es uno de los 3 primeros en hablar
    agg_dict['min_turn_number'] = ('turn_number', 'min')  # Turno más temprano
    agg_dict['mean_session_phase'] = ('start_time', lambda x: x.min())  # Momento de primera aparición

    # --- Booleanas: porcentaje (tasa) ---
    for col in bool_cols:
        agg_dict[f'pct_{col}'] = (col, 'mean')

    # --- Sentimiento (distribución porcentual) ---
    # Se calculará aparte

    # --- Emoción (distribución porcentual) ---
    # Se calculará aparte

    # ------------------------------------------------------------------
    # 5. Ejecutar la agregación
    # ------------------------------------------------------------------
    grouped = df.groupby('user_id')
    user_df = grouped.agg(**agg_dict)
    user_df = user_df.reset_index()

    # Rellenar NaN de std (usuarios con 1 sola intervención)
    user_df['std_duration'] = user_df['std_duration'].fillna(0)

    # Merge first turn number
    user_df = user_df.merge(first_turns, on='user_id', how='left')
    user_df['first_turn_number'] = user_df['first_turn_number'].fillna(999)

    # Normalizar mean_session_phase a [0,1] dentro de cada sesión
    # (ya es el start_time mínimo, lo normalizaremos después del merge con session)

    # ------------------------------------------------------------------
    # 6. Agregar distribución de sentimiento
    # ------------------------------------------------------------------
    if 'sentiment' in df.columns:
        sent_dummies = pd.get_dummies(df[['user_id', 'sentiment']].assign(val=1),
                                       columns=['sentiment'], prefix='sent')
        sent_cols = [c for c in sent_dummies.columns if c.startswith('sent_')]
        sent_rates = sent_dummies.groupby('user_id')[sent_cols].mean().reset_index()
        # Renombrar para claridad
        rename_map = {c: f'pct_{c}' for c in sent_cols}
        sent_rates = sent_rates.rename(columns=rename_map)
        user_df = user_df.merge(sent_rates, on='user_id', how='left')

    # ------------------------------------------------------------------
    # 7. Agregar distribución de emociones
    # ------------------------------------------------------------------
    if 'emotion' in df.columns:
        emo_dummies = pd.get_dummies(df[['user_id', 'emotion']].assign(val=1),
                                      columns=['emotion'], prefix='emo')
        emo_cols = [c for c in emo_dummies.columns if c.startswith('emo_')]
        emo_rates = emo_dummies.groupby('user_id')[emo_cols].mean().reset_index()
        rename_map = {c: f'pct_{c}' for c in emo_cols}
        emo_rates = emo_rates.rename(columns=rename_map)
        user_df = user_df.merge(emo_rates, on='user_id', how='left')

    # ------------------------------------------------------------------
    # 8. Extraer nombre de sesión limpio
    # ------------------------------------------------------------------
    user_df['session'] = user_df['user_id'].str.rsplit('__', n=1).str[0]
    user_df['speaker'] = user_df['user_id'].str.rsplit('__', n=1).str[1]

    # Rellenar NaN restantes con 0
    user_df = user_df.fillna(0)

    # ------------------------------------------------------------------
    # 9. Guardar
    # ------------------------------------------------------------------
    out_path = OUTPUT_DIR / "user_level_dataset.csv"
    user_df.to_csv(out_path, index=False)
    print(f"\n  ✅ Dataset guardado en: {out_path}")

    # ------------------------------------------------------------------
    # 10. Resumen
    # ------------------------------------------------------------------
    print(f"\n  📊 Resumen del dataset:")
    print(f"     Filas (usuarios): {len(user_df)}")
    print(f"     Columnas: {len(user_df.columns)}")
    print(f"     Distribución de género:")
    gender_counts = user_df['gender'].value_counts()
    for g, cnt in gender_counts.items():
        print(f"       {g}: {cnt} ({cnt/len(user_df)*100:.1f}%)")
    print(f"\n     Estadísticas descriptivas (variables clave):")
    key_cols = ['n_interventions', 'total_duration', 'mean_duration',
                'mean_wpm', 'mean_lexical_diversity', 'mean_conflict_score',
                'mean_assertiveness_score', 'pct_interrupts_previous',
                'pct_has_hedge', 'pct_is_question']
    key_cols = [c for c in key_cols if c in user_df.columns]
    print(user_df[key_cols].describe().round(3).to_string())

    return user_df


if __name__ == "__main__":
    build_user_dataset()
