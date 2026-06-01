"""
c04_cluster_enriched.py
=======================
Re-clustering con variables enriquecidas (Role, Specialty, citations)
+ análisis de sesgos de género por cada cluster resultante.

Entrada:  final_reports/resultados/csv/user_level_dataset_enriched.csv
Salida:   final_reports/resultados/csv/   (CSVs)
          final_reports/resultados/graficos/clustering_enriched/  (gráficos)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE / "final_reports" / "resultados" / "graficos" / "clustering_enriched"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e']

# ── 1. Cargar y preparar datos ─────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(CSV_DIR / "user_level_dataset_enriched.csv")
    print(f"Usuarios cargados: {len(df)}")

    # Columnas a excluir de las features
    exclude = {
        'user_id', 'gender', 'session', 'speaker',
        # Variables originales de texto/categoría
        'Role', 'Group', 'Country', 'Affiliation (Hospital)',
        'Specialty (ICU-Ane-Both)',
        'Number of citations', 'Year of qualification (specialty)',
        'Date of birth (DD/MM/YYYY)',
        # career_years: solo 1.7% cobertura → excluir
        'career_years',
        # session_stem si existiera
    }

    # Features numéricas existentes (comportamentales)
    behav_cols = [c for c in df.columns
                  if c not in exclude
                  and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
                  and df[c].var() > 0]

    # Nuevas features para incluir explícitamente
    new_features = [
        'role_known', 'is_moderator', 'is_speaker', 'is_public',
        'spec_known', 'is_ICU', 'is_Ane', 'is_Both',
        'log_citations',
    ]
    new_features = [f for f in new_features if f in df.columns]

    # Unir (evitando duplicados)
    feature_cols = list(dict.fromkeys(behav_cols + new_features))
    feature_cols = [c for c in feature_cols if df[c].var() > 0]

    print(f"Features behaviorales:  {len(behav_cols)}")
    print(f"Features nuevas:        {len(new_features)}")
    print(f"Total features:         {len(feature_cols)}")
    return df, feature_cols


# ── 2. K-Means con selección de k ─────────────────────────────────────────────
def run_kmeans(X, df, feature_cols):
    print("\n" + "="*60)
    print("  K-MEANS CLUSTERING (dataset enriquecido)")
    print("="*60)

    K_RANGE = range(2, 9)
    inertias, silhouettes = [], []

    for k in K_RANGE:
        km  = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, lbl)
        silhouettes.append(sil)
        print(f"  k={k}: Inertia={km.inertia_:.1f}, Silhouette={sil:.4f}")

    best_k = list(K_RANGE)[np.argmax(silhouettes)]
    print(f"\n  >> k optimo: {best_k} (Silhouette={max(silhouettes):.4f})")

    km_final = KMeans(n_clusters=best_k, n_init=20, random_state=42, max_iter=500)
    labels   = km_final.fit_predict(X)

    # Gráfico elbow + silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('K-Means enriquecido: selección de k', fontsize=14, fontweight='bold')
    ax1.plot(list(K_RANGE), inertias, 'o-', color='#3498db', linewidth=2, markersize=8)
    ax1.set_xlabel('k'); ax1.set_ylabel('Inercia'); ax1.set_title('Método del Codo')
    ax1.grid(True, alpha=0.3)
    ax2.plot(list(K_RANGE), silhouettes, 's-', color='#e74c3c', linewidth=2, markersize=8)
    ax2.axvline(x=best_k, color='#2ecc71', linestyle='--', linewidth=2, label=f'k={best_k}')
    ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette'); ax2.set_title('Silhouette Score')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_kmeans_elbow.png", dpi=200, bbox_inches='tight')
    plt.close()

    return labels, best_k, km_final


# ── 3. Visualización PCA ───────────────────────────────────────────────────────
def visualize(X, df, labels, best_k):
    pca = PCA(n_components=2, random_state=42)
    Xp  = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('PCA — Dataset enriquecido', fontsize=14, fontweight='bold')

    # Por cluster
    ax = axes[0]
    for c in range(best_k):
        m = labels == c
        ax.scatter(Xp[m, 0], Xp[m, 1], c=COLORS[c % len(COLORS)],
                   label=f'C{c}', alpha=0.6, s=35, edgecolors='white', lw=0.4)
    ax.set_title(f'Clusters (k={best_k})'); ax.legend(fontsize=8)
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
    ax.grid(True, alpha=0.2)

    # Por género
    ax = axes[1]
    gc = {'male': '#3498db', 'female': '#e74c3c'}
    for g, col in gc.items():
        m = df['gender'] == g
        ax.scatter(Xp[m, 0], Xp[m, 1], c=col, label=g.capitalize(),
                   alpha=0.5, s=35, edgecolors='white', lw=0.4)
    ax.set_title('Por Género'); ax.legend()
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
    ax.grid(True, alpha=0.2)

    # Por Role
    ax = axes[2]
    role_colors = {'Speaker': '#2ecc71', 'Moderator': '#3498db',
                   'public': '#e74c3c', None: '#cccccc'}
    df_temp = df.copy()
    df_temp['Role_plot'] = df_temp['Role'].fillna('unknown')
    for role in ['Moderator', 'Speaker', 'public', 'unknown']:
        m = df_temp['Role_plot'] == role
        col = role_colors.get(role, '#cccccc')
        ax.scatter(Xp[m, 0], Xp[m, 1], c=col, label=role,
                   alpha=0.7 if role != 'unknown' else 0.3,
                   s=50 if role != 'unknown' else 20,
                   edgecolors='white', lw=0.4)
    ax.set_title('Por Role'); ax.legend(fontsize=8)
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "02_pca_clusters_gender_role.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  PCA: PC1={var[0]*100:.1f}% + PC2={var[1]*100:.1f}% varianza explicada")


# ── 4. Profiling de clusters ───────────────────────────────────────────────────
def profile_clusters(df_profile, feature_cols, best_k):
    print("\n" + "="*60)
    print("  PROFILING DE CLUSTERS")
    print("="*60)

    # Tamaño
    sizes = df_profile['cluster'].value_counts().sort_index()
    for c, sz in sizes.items():
        print(f"  Cluster {c}: {sz} usuarios ({sz/len(df_profile)*100:.1f}%)")

    # Role por cluster
    print("\n  Role por cluster:")
    role_ct = pd.crosstab(df_profile['cluster'], df_profile['Role'].fillna('unknown'))
    print(role_ct.to_string())

    role_pct = pd.crosstab(df_profile['cluster'], df_profile['Role'].fillna('unknown'),
                            normalize='index') * 100
    print("\n  (porcentajes):")
    print(role_pct.round(1).to_string())
    role_pct.to_csv(CSV_DIR / "c04_role_by_cluster.csv")

    # Specialty por cluster
    if 'Specialty (ICU-Ane-Both)' in df_profile.columns:
        print("\n  Specialty por cluster:")
        spec_ct = pd.crosstab(df_profile['cluster'],
                               df_profile['Specialty (ICU-Ane-Both)'].fillna('unknown'))
        spec_pct = pd.crosstab(df_profile['cluster'],
                                df_profile['Specialty (ICU-Ane-Both)'].fillna('unknown'),
                                normalize='index') * 100
        print(spec_pct.round(1).to_string())
        spec_pct.to_csv(CSV_DIR / "c04_specialty_by_cluster.csv")

    # Group por cluster
    if 'Group' in df_profile.columns:
        print("\n  Group por cluster (top grupos):")
        grp_pct = pd.crosstab(df_profile['cluster'],
                               df_profile['Group'].fillna('unknown'),
                               normalize='index') * 100
        grp_pct.to_csv(CSV_DIR / "c04_group_by_cluster.csv")
        print(grp_pct.round(1).to_string())

    # Género por cluster
    print("\n  Género por cluster:")
    gen_ct  = pd.crosstab(df_profile['cluster'], df_profile['gender'])
    gen_pct = pd.crosstab(df_profile['cluster'], df_profile['gender'], normalize='index') * 100
    print(gen_ct.to_string())
    print(gen_pct.round(1).to_string())
    gen_pct.to_csv(CSV_DIR / "c04_gender_by_cluster.csv")

    if gen_ct.shape == (best_k, 2):
        chi2, p, dof, _ = chi2_contingency(gen_ct)
        print(f"\n  Chi²(género×cluster): χ²={chi2:.2f}, p={p:.4f}, dof={dof}")

    # Variables comportamentales clave por cluster
    key_vars = ['n_interventions', 'total_duration', 'mean_duration', 'mean_wpm',
                'mean_lexical_diversity', 'mean_conflict_score', 'mean_assertiveness_score',
                'pct_interrupts_previous', 'pct_has_hedge', 'pct_is_question',
                'pct_has_disagreement', 'is_top3_speaker', 'min_turn_number']
    key_vars = [v for v in key_vars if v in df_profile.columns]

    centroids = df_profile.groupby('cluster')[key_vars].mean()
    centroids.to_csv(CSV_DIR / "c04_cluster_centroids.csv")
    print("\n  Centroides (variables comportamentales):")
    print(centroids.round(3).to_string())

    # Heatmap
    global_m  = df_profile[key_vars].mean()
    global_s  = df_profile[key_vars].std().replace(0, 1)
    norm_cent = (centroids - global_m) / global_s
    short = {
        'n_interventions':'N Interv.','total_duration':'Dur. Total',
        'mean_duration':'Dur. Media','mean_wpm':'WPM',
        'mean_lexical_diversity':'Div. Léxica','mean_conflict_score':'Conflicto',
        'mean_assertiveness_score':'Asertividad','pct_interrupts_previous':'% Interrumpe',
        'pct_has_hedge':'% Hedge','pct_is_question':'% Preguntas',
        'pct_has_disagreement':'% Desacuerdo','is_top3_speaker':'Top-3 Hab.',
        'min_turn_number':'Turno 1ª'
    }
    col_labels = [short.get(c, c) for c in key_vars]

    fig, ax = plt.subplots(figsize=(16, max(4, best_k * 1.2)))
    im = ax.imshow(norm_cent.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f'Cluster {i}' for i in range(best_k)], fontsize=10)
    ax.set_title('Centroides normalizados por Cluster (enriquecido)', fontsize=13, fontweight='bold')
    for i in range(best_k):
        for j in range(len(key_vars)):
            val = norm_cent.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white' if abs(val) > 1 else 'black')
    plt.colorbar(im, ax=ax, label='Z-score')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_cluster_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()

    return centroids


# ── 5. Sesgos de género por cluster ───────────────────────────────────────────
def gender_bias_by_cluster(df_profile, feature_cols, best_k):
    print("\n" + "="*60)
    print("  SESGOS DE GÉNERO POR CLUSTER")
    print("="*60)

    bias_vars = [
        'n_interventions', 'mean_duration', 'total_duration', 'mean_wpm',
        'mean_lexical_diversity', 'mean_conflict_score', 'mean_assertiveness_score',
        'mean_echoing_score', 'pct_interrupts_previous', 'pct_interrupted_by_next',
        'pct_interruption_success', 'pct_has_hedge', 'pct_has_disagreement',
        'pct_has_agreement', 'pct_has_courtesy', 'pct_has_apology',
        'pct_is_question', 'pct_has_vulnerability', 'pct_is_mansplaining',
        'pct_is_backchannel', 'mean_overlap_duration', 'std_duration',
        'min_turn_number', 'is_top3_speaker'
    ]
    bias_vars = [v for v in bias_vars if v in df_profile.columns]

    all_results = []

    for c in range(best_k):
        cdata   = df_profile[df_profile['cluster'] == c]
        females = cdata[cdata['gender'] == 'female']
        males   = cdata[cdata['gender'] == 'male']
        n_f, n_m = len(females), len(males)

        # Determinar etiqueta del cluster basada en Role
        roles_in_cluster = cdata['Role'].fillna('unknown').value_counts()
        dominant = roles_in_cluster.index[0] if len(roles_in_cluster) else 'unknown'
        pct_panel = (cdata['Role'].isin(['Moderator', 'Speaker'])).mean() * 100
        pct_pub   = (cdata['Role'] == 'public').mean() * 100

        print(f"\n{'─'*55}")
        print(f"  CLUSTER {c}  ({n_f} mujeres / {n_m} hombres)")
        print(f"  Role → Panel:{pct_panel:.0f}%  Public:{pct_pub:.0f}%  "
              f"Unknown:{100-pct_panel-pct_pub:.0f}%")
        print(f"{'─'*55}")

        for var in bias_vars:
            f_vals = females[var].dropna()
            m_vals = males[var].dropna()
            if len(f_vals) < 3 or len(m_vals) < 3:
                continue
            try:
                u_stat, p_val = mannwhitneyu(f_vals, m_vals, alternative='two-sided')
            except Exception:
                continue

            pooled = np.sqrt(
                ((len(f_vals)-1)*f_vals.std()**2 + (len(m_vals)-1)*m_vals.std()**2) /
                (len(f_vals) + len(m_vals) - 2)
            )
            d = (f_vals.mean() - m_vals.mean()) / pooled if pooled > 0 else 0
            sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
                   else '*' if p_val < 0.05 else '')

            all_results.append({
                'cluster': c,
                'variable': var,
                'mean_female': round(f_vals.mean(), 4),
                'mean_male': round(m_vals.mean(), 4),
                'diff_pct': round((f_vals.mean()-m_vals.mean())/m_vals.mean()*100, 1)
                            if m_vals.mean() != 0 else 0,
                'cohens_d': round(d, 4),
                'p_value': round(p_val, 6),
                'significant': sig,
                'n_female': n_f, 'n_male': n_m,
                'pct_panel': round(pct_panel, 1),
                'pct_public': round(pct_pub, 1),
            })

            if sig:
                dir_label = "F>M" if d > 0 else "M>F"
                print(f"  {sig:>3} {var:<32} {dir_label}  d={d:+.3f}  p={p_val:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(CSV_DIR / "c04_gender_bias_by_cluster.csv", index=False)
    print(f"\n  ✅ Guardado en c04_gender_bias_by_cluster.csv")

    # ── Heatmap Cohen's d por cluster × variable ──────────────────────────────
    sig_vars = results_df[results_df['significant'] != '']['variable'].unique()
    if len(sig_vars) > 0:
        pivot = results_df[results_df['variable'].isin(sig_vars)].pivot_table(
            index='variable', columns='cluster', values='cohens_d', aggfunc='first'
        )
        pivot_p = results_df[results_df['variable'].isin(sig_vars)].pivot_table(
            index='variable', columns='cluster', values='p_value', aggfunc='first'
        )

        short_names = {
            'n_interventions':'N Interv.','mean_duration':'Dur. Media',
            'total_duration':'Dur. Total','mean_wpm':'WPM',
            'mean_lexical_diversity':'Div. Léxica','mean_conflict_score':'Conflicto',
            'mean_assertiveness_score':'Asertividad','mean_echoing_score':'Eco Léxico',
            'pct_interrupts_previous':'% Interrumpe','pct_interrupted_by_next':'% Interrumpido',
            'pct_interruption_success':'% Interr. Éxito','pct_has_hedge':'% Hedge',
            'pct_has_disagreement':'% Desacuerdo','pct_has_agreement':'% Acuerdo',
            'pct_has_courtesy':'% Cortesía','pct_has_apology':'% Disculpa',
            'pct_is_question':'% Preguntas','pct_has_vulnerability':'% Vulnerab.',
            'pct_is_mansplaining':'% Mansplaining','pct_is_backchannel':'% Backchannel',
            'mean_overlap_duration':'Solapamiento','std_duration':'Std Duración',
            'min_turn_number':'Turno 1ª','is_top3_speaker':'Top-3 Hab.'
        }
        row_labels = [short_names.get(v, v) for v in pivot.index]

        fig, ax = plt.subplots(figsize=(max(8, best_k * 2.5), max(6, len(pivot) * 0.5)))
        vmax = max(0.5, float(pivot.abs().max().max()))
        im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(best_k))
        ax.set_xticklabels([f'Cluster {c}' for c in pivot.columns],
                           fontsize=10, fontweight='bold')
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)

        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                d_val = pivot.values[i, j]
                p_val = pivot_p.values[i, j]
                if np.isnan(d_val):
                    continue
                sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
                       else '*' if p_val < 0.05 else '')
                color = 'white' if abs(d_val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{d_val:+.2f}{sig}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

        ax.set_title("Cohen's d por género dentro de cada cluster\n"
                     "(+ = Mujeres > Hombres, − = Hombres > Mujeres)",
                     fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label="Cohen's d")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "04_gender_bias_heatmap.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Heatmap guardado en 04_gender_bias_heatmap.png")

    return results_df


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  C04 — CLUSTERING ENRIQUECIDO + SESGOS DE GÉNERO")
    print("="*60)

    df, feature_cols = load_data()

    # Estandarizar
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].fillna(0))

    # K-Means
    labels, best_k, _ = run_kmeans(X, df, feature_cols)
    df['cluster'] = labels

    # Visualización PCA
    visualize(X, df, labels, best_k)

    # Profiling
    centroids = profile_clusters(df, feature_cols, best_k)

    # Sesgos de género por cluster
    results_df = gender_bias_by_cluster(df, feature_cols, best_k)

    # Guardar dataset final
    out = CSV_DIR / "user_level_enriched_with_clusters.csv"
    df.to_csv(out, index=False)

    # Resumen de hallazgos
    sig_total = results_df[results_df['significant'] != '']
    print(f"\n{'='*60}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Clusters encontrados: {best_k}")
    print(f"  Variables con sesgo significativo (total): {len(sig_total)}")
    for c in range(best_k):
        n_sig = len(sig_total[sig_total['cluster'] == c])
        print(f"    Cluster {c}: {n_sig} variables significativas")
    print(f"\n  Archivos:")
    print(f"    {out}")
    print(f"    {CSV_DIR}/c04_gender_bias_by_cluster.csv")
    print(f"    {FIG_DIR}/")


if __name__ == "__main__":
    main()
