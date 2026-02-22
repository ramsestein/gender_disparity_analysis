"""
c02_cluster_analysis.py
=======================
Análisis de cluster no supervisado sobre el dataset de usuarios únicos.
Técnicas: K-Means, DBSCAN, Jerárquico, PCA, t-SNE.
Incluye profiling de clusters y análisis de distribución de género.

Entrada:  final_reports/resultados/csv/user_level_dataset.csv
Salida:   final_reports/resultados/csv/  (CSVs)
          final_reports/resultados/graficos/clustering/  (Gráficos)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import f_oneway, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_PATH / "final_reports" / "resultados" / "csv"
FIG_DIR = BASE_PATH / "final_reports" / "resultados" / "graficos" / "clustering"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ====================================================================
# Paleta de colores premium
# ====================================================================
COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
GENDER_COLORS = {'male': '#3498db', 'female': '#e74c3c'}


def load_user_data():
    """Carga el dataset de usuarios y devuelve features numéricas."""
    path = CSV_DIR / "user_level_dataset.csv"
    df = pd.read_csv(path)
    print(f"  Usuarios cargados: {len(df)}")

    # Columnas a excluir del clustering
    exclude = ['user_id', 'gender', 'session', 'speaker']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    # Eliminar columnas con varianza 0
    variances = df[feature_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        print(f"  ⚠️  Eliminando {len(zero_var)} variables con varianza 0: {zero_var[:5]}...")
        feature_cols = [c for c in feature_cols if c not in zero_var]

    print(f"  Features para clustering: {len(feature_cols)}")
    return df, feature_cols


def standardize(df, feature_cols):
    """Estandariza las features con StandardScaler."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].fillna(0))
    return X, scaler


# ====================================================================
# K-MEANS
# ====================================================================
def run_kmeans(X, df, feature_cols):
    """K-Means con método del codo y Silhouette."""
    print("\n" + "=" * 60)
    print("  K-MEANS CLUSTERING")
    print("=" * 60)

    K_RANGE = range(2, 11)
    inertias = []
    silhouettes = []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        print(f"    k={k}: Inertia={km.inertia_:.1f}, Silhouette={sil:.4f}")

    # --- Gráfico Elbow + Silhouette ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('K-Means: Selección del número óptimo de clusters', fontsize=14, fontweight='bold')

    ax1.plot(list(K_RANGE), inertias, 'o-', color='#3498db', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de clusters (k)', fontsize=12)
    ax1.set_ylabel('Inercia (WCSS)', fontsize=12)
    ax1.set_title('Método del Codo', fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(K_RANGE), silhouettes, 's-', color='#e74c3c', linewidth=2, markersize=8)
    ax2.set_xlabel('Número de clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Marcar el óptimo
    best_k = list(K_RANGE)[np.argmax(silhouettes)]
    ax2.axvline(x=best_k, color='#2ecc71', linestyle='--', linewidth=2, label=f'Óptimo k={best_k}')
    ax2.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_kmeans_elbow_silhouette.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ K óptimo por Silhouette: {best_k} (score={max(silhouettes):.4f})")

    # --- Silhouette Plot detallado ---
    km_final = KMeans(n_clusters=best_k, n_init=20, random_state=42, max_iter=500)
    labels = km_final.fit_predict(X)
    sil_vals = silhouette_samples(X, labels)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10
    for i in range(best_k):
        cluster_vals = sil_vals[labels == i]
        cluster_vals.sort()
        size = cluster_vals.shape[0]
        y_upper = y_lower + size
        color = COLORS[i % len(COLORS)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals, facecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, f'C{i}', fontsize=11, fontweight='bold')
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_score(X, labels), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Silhouette coefficient', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_title(f'Silhouette Plot — K-Means (k={best_k})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_kmeans_silhouette_plot.png", dpi=200, bbox_inches='tight')
    plt.close()

    return labels, best_k, km_final


# ====================================================================
# DBSCAN
# ====================================================================
def run_dbscan(X, df):
    """DBSCAN con k-distance plot para selección de eps."""
    print("\n" + "=" * 60)
    print("  DBSCAN CLUSTERING")
    print("=" * 60)

    # --- k-distance plot ---
    k_neighbors = min(10, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    distances = np.sort(distances[:, k_neighbors - 1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, color='#3498db', linewidth=1.5)
    ax.set_xlabel('Puntos ordenados', fontsize=12)
    ax.set_ylabel(f'{k_neighbors}-NN Distance', fontsize=12)
    ax.set_title('K-Distance Plot para selección de eps', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Heurística: percentil 90 de las distancias
    eps_auto = np.percentile(distances, 90)
    ax.axhline(y=eps_auto, color='#e74c3c', linestyle='--', linewidth=2, label=f'eps sugerido = {eps_auto:.2f}')
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_dbscan_kdistance.png", dpi=200, bbox_inches='tight')
    plt.close()

    # --- Ejecutar DBSCAN con varios eps ---
    best_sil = -1
    best_eps = eps_auto
    best_labels = None

    for eps in [eps_auto * 0.7, eps_auto * 0.85, eps_auto, eps_auto * 1.15, eps_auto * 1.3]:
        for min_s in [3, 5, 7]:
            db = DBSCAN(eps=eps, min_samples=min_s)
            lbl = db.fit_predict(X)
            n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
            n_noise = (lbl == -1).sum()
            if n_clusters >= 2:
                # Silhouette solo sobre no-ruido
                mask = lbl != -1
                if mask.sum() > n_clusters:
                    sil = silhouette_score(X[mask], lbl[mask])
                    if sil > best_sil:
                        best_sil = sil
                        best_eps = eps
                        best_labels = lbl.copy()

    if best_labels is None:
        print("  ⚠️  DBSCAN no encontró clusters válidos. Usando eps por defecto.")
        db = DBSCAN(eps=eps_auto, min_samples=3)
        best_labels = db.fit_predict(X)

    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    n_noise = (best_labels == -1).sum()
    print(f"  ✅ DBSCAN: {n_clusters} clusters, {n_noise} outliers ({n_noise/len(X)*100:.1f}%)")
    if best_sil > -1:
        print(f"     Silhouette (sin outliers): {best_sil:.4f}")

    return best_labels


# ====================================================================
# CLUSTERING JERÁRQUICO
# ====================================================================
def run_hierarchical(X, df, best_k):
    """Clustering aglomerativo con dendrograma."""
    print("\n" + "=" * 60)
    print("  CLUSTERING JERÁRQUICO")
    print("=" * 60)

    # --- Dendrograma ---
    Z = linkage(X, method='ward')

    fig, ax = plt.subplots(figsize=(16, 7))
    max_display = min(50, len(X))
    dendrogram(Z, truncate_mode='lastp', p=max_display, ax=ax,
               color_threshold=Z[-(best_k - 1), 2],
               above_threshold_color='#95a5a6')
    ax.set_title(f'Dendrograma — Clustering Jerárquico (Ward)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Usuarios (o clústers de usuarios)', fontsize=12)
    ax.set_ylabel('Distancia', fontsize=12)
    ax.axhline(y=Z[-(best_k - 1), 2], color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Corte en k={best_k}')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_hierarchical_dendrogram.png", dpi=200, bbox_inches='tight')
    plt.close()

    # --- Corte ---
    hc = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    labels = hc.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"  ✅ Jerárquico (k={best_k}): Silhouette={sil:.4f}")

    return labels


# ====================================================================
# PCA + t-SNE
# ====================================================================
def run_dimensionality_reduction(X, df, km_labels, dbscan_labels, hc_labels, best_k):
    """PCA y t-SNE para visualización 2D."""
    print("\n" + "=" * 60)
    print("  REDUCCIÓN DE DIMENSIONALIDAD")
    print("=" * 60)

    # --- PCA ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_
    print(f"  PCA: Varianza explicada = {var_explained[0]:.3f} + {var_explained[1]:.3f} = {sum(var_explained):.3f}")

    # --- t-SNE ---
    perp = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    print(f"  t-SNE: completado (perplexity={perp})")

    # --- PCA por K-Means + Género (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Visualización 2D de Clusters de Usuarios', fontsize=16, fontweight='bold')

    # PCA — K-Means
    ax = axes[0, 0]
    for c in range(best_k):
        mask = km_labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS[c % len(COLORS)],
                   label=f'Cluster {c}', alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
    ax.set_title(f'PCA — K-Means (k={best_k})', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # PCA — Género
    ax = axes[0, 1]
    for g, color in GENDER_COLORS.items():
        mask = df['gender'] == g
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color,
                   label=g.capitalize(), alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    ax.set_title('PCA — por Género', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # t-SNE — K-Means
    ax = axes[1, 0]
    for c in range(best_k):
        mask = km_labels == c
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=COLORS[c % len(COLORS)],
                   label=f'Cluster {c}', alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
    ax.set_title(f't-SNE — K-Means (k={best_k})', fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # t-SNE — Género
    ax = axes[1, 1]
    for g, color in GENDER_COLORS.items():
        mask = df['gender'] == g
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color,
                   label=g.capitalize(), alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    ax.set_title('t-SNE — por Género', fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "05_pca_tsne_visualization.png", dpi=200, bbox_inches='tight')
    plt.close()

    # --- PCA: Varianza explicada acumulada ---
    pca_full = PCA(random_state=42)
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-', color='#3498db', linewidth=2, markersize=4)
    ax.axhline(y=0.9, color='#e74c3c', linestyle='--', linewidth=1.5, label='90% varianza')
    n_90 = np.argmax(cumvar >= 0.9) + 1
    ax.axvline(x=n_90, color='#2ecc71', linestyle='--', linewidth=1.5, label=f'{n_90} componentes')
    ax.set_xlabel('Número de componentes', fontsize=12)
    ax.set_ylabel('Varianza explicada acumulada', fontsize=12)
    ax.set_title('PCA — Varianza Explicada Acumulada', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_pca_variance_explained.png", dpi=200, bbox_inches='tight')
    plt.close()

    return X_pca, X_tsne


# ====================================================================
# PROFILING DE CLUSTERS
# ====================================================================
def profile_clusters(df, feature_cols, km_labels, best_k):
    """Análisis de las características de cada cluster."""
    print("\n" + "=" * 60)
    print("  PROFILING DE CLUSTERS")
    print("=" * 60)

    df_profile = df.copy()
    df_profile['cluster'] = km_labels

    # ------------------------------------------------------------------
    # 1. Medias por cluster
    # ------------------------------------------------------------------
    cluster_means = df_profile.groupby('cluster')[feature_cols].mean()
    cluster_means.to_csv(CSV_DIR / "cluster_centroids.csv")
    print(f"\n  📊 Centroides guardados en cluster_centroids.csv")

    # Tabla resumida con variables clave
    key_vars = ['n_interventions', 'total_duration', 'mean_duration', 'mean_wpm',
                'mean_lexical_diversity', 'mean_conflict_score', 'mean_assertiveness_score',
                'pct_interrupts_previous', 'pct_has_hedge', 'pct_has_disagreement',
                'pct_is_question', 'pct_has_courtesy', 'mean_echoing_score']
    key_vars = [v for v in key_vars if v in feature_cols]

    print("\n  Variables clave por cluster (medias):")
    print(cluster_means[key_vars].round(3).to_string())

    # ------------------------------------------------------------------
    # 2. ANOVA por variable
    # ------------------------------------------------------------------
    anova_results = []
    for col in feature_cols:
        groups = [df_profile[df_profile['cluster'] == c][col].dropna().values for c in range(best_k)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            try:
                f_stat, p_val = f_oneway(*groups)
                anova_results.append({
                    'variable': col,
                    'F_statistic': round(f_stat, 4),
                    'p_value': round(p_val, 6),
                    'significant': '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
                })
            except:
                pass

    anova_df = pd.DataFrame(anova_results).sort_values('p_value')
    anova_df.to_csv(CSV_DIR / "cluster_anova_results.csv", index=False)
    print(f"\n  📊 ANOVA — Top 10 variables más discriminantes:")
    print(anova_df.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Distribución de género por cluster
    # ------------------------------------------------------------------
    ct = pd.crosstab(df_profile['cluster'], df_profile['gender'])
    ct_pct = pd.crosstab(df_profile['cluster'], df_profile['gender'], normalize='index') * 100
    print(f"\n  📊 Distribución de género por cluster:")
    print(ct.to_string())
    print(f"\n  (Porcentajes):")
    print(ct_pct.round(1).to_string())

    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        try:
            chi2, p, dof, expected = chi2_contingency(ct)
            print(f"\n  Chi-cuadrado: χ²={chi2:.2f}, p={p:.6f}, dof={dof}")
        except:
            pass

    ct_pct.to_csv(CSV_DIR / "cluster_gender_distribution.csv")

    # ------------------------------------------------------------------
    # 4. Tamaño de clusters
    # ------------------------------------------------------------------
    cluster_sizes = df_profile['cluster'].value_counts().sort_index()
    print(f"\n  📊 Tamaño de clusters:")
    for c, size in cluster_sizes.items():
        print(f"     Cluster {c}: {size} usuarios ({size/len(df)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 5. Heatmap de centroides normalizados
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 8))
    # Normalizar centroides para heatmap (z-score respecto a la media global)
    global_means = df[feature_cols].mean()
    global_stds = df[feature_cols].std()
    global_stds[global_stds == 0] = 1
    norm_centroids = (cluster_means[key_vars] - global_means[key_vars]) / global_stds[key_vars]

    # Nombres cortos para las variables
    short_names = {
        'n_interventions': 'N Interv.',
        'total_duration': 'Duración Total',
        'mean_duration': 'Duración Media',
        'mean_wpm': 'WPM',
        'mean_lexical_diversity': 'Divers. Léxica',
        'mean_conflict_score': 'Conflictividad',
        'mean_assertiveness_score': 'Asertividad',
        'pct_interrupts_previous': '% Interrumpe',
        'pct_has_hedge': '% Hedge',
        'pct_has_disagreement': '% Desacuerdo',
        'pct_is_question': '% Preguntas',
        'pct_has_courtesy': '% Cortesía',
        'mean_echoing_score': 'Eco Léxico'
    }
    col_labels = [short_names.get(c, c) for c in key_vars]

    im = ax.imshow(norm_centroids.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f'Cluster {i}' for i in range(best_k)], fontsize=11)
    ax.set_title('Centroides normalizados por Cluster (K-Means)', fontsize=14, fontweight='bold')

    # Añadir valores en celdas
    for i in range(best_k):
        for j in range(len(key_vars)):
            val = norm_centroids.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if abs(val) > 1 else 'black')

    plt.colorbar(im, ax=ax, label='Z-score vs. media global')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_cluster_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # 6. Gráfico de barras de género por cluster
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(best_k)
    width = 0.35
    if 'female' in ct.columns and 'male' in ct.columns:
        ax.bar(x - width/2, ct['female'], width, label='Female', color='#e74c3c', alpha=0.85)
        ax.bar(x + width/2, ct['male'], width, label='Male', color='#3498db', alpha=0.85)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Número de usuarios', fontsize=12)
    ax.set_title('Distribución de Género por Cluster', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(best_k)])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_cluster_gender_bars.png", dpi=200, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # 7. Radar chart por cluster (variables clave)
    # ------------------------------------------------------------------
    if len(key_vars) >= 4:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(key_vars), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar

        for c in range(best_k):
            values = norm_centroids.loc[c, key_vars].values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {c}',
                    color=COLORS[c % len(COLORS)])
            ax.fill(angles, values, alpha=0.1, color=COLORS[c % len(COLORS)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_title('Perfil de Clusters (Radar)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "09_cluster_radar.png", dpi=200, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # 8. Distribución de clusters por sesión (homogeneidad)
    # ------------------------------------------------------------------
    print(f"\n  {'='*50}")
    print(f"  DISTRIBUCIÓN DE CLUSTERS POR SESIÓN")
    print(f"  {'='*50}")

    if 'session' in df_profile.columns:
        # Crosstab sesión × cluster
        ct_session = pd.crosstab(df_profile['session'], df_profile['cluster'])
        ct_session_pct = pd.crosstab(df_profile['session'], df_profile['cluster'], normalize='index') * 100

        # Chi-cuadrado para homogeneidad
        if ct_session.shape[0] >= 2 and ct_session.shape[1] >= 2:
            try:
                chi2_s, p_s, dof_s, exp_s = chi2_contingency(ct_session)
                print(f"\n  Chi-cuadrado (sesiones × clusters): χ²={chi2_s:.2f}, p={p_s:.6f}, dof={dof_s}")
                if p_s < 0.05:
                    print(f"  → Los clusters NO se distribuyen homogéneamente entre sesiones (p<0.05)")
                else:
                    print(f"  → Los clusters se distribuyen razonablemente uniforme entre sesiones (p≥0.05)")
            except:
                pass

        # Estadísticas por cluster: en cuántas sesiones aparecen
        for c in range(best_k):
            sessions_with_c = (ct_session[c] > 0).sum()
            total_sessions = len(ct_session)
            print(f"  Cluster {c}: presente en {sessions_with_c}/{total_sessions} sesiones ({sessions_with_c/total_sessions*100:.1f}%)")

        # Guardar tabla sesión×cluster
        ct_session_pct.to_csv(CSV_DIR / "cluster_session_distribution.csv")

        # Gráfico: % de cada cluster por sesión (heatmap de proporciones)
        fig, ax = plt.subplots(figsize=(8, max(8, len(ct_session) * 0.25)))
        # Solo mostrar el % del cluster 0 si hay 2 clusters (el otro es complementario)
        if best_k <= 3:
            im = ax.imshow(ct_session_pct.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=100)
        else:
            im = ax.imshow(ct_session_pct.values, cmap='RdYlBu_r', aspect='auto')

        ax.set_xticks(range(best_k))
        ax.set_xticklabels([f'C{i}' for i in range(best_k)], fontsize=10)
        # Simplificar nombres de sesión (solo primeros 30 chars)
        session_labels = [s[:35] + '...' if len(s) > 35 else s for s in ct_session_pct.index]
        ax.set_yticks(range(len(session_labels)))
        ax.set_yticklabels(session_labels, fontsize=6)
        ax.set_title('Distribución de Clusters por Sesión (%)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='% de usuarios en cluster')
        plt.tight_layout()
        fig.savefig(FIG_DIR / "10_cluster_session_heatmap.png", dpi=200, bbox_inches='tight')
        plt.close()

        # Boxplot de la proporción del cluster dominante por sesión
        fig, ax = plt.subplots(figsize=(10, 5))
        for c in range(best_k):
            ax.boxplot(ct_session_pct[c].values, positions=[c], widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor=COLORS[c % len(COLORS)], alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))
        ax.set_xticks(range(best_k))
        ax.set_xticklabels([f'Cluster {i}' for i in range(best_k)], fontsize=11)
        ax.set_ylabel('% de usuarios de la sesión', fontsize=12)
        ax.set_title('Variabilidad del % de cada Cluster entre Sesiones', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(FIG_DIR / "11_cluster_session_boxplot.png", dpi=200, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # 9. Análisis de "primer hablante" por cluster
    # ------------------------------------------------------------------
    print(f"\n  {'='*50}")
    print(f"  ANÁLISIS DE PRIMEROS HABLANTES (TOP 3) POR CLUSTER")
    print(f"  {'='*50}")

    if 'is_top3_speaker' in df_profile.columns:
        # Proporción de primeros hablantes por cluster
        first_speaker_rates = df_profile.groupby('cluster')['is_top3_speaker'].agg(['mean', 'sum', 'count'])
        first_speaker_rates.columns = ['pct_top3_speaker', 'n_top3_speakers', 'n_total']
        first_speaker_rates['pct_top3_speaker'] *= 100
        print(f"\n  Tasa de top-3 primeros hablantes por cluster:")
        for c, row in first_speaker_rates.iterrows():
            print(f"    Cluster {c}: {row['n_top3_speakers']:.0f}/{row['n_total']:.0f} entre los 3 primeros ({row['pct_top3_speaker']:.1f}%)")

        # Chi-cuadrado: ¿la proporción de primeros hablantes difiere entre clusters?
        ct_first = pd.crosstab(df_profile['cluster'], df_profile['is_top3_speaker'])
        if ct_first.shape[0] >= 2 and ct_first.shape[1] >= 2:
            try:
                chi2_f, p_f, dof_f, _ = chi2_contingency(ct_first)
                print(f"\n  Chi-cuadrado (cluster × top3_hablante): χ²={chi2_f:.2f}, p={p_f:.6f}")
                if p_f < 0.05:
                    print(f"  → Hay diferencia significativa en quién inicia la conversación por cluster")
                else:
                    print(f"  → No hay diferencia significativa (ambos clusters inician por igual)")
            except:
                pass

        first_speaker_rates.to_csv(CSV_DIR / "cluster_top3_speaker_rates.csv")

    if 'min_turn_number' in df_profile.columns:
        # Turno medio de primera aparición por cluster
        print(f"\n  Turno de primera intervención por cluster:")
        for c in range(best_k):
            cluster_data = df_profile[df_profile['cluster'] == c]
            mean_turn = cluster_data['min_turn_number'].mean()
            median_turn = cluster_data['min_turn_number'].median()
            print(f"    Cluster {c}: Turno medio={mean_turn:.1f}, Mediana={median_turn:.0f}")

        # Test Mann-Whitney para diferencia de turnos
        if best_k == 2:
            from scipy.stats import mannwhitneyu
            g0 = df_profile[df_profile['cluster'] == 0]['min_turn_number']
            g1 = df_profile[df_profile['cluster'] == 1]['min_turn_number']
            u_stat, p_turn = mannwhitneyu(g0, g1, alternative='two-sided')
            print(f"\n  Mann-Whitney U (turno 1ª intervención): U={u_stat:.0f}, p={p_turn:.6f}")

        # Gráfico: distribución del turno de primera aparición por cluster
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Boxplot del turno de primera aparición
        bp_data = [df_profile[df_profile['cluster'] == c]['min_turn_number'].values for c in range(best_k)]
        bp = ax1.boxplot(bp_data, labels=[f'Cluster {c}' for c in range(best_k)],
                         patch_artist=True, showfliers=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
        ax1.set_ylabel('Turno de primera intervención', fontsize=12)
        ax1.set_title('¿Cuándo intervienen por primera vez?', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Barras: % de primeros hablantes por cluster
        if 'is_top3_speaker' in df_profile.columns:
            pcts = [df_profile[df_profile['cluster'] == c]['is_top3_speaker'].mean() * 100 for c in range(best_k)]
            bars = ax2.bar(range(best_k), pcts, color=[COLORS[c % len(COLORS)] for c in range(best_k)], alpha=0.85)
            for bar, pct in zip(bars, pcts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{pct:.1f}%', ha='center', fontsize=12, fontweight='bold')
            ax2.set_xticks(range(best_k))
            ax2.set_xticklabels([f'Cluster {c}' for c in range(best_k)])
            ax2.set_ylabel('% top-3 primeros hablantes', fontsize=12)
            ax2.set_title('¿Están entre los 3 primeros en hablar?', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(FIG_DIR / "12_cluster_first_speaker.png", dpi=200, bbox_inches='tight')
        plt.close()

    return df_profile


# ====================================================================
# INTERPRETACIÓN AUTOMÁTICA
# ====================================================================
def interpret_clusters(df_profile, feature_cols, best_k):
    """Genera una interpretación textual de cada cluster."""
    print("\n" + "=" * 60)
    print("  INTERPRETACIÓN DE CLUSTERS")
    print("=" * 60)

    global_means = df_profile[feature_cols].mean()
    interpretations = []

    for c in range(best_k):
        cluster_data = df_profile[df_profile['cluster'] == c]
        cluster_means = cluster_data[feature_cols].mean()
        diffs = (cluster_means - global_means) / global_means.replace(0, 1)

        # Variables más altas y más bajas
        top_high = diffs.nlargest(5)
        top_low = diffs.nsmallest(5)

        n = len(cluster_data)
        pct_female = (cluster_data['gender'] == 'female').mean() * 100

        desc = f"\n  Cluster {c} ({n} usuarios, {pct_female:.0f}% femenino):"
        desc += f"\n    ↑ Destacan por:"
        for var, val in top_high.items():
            if val > 0.1:
                desc += f"\n      • {var}: +{val*100:.0f}% sobre media"
        desc += f"\n    ↓ Bajo en:"
        for var, val in top_low.items():
            if val < -0.1:
                desc += f"\n      • {var}: {val*100:.0f}% bajo media"

        print(desc)
        interpretations.append({
            'cluster': c,
            'n_users': n,
            'pct_female': round(pct_female, 1),
            'top_features_high': ', '.join([f"{v}(+{d*100:.0f}%)" for v, d in top_high.items() if d > 0.1]),
            'top_features_low': ', '.join([f"{v}({d*100:.0f}%)" for v, d in top_low.items() if d < -0.1])
        })

    interp_df = pd.DataFrame(interpretations)
    interp_df.to_csv(CSV_DIR / "cluster_interpretation.csv", index=False)
    return interp_df


# ====================================================================
# CRUCE CLUSTER × GÉNERO
# ====================================================================
def analyze_cluster_gender_cross(df_profile, feature_cols, best_k):
    """Analiza diferencias de género DENTRO de cada cluster."""
    print("\n" + "=" * 60)
    print("  CRUCE CLUSTER × GÉNERO")
    print("=" * 60)

    from scipy.stats import mannwhitneyu

    cluster_names = {0: 'Moderadores/Panelistas', 1: 'Público/Audiencia'}

    # Variables clave para el análisis de género
    key_vars = [
        'n_interventions', 'mean_duration', 'total_duration', 'mean_wpm',
        'mean_lexical_diversity', 'mean_conflict_score', 'mean_assertiveness_score',
        'mean_echoing_score', 'pct_interrupts_previous', 'pct_interrupted_by_next',
        'pct_interruption_success', 'pct_has_hedge', 'pct_has_disagreement',
        'pct_has_agreement', 'pct_has_courtesy', 'pct_has_apology',
        'pct_is_question', 'pct_has_vulnerability', 'pct_is_mansplaining',
        'pct_is_backchannel', 'mean_overlap_duration', 'std_duration',
        'min_turn_number', 'is_top3_speaker'
    ]
    key_vars = [v for v in key_vars if v in feature_cols or v in df_profile.columns]

    all_results = []

    for c in range(best_k):
        cname = cluster_names.get(c, f'Cluster {c}')
        cluster_data = df_profile[df_profile['cluster'] == c]
        females = cluster_data[cluster_data['gender'] == 'female']
        males = cluster_data[cluster_data['gender'] == 'male']

        n_f, n_m = len(females), len(males)
        print(f"\n  {'─'*50}")
        print(f"  CLUSTER {c} — {cname}")
        print(f"  {'─'*50}")
        print(f"  Mujeres: {n_f} | Hombres: {n_m}")

        cluster_results = []

        for var in key_vars:
            if var not in cluster_data.columns:
                continue
            f_vals = females[var].dropna()
            m_vals = males[var].dropna()

            if len(f_vals) < 3 or len(m_vals) < 3:
                continue

            # Mann-Whitney U
            try:
                u_stat, p_val = mannwhitneyu(f_vals, m_vals, alternative='two-sided')
            except:
                continue

            # Cohen's d
            pooled_std = np.sqrt(((len(f_vals)-1)*f_vals.std()**2 + (len(m_vals)-1)*m_vals.std()**2) /
                                  (len(f_vals) + len(m_vals) - 2))
            cohens_d = (f_vals.mean() - m_vals.mean()) / pooled_std if pooled_std > 0 else 0

            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))

            cluster_results.append({
                'cluster': c,
                'cluster_name': cname,
                'variable': var,
                'mean_female': round(f_vals.mean(), 4),
                'mean_male': round(m_vals.mean(), 4),
                'diff_pct': round((f_vals.mean() - m_vals.mean()) / m_vals.mean() * 100, 1) if m_vals.mean() != 0 else 0,
                'cohens_d': round(cohens_d, 4),
                'U_statistic': round(u_stat, 1),
                'p_value': round(p_val, 6),
                'significant': sig
            })

        cluster_df = pd.DataFrame(cluster_results).sort_values('p_value')
        all_results.extend(cluster_results)

        # Mostrar significativos
        sig_df = cluster_df[cluster_df['significant'] != '']
        if len(sig_df) > 0:
            print(f"\n  Variables con diferencia significativa de género (p<0.05):")
            for _, row in sig_df.iterrows():
                direction = "F>M" if row['cohens_d'] > 0 else "M>F"
                print(f"    {row['significant']:>3} {row['variable']:<30} {direction}  d={row['cohens_d']:+.3f}  p={row['p_value']:.4f}")
        else:
            print(f"\n  No se encontraron diferencias de género significativas en este cluster.")

    # Guardar resultados completos
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(CSV_DIR / "cluster_gender_cross_analysis.csv", index=False)
    print(f"\n  📊 Resultados guardados en cluster_gender_cross_analysis.csv")

    # ------------------------------------------------------------------
    # Gráfico 1: Heatmap de Cohen's d por cluster × variable
    # ------------------------------------------------------------------
    if len(all_results) > 0:
        pivot_d = results_df.pivot_table(index='variable', columns='cluster_name',
                                          values='cohens_d', aggfunc='first')
        pivot_p = results_df.pivot_table(index='variable', columns='cluster_name',
                                          values='p_value', aggfunc='first')

        # Filtrar solo variables con al menos un resultado significativo
        sig_vars = results_df[results_df['significant'] != '']['variable'].unique()
        if len(sig_vars) > 0:
            pivot_d_sig = pivot_d.loc[pivot_d.index.isin(sig_vars)]
            pivot_p_sig = pivot_p.loc[pivot_p.index.isin(sig_vars)]
        else:
            pivot_d_sig = pivot_d.head(15)
            pivot_p_sig = pivot_p.head(15)

        # Nombres cortos
        short_names = {
            'n_interventions': 'N Interv.', 'mean_duration': 'Duración Media',
            'total_duration': 'Dur. Total', 'mean_wpm': 'WPM',
            'mean_lexical_diversity': 'Divers. Léxica', 'mean_conflict_score': 'Conflicto',
            'mean_assertiveness_score': 'Asertividad', 'mean_echoing_score': 'Eco Léxico',
            'pct_interrupts_previous': '% Interrumpe', 'pct_interrupted_by_next': '% Interrumpido',
            'pct_interruption_success': '% Interr. Exitosa', 'pct_has_hedge': '% Hedge',
            'pct_has_disagreement': '% Desacuerdo', 'pct_has_agreement': '% Acuerdo',
            'pct_has_courtesy': '% Cortesía', 'pct_has_apology': '% Disculpa',
            'pct_is_question': '% Preguntas', 'pct_has_vulnerability': '% Vulnerabilidad',
            'pct_is_mansplaining': '% Mansplaining', 'pct_is_backchannel': '% Backchannel',
            'mean_overlap_duration': 'Dur. Solapamiento', 'std_duration': 'Std Duración',
            'min_turn_number': 'Turno 1ª Interv.', 'is_top3_speaker': 'Top-3 Hablante'
        }
        row_labels = [short_names.get(v, v) for v in pivot_d_sig.index]

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_d_sig) * 0.45)))
        im = ax.imshow(pivot_d_sig.values, cmap='RdBu_r', aspect='auto',
                       vmin=-max(0.5, abs(pivot_d_sig.values).max()),
                       vmax=max(0.5, abs(pivot_d_sig.values).max()))

        ax.set_xticks(range(len(pivot_d_sig.columns)))
        ax.set_xticklabels(pivot_d_sig.columns, fontsize=11, fontweight='bold')
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)

        # Añadir valores y asteriscos
        for i in range(len(pivot_d_sig)):
            for j in range(len(pivot_d_sig.columns)):
                d_val = pivot_d_sig.values[i, j]
                p_val = pivot_p_sig.values[i, j]
                if np.isnan(d_val):
                    continue
                sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
                label = f'{d_val:+.2f}{sig}'
                color = 'white' if abs(d_val) > 0.3 else 'black'
                ax.text(j, i, label, ha='center', va='center', fontsize=8,
                        fontweight='bold', color=color)

        ax.set_title('Cohen\'s d por Género dentro de cada Cluster\n(+ = Mujeres > Hombres, − = Hombres > Mujeres)',
                     fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label="Cohen's d (F − M)")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "13_cluster_gender_cross_heatmap.png", dpi=200, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Gráfico 2: Barras pareadas (F vs M) para variables clave por cluster
    # ------------------------------------------------------------------
    top_vars = ['mean_conflict_score', 'mean_assertiveness_score', 'pct_interrupts_previous',
                'pct_has_hedge', 'pct_is_question', 'pct_has_disagreement',
                'pct_is_mansplaining', 'pct_has_courtesy', 'mean_duration', 'n_interventions']
    top_vars = [v for v in top_vars if v in df_profile.columns]

    if len(top_vars) > 0:
        fig, axes = plt.subplots(1, best_k, figsize=(8 * best_k, 8))
        if best_k == 1:
            axes = [axes]

        for c in range(best_k):
            ax = axes[c]
            cname = cluster_names.get(c, f'Cluster {c}')
            cluster_data = df_profile[df_profile['cluster'] == c]
            females = cluster_data[cluster_data['gender'] == 'female']
            males = cluster_data[cluster_data['gender'] == 'male']

            f_means = [females[v].mean() for v in top_vars]
            m_means = [males[v].mean() for v in top_vars]

            x = np.arange(len(top_vars))
            width = 0.35
            ax.barh(x - width/2, f_means, width, label=f'Mujeres (n={len(females)})',
                    color='#e74c3c', alpha=0.85)
            ax.barh(x + width/2, m_means, width, label=f'Hombres (n={len(males)})',
                    color='#3498db', alpha=0.85)

            labels = [short_names.get(v, v) for v in top_vars]
            ax.set_yticks(x)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_title(f'{cname}\n(Cluster {c})', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='x')

            # Marcar diferencias significativas
            for i, var in enumerate(top_vars):
                entry = results_df[(results_df['cluster'] == c) & (results_df['variable'] == var)]
                if len(entry) > 0 and entry.iloc[0]['significant'] != '':
                    ax.text(max(f_means[i], m_means[i]) * 1.02, i,
                            entry.iloc[0]['significant'], fontsize=10, fontweight='bold',
                            color='#e74c3c', va='center')

        plt.suptitle('Comparación de Género por Cluster — Variables Clave',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "14_cluster_gender_bars_comparison.png", dpi=200, bbox_inches='tight')
        plt.close()

    print(f"\n  ✅ Análisis cluster × género completado")
    return results_df


# ====================================================================
# EFECTO LLAMADA DE GÉNERO (Gender Attraction Effect)
# ====================================================================
def analyze_gender_attraction(df_profile, best_k):
    """
    Analiza si el género de los moderadores (Cluster 0) en una sesión
    predice el género de la audiencia (Cluster 1) en esa misma sesión.
    ¿Ejercen los 'popes' un efecto llamada para su género?
    """
    print("\n" + "=" * 60)
    print("  EFECTO LLAMADA DE GÉNERO")
    print("  ¿El género de moderadores predice el de la audiencia?")
    print("=" * 60)

    from scipy.stats import spearmanr, pearsonr, fisher_exact, mannwhitneyu

    # Separar moderadores y audiencia
    mods = df_profile[df_profile['cluster'] == 0].copy()
    auds = df_profile[df_profile['cluster'] == 1].copy()

    # Calcular % femenino por sesión para cada cluster
    mod_gender = mods.groupby('session').apply(
        lambda g: pd.Series({
            'pct_female_mods': (g['gender'] == 'female').mean() * 100,
            'n_mods': len(g),
            'n_female_mods': (g['gender'] == 'female').sum(),
            'n_male_mods': (g['gender'] == 'male').sum()
        })
    ).reset_index()

    aud_gender = auds.groupby('session').apply(
        lambda g: pd.Series({
            'pct_female_auds': (g['gender'] == 'female').mean() * 100,
            'n_auds': len(g),
            'n_female_auds': (g['gender'] == 'female').sum(),
            'n_male_auds': (g['gender'] == 'male').sum()
        })
    ).reset_index()

    # Merge por sesión (solo sesiones con ambos clusters)
    merged = mod_gender.merge(aud_gender, on='session', how='inner')
    n_sessions = len(merged)
    print(f"\n  Sesiones con moderadores Y audiencia: {n_sessions}")

    if n_sessions < 5:
        print("  ⚠️  Pocas sesiones para análisis, resultados poco fiables.")

    # ------------------------------------------------------------------
    # 1. Correlación continua: %F moderadores vs %F audiencia
    # ------------------------------------------------------------------
    r_pearson, p_pearson = pearsonr(merged['pct_female_mods'], merged['pct_female_auds'])
    r_spearman, p_spearman = spearmanr(merged['pct_female_mods'], merged['pct_female_auds'])

    print(f"\n  📊 Correlación entre % femenino de moderadores y audiencia:")
    print(f"     Pearson:  r={r_pearson:.4f}, p={p_pearson:.4f}")
    print(f"     Spearman: ρ={r_spearman:.4f}, p={p_spearman:.4f}")

    if p_spearman < 0.05:
        direction = "positiva" if r_spearman > 0 else "negativa"
        print(f"  → Correlación {direction} significativa: SÍ hay efecto llamada de género")
    else:
        print(f"  → No se detecta correlación significativa (p≥0.05)")

    # ------------------------------------------------------------------
    # 2. Análisis categórico: moderadores mayoría-F vs mayoría-M
    # ------------------------------------------------------------------
    # Clasificar sesiones según el género dominante de moderadores
    merged['mod_majority'] = merged['pct_female_mods'].apply(
        lambda x: 'Mayoría F' if x > 50 else ('Mayoría M' if x < 50 else 'Equitativo')
    )

    print(f"\n  📊 Sesiones por género dominante de moderadores:")
    for cat in ['Mayoría F', 'Equitativo', 'Mayoría M']:
        subset = merged[merged['mod_majority'] == cat]
        if len(subset) > 0:
            avg_aud_f = subset['pct_female_auds'].mean()
            print(f"    {cat}: {len(subset)} sesiones → %F audiencia = {avg_aud_f:.1f}%")

    # Test: ¿Difiere el % femenino de audiencia entre sesiones con mods mayoritariamente F vs M?
    group_f = merged[merged['mod_majority'] == 'Mayoría F']['pct_female_auds']
    group_m = merged[merged['mod_majority'] == 'Mayoría M']['pct_female_auds']

    if len(group_f) >= 3 and len(group_m) >= 3:
        u_stat, p_mw = mannwhitneyu(group_f, group_m, alternative='two-sided')
        print(f"\n  Mann-Whitney U (audiencia %F: mods-F vs mods-M): U={u_stat:.0f}, p={p_mw:.4f}")
        if p_mw < 0.05:
            print(f"  → La composición de género de moderadores SÍ afecta a la audiencia")
        else:
            print(f"  → No se detecta diferencia significativa")

    # ------------------------------------------------------------------
    # 3. Fisher's Exact Test (2×2 simplificado)
    # ------------------------------------------------------------------
    # Crear tabla 2×2: (mod_mayoría_F/M) × (aud_mayoría_F/M)
    merged['aud_majority'] = merged['pct_female_auds'].apply(
        lambda x: 'Mayoría F' if x > 50 else 'Mayoría M'
    )
    fisher_data = merged[merged['mod_majority'].isin(['Mayoría F', 'Mayoría M'])]

    if len(fisher_data) >= 4:
        ct_fisher = pd.crosstab(fisher_data['mod_majority'], fisher_data['aud_majority'])
        print(f"\n  📊 Tabla de contingencia (mayorías):")
        print(f"  {ct_fisher.to_string()}")

        if ct_fisher.shape == (2, 2):
            odds, p_fisher = fisher_exact(ct_fisher)
            print(f"\n  Fisher's Exact: OR={odds:.3f}, p={p_fisher:.4f}")
            if p_fisher < 0.05:
                print(f"  → Efecto llamada CONFIRMADO (odds ratio={odds:.2f})")
            else:
                print(f"  → No significativo por Fisher's exact")

    # Guardar datos del efecto llamada
    merged.to_csv(CSV_DIR / "gender_attraction_effect.csv", index=False)

    # ------------------------------------------------------------------
    # Gráfico 1: Scatter plot %F moderadores vs %F audiencia
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Scatter con regresión
    ax1.scatter(merged['pct_female_mods'], merged['pct_female_auds'],
                s=merged['n_auds'] * 15, alpha=0.6, c='#9b59b6', edgecolors='white', linewidth=1)

    # Línea de regresión
    if n_sessions >= 5:
        z = np.polyfit(merged['pct_female_mods'], merged['pct_female_auds'], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(merged['pct_female_mods'].min(), merged['pct_female_mods'].max(), 100)
        ax1.plot(x_line, p_line(x_line), '--', color='#e74c3c', linewidth=2,
                 label=f'Regresión (r={r_pearson:.3f}, p={p_pearson:.3f})')

    # Línea diagonal de referencia
    ax1.plot([0, 100], [0, 100], ':', color='gray', alpha=0.5, label='Identidad')

    ax1.set_xlabel('% Mujeres en Moderadores (Cluster 0)', fontsize=12)
    ax1.set_ylabel('% Mujeres en Audiencia (Cluster 1)', fontsize=12)
    ax1.set_title('Efecto Llamada de Género\n¿Moderadoras atraen audiencia femenina?',
                   fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)

    # Anotación con resultado
    result_text = f"Spearman ρ = {r_spearman:.3f}\np = {p_spearman:.4f}"
    ax1.text(0.05, 0.95, result_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ------------------------------------------------------------------
    # Gráfico 2: Barras agrupadas por tipo de sesión
    # ------------------------------------------------------------------
    categories = ['Mayoría F', 'Equitativo', 'Mayoría M']
    cats_present = [c for c in categories if c in merged['mod_majority'].values]

    x = np.arange(len(cats_present))
    means_aud_f = []
    means_aud_m = []
    counts = []

    for cat in cats_present:
        subset = merged[merged['mod_majority'] == cat]
        means_aud_f.append(subset['pct_female_auds'].mean())
        means_aud_m.append(100 - subset['pct_female_auds'].mean())
        counts.append(len(subset))

    width = 0.35
    ax2.bar(x - width/2, means_aud_f, width, label='% Mujeres en Audiencia',
            color='#e74c3c', alpha=0.85)
    ax2.bar(x + width/2, means_aud_m, width, label='% Hombres en Audiencia',
            color='#3498db', alpha=0.85)

    for i, (mf, mm, cnt) in enumerate(zip(means_aud_f, means_aud_m, counts)):
        ax2.text(i - width/2, mf + 1, f'{mf:.0f}%', ha='center', fontsize=10, fontweight='bold')
        ax2.text(i + width/2, mm + 1, f'{mm:.0f}%', ha='center', fontsize=10, fontweight='bold')
        ax2.text(i, -5, f'n={cnt}', ha='center', fontsize=9, color='gray')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{c}\n(moderadores)' for c in cats_present], fontsize=10)
    ax2.set_ylabel('% Género en Audiencia', fontsize=12)
    ax2.set_title('Composición de Audiencia según\nGénero Dominante de Moderadores',
                   fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "15_gender_attraction_effect.png", dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n  ✅ Análisis efecto llamada de género completado")
    return merged


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("  ANÁLISIS DE CLUSTERS — USUARIOS ÚNICOS")
    print("=" * 60)

    # 1. Cargar datos
    df, feature_cols = load_user_data()
    X, scaler = standardize(df, feature_cols)

    # 2. K-Means
    km_labels, best_k, km_model = run_kmeans(X, df, feature_cols)
    df['cluster_kmeans'] = km_labels

    # 3. DBSCAN
    dbscan_labels = run_dbscan(X, df)
    df['cluster_dbscan'] = dbscan_labels

    # 4. Jerárquico
    hc_labels = run_hierarchical(X, df, best_k)
    df['cluster_hierarchical'] = hc_labels

    # 5. Visualización
    X_pca, X_tsne = run_dimensionality_reduction(X, df, km_labels, dbscan_labels, hc_labels, best_k)

    # 6. Profiling (con K-Means como referencia principal)
    df_profile = profile_clusters(df, feature_cols, km_labels, best_k)

    # 7. Interpretación
    interpret_clusters(df_profile, feature_cols, best_k)

    # 7b. Cruce cluster × género
    analyze_cluster_gender_cross(df_profile, feature_cols, best_k)

    # 7c. Efecto llamada de género
    analyze_gender_attraction(df_profile, best_k)

    # 8. Guardar dataset final con asignaciones de cluster
    df.to_csv(CSV_DIR / "user_level_with_clusters.csv", index=False)
    print(f"\n  ✅ Dataset con clusters guardado en user_level_with_clusters.csv")

    # 9. Comparación entre métodos
    print("\n" + "=" * 60)
    print("  COMPARACIÓN ENTRE MÉTODOS")
    print("=" * 60)

    from sklearn.metrics import adjusted_rand_score
    ari_km_hc = adjusted_rand_score(km_labels, hc_labels)
    mask_db = dbscan_labels != -1
    if mask_db.sum() > 0 and len(set(dbscan_labels[mask_db])) >= 2:
        ari_km_db = adjusted_rand_score(km_labels[mask_db], dbscan_labels[mask_db])
    else:
        ari_km_db = float('nan')

    print(f"  ARI K-Means vs Jerárquico: {ari_km_hc:.4f}")
    print(f"  ARI K-Means vs DBSCAN (sin outliers): {ari_km_db:.4f}")

    comparison = pd.DataFrame({
        'Method': ['K-Means', 'DBSCAN', 'Hierarchical'],
        'N_clusters': [best_k,
                       len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                       best_k],
        'Silhouette': [
            round(silhouette_score(X, km_labels), 4),
            round(silhouette_score(X[mask_db], dbscan_labels[mask_db]), 4) if mask_db.sum() > 0 and len(set(dbscan_labels[mask_db])) >= 2 else float('nan'),
            round(silhouette_score(X, hc_labels), 4)
        ]
    })
    comparison.to_csv(CSV_DIR / "cluster_method_comparison.csv", index=False)
    print(f"\n{comparison.to_string(index=False)}")

    print("\n" + "=" * 60)
    print("  ✅ ANÁLISIS COMPLETADO")
    print("=" * 60)
    print(f"  Gráficos: {FIG_DIR}")
    print(f"  CSVs: {CSV_DIR}")


if __name__ == "__main__":
    main()
