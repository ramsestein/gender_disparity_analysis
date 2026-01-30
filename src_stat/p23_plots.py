import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pub_utils import OUTPUT_DIR, FIG_DIR
import os

# Set global style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})
OS_DPI = 300

def generate_publication_plots():
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Figure 1: Forest plot of effect sizes (P01)
    if os.path.exists(OUTPUT_DIR / 'p01_effect_sizes_table.csv'):
        es_df = pd.read_csv(OUTPUT_DIR / 'p01_effect_sizes_table.csv')
        # Order by magnitude
        es_df = es_df.sort_values('effect_size', ascending=True)
        plt.figure(figsize=(10, 8))
        plt.errorbar(es_df['effect_size'], range(len(es_df)), 
                     xerr=[es_df['effect_size'] - es_df['es_ci_lower'], 
                           es_df['es_ci_upper'] - es_df['effect_size']],
                     fmt='o', color='teal', capsize=5)
        plt.axvline(0, color='red', linestyle='--')
        plt.yticks(range(len(es_df)), es_df['variable'])
        plt.xlabel("Hedges' g / Cramr's V (Male - Female)")
        plt.title('Figure 1: Effect Sizes for Behavioral Metrics')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'figure_01_forest_effect_sizes.png', dpi=OS_DPI)
        plt.close()

    # Figure 2: Odds Ratios Forest Plot (P07)
    if os.path.exists(OUTPUT_DIR / 'p07_odds_ratios_table.csv'):
        or_df = pd.read_csv(OUTPUT_DIR / 'p07_odds_ratios_table.csv')
        or_df = or_df.sort_values('odds_ratio')
        plt.figure(figsize=(10, 8))
        plt.errorbar(or_df['odds_ratio'], range(len(or_df)),
                     xerr=[or_df['odds_ratio'] - or_df['or_ci_lower'], 
                           or_df['or_ci_upper'] - or_df['odds_ratio']],
                     fmt='o', color='darkblue', capsize=5)
        plt.axvline(1, color='red', linestyle='--')
        plt.yticks(range(len(or_df)), or_df['feature'])
        plt.xlabel('Odds Ratio (95% CI)')
        plt.title('Figure 2: Predictive Factors for Interruption Success')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'figure_02_forest_odds_ratios.png', dpi=OS_DPI)
        plt.close()

    # Figure 3: Dual Heatmap (P10) - Already created in P10 script but renamed here for consistency
    # (Actually P10 generates it as figure_p10_...)
    
    # Figure 5: Appropriation Matrix (P05)
    if os.path.exists(OUTPUT_DIR / 'p05_appropriation_matrix.csv'):
        app = pd.read_csv(OUTPUT_DIR / 'p05_appropriation_matrix.csv', index_col=0) 
        # (Reloading manually as it's small)
        plt.figure(figsize=(8, 6))
        sns.heatmap(app, annot=True, fmt='.1f', cmap='Reds')
        plt.title('Figure 5: Appropriation Rate Matrix (%)')
        plt.ylabel('Previous Speaker Gender')
        plt.xlabel('Current Speaker Gender')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'figure_05_heatmap_appropriation.png', dpi=OS_DPI)
        plt.close()

    # Figure 6: Boundary Speakers (P06)
    if os.path.exists(OUTPUT_DIR / 'p06_boundary_speakers_distribution.csv'):
        dist = pd.read_csv(OUTPUT_DIR / 'p06_boundary_speakers_distribution.csv')
        # ... logic as in P06 ...
        pass

    print(" Figure suite generated in /resultados/graficos/")

if __name__ == "__main__":
    generate_publication_plots()
