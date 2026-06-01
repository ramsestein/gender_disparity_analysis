# Gender Disparity Analysis Pipeline

**Automated Analysis of Gender Disparity in Conferences and Events**

This software processes video recordings from conferences, congresses, and academic events to automatically analyze participation by gender, generating metrics on speaking time, interruptions, and conversational dynamics. The pipeline includes nominal speaker identification, external gender classifier validation, explainable multivariate predictive modeling, Bayesian estimation, and formal robustness analysis against classification error.

## 📜 License

This project is distributed under the **MIT License** (see [LICENSE](LICENSE) file).

The code is released publicly to enable:
- ✅ Independent audits of methodologies
- ✅ Reproducibility of studies
- ✅ Continuous improvement by the community
- ✅ Democratic access to gender analysis tools

## 📋 Features

### Audio Pipeline
- **Audio extraction** from video files
- **Normalization** of audio levels (EBU R128, −23 LUFS)
- **Speaker diarization** (Pyannote.audio 3.1, ECAPA-TDNN)
- **Transcription** with Whisper large-v2 (speech-to-text)
- **Gender classification** with dual acoustic validation (pitch F0 + wav2vec2-xlsr-53)
- **Sentiment and emotion analysis** with BERT models (pysentimiento)
- **Power and conflict indices** calculated automatically

### Enrichment and Validation
- **Nominal speaker identification** against external registry (RapidFuzz, fuzzy matching)
- **External gender validation** by given name (gender_guesser, accuracy 94.3%)
- **Professional metadata** incorporated: role, topic area, country, specialty, citations
- **Formal correction** for classification error (Rogan-Gladen + Monte Carlo, N=1,000)

### Statistical Analysis
- **Non-parametric tests** with effect sizes and FDR correction
- **Mixed effects models** (LMM/GLMM) with ICC per variable
- **Bayesian estimation** via MCMC Metropolis-Hastings (20,000 samples)
- **Markov chains** of gender transitions in speaking turns
- **Stratified analysis** by role, specialty, topic area, and moderation composition

### Explainable Predictive Modeling
- **Regularized regression** (Ridge L2, Lasso L1) with cross-validation
- **CART** (decision tree, `max_depth=4`) with AUC in stratified cross-validation
- **Gradient Boosting** (GBM, 200 trees) with ROC curves and MDI feature importance
- **MILP** Best-Subset L1 with K=6 (PuLP + CBC solver), global optimum guarantee
- **SHAP values** for individual interpretability (TreeExplainer)
- **Detailed academic reports** in Markdown

## 📚 Detailed Research Documentation

For an in-depth understanding of the methodology and findings, consult the following documents:

- 🔬 [**Complete Methodology**](final_reports/resultados/METHOD_EN.md): Technical detail of the pipeline and statistical framework (14 sections, including nominal validation, explainable predictive modeling, and sensitivity analysis).
- 📊 [**Study Results**](final_reports/resultados/SUMMARY_RESULTS_EN.md): Synthesis of findings with up-to-date tables and multi-level discussion.
- 📖 [**Variable Dictionary**](final_reports/resultados/VARIABLE_DICTIONARY.md): Operational definitions of all metrics (12 sections + glossary).
- 📉 [**Extensive Statistical Report**](final_reports/resultados/STATISTICAL_RESULTS.md): 69+ sections with inferential analysis, clusters, mixed effects, explainable models, Bayesian, Markov, and sensitivity.

## 🚀 Quick Installation

### Prerequisites
- **Python 3.8 or higher** installed on your system
  - Windows: Download from [python.org](https://www.python.org/downloads/)
  - Mac: `brew install python3` or download from python.org
  - Linux: `sudo apt-get install python3 python3-venv python3-pip`

### Automatic Installation

1. **Download or clone this project**

2. **Place your videos** in the `video/` folder within the project. You must create it and name it accordingly.

3. **Run the installation and pipeline script:**

**Windows:**
```bash
run_pipeline.bat
```

**Linux/Mac:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

The script automatically:
- ✅ Creates the Python virtual environment
- ✅ Installs all necessary dependencies
- ✅ Asks for your Hugging Face token (guides you on how to obtain it)
- ✅ Runs the complete pipeline

**That's it!** The script will guide you step by step.

## 📁 Proyect´s structure

```
gender_diaparity/
├── video/                      # Input videos
├── fuentes/                    # Working directories (auto-generated)
│   ├── audio/                  # Extracted audio
│   ├── audio_normalized/       # Normalized audio
│   ├── diarization/           # Speaker segments
│   ├── transcription/         # Transcriptions
│   └── gender_classification/ # Gender classification
├── final_reports/             # Final reports
│   ├── csv/                   # CSV reports
│   ├── excel/                 # Excel reports
│   └── resultados/            # Study documentation
│       ├── METHOD.md / METHOD_EN.md          # Complete methodology
│       ├── SUMMARY_RESULTS.md / SUMMARY_RESULTS_EN.md # Results synthesis
│       ├── INFORME_STAT.md / STATISTICAL_RESULTS.md    # Statistical report (50 sections)
│       ├── DICCIONARIO_VARIABLES.md / VARIABLE_DICTIONARY.md # Variable dictionary and glossary
│       ├── csv/               # 82 statistical backup CSVs
│       └── graficos/          # Visualizations (40+ graphics)
├── logs/                      # Processing logs
├── src/                       # Audio pipeline scripts
│   ├── 01_video_to_audio.py
│   ├── 02_normalize_audio.py
│   ├── 03_diarization.py
│   ├── 04_transcription.py
│   ├── 05_gender_classification.py
│   └── 06_final_report.py
├── src_stat/                  # Statistical analysis scripts (22 modules)
│   ├── c01_build_user_dataset.py      # User-level dataset (652 obs.)
│   ├── c02_cluster_analysis.py        # Clustering + attraction effect
│   ├── c03_enrich_speakers.py         # Nominal identification (RapidFuzz)
│   ├── c04_cluster_enriched.py        # Enriched clustering (64 features)
│   ├── c05_validate_gender.py         # Gender validation by name
│   ├── c06_sensitivity_analysis.py    # Rogan-Gladen + Monte Carlo correction
│   ├── c07_new_vars_bias_predictors.py # Bias predictors
│   ├── c08_mixed_effects.py           # LMM/GLMM with ICC
│   ├── c10_interruption_temporal.py   # Interruption networks and temporal
│   ├── c11_propensity_matching.py     # Propensity score matching
│   ├── c12_cart_gbm_milp.py           # Predictive CART, GBM and MILP
│   ├── c14_shap.py                    # SHAP values
│   ├── c15_interactions.py            # Interaction effects
│   ├── c16_session_bias.py            # Session-level bias (bias_score)
│   ├── c17_markov_turns.py            # Markov chains of turns
│   ├── c18_geography.py               # Geographic analysis
│   ├── c19_bayesian.py                # MCMC Metropolis-Hastings
│   ├── c20_power_analysis.py          # Power analysis
│   ├── c21_female_moderator.py        # Female moderator effect
│   └── c22_enrich_excel_and_graphs.py # Explainable graphics + enriched Excel
├── run_pipeline.bat           # Run pipeline (Windows)
├── run_pipeline.sh            # Run pipeline (Linux/Mac)
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🎯 Usage

### Option 1: Run Complete Pipeline (Recommended)

**Windows:**
```bash
run_pipeline.bat
```

**Linux/Mac:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Option 2: Run Individual Scripts

```bash
# 1. Extract audio from videos
python src/01_video_to_audio.py

# 2. Normalize audio
python src/02_normalize_audio.py

# 3. Speaker diarization
python src/03_diarization.py

# 4. Transcription
python src/04_transcription.py

# 5. Gender classification
python src/05_gender_classification.py

# 6. Generate final reports
python src/06_final_report.py
```

## 📊 Final Report Format

Reports include the following columns:

| Column | Description |
|---------|-------------|
| `intervention_id` | Unique intervention ID |
| `start_time` | Start time (seconds) |
| `end_time` | End time (seconds) |
| `duration` | Intervention duration (seconds) |
| `speaker` | Speaker ID (SPEAKER_00, SPEAKER_01, etc.) |
| `gender` | Classified gender (male/female) |
| `gender_confidence` | Classification confidence (0-1) |
| `text` | Text transcription |
| `has_overlap` | Whether there is overlap with another speaker |
| `overlap_duration` | Overlap duration (seconds) |
| `interrupts_previous` | Whether it interrupts the previous speaker |
| `interrupted_by_next` | Whether interrupted by the next speaker |
| `turn_number` | Turn number in the conversation |
| `sentiment` | Affective tone (Positive/Negative/Neutral) |
| `emotion` | Emotional category (joy, anger, etc.) |
| `conflict_score` | Degree of conflict in the turn |
| `assertiveness_score` | Level of direct assertiveness |

## 🧪 Study Methodology

The analysis is based on a multidimensional framework of **14 blocks** covering from signal processing to formal robustness:

1.  **Ingestion and Normalization:** EBU R128 (-23 LUFS).
2.  **Advanced Diarization:** Pyannote 3.1 with ECAPA-TDNN embeddings.
3.  **Dual Acoustic Gender Validation:** Pitch (F0) analysis + wav2vec2-xlsr-53.
4.  **Aligned Transcription:** OpenAI Whisper large-v2 per speaker.
5.  **Nominal Identification:** RapidFuzz fuzzy matching against ESICM registry; gender validation by given name (gender_guesser, accuracy 94.3% over N=88 comparables).
6.  **Linguistic Analysis:** Grammatical (spaCy), pragmatic (hedges, agreements, courtesy), academic authority and affective (pysentimiento).
7.  **Composite Indices:** Conflict, assertiveness, interruption climate, idea appropriation, explaining pattern.
8.  **Frequentist Statistical Framework:** Mann-Whitney U, chi-squared, mixed effects models (LMM/GLMM), ICC, FDR, bootstrap, Wilson Score.
9.  **Bayesian Estimation:** MCMC Metropolis-Hastings (20,000 samples, 25% burn-in), 95% HDI, directional probability.
10. **Sequence Analysis:** First-order Markov chains on gender turns; chi-squared randomness test.
11. **Enriched Clustering:** K-Means/Hierarchical/DBSCAN on 64 features (55 behavioral + 9 professional metadata); Silhouette Score; cluster x gender cross.
12. **Explainable Predictive Modeling:** Ridge/Lasso regression, CART (AUC=0.85-0.99), GBM, MILP Best-Subset L1 (PuLP/CBC), SHAP values.
13. **Stratified Analysis:** By role, specialty, ESICM topic area, region, and moderation composition.
14. **Formal Sensitivity Analysis:** Cohen's d correction (Rogan-Gladen, kappa=0.894) + p-value correction (Monte Carlo N=1,000, conservative P75).

## 📈 Main Results

The study analyzes **12,138 interventions** from **652 participants** across **75 sessions**. Findings are articulated across three qualitatively distinct levels:

### At the intervention level (N=12,138)
- **Discursive Segregation:** Intra-gender transitions of 85.4% (M->M) and 74.8% (F->F); self-clustering excess of 0.279 over expected under randomness (chi2=4,362; p<0.0001).
- **Small but replicable effects:** Women present slightly longer interventions (+1.18s) and greater expression of disagreement; men show greater lexical diversity. All effects are of negligible magnitude (|g|<0.08) but with total convergence between frequentist, FDR, and Bayesian frameworks (P(direction)=1.000).
- **Equity:** 15% of sessions achieve functional parity >=0.85.

### At the user level (n=652, enriched clustering)
- **Clustering with 64 features:** K-Means k=2 (Silhouette=0.327) exactly separates identified panelists (n=94) from anonymous audience (n=558), with no gender bias in assignment (chi2=3.25; p=0.072).
- **Robust mansplaining:** d=-0.68 in identified panelists, d=-0.32 in audience; both significant and robust to classification error (Rogan-Gladen corr. d_corr approx +12%).
- **Explainable models:** CART AUC=0.849 / GBM AUC=0.879 for mansplaining; CART AUC=0.967 / GBM AUC=0.975 for being interrupted (women). Gender SHAP is 2.4x that of the second predictor.
- **MILP Best-Subset L1 (K=6):** Confirms Gender, Lexical diversity, Courtesy, Start turn, Echoing, and SD duration as the minimal informative subset.
- **Stratification:** Mansplaining effect d=-1.23 in Anaesthesiology, d=+2.31 in Neurology sessions; female moderators reduce interruptions of female speakers by half without eliminating global mansplaining.

### At the session level (n=75)
- **Omnipresent bias:** No structural predictor (% women, N speakers, female moderator, log-citations) correlates significantly with the composite bias_score -- bias is homogeneous and context-independent.
- **Paradoxical co-moderation effect:** Sessions with mixed co-moderation presented the highest mansplaining bias (d=+0.73) and interruption bias (d=-0.98).
- **Formal robustness:** 85.7% of significant findings are robust after conservative Monte Carlo correction (75th percentile).

## 🔧 Advanced Configuration

### Change Whisper model

In `src/04_transcription.py`, line ~443:
```python
model_size="base"  # Options: tiny, base, small, medium, large
```

- `tiny`: Fastest, lower accuracy
- `base`: Balanced (default)
- `small`: Good accuracy
- `medium`: High accuracy
- `large`: Maximum accuracy, slower

### Disable noise reduction

In `src/02_normalize_audio.py`, line ~273:
```python
apply_noise_reduction=False  # Change to False
```

## 📈 Performance

Approximate times per 20-minute audio (CPU):

| Script | Time |
|--------|------|
| 01 - Extraction | ~2 min |
| 02 - Normalization | ~50s |
| 03 - Diarization | ~20 min |
| 04 - Transcription | ~30s |
| 05 - Gender | ~2 min |
| 06 - Reports | 1s |
| **Total** | **~25 min** |

**Memory Note:** The diarization script (`03_diarization.py`) includes automatic RAM optimization and real-time monitoring. It releases resources after each audio to avoid saturation.

## 🎓 Gender Classification and Validation

### Acoustic classification (hybrid approach)

1. **Pitch (F0) analysis** -- fast baseline
   - Male: < 165 Hz
   - Female: >= 165 Hz

2. **Pre-trained Model (Wav2Vec2-xlsr-53)** -- final decision
   - Validated empirical accuracy: 94.3%

**Decision logic:**
- If both agree -> High confidence (99.65% of corpus)
- If they disagree -> Human audit for resolution
