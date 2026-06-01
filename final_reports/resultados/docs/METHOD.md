# Methodology: Automated Analysis of Gender Disparity in Scientific Debates

---

## 1. Study Summary

### 1.1 Objective

To objectively and automatically quantify the dynamics of verbal participation by gender in scientific debates in the field of critical care medicine, implementing a computational pipeline that integrates audio signal processing, deep learning, and rigorous statistical analysis.

### 1.2 Design

Retrospective observational study of automated conversational analysis based on recordings of scientific debate sessions.

### 1.3 Analysis Units

Work is conducted with three complementary units:

- **Intervention**: continuous segment of speech from a single speaker, delimited by silence exceeding 0.5 seconds or by speaker change identified through automatic diarization. Constitutes the primary analysis unit (N=12,138).
- **User**: each unique participant within each session (`user_id = session + "__" + speaker`), unit used for clustering analyses and aggregated predictive modeling (N=652).
- **Session**: the complete debate, unit used for analyzing structural predictors of bias (N=75).

---

## 2. Audio Processing Pipeline

### 2.1 Ingestion and Normalization

The process begins with extraction of the audio stream from video files using standard multimedia processing tools. To ensure robustness of subsequent models, rigorous standardization is applied:

**Standardization Parameters:**
- Format: WAV (PCM 16-bit)
- Sampling frequency: 16 kHz
- Channels: Mono

**Loudness Normalization:** The EBU R128 standard is implemented, adjusting integrated loudness to -23 LUFS. This normalization mitigates acoustic variations arising from different microphone distances, speakers' vocal characteristics, and heterogeneous recording conditions. Previous studies indicate that this normalization improves diarization accuracy by 15-20% in environments with acoustic variability.

### 2.2 Speaker Diarization

For speaker segmentation and identity assignment, **Pyannote.audio 3.1** is used, a system based on segmentation models and voice embeddings.

**System Architecture:**

1. **Voice Activity Detection (VAD):** Identification of regions with speech versus silence or background noise.
2. **Embedding Extraction:** Neural networks (ECAPA-TDNN architecture) are employed to extract 192-dimensional identity vectors from each speech segment. These vectors capture the distinctive acoustic characteristics of each speaker.
3. **Clustering:** Agglomerative Hierarchical Clustering is applied to identity vectors to group segments corresponding to the same speaker, assigning unique identifiers (e.g., SPEAKER_01, SPEAKER_02).
4. **Overlap Detection:** The system natively detects moments when two or more people speak simultaneously, allowing quantification of both the presence and duration of overlapping speech.

### 2.3 Transcription and Temporal Alignment

For speech-to-text conversion, **OpenAI Whisper** (large-v2 model) is used, recognized for its robustness to accent variations and audio quality.

**Dynamic Segmentation Strategy:** Instead of transcribing the complete audio monolithically, the system individually processes segments pre-identified by diarization. This ensures that each text fragment is correctly associated with its corresponding speaker.

**Temporal Alignment:** A cross-referencing of timestamps between diarization boundaries and transcription segments is performed, guaranteeing that each word is assigned to the correct speaker with millisecond precision.

### 2.4 Post-processing and Cleaning

**Intervention Merging:** Consecutive segments from the same speaker separated by gaps shorter than 0.5 seconds are merged into a single intervention, avoiding artificial discourse fragmentation.

**Micro-segment Filtering:** Interventions with duration shorter than 0.3 seconds, typically corresponding to noises, coughs, or recording artifacts, are removed.

**Text Normalization:** Encoding artifacts are corrected and text format is standardized to UTF-8.

---

## 3. Gender Classification System

### 3.1 Dual Validation Architecture

To maximize gender classification accuracy, a cross-validation system is implemented using two independent and complementary methods:

**Method 1 - Fundamental Frequency (F0) Analysis:**

The Parselmouth interface (Python wrapper for Praat) is used to extract the median fundamental frequency of each speaker. Pitch or F0 is the acoustic correlate of vocal cord vibration and presents systematic differences between male and female voices.

A threshold of 165 Hz is established based on acoustic phonetics literature, where typical ranges are:
- Male voices: 85-180 Hz
- Female voices: 165-255 Hz

**Method 2 - Deep Learning:**

The wav2vec2-large-xlsr-53 model, specifically fine-tuned for gender recognition, is employed. This model, based on Transformer architecture, was pre-trained on 53 languages and subsequently fine-tuned for the gender classification task, conferring robustness to accent variations and atypical vocal characteristics.

**Robustness through accumulation:** To avoid misclassifications based on non-representative fragments, the system concatenates and analyzes up to 10 seconds of audio from each speaker before rendering a judgment.

### 3.2 Decision Logic

- **Concordance:** If both methods agree on their classification, gender is assigned with maximum confidence.
- **Discrepancy:** In cases where methods differ, the case is referred for human evaluation by a researcher who listens to the speaker's audio and determines the correct classification.

### 3.3 Concordance Between Acoustic Methods

Of all classified speakers, 99.65% showed concordance between both acoustic methods. The 42 discrepancy cases (0.35%) were resolved through human evaluation, ensuring consistent internal classification. This internal concordance is a necessary but not sufficient condition: external validation of classifier quality is addressed in Section 4.

---

## 4. Methodological Validation through Nominal Identification

To validate the robustness of the pipeline beyond internal coherence between acoustic methods, a process of nominal speaker identification against an external reference database is performed. This step constitutes an **independent validation** that allows (a) confirming the accuracy of the gender classifier through a completely different information channel than the acoustic one, and (b) enriching the dataset with professional metadata (role, specialty, topic area, academic citations) that will be used in subsequent stratified analyses.

### 4.1 Nominal Identification against ESICM Reference Database

A matching process is constructed between speaker names detected in transcriptions and the official nominal database of panelists from the European Society of Intensive Care Medicine (ESICM), which contains verified metadata of participants in the analyzed sessions.

**Matching Procedure:** A lexical similarity matching is implemented in six sequential phases using the **RapidFuzz** library (optimized extension of FuzzyWuzzy based on Levenshtein distance):

1. String normalization: lowercasing, removal of diacritics, spaces, and non-alphabetic characters.
2. Exact match on the normalized string.
3. Token sort ratio match (insensitive to word order).
4. Token set ratio match (insensitive to redundant tokens).
5. Partial substring match.
6. Manual review of cases with scores between 60 and 80 to detect severe transcription errors.

A minimum score threshold of 70/100 is established to accept automatic matching. Cases below this threshold are manually inspected.

**Process Coverage:**

| Category | N | % |
|-----------|---|---|
| Automatically identified | 90 | 13.8% |
| Manually corrected | 4 | 0.6% |
| Unidentified (anonymous audience) | 558 | 85.6% |
| **Total unique speakers** | **652** | **100%** |

The 558 unidentified speakers correspond mostly to audience interventions during Q&A sessions, where the intervenient's name is not recorded in official registers.

**Enriched Variables:** For the 94 identified speakers, six professional metadata variables were incorporated:

- `Role`: Speaker / Moderator / Public
- `Group`: one of 9 ESICM topic areas (Sepsis/Infection, Respiratory, Haemodynamics, Neuro, Renal/Metabolic, Ethics/EoL/Communication, Education/Professional Development, AI/Digital Health, Perioperative/Emergency)
- `Country`: primary affiliation country
- `Affiliation`: hospital institution
- `Specialty`: ICU / Anaesthesiology / Both / Other
- `Number of citations`: h-derived bibliometric index, transformed using `log1p` for statistical analyses.

### 4.2 Gender Validation by Given Name

Nominal identification allows external validation of the acoustic gender classifier using a completely independent information source: the speaker's given name.

**Procedure:** The acoustic gender assignment (variable `gender` derived from Pyannote + wav2vec2 + Parselmouth) is cross-referenced with the gender inferred from the first name of each `matched_person` using the **`gender_guesser`** library (based on a name database with gender labels by country). Six speakers whose names are ambiguous for the detector are excluded from analysis (Jean-Louis, Miklos, Zeljka, Lisa-Marie, Hannah, Bairbre), leaving 88 comparable speakers.

**Confusion Matrix:**

| | Voice assigns: female | Voice assigns: male |
|--|---|---|
| **Name: female** | 37 ✓ | 1 ✗ |
| **Name: male** | 4 ✗ | 46 ✓ |

**Validation Metrics:**

| Metric | Value |
|---------|-------|
| N comparable | 88 |
| Global accuracy | 94.3% (83/88) |
| Sensitivity for women | 97.4% (37/38) |
| Specificity for men | 92.0% (46/50) |
| Error rate | 5.7% (5/88) |

**Detected and corrected discordances:** The 5 discrepant cases were audited auditorily and corrected directly in the enriched CSVs. Notably, the case of Lennie Derde (male) was erroneously classified as female in two independent sessions, indicating an atypical voice pattern systematically misclassified by the acoustic model — a finding that justifies the subsequent sensitivity analyses for classification error (Section 12).

**Methodological implication:** The global accuracy of 94.3% obtained through cross-validation with a completely independent information channel (nominal text vs. acoustic signal) constitutes evidence of the validity of the gender classification pipeline. The 5.7% error rate is subsequently used as a parameter to formally correct effect sizes and observed p-values (Section 12).

### 4.3 ESICM Next Members Analysis

As a complementary analysis, the list of 2,339 members of the **ESICM Next** program (young researchers selected by the society) was cross-referenced against the 94 nominally identified speakers, to evaluate whether Next program participants exhibited differentiated behavioral patterns.

**Matching Procedure:** A three-phase matching was implemented:
1. **Token-subset**: all words in the speaker's name had to be contained in the Next Member's name (allowing missing surnames but no extra words).
2. **Complementary fuzzy**: `token_set_ratio >= 85` (RapidFuzz) to capture minor spelling errors.
3. **Manual review**: For speakers not captured by automated methods, transcriptions were manually inspected for self-introductions and third-party introductions by co-moderators/co-panelists.

**Result:** 15 of the 2,339 Next Members (0.6%) were identified as participants in the recorded sessions. 7 were matched automatically via fuzzy matching; 8 additional members were identified through manual review of transcriptions (self-introductions and third-party introductions), required because Whisper ASR transcription errors distorted their names beyond the fuzzy matching threshold. Of the 15, 12 acted as moderators and 3 as speakers; none appeared as anonymous audience members.

A Mann-Whitney U comparison of Next Members (486 interventions) versus non-Next identified moderators (765 interventions) found one statistically significant difference: **Next Members emitted interruptions at a significantly lower rate** (0.014 vs 0.041; p = 0.009, Mann-Whitney U, r = 0.074). No significant differences were found for duration, received interruptions, or overlap (all p > 0.20).

**Methodological conclusion:** With n=15 identified Next Members and 486 interventions, the sample is sufficient to detect the one significant finding (lower emitted interruption rate). This may reflect a behavioral profile typical of junior researchers, or partly the predominantly female composition of the identified NEXT group (78% female interventions). Stratified analysis by Next program membership remains exploratory.

---

## 5. Linguistic Variable Extraction

### 5.1 Grammatical Analysis

The spaCy natural language processing model (en_core_web_lg) is employed to extract grammatical metrics from each intervention:

**Basic Metrics:**
- **Word count:** Total number of lexical tokens, excluding punctuation.
- **Speech rate (WPM):** Words per minute, calculated as the ratio between word count and intervention duration.
- **Lexical diversity (TTR):** Type-Token Ratio, measuring vocabulary richness as the proportion of unique words over total words. Values close to 1 indicate greater lexical variety.

**Question detection:** Interrogative interventions are identified through the presence of question marks or interrogative words (what, who, when, where, why, how) in initial position.

**Imperative count:** Verbs in imperative mood are detected through morphosyntactic analysis, identifying direct orders or instructions.

### 5.2 Pragmatic Analysis

Lexical pattern dictionaries are implemented to identify pragmatic markers relevant to conversational power dynamics:

**Hedges:** Expressions that reduce the assertive force of the utterance, such as "I think", "maybe", "perhaps", "probably", "sort of", "kind of", "seems", "actually", "just".

**Apology markers:** Explicit apology expressions such as "sorry", "I apologize", "excuse me", "pardon".

**Courtesy markers:** Deference expressions such as "please", "thank you", "kindly", "I appreciate".

**Vulnerability markers:** Admissions of uncertainty or lack of knowledge such as "I'm not sure", "I don't know", "I'm uncertain".

**Agreement markers:** Expressions of conformity such as "I agree", "absolutely", "exactly", "that's right", "good point".

**Disagreement markers:** Expressions of discrepancy such as "I disagree", "I don't think", "however", "on the contrary", "not necessarily".

### 5.3 Academic Authority

**Title usage:** Detection of academic titles (Doctor, Professor, PhD) when referring to other participants or oneself.

**Idea attribution:** Explicit recognition of authorship of others' ideas through expressions such as "as Dr. X said", "according to", "building on what X mentioned".

### 5.4 Affective Analysis

The pysentimiento library, based on fine-tuned BERT models for Spanish and English, is used for:

**Sentiment analysis:** Classification of each intervention into Positive, Negative, or Neutral categories.

**Emotion analysis:** Classification into seven emotional categories: joy, anger, sadness, fear, surprise, disgust, and others.

### 5.5 Echoing Score

Lexical overlap between consecutive interventions is calculated as a proxy for thematic continuity or idea repetition. The index measures Jaccard similarity between the sets of content word lemmas (nouns and adjectives) of the current and preceding intervention.

A high value indicates that the speaker is picking up vocabulary from the previous speaker, which can be interpreted as validation, elaboration, or, in the absence of attribution, potential idea appropriation.

---

## 6. Composite Index Construction

### 6.1 Conflict Score

Weighted sum capturing the level of confrontation in an intervention, positively weighting the presence of disagreements, imperatives, and interruptions, and negatively weighting the presence of agreements and courtesy. Allows identification of high-conflict interventions (90th percentile) for extreme pattern analysis.

### 6.2 Assertiveness Score

Evaluates the direct versus mitigated communicative style of each intervention, positively weighting the use of imperatives and direct disagreements, and negatively weighting the use of hedges and apologies.

### 6.3 Interruption Climate

Rolling window metric measuring the density of interruptions in the 5 turns preceding each intervention. Allows classification of the conversational climate as:
- **Calm:** Low interruption density (<10%)
- **Hostile:** High interruption density (>30%)
- **Neutral:** Intermediate values

### 6.4 Idea Appropriation Detection

Appropriation is operationalized as the combination of high echoing score (>0.3, indicating repetition of content from the previous speaker) with absence of explicit attribution. This definition captures situations where a speaker picks up ideas from another without acknowledging their origin.

### 6.5 Explaining Pattern Detection

A symmetric heuristic rule is implemented to detect both "mansplaining" (man explaining to woman) and the reverse pattern "womansplaining" (woman explaining to man). The conditions are:

1. Previous intervention of short duration (<10 seconds)
2. Current intervention of long duration (>15 seconds)
3. Presence of correction or disagreement markers
4. Cross-gender transition (M→F or F→M)

The symmetric application of the same rule to both directions allows an unbiased comparison of the prevalence of this pattern in each gender.

---

## 7. Statistical Framework

### 7.1 Parametric Assumption Verification

Before selecting appropriate statistical tests, compliance with assumptions is systematically verified:

**Normality:** The Shapiro-Wilk test is applied to each continuous variable, segregated by gender. Since no variable meets the normality assumption (p < 0.001 in all cases), non-parametric tests are chosen.

**Homoscedasticity:** Levene's test is applied to evaluate equality of variances between groups. Detected violations reinforce the decision to employ non-parametric methods.

### 7.2 Group Comparison Tests

**Continuous variables:** The Mann-Whitney U test is used, the non-parametric equivalent of the t-test, comparing rank distributions between groups without assuming normality.

**Categorical variables:** The Chi-square test of independence is used to evaluate association between gender and binary or nominal variables.

### 7.3 Effect Sizes

Beyond statistical significance, measures of effect magnitude are reported:

**Hedges' g (continuous variables):** Corrected version of Cohen's d that adjusts for sample size. Interpreted according to conventional thresholds:
- < 0.2: Negligible
- 0.2-0.5: Small
- 0.5-0.8: Medium
- > 0.8: Large

**Cramér's V (categorical variables):** Association measure for contingency tables, normalized between 0 and 1. Interpreted as:
- < 0.1: Negligible
- 0.1-0.3: Small
- 0.3-0.5: Medium
- > 0.5: Large

### 7.4 Multiple Comparison Correction

Given the high number of tests performed, the false positive rate is controlled using the Benjamini-Hochberg method (False Discovery Rate). This procedure orders p-values and applies an adaptive threshold that maintains the expected proportion of false positives below the specified α level (0.05).

Both original and corrected p-values are reported, indicating which findings survive the correction.

### 7.5 Mixed Effects Models

Interventions are nested within sessions, violating the assumption of independence of observations. To correct this hierarchical structure, mixed effects models are employed:

**Linear Mixed Models (LMM):** For continuous dependent variables, with gender as fixed effect and session as random effect (random intercept).

**Generalized Linear Mixed Models (GLMM):** For binary dependent variables, using logit link function.

These models provide estimates of the gender effect that appropriately control for intraclass correlation.

### 7.6 Intraclass Correlation Coefficient (ICC)

ICC is calculated for each variable as the proportion of variance explained by differences between sessions relative to total variance. A low ICC (<0.05) indicates that most variability is intra-session (between individuals), validating that observed behaviors are stable traits and not artifacts of particular sessions.

### 7.7 Confidence Intervals

All estimates of differences between groups are accompanied by 95% confidence intervals, calculated using:

**Bootstrap:** For mean differences, through resampling with replacement (10,000 iterations).

**Wilson Score:** For proportions, a method providing better coverage than the Wald interval, especially with extreme proportions.

### 7.8 Bayesian Estimation of Gender Effects

As a complement to the classical frequentist framework, Bayesian estimation of main effects is performed, allowing complete posterior distributions and credibility intervals.

**Algorithm:** Markov Chain Monte Carlo with Metropolis-Hastings sampler, configured with 20,000 samples and a burn-in period of 25% (5,000 samples discarded to ensure convergence from initial state).

**Prior specification:** A weakly informative Normal(0, 0.5) prior is used on the gender effect, centered at zero (null hypothesis of no difference) with moderate standard deviation that does not impose a strong direction but penalizes extreme effects a priori.

**Posterior inference:** For each variable of interest, the following are reported:
- Posterior expectation E[δ_M-F]
- 95% Highest Density Interval (HDI)
- Directional posterior probability P(δ > 0 | data) or P(δ < 0 | data)

**Bayesian decision criterion:** Robust evidence is considered when the 95% HDI does not include zero and the directional probability exceeds 0.95. Bayesian estimation is complementary to the frequentist framework: it allows quantifying effect uncertainty without resorting to the significant/non-significant binary.

### 7.9 Sequence Analysis using Markov Chains

To evaluate whether the succession of genders in speaking turns is random or shows structure, the sequence is modeled as a first-order Markov chain over the state {Man, Woman}.

**Transition matrix construction:** P(g_{t+1} | g_t) is empirically calculated over the 12,063 available turn transitions across 75 sessions, generating a 2×2 matrix with observed conditional probabilities.

**Randomness test:** The observed matrix is contrasted against the null hypothesis of transition independent of previous gender (i.e., P(g_{t+1} | g_t) = P(g_{t+1}) = base rate). The χ² test evaluates whether deviations from independence are statistically significant.

**Same-gender clustering quantification:** The excess probability of intra-gender transition relative to expected under randomness is calculated:

```
Excess = P(same gender | observed) − P(same gender | random)
```

where the expected probability under randomness is derived from marginal base rates of each gender in the corpus. A positive excess indicates discursive segregation (speakers tend to follow same-gender speakers).

---

## 8. Multidimensional Analysis Framework

The analysis is structured in a modular framework that systematically addresses different dimensions of gender dynamics, organized in thematic blocks.

### 8.1 Statistical Rigor

- Comprehensive effect size calculation.
- FDR correction application.
- Estimation through mixed effects models.

### 8.2 Relational Dynamics

- Question-response asymmetry analysis: quantification of differences in length, quality, and response rate to questions according to gender of questioner and responder (2×2 Q-A pair matrix).
- Appropriation and amplification matrices: identification of directional patterns in idea uptake (echoing score) with or without explicit attribution.
- Power positions: analysis of gender distribution of first and last speaker in each session, contrasted using binomial test.

### 8.3 Predictive Modeling of Binary Variables

Logistic regression model to predict interruption success, evaluating the incremental contribution of gender as a predictor through comparison of complete vs. reduced models (without gender), reporting AUC and accuracy.

### 8.4 Assumption Validation

- Automatic verification of parametric assumptions for each variable.
- Partial correlation calculation controlling for confounders.
- Complete confidence interval construction (bootstrap + Wilson Score).

### 8.5 Contextual and Temporal Analysis

- Session climate effect (calm/hostile/neutral) on gender dynamics through factorial ANOVA Gender × Climate.
- Symmetric evaluation of explaining pattern (mansplaining/womansplaining).
- "Sticky floor" analysis: time to first intervention by gender.
- Turn transition matrix (Section 7.9).
- Evolution of metrics across temporal quartiles of the session.
- Potential backlash to assertiveness analysis through interaction terms.

### 8.6 Stratified Analysis by Subgroups

Leveraging enriched metadata derived from nominal identification (Section 4), stratified analysis of gender bias is performed across four contextual dimensions:

**8.6.1 By speaker role (Speaker / Moderator / Public):**
Cohen's d effect sizes are calculated separately for each role, evaluating whether role moderates the magnitude of bias. Role functions as a structural factor: moderators control the turn, speakers present content, and the audience registers questions.

**8.6.2 By medical specialty (ICU / Anaesthesiology / Both / Other):**
Kruskal-Wallis is applied to evaluate whether specialty per se predicts differences in communicative variables, independently of gender. Subsequently, the gender gap (Cohen's d M vs F) within each specialty is calculated to detect moderation.

**8.6.3 By topic area (9 ESICM groups):**
Bias analysis within each of the 9 topic areas assigned to sessions (propagated to all speakers in each session, n=388 speakers in sessions with assigned group). Female representation per group and intra-group gender effects are reported. Kruskal-Wallis is applied to evaluate the influence of topic area per se on style variables.

**8.6.4 By geographic region:**
Comparison of gender gap between Continental Europe, Southern Europe, Anglo-Saxon world, and "Unknown". Directional differences are reported with the caveat that 86% of speakers lack an identified country, limiting power for this dimension.

**8.6.5 By session moderation composition:**
Classification of 75 sessions into four categories according to moderation gender (no identified moderator / male-only moderator / female-only moderator / mixed co-moderation). The effect of moderation composition on the gender gap of mansplaining and interruption is evaluated.

### 8.7 Session-Level Analysis

To evaluate structural homogeneity of bias across sessions and detect predictors of more biased sessions, a session-level dataset is constructed (n=60 sessions with ≥2 men and ≥2 women) with the following specification:

**Dependent variable — composite `bias_score`:**
Z-standardized average of three main bias components:
- Mansplaining gap (mansplaining_M − mansplaining_F)
- Interruption gap (interruption_F − interruption_M)
- Lexical diversity gap (lexdiv_M − lexdiv_F)

**Predictors evaluated:**
- Percentage of women in the session
- Total number of speakers
- Presence of female moderator (binary)
- Logarithm of mean speaker citations

Spearman correlations between each predictor and `bias_score` are reported. This approach allows contrasting the hypothesis that bias is **omnipresent** (does not correlate with composition) against the hypothesis that it is **structurally modulable**.

### 8.8 Robustness and Sensitivity

- Identification and analysis of extreme cases.
- Subgroup analysis by conflict level.
- Post-hoc power analysis with calculation of N required to reach 80% and 90% power.
- Sensitivity analysis under variation of thresholds in operationalized metrics.
- Comparison of models with and without gender as predictor.

---

## 9. Cluster Analysis: Participation Role Segmentation

### 9.1 User-Level Dataset Construction

To complement the intervention-level analysis, an aggregated user-level dataset is constructed, where each unique participant within each session (`user_id = session + "__" + speaker`) constitutes an observation. **64 features** are aggregated, combining behavioral metrics with professional metadata:

**Behavioral features (55):**
- **Counts:** Number of interventions per user.
- **Means and standard deviations:** Duration, word count, speech rate, lexical diversity, conflict, assertiveness, and echoing scores.
- **Pragmatic marker percentages:** Proportion of interventions with hedge, question, agreement, disagreement, courtesy, apology, etc.
- **Emotional and sentiment distributions:** Proportion of interventions in each category.
- **First intervention metrics:**
  - `is_top3_speaker`: indicator of whether the user is among the 3 first distinct speakers in the session (ordered by `start_time`).
  - `min_turn_number`: earliest turn in which the user intervenes.

**Professional metadata features (9):** From nominal identification (Section 4):
- `role_known`, `is_moderator`, `is_speaker`, `is_public`
- `spec_known`, `is_ICU`, `is_Ane`, `is_Both`
- `log_citations` (logarithm of citation count, coded as log1p)

The variable `career_years` is excluded due to insufficient coverage (1.7% of speakers).

### 9.2 Clustering Algorithms

Three unsupervised learning algorithms are applied over the 64 z-score standardized features, allowing results to be contrasted through different approaches:

**K-Means:** Partitioning algorithm minimizing intra-cluster inertia. k=2 to k=10 are evaluated, selecting optimal k by maximum Silhouette Score.

**Hierarchical Clustering:** Agglomerative method (Ward's linkage) building a complete dendrogram, cut at the same optimal k as K-Means for direct comparison. Adjusted Rand Index (ARI) is calculated to quantify concordance between methods.

**DBSCAN:** Density-based algorithm not requiring k specification and allowing outlier detection (users with atypical behavior). Parameters `eps` and `min_samples` are determined through k-neighbor distance distribution analysis.

### 9.3 Optimal Cluster Number Selection

Silhouette Score evaluation for k between 2 and 10 identifies k=2 as the optimal solution (Silhouette=0.327), notably superior to solutions with k≥3 (Silhouette ≤0.09). The second best candidate (k=3) shows an abrupt Silhouette drop, confirming that the latent structure of the corpus is dichotomous and does not admit a finer robust segmentation.

### 9.4 Dimensionality Reduction and Visualization

- **PCA (Principal Component Analysis):** Linear 2D projection for visualization, reporting accumulated explained variance.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Non-linear 2D projection with perplexity=30, preserving local neighborhood structure for a more faithful visual representation of groupings.

### 9.5 Cluster Profiling and Interpretation

Cluster profiling is performed through:

- **One-way ANOVA** on each feature to identify the most discriminating variables between clusters (ordered by F-statistic).
- **Normalized centroid analysis** visualized as heatmap and radar chart.
- **Gender distribution per cluster** evaluated through Chi-square independence test.

**Empirical interpretation of result (k=2):**
- **Cluster 0 — Identified panelists (n=94, 14.4%):** Groups all nominally identified speakers (35.1% moderators, 44.7% panelists, 20.2% registered audience). 47.9% are women. Concentrates 100% of specialty and citation information.
- **Cluster 1 — Anonymous audience (n=558, 85.6%):** Groups all unidentified speakers, predominantly audience interventions during Q&A. 37.5% are women.

The χ² test of gender × cluster (χ²=3.25, p=0.072) shows **absence of gender bias** in cluster assignment: gender composition is proportionally similar between identified panelists and anonymous audience.

### 9.6 Intra-Cluster Gender Difference Analysis

To determine whether gender differences persist within each structural role, a stratified analysis by cluster is performed:

- **Mann-Whitney U** per variable and cluster: compares men vs women within each cluster.
- **Cohen's d** as effect size measure for each comparison.
- **Significance levels reported:** p<0.05 (*), p<0.01 (**), p<0.001 (***).

This analysis allows verifying that the main patterns (mansplaining M>F, hedge F>M, lexical diversity M>F) are robust to structural role and not solely explained by male overrepresentation among panelists. Significant findings from each cluster are subsequently subjected to sensitivity analysis for classification error (Section 12).

### 9.7 Session Homogeneity Analysis

To evaluate whether segmentation is an artifact of specific sessions, cluster distribution by session is analyzed through Chi-square homogeneity test (sessions × clusters) and session × cluster proportion heatmap.

---

## 10. Predictive Modeling and Explainable Machine Learning

To complement bivariate tests (Section 7) with a multivariate analysis quantifying relative contributions and interactions between predictors, a hierarchy of predictive models is implemented over the 652 unique speakers. Outcome selection is based on theoretical relevance:

- **Outcome 1:** `pct_is_mansplaining` — proportion of speaker interventions classified as mansplaining (analyzed on the complete sample with gender as predictor).
- **Outcome 2:** `pct_interrupted_by_next` — proportion of speaker interventions interrupted by the next turn (analyzed only on women, n=254, with remaining variables as predictors).

### 10.1 Multivariate Regression with Regularization

Three linear regression variants are fitted over 40-41 predictors (aggregated behavioral variables + professional metadata) to obtain a first characterization of linear relationships:

**Ordinary Least Squares (OLS):** Standard linear regression as reference model. Allows reporting traditional frequentist p-values for each predictor but is sensitive to multicollinearity.

**Ridge (L2 regularization):** Penalty proportional to the square of coefficients. Reduces estimator variance in the presence of multicollinearity and attenuates unstable coefficients without eliminating them. Hyperparameter α selected by cross-validation.

**Lasso (L1 regularization):** Penalty proportional to the absolute value of coefficients. Performs automatic variable selection (some coefficients become exactly zero). Hyperparameter α selected by cross-validation.

**Reported metrics:** R² and adjusted R² for OLS; R² for Ridge and Lasso; standardized β coefficients for all models. Consistency of a predictor across the three methods (OLS + Ridge + Lasso) is interpreted as evidence of robustness. Predictors surviving in Lasso (with non-zero coefficient) constitute the minimal informative subset.

### 10.2 Explainable Non-linear Models: CART and Gradient Boosting

Linear models do not capture multiplicative interactions or discontinuities. To overcome this limitation, two non-linear models maintaining high interpretability are fitted:

**CART (Classification and Regression Tree):** Binary decision tree with the following hyperparameters:
- `max_depth=4` (depth limitation to preserve interpretability)
- `min_samples_leaf=15` (avoids overfitting to small leaves)
- `class_weight='balanced'` (corrects class imbalance in binarized outcomes)

The tree generates explicit IF-THEN rules describing paths to each prediction and allows identifying hierarchical interactions between variables (e.g., "male gender AND high disagreement AND high duration → mansplaining").

**GBM (Gradient Boosting Machine):** Ensemble of sequential trees with hyperparameters:
- 200 trees
- maximum depth = 3 (weak learners)
- learning rate = 0.05

Captures interactions not representable by a single tree. Feature importance based on MDI (Mean Decrease in Impurity) is reported.

**Outcome binarization for classification:** To use AUC as evaluation metric, continuous outcomes are binarized at the upper quartile (1 if speaker is in top 25%, 0 otherwise). AUC metrics are calculated through stratified 5-fold cross-validation.

### 10.3 Optimization through Mixed Integer Linear Programming (MILP)

As a third approach, optimal global variable selection is implemented through mathematical programming. Unlike Lasso (continuous regularization) and greedy feature selection algorithms, MILP guarantees finding the combination of exactly K variables that minimizes sample error.

**Mathematical formulation (Best-Subset L1 Regression with K=6):**

```
min  (1/n) * Σ_i |y_i - (X_i · β + b)|
s.t. |β_j| <= M · z_j    (big-M: if z_j=0 then β_j=0)
     Σ_j z_j <= K          (at most K active features)
     z_j ∈ {0,1}            (binary integer variables)
```

where `K=6` defines the maximum number of active variables, `M` is a large constant for the big-M formulation, and `z_j` is the selection indicator variable for the j-th feature.

**Implementation:** Solved with the **PuLP** library on the **CBC** solver (COIN-OR Branch and Cut), open source. A balanced sample (n+ = n−) is used to avoid bias toward the majority class.

**Justification of approach:** MILP formally solves an NP-hard problem, guaranteeing global optimum unlike heuristic methods. The `K=6` constraint represents an explicit trade-off between interpretability (few variables) and explanatory power. L1 error (not L2) is robust to outliers.

### 10.4 SHAP Values: Individual Interpretability

To complement global importance rankings (which may be affected by correlations between features) with rigorous individual-level explanations, **SHAP values** (SHapley Additive exPlanations) are applied on the GBM model.

**Theoretical foundation:** SHAP values assign to each predictor its exact marginal contribution to the prediction of each observation, based on the Shapley value from cooperative game theory. They satisfy formal properties of efficiency, symmetry, dummy, and additivity, conferring unique mathematical guarantees among interpretability methods.

**Reported metrics:**
- **Mean absolute SHAP** per feature: global importance ranking with interpretation robust to collinearities.
- **SHAP distributions** per feature: dispersion of contribution across observations.

### 10.5 Triangulation between Methods

Robust predictors are identified by **consensus between methods**. Considered:
- **Complete consensus (3/3):** predictor selected by CART, GBM, and MILP simultaneously.
- **Majority consensus (2/3):** predictor selected by at least two of the three methods.

Triangulation between OLS+Ridge+Lasso (linear), CART+GBM (non-linear), and MILP (combinatorial optimization) provides evidence that identified predictors are intrinsic to data structure and not artifacts of a particular method.

### 10.6 Comparison of Linear vs Non-linear Models

The increase in predictive power from OLS to CART/GBM is systematically reported:

| Outcome | OLS (R²) | CART (AUC) | GBM (AUC) |
|---------|---------|------------|-----------|
| Mansplaining | reference | reported | reported |
| Being interrupted (women) | reference | reported | reported |

A substantial AUC improvement over linear R² is indicative of the presence of multiplicative interactions justifying non-linear models.

---

## 11. Validation and Quality Control

### 11.1 Post-hoc Power Analysis

Statistical power achieved for each variable is calculated, determining the probability of detecting effects of different magnitudes given the available sample size. Additionally, the N required to reach 80% and 90% power under the observed effect is reported, allowing planning of future replications.

**Interpretation:** high power (>80%) to detect small effects indicates that absence of significance genuinely reflects absence of differences, while low power suggests the study might not have detected existing effects.

### 11.2 Sensitivity Analysis to Operational Thresholds

Robustness of metrics operationalized through thresholds (such as echoing index for appropriation) is evaluated by systematically varying said thresholds and observing stability of results.

### 11.3 Comparison of Models with and without Gender

To evaluate the specific contribution of gender as a predictor, complete models (with gender) versus reduced models (without gender) are compared using discrimination (AUC) and calibration (accuracy) metrics. A minimal difference between models indicates that gender does not provide additional predictive information beyond communicative style variables.

### 11.4 Triangulation between Inferential Frameworks

Main findings are contrasted across three independent inferential frameworks:
- **Classical frequentist** (Mann-Whitney U, χ², LMM/GLMM).
- **Corrected frequentist** (FDR Benjamini-Hochberg).
- **Bayesian** (95% HDI and directional posterior probability).

Convergence of the three frameworks on the same significant effects reinforces the robustness of conclusions.

---

## 12. Sensitivity Analysis for Classification Error

Validation in Section 4 estimated a gender classifier error rate of 5.7% (5/88 speakers with misclassified voice according to nominal validation criterion). Although this rate is low, misclassification of a binary predictor produces two systematic statistical effects:

1. **Attenuation of effect sizes**: observed effects are smaller than real effects (d_obs < d_real).
2. **Inflation of p-values**: statistical power decreases, increasing risk of type II error.

To formally correct both effects on cluster analysis findings (Section 9.6) and predictive models (Section 10), two complementary methods are applied.

### 12.1 Cohen's d Correction through Rogan-Gladen

**Procedure:** The correction factor kappa is defined from sensitivity and specificity observed in nominal validation:

```
kappa = Se + Sp − 1
      = 0.974 + 0.920 − 1
      = 0.894
```

where:
- **Se = 0.974** (sensitivity for women) = P(classified F | true F)
- **Sp = 0.920** (specificity for men) = P(classified M | true M)

The corrected effect size is obtained by:

```
d_corrected = d_observed / kappa
```

This equates to a multiplicative factor of 1.119 (≈ +12%) on the observed d, approximately recovering the real effect under the assumption of non-differential error.

### 12.2 p-value Correction through Monte Carlo Simulation

**Procedure:** To correct p-value inflation, **N=1,000 Monte Carlo simulations** are run. In each iteration:

1. Genders of the **94 validated speakers** (Section 4.2) remain invariant (reference information).
2. Genders of the **558 non-validated speakers** are randomly flipped with empirical error rates:
   - P(woman → classified as man) = 1 − Se = 0.026
   - P(man → classified as woman) = 1 − Sp = 0.080
3. p-values for each test are recalculated on the perturbed dataset.

**Decision criterion:** The adjusted p-value (`p_adj`) corresponds to the **75th percentile of the empirical distribution of simulated p-values**. This choice is deliberately conservative: it requires that at least 75% of simulations produce an equally or more significant result than the adjusted one, assuming unfavorable classification error scenarios.

### 12.3 Robustness Criterion

A finding is considered **robust to classification error** if:

- `p_adj < 0.05` (maintains significance after Monte Carlo correction)
- `|d_corrected| > 0.2` (maintains non-negligible magnitude after Rogan-Gladen correction)

### 12.4 Results Reporting

For each significant finding, a table with five columns is reported:

| Variable | d_obs | d_corr | p_obs | p_adj | Robust |

The `Robust` column indicates whether the finding meets both criteria. Non-robust findings are retained in the report but explicitly labeled as sensitive to classification error. This practice ensures that main conclusions are explicitly supported by formal quantification of measurement error.

---

## 13. Ethical Considerations and Limitations

### 13.1 Gender Classification

The system implements a binary classification (male/female) based on acoustic characteristics, externally validated by given name. This approach:
- Does not capture non-binary gender identities.
- Assumes correspondence between vocal characteristics and gender.
- Discrepant cases were resolved through human evaluation and nominal validation.
- Residual error rate (5.7%) is formally addressed through sensitivity analysis in Section 12.

### 13.2 Generalization Limitations

- **Specific context:** data come from scientific debates in critical care medicine, limiting extrapolation to other professional or cultural contexts.
- **Language:** analysis is restricted to sessions in English.
- **Format:** analyzed debates have a structured format with forced gender parity in panel composition, which may differ from less regulated contexts.
- **Nominal enrichment coverage:** only 14.4% of speakers were nominally identified; stratified analyses by role, specialty, citations, and region apply exclusively to this subset, while main analyses (intervention and complete user) cover the entire corpus.

### 13.3 Interpretive Limitations

- **Causality:** the observational design does not allow establishing causal relationships between gender and communicative behavior.
- **Unmeasured variables:** factors such as prior experience in international debates or specific training in scientific communication were not controlled.
- **Topic area assignment:** ESICM groups are assigned at session level and propagated to all speakers, which may introduce noise in stratified analyses by group.

---

## 14. Reproducibility

### 14.1 Technology Stack

The pipeline is implemented in Python 3.10+ using the following main tools:

| Component | Tool |
|------------|-------------|
| Audio processing | ffmpeg, librosa |
| Loudness normalization | pyloudnorm (EBU R128) |
| Speaker diarization | Pyannote.audio 3.1 |
| Automatic transcription | OpenAI Whisper (large-v2) |
| Acoustic gender classification | Parselmouth (Praat), wav2vec2-large-xlsr-53 |
| Nominal identification | RapidFuzz |
| Nominal gender validation | gender_guesser |
| Linguistic analysis | spaCy (en_core_web_lg) |
| Affective analysis | pysentimiento |
| Classical statistical analysis | statsmodels, scipy |
| Mixed effects models | statsmodels (LMM, GLMM) |
| Bayesian estimation (MCMC) | ad-hoc Metropolis-Hastings implementation |
| Regularized linear modeling | scikit-learn (Ridge, Lasso) |
| Explainable non-linear modeling | scikit-learn (DecisionTreeClassifier, GradientBoostingClassifier) |
| MILP optimization | PuLP + CBC solver |
| Individual interpretability | SHAP |
| Clustering and dimensionality reduction | scikit-learn (KMeans, AgglomerativeClustering, DBSCAN, PCA, t-SNE) |
| Visualization | matplotlib, seaborn |

### 14.2 Variable Dictionary

An exhaustive dictionary of all extracted and calculated variables is provided, specifying for each:
- Technical name
- Data type
- Range of possible values
- Operational definition
- Calculation method

This dictionary, together with the pipeline code, intermediate CSVs, and enriched CSVs with nominal metadata, guarantees complete replicability of the study.

### 14.3 Analysis Levels and Traceability

The study operates on three clearly differentiated aggregation levels, all derived from the same base corpus:

| Level | N | Main use |
|-------|---|---------------|
| Intervention | 12,138 | Bivariate tests, mixed models, transitions, temporal analysis |
| User | 652 | Clustering, multivariate predictive models, MILP, SHAP |
| Session | 75 | Structural predictors of bias, moderation effect |

Each analysis explicitly identifies its aggregation level, avoiding confusion between intervention-level effects and structural effects at speaker or session level.

---

*Methodological document prepared according to scientific publication standards to ensure transparency and replicability.*
``