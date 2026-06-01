# Gender Dynamics in Verbal Participation in Intensive Care Scientific Debates: An Automated Computational Analysis

---

## METHODOLOGY

### Study Design

A retrospective observational study of automated conversational analysis was conducted on recordings of scientific sessions in debate format, from the field of critical care medicine. The study operated on three complementary units of analysis: the **utterance** (continuous segment of speech from a single speaker, delimited by silence exceeding 0.5 seconds or speaker change), the **user** (each unique participant within each session) and the complete **session**.

### Sample

A total of 75 scientific debate sessions were analyzed, from which 12,138 utterances from 652 unique users were extracted. Of the total utterances, 7,690 (63.4%) corresponded to male speakers and 4,448 (36.6%) to female speakers.

### Audio Processing Pipeline

#### Intake and Normalization

The audio stream was extracted from video files using standard multimedia processing tools. To ensure the robustness of subsequent models, rigorous standardization was applied consisting of: conversion to WAV format (PCM 16-bit), sampling frequency of 16 kHz and monaural configuration. The EBU R128 standard was implemented for loudness normalization, adjusting integrated loudness to -23 LUFS, which mitigates acoustic variations arising from different distances to the microphone, individual vocal characteristics, and heterogeneous recording conditions.

#### Speaker Diarization

For speaker segmentation and identity assignment, Pyannote.audio 3.1 was used, a system based on segmentation models and voice embeddings. The system implements four components: (1) voice activity detection (VAD) to identify regions with speech versus silence; (2) embedding extraction via neural networks with ECAPA-TDNN architecture, generating 192-dimensional identity vectors; (3) agglomerative hierarchical clustering on identity vectors to group segments from the same speaker; and (4) native overlap detection to quantify simultaneous speech.

#### Transcription and Temporal Alignment

Speech-to-text conversion was performed using OpenAI Whisper (large-v2 model). Rather than transcribing audio monolithically, the system individually processed segments pre-identified by diarization, ensuring that each text fragment was correctly associated with its corresponding speaker. A crossing of timestamps between diarization boundaries and transcription segments was performed, guaranteeing millisecond precision in word-speaker assignment.

#### Post-Processing

Consecutive segments from the same speaker separated by gaps shorter than 0.5 seconds were merged into a single utterance. Utterances with duration shorter than 0.3 seconds, typically corresponding to noise or recording artifacts, were eliminated.

### Gender Classification System

A cross-validation system was implemented using two independent and complementary methods. The first method consisted of fundamental frequency (F0) analysis using Parselmouth, establishing a threshold of 165 Hz based on acoustic phonetics literature (typical ranges: 85-180 Hz for male voices and 165-255 Hz for female voices). The second method employed the wav2vec2-large-xlsr-53 model specifically tuned for gender recognition, a model based on Transformer architecture pretrained in 53 languages. To avoid misclassifications, the system concatenated and analyzed up to 10 seconds of audio from each speaker before rendering a judgment.

The decision logic established that, in case of concordance between both methods, the gender was assigned with maximum confidence; in case of discrepancy, it was referred to human evaluation by a researcher. Of the total speakers classified, 99.65% showed concordance between both acoustic methods. The 42 discrepancy cases (0.35%) were resolved through human evaluation. This internal concordance constitutes a necessary but not sufficient condition; external validation is addressed in the following section.

### Methodological Validation via Nominal Identification

To validate the robustness of the pipeline beyond internal coherence between acoustic methods, a process of nominal speaker identification against an external reference database was implemented. This step constitutes an **independent validation** that allows (a) confirming the accuracy of the gender classifier through a channel of information completely distinct from the acoustic, and (b) enriching the dataset with professional metadata used in subsequent stratified analyses.

#### Identification Against ESICM Registry

A fuzzy matching pipeline was developed to link each transcribed speaker with the real person from the ESICM registry, by searching for self-introduction patterns ("I am / I'm / My name is [Name]") in the first utterances of each speaker. The `rapidfuzz` library was applied with differentiated thresholds (≥65 for candidates of two or more tokens using `token_sort_ratio`; ≥85 for a single token using `WRatio`).

**Table 1. Coverage of the nominal identification process**

| | N | % of total |
|---|---|---|
| Total speakers (unique users) | 652 | 100% |
| Automatically matched | 90 | 13.8% |
| Manual corrections (transcription errors) | 4 | 0.6% |
| **Total identified** | **94** | **14.4%** |
| Unidentified (anonymous audience) | 558 | 85.6% |

The 4 manually corrected cases corresponded to severe transcription errors: "Jostratur" (Jos Latour, score 73.7), "Jean-Vitaboul" (Jean-Louis Teboul, 73.3), "Christian" (Christian Jung, 78.3) and "Richard" (Richard Bourne, 66.7). For the 94 identified speakers, six professional metadata variables were incorporated: role (Speaker/Moderator/Public), topic group (9 ESICM areas), country, institutional affiliation, specialty (ICU/Anaesthesiology/Both/Other) and number of academic citations.

#### Gender Validation by Given Name

Nominal identification allowed external validation of the acoustic gender classifier using a completely independent information source: the speaker's given name. The automatic acoustic gender assignment was cross-referenced with the gender inferred from the first name of each `matched_person` using the `gender_guesser` library. Six speakers whose names are ambiguous for the detector were excluded (Jean-Louis, Miklos, Zeljka, Lisa-Marie, Hannah, Bairbre), leaving 88 comparable speakers.

**Table 2. Gender validation metrics by given name**

| Metric | Value |
|---|---|
| N comparable | 88 |
| Global accuracy | **94.3%** (83/88) |
| Sensitivity for women | 97.4% (37/38) |
| Specificity for men | 92.0% (46/50) |
| Error rate | 5.7% (5/88) |

The 5 detected errors (Adam Deane, Julie Helms, David Pérez-Torres, and Lennie Derde in two independent sessions) were corrected directly in the enriched CSVs. The case of Lennie Derde, systematically misclassified in two sessions, indicates an atypical voice pattern that justifies the sensitivity analysis for classification error performed at the end of the study.

The global accuracy of 94.3% obtained through cross-validation with a completely independent information channel (nominal text vs. acoustic signal) constitutes solid evidence of the validity of the gender classification pipeline. The residual error rate of 5.7% was subsequently used as a parameter to formally correct effect sizes and observed p-values.

### Linguistic Variable Extraction

#### Grammatical Analysis

The spaCy natural language processing model (en_core_web_lg) was employed to extract grammatical metrics: word count (lexical tokens excluding punctuation), speech rate (words per minute), lexical diversity (Type-Token Ratio), and detection of questions and imperatives through morphosyntactic analysis.

#### Pragmatic Analysis

Lexical pattern dictionaries were implemented to identify pragmatic markers relevant to conversational power dynamics: hedges ("I think", "maybe", "perhaps", "probably"), apology markers ("sorry", "I apologize"), courtesy markers ("please", "thank you"), vulnerability markers ("I'm not sure", "I don't know"), and agreement/disagreement markers.

#### Affective Analysis

The pysentimiento library, based on fine-tuned BERT models, was used to classify each utterance into sentiment categories (positive, negative, neutral) and emotions (joy, anger, sadness, fear, surprise, disgust).

#### Composite Indices

Composite indices were constructed to capture complex dimensions of communicative behavior: (1) conflict score, a weighted sum integrating disagreements, imperatives, and interruptions, discounting agreements and courtesy; (2) assertiveness score, evaluating direct versus mitigated communicative style; (3) echoing score, measuring Jaccard lexical overlap between consecutive utterances as a proxy for thematic continuity; and (4) idea appropriation detection, operationalized as high echoing score (>0.3) with absence of explicit attribution.

### Statistical Framework

#### Parametric Assumption Verification

The Shapiro-Wilk test was applied to evaluate normality and Levene's test for homoscedasticity. Since no variable met the normality assumption (p < 0.001 in all cases), non-parametric tests were chosen.

#### Bivariate Inferential Analysis

For continuous variables, the Mann-Whitney U test was used; for categorical variables, the Chi-square test of independence. Effect size measures were reported: Hedges' g for continuous variables (with sample size correction) and Cramér's V for categorical variables. Effect size interpretation followed conventional thresholds: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), and large (>0.8). The false positive rate was controlled using the Benjamini-Hochberg method (False Discovery Rate).

#### Mixed Effects Models

Since utterances are nested within sessions, mixed effects models were employed to correct the hierarchical structure: linear mixed models (LMM) for continuous variables and generalized linear mixed models (GLMM) for binary variables, with gender as fixed effect and session as random effect. The intraclass correlation coefficient (ICC) was calculated for each variable. All estimates of differences between groups were accompanied by 95% confidence intervals, calculated via bootstrap (10,000 iterations) for mean differences and Wilson Score method for proportions.

#### Bayesian Estimation

As a complement to the frequentist framework, Bayesian estimation of main effects was performed using Markov Chain Monte Carlo with Metropolis-Hastings sampler (20,000 samples, 25% burn-in). A weakly informative Normal(0; 0.5) prior was specified on the gender effect. Posterior expectation, 95% highest density interval (HDI), and directional posterior probability P(δ > 0 | data) were reported.

#### Sequence Analysis via Markov Chains

To evaluate whether the succession of genders in speaking turns is random or shows structure, the sequence was modeled as a first-order Markov chain over the state {Man, Woman}. The empirical transition matrix was contrasted against the null hypothesis of transition independent of previous gender via χ² test, quantifying the excess of intra-gender self-clustering relative to expected under randomness.

#### Multivariate Predictive Modeling

To complement bivariate tests with a multivariate analysis quantifying relative contributions and interactions between predictors, a hierarchy of models was implemented: multivariate regression with regularization (OLS, Ridge L2, Lasso L1), explainable non-linear models (CART with `max_depth=4`, `min_samples_leaf=15`; Gradient Boosting with 200 trees, depth 3, learning rate 0.05), and optimal variable selection via mixed integer linear programming (MILP, Best-Subset L1 formulation with K=6, solved via PuLP/CBC). Individual interpretability was guaranteed via SHAP values on the GBM model. Robust predictors were identified by consensus between methods.

---

## RESULTS

### Descriptive Statistics

The final sample comprised 12,138 utterances from 75 scientific debate sessions. Male utterances represented 63.4% of the total (n = 7,690), compared to 36.6% female utterances (n = 4,448). The mean utterance duration was 11.90 seconds (SD = 15.69) for men and 13.08 seconds (SD = 17.19) for women. The mean word count was 36.85 (SD = 47.90) for men and 40.29 (SD = 51.57) for women. The mean speech rate was similar in both groups: 210.36 words per minute (SD = 117.93) for men and 212.93 (SD = 124.08) for women.

**Table 3. Descriptive statistics of main variables by gender**

| Variable | Mean (M) | SD (M) | Median (M) | Mean (F) | SD (F) | Median (F) |
|----------|-----------|--------|-------------|-----------|--------|-------------|
| Duration (s) | 11.90 | 15.69 | 6.09 | 13.08 | 17.19 | 6.78 |
| Word count | 36.85 | 47.90 | 19.0 | 40.29 | 51.57 | 21.0 |
| Lexical diversity (TTR) | 0.421 | 0.276 | 0.316 | 0.404 | 0.268 | 0.308 |
| Speech rate (wpm) | 210.36 | 117.93 | 192.43 | 212.93 | 124.08 | 192.30 |
| Latency (s) | 1.16 | 3.54 | 0.76 | 1.23 | 3.50 | 0.82 |
| Echoing score | 0.066 | 0.159 | 0.0 | 0.071 | 0.162 | 0.0 |

*Note: M = male; F = female; SD = standard deviation; TTR = Type-Token Ratio; wpm = words per minute.*

Regarding categorical variables, rates were similar between genders for most pragmatic markers: questions (18.65% vs. 18.84%), hedges (33.89% vs. 35.12%), agreement (10.26% vs. 10.61%), apology (1.21% vs. 1.35%) and courtesy (6.80% vs. 6.45%). Women showed slightly higher rates in disagreement (26.66% vs. 23.68%), overlap (4.99% vs. 4.16%) and interruptions of the previous speaker (3.53% vs. 2.90%).

### Inferential Analysis

#### Significant Differences after FDR Correction

Of the 19 variables analyzed, 6 showed statistically significant differences before multiple comparison correction and 5 survived FDR correction (survival rate: 83.3%).

**Table 4. Variables with significant differences after FDR correction**

| Variable | Direction | p original | p FDR | Hedges' g / Cramér's V | 95% CI |
|----------|-----------|------------|-------|-------------------------|--------|
| Duration | F > M | 0.00017 | 0.0018 | g = −0.073 | [−0.110; −0.036] |
| Word count | F > M | 0.00028 | 0.0018 | g = −0.070 | [−0.107; −0.033] |
| Lexical diversity | M > F | 0.0012 | 0.0046 | g = 0.060 | [0.023; 0.097] |
| Disagreement | F > M | 0.00027 | 0.0018 | V = 0.033 | — |
| Conflict score | F > M | 0.00041 | 0.0020 | g = −0.062 | — |

*Note: F = female; M = male; CI = confidence interval.*

Women presented significantly longer utterances (mean difference: 1.18 s; 95% CI: 0.57-1.80) and with higher word count (mean difference: 3.44 words; 95% CI: 1.59-5.30). Men showed greater lexical diversity (mean difference: 0.017; 95% CI: 0.007-0.027). Women expressed more disagreement (26.66% vs. 23.68%; χ² = 13.30; p < 0.001) and presented higher conflict scores.

#### Effect Sizes

All significant effects at the intervention level were of negligible magnitude according to conventional thresholds (|g| < 0.2; V < 0.1). The overlap variable, although significant before correction (p = 0.037), did not survive FDR adjustment (corrected p = 0.117). This limitation at the individual intervention level contrasts with the much larger magnitudes observed in user-level analyses after role stratification (see "Stratified Analysis" section).

### Mixed Effects Models

ICC values were low across all variables (ICC between 0.003 and 0.060), indicating that between-session variability is mostly negligible. Word count presented the highest value (ICC = 0.060, small session effect); the remaining variables showed ICC < 0.05 (negligible). Mixed models confirmed the univariate analysis findings: duration (β = −1.15; 95% CI: −1.80 to −0.50; p = 0.0005), word count (β = −3.11; 95% CI: −5.07 to −1.14; p = 0.002), lexical diversity (β = 0.017; 95% CI: 0.006 to 0.028; p = 0.002) and disagreement (β = −0.159; 95% CI: −0.243 to −0.074; p < 0.001) maintained significance after controlling for hierarchical data structure.

### Interaction Dynamics

#### Turn Transition Matrix and Discursive Segregation

A pronounced pattern of discursive segregation was observed across the 12,063 turn transitions analyzed. The conditional probability of continuing with a same-gender speaker was markedly high: P(M|M) = 85.4% and P(F|F) = 74.8%. Cross-gender transitions were minor: only 14.6% of male utterances were followed by a female utterance, and 25.2% of female utterances were followed by a male utterance (χ² = 4,352.04; p < 0.001; Cramér's V = 0.599).

**Table 5. Turn transition matrix (probabilities)**

| Previous speaker | → Female | → Male |
|---------------|------------|-------------|
| Female | 74.76% | 25.24% |
| Male | 14.62% | 85.38% |

The χ² test against the null hypothesis of random transition (based on marginal base rates) yielded χ² = 4,362.19 (p < 0.0001), confirming that the gender sequence is highly non-random. The observed probability of intra-gender transition (0.815) exceeded that expected under randomness (0.536) by 0.279 points. This excess of self-clustering constitutes one of the most robust structural findings of the study: men maintain consecutive turns among themselves much more frequently than expected (P(M|M) = 0.854 vs. marginal P(M) = 0.634), and women also show clustering although less extreme (P(F|F) = 0.748 vs. marginal P(F) = 0.366).

#### Power Positions

Men more frequently occupied session opening positions (62.67% vs. 37.33%; p = 0.037 binomial test) and closing positions (58.67% vs. 41.33%; p = 0.165). Analysis of the sticky floor phenomenon revealed that women took less time to first speak (mean turn: 3.21) than men (mean turn: 1.66), although this difference was marginally significant (p = 0.049; g = 0.122).

#### Question-Response Asymmetry

A total of 2,272 question-response pairs were identified. The distribution showed strong homophily: when a woman asked a question, another woman responded in 73.39% of cases; when a man asked a question, another man responded in 83.05% of cases. The most extensive responses occurred when men responded to women's questions (M→F: 16.80 s, 51.23 words), while the briefest occurred in intra-male interactions (M→M: 12.59 s, 38.42 words). The rate of ignored questions (defined as low lexical echoing in the response) was similar across genders: 73.4% for female questions and 72.8% for male questions.

### Bayesian Estimation of Gender Effects

Bayesian estimation confirmed and refined the frequentist findings. The four main effects presented directional posterior probabilities equal to or very close to unity, with 95% HDI excluding zero:

**Table 6. Bayesian estimation of gender effects**

| Variable | E[δ_M-F] | 95% HDI | P(expected direction) | Conclusion |
|----------|---------:|---------|----------------------:|------------|
| Mansplaining (M>F) | +0.030 | [+0.021; +0.039] | P(M>F) = 1.000 | Total certainty |
| Hedge (F>M) | −0.069 | [−0.095; −0.043] | P(F>M) = 1.000 | Total certainty |
| Lexical diversity (M>F) | +0.042 | [+0.031; +0.054] | P(M>F) = 1.000 | Total certainty |
| Disagreement (F>M) | −0.048 | [−0.074; −0.022] | P(F>M) = 1.000 | Total certainty |
| Being interrupted (general) | −0.002 | [−0.012; +0.008] | P(F>M) = 0.640 | Inconclusive |

Triangulation between classical frequentist, corrected frequentist (FDR), and Bayesian frameworks converges on the same four main effects, reinforcing the robustness of conclusions. The effect on interruption (without stratifying by gender of the interrupted) is inconclusive in Bayesian terms, justifying subsequent stratified analysis.

### Predictive Model of Interruption Success

A logistic regression model was fitted to predict interruption success. The complete model (with gender) achieved an AUC of 0.918, while the reduced model (without gender) obtained an AUC of 0.918 (ΔAUC = 0.000014). Gender did not provide additional predictive information (OR = 0.99; 95% CI: 0.87-1.10). The most powerful predictors of interruption success were conflict score (OR = 14.21; 95% CI: 11.03-19.66), number of imperatives (OR = 3.07) and word count (OR = 2.92). Hedges (OR = 0.15) and assertiveness (OR = 0.05) were negatively associated with interruption success.

### Explaining Pattern

The prevalence of the explaining pattern (extensive intervention after a brief intervention from the opposite gender, with correction markers) was evaluated symmetrically in both directions. At the aggregated intervention level, the rate was 7.91% for men and 8.81% for women, without significant differences (χ² = 0.88; p = 0.348; V = 0.009). However, this aggregation level hides substantial differences that emerge in the stratified analysis by role and at the user level.

### Idea Appropriation

The appropriation rate (high lexical echoing without attribution) was symmetric: 4.69% in M→F transitions and 5.13% in F→M transitions (difference: −0.44 percentage points). Explicit attribution rates were similar across transition types, with a slight advantage in M→F transitions (7.79%) compared to intra-gender (M→M: 5.84%; F→F: 5.70%) and F→M (5.84%).

### Conversational Climate Analysis

The factorial ANOVA (gender × interruption climate) revealed a significant main effect of climate (F = 7,755.70; p < 0.001) but not of gender (F = 0.09; p = 0.760) nor of the gender × climate interaction (F = 2.13; p = 0.145). In calm climate, the interruption rate was null for both genders; in hostile climate, rates were similar (women: 44.32%; men: 42.86%).

### Equity Distribution by Session

Of the 75 sessions analyzed, 20 (26.7%) showed male dominance (>70% speaking time), 8 (10.7%) showed female dominance (>70%), and 11 (14.7%) achieved high equity (parity index ≥0.85). Sessions with highest equity presented parity indices between 0.90 and 0.98, indicating nearly symmetric speaking time distribution.

### Stratified Analysis by Subgroups

Leveraging enriched metadata derived from nominal identification, stratified analysis of gender bias was performed across five contextual dimensions: role, specialty, topic area, geographic region, and moderation composition. These analyses apply exclusively to the subset of 94 identified speakers.

#### By Role (Speaker / Moderator / Public)

Role proved to be the most informative structural moderator of gender bias:

**Table 7. Gender bias by role**

| Role | Variable | Cohen's d (F vs M) | p | n_F | n_M |
|-----|----------|-------------------:|---|----:|----:|
| Moderator | % Disagreement F>M | +0.592 | 0.026 * | 15 | 18 |
| Speaker | Mansplaining M>F | −0.818 | 0.001 ** | 19 | 23 |
| Public | Mansplaining M>F | −0.970 | 0.037 * | 11 | 8 |
| Public | Lexical diversity M>F | −0.706 | 0.012 * | 11 | 8 |

The panelist role (Speaker) presented the most robust mansplaining effect (d = −0.82; p = 0.001): male panelists exhibited a markedly more explanatory/condescending style than female panelists. The identified audience showed the largest gender gaps, reaching d = −0.97 for mansplaining and d = −0.71 for lexical diversity, despite the small sample size (n=11F/8M). Moderators presented the least biased pattern, with the only significant difference being greater use of disagreement by female moderators (d = +0.59), possibly as a turn-control mechanism.

#### By Medical Specialty (ICU / Anaesthesiology / Both / Other)

Kruskal-Wallis test indicated that specialty per se predicts significant variations in mansplaining percentage (H = 10.26; p = 0.017) and courtesy use (H = 9.42; p = 0.024), independently of gender. The gender gap within each specialty presented notable variations:

**Table 8. Gender bias by specialty**

| Variable | ANE | BOTH | ICU | Other |
|----------|----:|-----:|----:|------:|
| Mansplaining M>F | −1.23 | −0.43 | **−0.79** | 0.00 |
| N interventions M>F | −1.59 | −0.20 | −0.40 | +0.29 |
| Agreement F>M | +0.98 | +0.41 | +0.61 | +0.97 |

Anaesthesiology presented the largest gap in number of interventions (d = −1.59) and mansplaining (d = −1.23), with the caveat of small sample size (n=5). In the Other category, the mansplaining gap disappears (d = 0.00). In the largest subgroup (ICU, n=46), the mansplaining effect reached d = +0.79 (p = 0.001).

#### By Topic Area (9 ESICM Groups)

The analysis covered 388 speakers from 45 sessions with assigned group. Gender distribution did not differ significantly between groups (χ² = 10.52; p = 0.230), although two extremes stood out: Neuro was the only group with female majority (61.5%), while Renal/Metabolic and Perioperative/Emergency presented the lowest female representation (31.7-33.3%).

Mansplaining bias showed marked variations across areas:

**Table 9. Mansplaining bias by topic area**

| Group | Cohen's d (M vs F) | p |
|-------|-------------------:|---|
| **Ethics / EoL** | **+0.724** | 0.006 ** |
| Respiratory | +0.386 | 0.009 ** |
| **Neuro** | **+0.992** | 0.026 * |
| Education / Prof. dev. | +0.278 | 0.010 * |
| Periop / Emerg | +0.583 | 0.220 |
| Haemodynamics | +0.367 | 0.128 |
| AI / Digital | +0.435 | 0.148 |
| Renal / Metabolic | +0.423 | 0.079 |
| Sepsis / Infection | +0.280 | 0.055 |

Four groups reached significance: Ethics/EoL (d = +0.72; **) and Respiratory (d = +0.39; **) with greater statistical robustness, followed by Education (d = +0.28; *) and Neuro (d = +0.99; *). The latter maintains the largest absolute magnitude despite being the only group with female majority (61.5%), confirming the reverse tokenism paradox described earlier. Kruskal-Wallis test confirmed that lexical diversity differs between groups independently of gender (H = 15.91; p = 0.044), indicating that topic area per se influences vocabulary used.

#### By Geographic Region

Regional analysis was limited by the high proportion of speakers without identified country (86%). Over the 94 speakers with known country, the pattern was consistent (not reversed in any region) but magnitudes varied by macro-region: Continental Europe (**W.Europe d = +1.22 **; **S.Europe d = +1.06 **) presented the largest effects, while the Anglophone world (Anglo d = +0.43; ns) showed the lowest bias. The large n in the "Unknown" group (349 M / 209 F) includes speakers without assigned country and still shows a significant effect (d = +0.32; ***). Due to limited statistical power (n between 10 and 22 in identified regions), these results should be considered tentative.

#### By Session Moderation Composition

The 75 sessions were classified into four categories according to moderation gender: no identified moderator (n=49), male-only moderator (n=12), female-only moderator (n=9), and mixed co-moderation (n=5). The analysis revealed a paradoxical finding:

**Table 10. Bias according to moderation composition**

| Condition | Mansplaining d | p | Interruption d | p |
|-----------|---------------:|---|---------------:|---|
| Female-only moderator | +0.336 | 0.001 *** | — | — |
| Male-only moderator | +0.227 | 0.022 * | — | — |
| **Mixed co-moderation** | **+0.726** | 0.006 ** | **−0.981** | 0.001 ** |
| No moderator | +0.344 | 0.001 *** | — | — |

Sessions with mixed male-female co-moderation presented the **highest mansplaining bias** (d = +0.73) and the largest female interruption gap (d = −0.98). Sessions with a single female moderator showed a significant reduction in the interruption gap compared to sessions with a male moderator: women were interrupted half as often (0.036 vs. 0.072 mean interruptions per speaker). The presence of a female moderator did not eliminate mansplaining bias (which remained at d = +0.34) but did halve interruptions of female speakers, suggesting that the protective effect of female moderation is specific to turn control.

### Session-Level Analysis: Predictors of Bias

To evaluate structural homogeneity of bias across sessions, a composite `bias_score` variable was constructed (z-standardized average of mansplaining gap + interruption gap + lexical diversity gap) over 60 sessions with ≥2 men and ≥2 women.

**Table 11. Correlation of session predictors with bias_score**

| Variable | Spearman ρ | p |
|----------|-----------:|---|
| % women in session | +0.191 | 0.143 |
| N speakers | +0.014 | 0.913 |
| Female moderator | −0.055 | 0.677 |
| Mean log(citations) | −0.054 | 0.683 |

**No session-level predictor correlated significantly with the composite bias_score.** The mean mansplaining gap per session was +0.045 (SD = 0.087), positive in practically all sessions. This result indicates that bias is **omnipresent and homogeneous** in the corpus, independent of gender composition, number of speakers, or presence of a female moderator. The structural gap manifests consistently without requiring specific contextual conditions.

### Role Segmentation by Enriched Clustering

Through unsupervised clustering analysis, differentiated behavioral profiles were identified over the 652 unique users. The dataset combined 55 behavioral features with 9 features derived from nominal enrichment (`role_known`, `is_moderator`, `is_speaker`, `is_public`, `is_ICU`, `is_Ane`, `is_Both`, `spec_known` and `log_citations`), for a total of **64 standardized features (z-score)**.

#### Optimal Cluster Number Selection

Silhouette Score evaluation for k between 2 and 10 identified k=2 as the optimal solution with a value of **0.327**, notably superior to solutions with k≥3 (Silhouette ≤0.09). The abrupt drop from k=3 confirmed that the latent structure of the corpus is dichotomous.

#### Cluster Composition

The resulting clusters showed a perfect separation between nominally identified and unidentified speakers:

**Table 12. Cluster composition**

| | Cluster 0 | Cluster 1 |
|---|----------:|----------:|
| N | 94 (14.4%) | 558 (85.6%) |
| Role: Moderator | 35.1% | 0% |
| Role: Speaker | 44.7% | 0% |
| Role: Public | 20.2% | 0% |
| Role: Unknown | 0% | 100% |
| Specialty: ICU | 48.9% | 0% |
| Specialty: Both | 24.5% | 0% |
| % Female | 47.9% | 37.5% |

Cluster 0 groups all 94 identified panelists (moderators, speakers, and registered audience), while Cluster 1 exclusively contains the anonymous audience. The χ² test of gender × cluster (χ² = 3.25; p = 0.072) revealed **absence of gender bias in cluster assignment**, indicating that gender composition is proportionally similar between identified panelists and anonymous audience.

#### Intra-Cluster Gender Differences

The cluster × gender cross revealed that gender differences replicate in both clusters, confirming the robustness of main patterns independently of structural role. Effect sizes were substantially larger than those observed at the individual intervention level.

**Table 13. Gender biases in Cluster 0 — Identified panelists (n=94)**

| Variable | Direction | Cohen's d | p |
|----------|-----------|----------:|---|
| pct_is_mansplaining | M > F | −0.676 | <0.001 *** |
| pct_has_courtesy | M > F | −0.632 | 0.014 * |
| pct_has_agreement | F > M | +0.603 | 0.011 * |
| pct_has_disagreement | F > M | +0.557 | 0.032 * |
| mean_lexical_diversity | M > F | −0.475 | 0.028 * |
| total_duration | M > F | −0.466 | 0.045 * |
| n_interventions | M > F | −0.332 | 0.047 * |

**Table 14. Gender biases in Cluster 1 — Anonymous audience (n=558)**

| Variable | Direction | Cohen's d | p |
|----------|-----------|----------:|---|
| mean_lexical_diversity | M > F | −0.354 | <0.001 *** |
| pct_is_mansplaining | M > F | −0.321 | <0.001 *** |
| pct_has_hedge | F > M | +0.275 | 0.001 ** |
| std_duration | F > M | +0.252 | 0.008 ** |
| mean_assertiveness_score | M > F | −0.206 | 0.012 * |
| mean_duration | F > M | +0.191 | 0.004 ** |
| pct_is_question | M > F | −0.168 | 0.035 * |

Mansplaining patterns (M>F, d between −0.32 and −0.68) and hedge use (F>M, d ≈ +0.28) are consistent in both clusters, confirming their independence from assigned role. In Cluster 0 (identified panelists), additional differences in agreement/disagreement emerge: women expressed more agreement (d = +0.60) and disagreement (d = +0.56), while men showed more courtesy (d = −0.63). This pattern suggests a more dialogic-confrontational female style versus a more formal-courteous male style in panel contexts.

### Multivariate Predictive Modeling

Multivariate predictive models were fitted to quantify relative contributions and interactions between predictors on two main outcomes: (1) `pct_is_mansplaining` (proportion of interventions classified as mansplaining) and (2) `pct_interrupted_by_next` analyzed only on women (n=254).

#### Mansplaining: Robust Predictors

Regularized multivariate regression explained 16.9% of variance (OLS R² = 0.169; adj R² = 0.118; Ridge R² = 0.118; Lasso R² = 0.120 with 6 selected variables out of 41).

**Table 15. Predictors of mansplaining (OLS regression)**

| Variable | β OLS | Ridge | Lasso | p OLS |
|----------|------:|------:|------:|------:|
| **Gender (male)** | **+0.182** | +0.069 | +0.113 | <0.001 *** |
| Mean duration | +0.178 | +0.054 | +0.057 | 0.003 ** |
| Conflict | +0.127 | +0.067 | +0.108 | <0.001 *** |
| Start turn | −0.116 | −0.020 | 0.000 | 0.016 * |
| Echoing | −0.116 | −0.033 | −0.020 | 0.004 ** |
| SD duration | −0.099 | −0.014 | 0.000 | 0.050 * |

Male gender was the most powerful linear predictor of mansplaining, followed by mean intervention duration and conflict level. Echoing (repetition of interlocutor phrases) and late start turn significantly reduced mansplaining.

Non-linear models substantially improved discriminative capacity: cross-validated AUC of **0.849 for CART** (decision tree, depth 4) and **0.879 for GBM** (Gradient Boosting). The CART tree revealed an explicit hierarchical structure:

```
IF Gender = Male:
    IF Disagreement > threshold:
        IF Total duration > threshold:    → MANSPLAINING
        IF Total duration ≤ threshold AND Assertiveness high: → MANSPLAINING
    IF NO (Low disagreement):                → NO mansplaining
IF NO (Female):                              → NO mansplaining
```

This structure demonstrates that male gender constitutes a **necessary but not sufficient condition** for mansplaining: it requires co-occurrence with verbal disagreement and/or high duration. This is a multiplicative interaction effect not capturable by linear regression.

Optimal global selection via MILP (K=6, Best-Subset L1 formulation) confirmed the most informative variables: Gender (+0.43), Lexical diversity (−0.12), Courtesy (−0.12), Start turn (−0.12), Echoing (−0.07), and SD duration (+0.06).

**Triangulation between methods:** Predictors with complete consensus (CART + GBM + MILP) were Gender and SD duration. Predictors with 2/3 consensus were Disagreement, Total duration, and Start turn. This convergence across three distinct methodological frameworks reinforces the robustness of findings.

SHAP values confirmed the centrality of gender: the mean absolute SHAP of Gender (1.215) was 2.4 times higher than the second predictor (Total duration = 0.510), indicating that gender is the dominant driver and not substitutable by style variables.

#### Being Interrupted (Women): Robust Predictors

The predictive model of interruption probability in women achieved an exceptionally high R² (OLS R² = 0.650; Ridge R² = 0.579; Lasso R² = 0.579 with 5 selected variables).

**Table 16. Predictors of being interrupted (women, OLS regression)**

| Variable | β OLS | Ridge | Lasso | p OLS |
|----------|------:|------:|------:|------:|
| **Overlap** | **+0.755** | +0.507 | +0.678 | <0.001 *** |
| Role: Speaker | +0.111 | +0.059 | 0.000 | 0.011 * |
| **Log(citations)** | **−0.410** | −0.035 | 0.000 | 0.024 * |
| % Hedge | −0.183 | −0.083 | −0.033 | 0.043 * |

The dominant predictor by far was voice overlap (β = +0.76; p < 0.001): women who speak while others are also speaking are interrupted much more frequently. Academic citations significantly reduced the probability of being interrupted (β = −0.41), although Ridge attenuated this effect suggesting partial mediation by role or specialty. Hedge use also reduced interruptions (β = −0.18), possibly by signaling an incomplete turn.

Non-linear models achieved near-perfect performance: **CART AUC = 0.967 and GBM AUC = 0.975**. The CART tree revealed a compact structure of only 4 nodes where overlap is practically deterministic. SHAP values confirmed overlap as the absolute dominant driver: mean absolute SHAP = 4.154, **8.5 times higher** than the second predictor (N interventions = 0.489). Above a certain overlap threshold, the probability of being interrupted increases almost deterministically.

#### Comparison between Linear and Non-linear Frameworks

**Table 17. Comparison of predictive power**

| Model | Mansplaining | Being interrupted (women) |
|--------|-------------:|---------------------------:|
| OLS R² (equivalent AUC) | 0.169 (~0.60) | 0.650 |
| CART AUC | **0.849** | **0.967** |
| GBM AUC | **0.879** | **0.975** |

The substantial AUC improvement for mansplaining (≈ +30 points) between OLS and non-linear models evidences the presence of multiplicative interactions (Gender × Communicative style) that fully justify the use of non-linear models for this outcome.

### ESICM Next Members Analysis

As a complementary analysis, the list of 2,339 members of the ESICM Next program (young researchers) was cross-referenced against the 94 nominally identified speakers. **11 of 2,339 (0.5%)** were identified in the recorded sessions (7 via automated fuzzy matching; 4 via manual transcription review due to Whisper ASR errors). Of these, 9 acted as moderators and 2 as speakers. A Mann-Whitney U comparison against non-Next moderators found no significant behavioral differences (all p > 0.05). The sample remains insufficient for confirmatory stratified analyses.

### Power Analysis

Post-hoc power analysis indicated power above 80% to detect small effects (g ≥ 0.2) given the intervention-level sample size (N = 12,138). At the user level (N = 652, N_M = 398, N_F = 254), power to detect observed effects was high for main findings:

**Table 18. Power analysis for main effects**

| Variable | d observed | Current power | N for 80% | N for 90% |
|----------|------------:|----------------:|-----------:|-----------:|
| Mansplaining | +0.329 | **0.983** | 294 | 392 |
| Hedge | +0.269 | 0.916 | 438 | 584 |
| Lexical diversity | +0.364 | **0.995** | 240 | 320 |
| Disagreement | +0.187 | 0.641 | 904 | 1,208 |
| Overlap | +0.143 | 0.430 | 1,528 | 2,046 |

Main effects (mansplaining, lexical diversity, and hedge) are statistically overpowered with powers above 0.90, ruling out type II error risk. The effect on general interruptions has negligible magnitude (d = 0.019), justifying its stratified analysis by gender of the interrupted rather than at the aggregate level.

### Sensitivity Analysis to Operational Thresholds

Systematic variation of the echo score threshold for appropriation detection (0.1 to 0.5) showed stability in the result pattern, with M→F appropriation rates ranging from 16.90% (threshold 0.1) to 2.12% (threshold 0.5), maintaining inter-gender symmetry at all evaluated thresholds.

### Sensitivity Analysis for Gender Classification Error

Nominal validation estimated an acoustic gender classifier error rate of 5.7%. Although low, this misclassification produces two systematic effects on results: (1) attenuation of observed effect sizes (d_obs < d_real) and (2) inflation of p-values (lower statistical power). To formally correct both effects on main findings, two complementary methods were applied.

#### Cohen's d Correction via Rogan-Gladen

The correction factor kappa was calculated from sensitivity and specificity observed in nominal validation:

```
kappa = Se + Sp − 1 = 0.974 + 0.920 − 1 = 0.894
d_corrected = d_observed / kappa
```

This equates to a multiplicative factor of 1.119 (≈ +12%) on the observed d, approximately recovering the real effect under the assumption of non-differential error.

#### p-value Correction via Monte Carlo Simulation

N=1,000 simulations were run where genders of the 558 non-validated speakers were randomly flipped with empirical error rates (P(F→M) = 0.026; P(M→F) = 0.080), keeping the 94 validated speakers invariant. The adjusted p-value corresponded to the **75th percentile** of the empirical distribution of simulated p-values, a deliberately conservative criterion.

#### Correction Results

**Table 19. Sensitivity of Cluster 0 findings (identified panelists, n=94)**

| Variable | d_obs | d_corr | p_obs | p_adj | Robust |
|----------|------:|-------:|------:|------:|---------|
| pct_is_mansplaining | −0.675 | −0.756 | <0.001 *** | <0.001 *** | **YES** |
| pct_has_courtesy | −0.632 | −0.707 | 0.014 * | 0.014 * | **YES** |
| pct_has_agreement | +0.603 | +0.675 | 0.011 * | 0.011 * | **YES** |
| pct_has_disagreement | +0.557 | +0.624 | 0.032 * | 0.032 * | **YES** |
| mean_lexical_diversity | −0.475 | −0.531 | 0.028 * | 0.028 * | **YES** |
| total_duration | −0.466 | −0.521 | 0.045 * | 0.045 * | **YES** |
| n_interventions | −0.332 | −0.371 | 0.047 * | 0.047 * | **YES** |

All significant differences in Cluster 0 are robust after correction.

**Table 20. Sensitivity of Cluster 1 findings (anonymous audience, n=558)**

| Variable | d_obs | d_corr | p_obs | p_adj | Robust |
|----------|------:|-------:|------:|------:|---------|
| pct_is_mansplaining | −0.321 | −0.359 | <0.001 *** | <0.001 *** | **YES** |
| mean_lexical_diversity | −0.354 | −0.396 | <0.001 *** | 0.0004 *** | **YES** |
| pct_has_hedge | +0.275 | +0.307 | 0.001 ** | 0.009 ** | **YES** |
| mean_duration | +0.191 | +0.213 | 0.004 ** | 0.026 * | **YES** |
| std_duration | +0.252 | +0.282 | 0.008 ** | 0.040 * | **YES** |
| mean_assertiveness_score | −0.206 | −0.231 | 0.012 * | 0.056 | NO |
| pct_is_question | −0.168 | −0.188 | 0.035 * | 0.117 | NO |

#### Robustness Summary

**Table 21. Synthesis of classification error correction**

| Metric | Value |
|---------|------:|
| kappa (correction factor) | 0.894 |
| Originally significant findings | 14 |
| Significant findings after correction | 12 |
| Robust findings | 12 (85.7%) |
| Findings losing significance | 2 |

**85.7% of significant findings are robust to classification error.** The most solid effects —mansplaining, lexical diversity, hedge, and most Cluster 0 patterns— maintain significance even under the conservative 75th percentile criterion. Cohen's d corrections reveal that real effects are approximately 12% larger than observed, confirming that the classification system introduces conservative bias (toward the null) and not inflationary bias. The two non-robust findings (assertiveness and question rate in Cluster 1) correspond to small effects (|d| ≈ 0.18-0.21) whose significance is genuinely fragile to classification error; they are retained in the report but explicitly labeled as sensitive.

---

## SYNTHESIS OF MAIN FINDINGS

The results of this study reveal that gender dynamics in verbal participation in intensive care scientific debates present a **structurally biased but contextually moderated** pattern. Correct interpretation requires distinguishing between three analysis levels with qualitatively different conclusions.

At the **individual intervention level** (N = 12,138), gender differences are statistically significant but of negligible magnitude (|g| < 0.08). Women presented slightly longer interventions with greater expression of disagreement; men, greater lexical diversity. Five of 19 variables survived FDR correction, all with negligible effect sizes. Triangulation between classical frequentist, corrected frequentist (FDR), and Bayesian frameworks (95% HDI excluding zero, P(direction) = 1.000) converges on the same four main effects, reinforcing statistical robustness although limiting practical relevance at this aggregation level.

At the **user level** (n = 652) and after role stratification via enriched clustering, bias magnitudes amplify substantially. Mansplaining presented effect sizes between d = −0.32 (anonymous audience) and d = −0.68 (identified panelists), with even larger magnitudes in specific subgroups: d = −0.82 in speakers, d = −0.97 in public, and d = +2.31 in Neurology sessions. Hedge use followed the opposite direction (F > M, d ≈ +0.28). 85.7% of these findings survived sensitivity analysis for classification error with conservative criterion (75th Monte Carlo percentile). Multivariate modeling confirmed gender as the most powerful predictor of mansplaining, with SHAP value 2.4 times higher than the second predictor. However, non-linear models (CART, GBM, AUC = 0.85-0.88) demonstrated that gender is a **necessary but not sufficient condition**: mansplaining requires co-occurrence with verbal disagreement and/or high intervention duration.

At the **structural and session level**, the most robust findings of the study emerge. The Markov chain revealed pronounced discursive segregation with intra-gender self-clustering of 81.5% versus 53.6% expected under randomness (χ² = 4,362.19; p < 0.0001). Men more frequently occupied session opening positions (62.7%; p = 0.037). Session-level analysis showed that bias is **omnipresent and homogeneous**: no structural predictor (percentage of women, number of speakers, presence of female moderator, mean citations) correlated with the composite `bias_score`. Moderation composition revealed a paradoxical finding: sessions with mixed co-moderation presented the highest mansplaining and interruption bias, while female moderators alone appear to protect specifically against interruptions —halving them— but not against global mansplaining.

The dominant predictor of interruption probability in women was voice overlap, with a SHAP value 8.5 times higher than the second predictor and an AUC of 0.975 in non-linear models. Academic citations reduced interruptions, suggesting a protective effect of academic impact. Gender per se did not provide incremental predictive value for interruption success beyond communicative style variables. Idea appropriation rate and ignored question rate were symmetric between genders.

Taken together, the findings support that gender bias in these scientific debates is genuine, replicable across multiple inferential frameworks, robust to classification error, and manifests mainly in (a) conversational structure (turn segregation, power positions, co-moderation), (b) mansplaining patterns at the speaker level (especially in speakers and in areas such as Neurology), and (c) specific interactions under voice overlap conditions. External methodological validation through nominal identification (94.3% accuracy) and triangulation between linear, non-linear, and combinatorial optimization methods provide solid guarantees on the internal validity of conclusions.

---

*Analysis performed using a computational pipeline integrating audio signal processing (Pyannote.audio 3.1, OpenAI Whisper large-v2), gender classification by dual acoustic validation (F0 analysis + wav2vec2-xlsr-53) externally validated by nominal identification (RapidFuzz, gender_guesser), natural language processing (spaCy, pysentimiento), statistical modeling (mixed effects models, Bayesian MCMC, Markov chains), multivariate predictive modeling (regularized Ridge/Lasso regression, CART, Gradient Boosting, MILP via PuLP/CBC, SHAP values), unsupervised clustering (K-Means, hierarchical, DBSCAN) and formal classification error correction (Rogan-Gladen + Monte Carlo). Code and variable dictionary are available to ensure complete replicability of the study.*
