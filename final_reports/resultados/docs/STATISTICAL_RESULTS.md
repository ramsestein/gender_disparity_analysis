# STATISTICAL RESULTS: GENDER DISPARITY IN SCIENTIFIC DEBATES

---

## 1. SAMPLE STATISTICS

| Metric | Value |
|---------|-------|
| Total sessions | 75 |
| Total interventions | 12,138 |
| Male interventions | 7,690 (63.4%) |
| Female interventions | 4,448 (36.6%) |

---

## 2. DESCRIPTIVE STATISTICS BY GENDER

### 2.1 Numeric Variables

| Variable | Mean M | SD M | Median M | Mean F | SD F | Median F |
|----------|---------|------|-----------|---------|------|-----------|
| duration (s) | 11.90 | 15.69 | 6.09 | 13.08 | 17.19 | 6.78 |
| word_count | 36.85 | 47.90 | 19.0 | 40.29 | 51.57 | 21.0 |
| lexical_diversity | 0.421 | 0.276 | 0.316 | 0.404 | 0.268 | 0.308 |
| wpm | 210.36 | 117.93 | 192.43 | 212.93 | 124.08 | 192.30 |
| latency_s | 1.16 | 3.54 | 0.76 | 1.23 | 3.50 | 0.82 |
| overlap_duration (s) | 0.25 | 2.37 | 0.0 | 0.32 | 2.76 | 0.0 |
| echoing_score | 0.066 | 0.159 | 0.0 | 0.071 | 0.162 | 0.0 |
| num_imperatives | 0.044 | 0.214 | 0.0 | 0.040 | 0.208 | 0.0 |

### 2.2 Categorical Variables (Rates)

| Variable | Rate M | Rate F |
|----------|--------|--------|
| is_question | 18.65% | 18.84% |
| has_hedge | 33.89% | 35.12% |
| has_disagreement | 23.68% | 26.66% |
| has_agreement | 10.26% | 10.61% |
| has_courtesy | 6.80% | 6.45% |
| has_apology | 1.21% | 1.35% |
| has_title | 5.47% | 6.09% |
| has_attribution | 5.84% | 6.23% |
| has_vulnerability | 2.00% | 2.23% |
| has_overlap | 4.16% | 4.99% |
| interrupts_previous | 2.90% | 3.53% |
| interrupted_by_next | 3.99% | 4.65% |
| interruption_success | 1.35% | 1.51% |
| is_backchannel | 0.90% | 0.92% |

---

## 3. PARAMETRIC ASSUMPTION VERIFICATION

| Variable | Shapiro p (M) | Shapiro p (F) | Levene p | Normality | Homoscedasticity | Applied Test |
|----------|---------------|---------------|----------|------------|------------------|---------------|
| duration | 2.44e-80 | 1.46e-67 | 0.0003 | No | No | Mann-Whitney U |
| word_count | 1.97e-80 | 1.70e-67 | 0.0023 | No | No | Mann-Whitney U |
| lexical_diversity | 4.60e-77 | 6.19e-66 | 0.0145 | No | No | Mann-Whitney U |
| wpm | 3.14e-90 | 1.94e-78 | 0.6311 | No | Yes | Mann-Whitney U |
| latency_s | 7.13e-98 | 1.90e-81 | 0.0345 | No | No | Mann-Whitney U |
| overlap_duration | 1.22e-105 | 1.87e-90 | 0.1176 | No | Yes | Mann-Whitney U |
| echoing_score | 6.55e-92 | 3.77e-77 | 0.0962 | No | Yes | Mann-Whitney U |
| conflict_score | 1.03e-73 | 8.59e-61 | 0.0003 | No | No | Mann-Whitney U |
| assertiveness_score | 5.57e-77 | 1.84e-63 | 0.2346 | No | Yes | Mann-Whitney U |

---

## 4. EFFECT SIZES

### 4.1 Numeric Variables (Hedges' g)

| Variable | Hedges' g | 95% CI Lower | 95% CI Upper | Magnitude |
|----------|-----------|-----------------|-----------------|----------|
| duration | -0.073 | -0.110 | -0.036 | Negligible |
| word_count | -0.070 | -0.107 | -0.033 | Negligible |
| lexical_diversity | 0.060 | 0.023 | 0.097 | Negligible |
| wpm | -0.021 | -0.058 | 0.016 | Negligible |
| latency_s | -0.021 | -0.058 | 0.016 | Negligible |
| overlap_duration | -0.029 | -0.066 | 0.007 | Negligible |
| echoing_score | -0.031 | -0.068 | 0.006 | Negligible |
| num_imperatives | 0.020 | -0.017 | 0.057 | Negligible |

### 4.2 Categorical Variables (Cramér's V)

| Variable | Cramér's V | χ² | p-value | Magnitude |
|----------|-------------|-----|---------|----------|
| has_disagreement | 0.033 | 13.30 | 0.00027 | Negligible |
| has_overlap | 0.019 | 4.36 | 0.0369 | Negligible |
| interrupts_previous | 0.017 | 3.48 | 0.0621 | Negligible |
| interrupted_by_next | 0.015 | 2.88 | 0.0897 | Negligible |
| has_hedge | 0.012 | 1.83 | 0.1758 | Negligible |
| has_courtesy | 0.006 | 0.50 | 0.4814 | Negligible |
| interruption_success | 0.006 | 0.38 | 0.5397 | Negligible |
| has_agreement | 0.005 | 0.34 | 0.5615 | Negligible |
| has_apology | 0.005 | 0.34 | 0.5622 | Negligible |
| is_question | 0.002 | 0.06 | 0.8122 | Negligible |

---

## 5. MULTIPLE COMPARISON CORRECTION (FDR)

### 5.1 Benjamini-Hochberg Correction Summary

| Metric | Value |
|---------|-------|
| Significant variables before FDR | 6/19 |
| Significant variables after FDR | 5/19 |
| Survival rate | 83.3% |

### 5.2 Significant Variables Post-FDR

| Variable | Original p | Corrected p FDR | Original Significant | Post-FDR Significant |
|----------|------------|-----------------|------------------------|------------------------|
| duration | 0.000165 | 0.00176 | Yes | Yes |
| word_count | 0.000278 | 0.00176 | Yes | Yes |
| has_disagreement | 0.000265 | 0.00176 | Yes | Yes |
| conflict_score | 0.000412 | 0.00196 | Yes | Yes |
| lexical_diversity | 0.00122 | 0.00463 | Yes | Yes |
| has_overlap | 0.0369 | 0.1168 | Yes | No |

### 5.3 Complete FDR Correction Table

| Variable | Original p | p FDR | Orig. Sig. | FDR Sig. | Survives |
|----------|------------|-------|---------------|----------|-----------|
| duration | 0.000165 | 0.00176 | Yes | Yes | Yes |
| word_count | 0.000278 | 0.00176 | Yes | Yes | Yes |
| lexical_diversity | 0.00122 | 0.00463 | Yes | Yes | Yes |
| conflict_score | 0.000412 | 0.00196 | Yes | Yes | Yes |
| has_disagreement | 0.000265 | 0.00176 | Yes | Yes | Yes |
| has_overlap | 0.0369 | 0.1168 | Yes | No | No |
| interrupts_previous | 0.0621 | 0.1685 | No | No | No |
| interrupted_by_next | 0.0897 | 0.2058 | No | No | No |
| echoing_score | 0.0975 | 0.2058 | No | No | No |
| overlap_duration | 0.1331 | 0.2529 | No | No | No |
| has_hedge | 0.1758 | 0.3036 | No | No | No |
| wpm | 0.2629 | 0.3925 | No | No | No |
| latency_s | 0.2686 | 0.3925 | No | No | No |
| has_courtesy | 0.4814 | 0.5934 | No | No | No |
| assertiveness_score | 0.5428 | 0.5934 | No | No | No |
| interruption_success | 0.5397 | 0.5934 | No | No | No |
| has_agreement | 0.5615 | 0.5934 | No | No | No |
| has_apology | 0.5622 | 0.5934 | No | No | No |
| is_question | 0.8122 | 0.8122 | No | No | No |

---

## 6. CONFIDENCE INTERVALS (95%)

### 6.1 Numeric Variables

| Variable | Mean M | Mean F | Difference (M-F) | CI Lower | CI Upper | Crosses Zero |
|----------|---------|---------|------------------|-------------|-------------|------------|
| duration | 11.90 | 13.08 | -1.18 | -1.80 | -0.57 | No |
| word_count | 36.85 | 40.29 | -3.44 | -5.30 | -1.59 | No |
| lexical_diversity | 0.421 | 0.404 | 0.017 | 0.007 | 0.027 | No |
| wpm | 210.36 | 212.93 | -2.57 | -7.07 | 1.93 | Yes |
| latency_s | 1.16 | 1.23 | -0.07 | -0.20 | 0.06 | Yes |
| overlap_duration | 0.25 | 0.32 | -0.07 | -0.17 | 0.02 | Yes |
| echoing_score | 0.066 | 0.071 | -0.005 | -0.011 | 0.001 | Yes |
| conflict_score | 0.376 | 0.438 | -0.062 | -0.096 | -0.028 | No |
| assertiveness_score | 1.974 | 1.982 | -0.008 | -0.033 | 0.017 | Yes |
| num_imperatives | 0.044 | 0.040 | 0.004 | -0.004 | 0.012 | Yes |

### 6.2 Categorical Variables

| Variable | Rate M | CI M Lower | CI M Upper | Rate F | CI F Lower | CI F Upper | Difference |
|----------|--------|----------|----------|--------|----------|----------|------------|
| has_overlap | 4.16% | 3.74% | 4.63% | 4.99% | 4.39% | 5.67% | -0.83% |
| interrupts_previous | 2.90% | 2.55% | 3.30% | 3.53% | 3.03% | 4.11% | -0.63% |
| interrupted_by_next | 3.99% | 3.58% | 4.45% | 4.65% | 4.07% | 5.31% | -0.66% |
| interruption_success | 1.35% | 1.12% | 1.64% | 1.51% | 1.19% | 1.91% | -0.15% |
| has_hedge | 33.89% | 32.84% | 34.95% | 35.12% | 33.73% | 36.53% | -1.23% |
| has_disagreement | 23.68% | 22.74% | 24.64% | 26.66% | 25.38% | 27.98% | -2.98% |
| has_agreement | 10.26% | 9.60% | 10.96% | 10.61% | 9.74% | 11.55% | -0.35% |
| has_apology | 1.21% | 0.99% | 1.48% | 1.35% | 1.05% | 1.73% | -0.14% |
| has_courtesy | 6.80% | 6.26% | 7.39% | 6.45% | 5.77% | 7.21% | 0.35% |
| is_question | 18.65% | 17.79% | 19.53% | 18.84% | 17.72% | 20.02% | -0.19% |

---

## 7. MIXED EFFECTS MODELS

### 7.1 Intraclass Correlation Coefficient (ICC)

| Variable | ICC | Between-Session Variance | Within-Session Variance | Interpretation |
|----------|-----|-------------------------|----------------------|----------------|
| duration | 0.047 | 12.45 | 254.42 | Negligible |
| word_count | 0.060 | 146.94 | 2312.21 | Small session effect |
| lexical_diversity | 0.010 | 0.001 | 0.074 | Negligible |
| wpm | 0.019 | 274.81 | 14194.14 | Negligible |
| latency_s | 0.011 | 0.141 | 12.32 | Negligible |
| overlap_duration | 0.013 | 0.083 | 6.26 | Negligible |
| echoing_score | 0.011 | 0.000 | 0.025 | Negligible |
| conflict_score | 0.020 | 0.017 | 0.83 | Negligible |
| assertiveness_score | 0.003 | 0.001 | 0.46 | Negligible |

### 7.2 Mixed Model Results

| Variable | Naive p | Mixed p | Coefficient | CI Lower | CI Upper | Type |
|----------|---------|---------|-------------|-------------|-------------|------|
| duration | 0.000165 | 0.000512 | -1.150 | -1.799 | -0.501 | LMM |
| word_count | 0.000278 | 0.00193 | -3.105 | -5.068 | -1.142 | LMM |
| lexical_diversity | 0.00122 | 0.00175 | 0.017 | 0.006 | 0.028 | LMM |
| wpm | 0.263 | 0.380 | -2.140 | -6.915 | 2.636 | LMM |
| latency_s | 0.269 | 0.688 | -0.029 | -0.168 | 0.111 | LMM |
| overlap_duration | 0.133 | 0.0126 | -0.127 | -0.227 | -0.027 | LMM |
| echoing_score | 0.0975 | 0.468 | -0.002 | -0.009 | 0.004 | LMM |
| conflict_score | 0.000412 | 0.000912 | -0.062 | -0.098 | -0.025 | LMM |
| assertiveness_score | 0.543 | 0.699 | -0.005 | -0.031 | 0.021 | LMM |
| has_overlap | 0.0369 | 0.0332 | -0.191 | -0.366 | -0.015 | GLMM |
| interrupts_previous | 0.0621 | 0.0553 | -0.203 | -0.411 | 0.005 | GLMM |
| interrupted_by_next | 0.0897 | 0.0814 | -0.160 | -0.341 | 0.020 | GLMM |
| has_disagreement | 0.000265 | 0.000245 | -0.159 | -0.243 | -0.074 | GLMM |
| has_hedge | 0.176 | 0.170 | -0.054 | -0.132 | 0.023 | GLMM |
| has_agreement | 0.562 | 0.541 | -0.038 | -0.158 | 0.083 | GLMM |
| has_apology | 0.562 | 0.507 | -0.111 | -0.437 | 0.216 | GLMM |
| has_courtesy | 0.481 | 0.458 | 0.056 | -0.093 | 0.205 | GLMM |
| is_question | 0.812 | 0.794 | -0.013 | -0.107 | 0.082 | GLMM |
| interruption_success | 0.540 | 0.488 | -0.109 | -0.419 | 0.200 | GLMM |

---

## 8. QUESTION-RESPONSE ASYMMETRY

### 8.1 Response Duration by Questioner Gender

| Questioner Gender | Mean (s) | Median (s) | SD | Mixed p | Coefficient |
|--------------------|-----------|-------------|-----|---------|-------------|
| Female | 14.17 | 6.31 | 19.55 | 0.381 | -0.755 |
| Male | 13.30 | 6.29 | 17.99 | 0.381 | -0.755 |

### 8.2 Response Matrix: Who Responds to Whom? (%)

| Questioner \ Responder | Female | Male |
|---------------------------|----------|-----------|
| Female | 73.39% | 26.61% |
| Male | 16.95% | 83.05% |

### 8.3 Ignored Question Rate

| Questioner Gender | Ignored Rate |
|--------------------|---------------|
| Female | 73.39% |
| Male | 74.20% |

---

## 9. IDEA APPROPRIATION

### 9.1 Appropriation Matrix (%)

| Previous Gender \ Current Gender | Female | Male |
|-------------------------------|----------|-----------|
| Female | 7.44% | 4.69% |
| Male | 5.13% | 6.24% |

### 9.2 Echoing Score and Attribution

| Transition | Echoing Score | Attribution Rate |
|------------|---------------|-----------------|
| F → F | 0.074 | 5.70% |
| F → M | 0.061 | 5.84% |
| M → F | 0.063 | 7.79% |
| M → M | 0.067 | 5.84% |

### 9.3 Appropriation Asymmetry

| Metric | Value |
|---------|-------|
| Appropriation M→F | 4.69% |
| Appropriation F→M | 5.13% |
| Asymmetry (M→F - F→M) | -0.44 |

---

## 10. POWER POSITIONS

### 10.1 Distribution at Session Boundaries

| Position | % Male | p-value (Binomial) |
|----------|-------------|-------------------|
| First speaker | 62.67% | 0.0370 |
| Last speaker | 58.67% | 0.1654 |

---

## 11. TURN TRANSITION MATRIX

### 11.1 Transition Probabilities (%)

| From \ To | Female | Male |
|---------------|----------|-----------|
| Female | 74.76% | 25.24% |
| Male | 14.62% | 85.38% |

### 11.2 Independence Test

| Metric | Value |
|---------|-------|
| χ² | 4352.04 |
| p-value | 0.0 |
| Cramér's V | 0.599 |

### 11.3 Characteristics of Cross-Gender vs Intra-Gender Transitions

| Transition Type | Mean Duration | Conflict Score | Echoing Score |
|-----------------|----------------|----------------|---------------|
| Intra-gender | 11.93 | 0.394 | 0.069 |
| Inter-gender | 14.07 | 0.419 | 0.062 |

---

## 12. PREDICTIVE MODEL OF INTERRUPTION SUCCESS

### 12.1 Model Comparison

| Metric | Full Model | Model without Gender | Delta |
|---------|-----------------|-------------------|-------|
| AUC | 0.9184 | 0.9184 | 0.000014 |
| Accuracy | 0.9855 | 0.9856 | -0.00008 |

### 12.2 Odds Ratios of Full Model

| Variable | Coefficient | Odds Ratio | 95% CI Lower | 95% CI Upper |
|----------|-------------|------------|-----------------|-----------------|
| conflict_score | 2.654 | 14.21 | 11.03 | 19.66 |
| num_imperatives | 1.121 | 3.07 | 2.51 | 3.87 |
| word_count | 1.073 | 2.92 | 1.23 | 4.94 |
| phase_Q3 | 0.203 | 1.22 | 0.97 | 1.60 |
| phase_Q4 | 0.030 | 1.03 | 0.79 | 1.30 |
| phase_Q2 | 0.008 | 1.01 | 0.81 | 1.33 |
| gender_bin (M=1) | -0.005 | 0.99 | 0.87 | 1.10 |
| latency_s | -0.259 | 0.77 | 0.69 | 0.82 |
| wpm | -0.596 | 0.55 | 0.39 | 0.75 |
| lexical_diversity | -0.976 | 0.38 | 0.28 | 0.48 |
| duration | -1.421 | 0.24 | 0.12 | 0.62 |
| has_hedge | -1.892 | 0.15 | 0.11 | 0.20 |
| assertiveness_score | -2.985 | 0.05 | 0.03 | 0.06 |

---

## 13. CLIMATE AND INTERACTION ANALYSIS

### 13.1 Factorial ANOVA (Gender × Climate)

| Factor | Sum of Squares | df | F | p-value |
|--------|----------------|-----|------|---------|
| Gender | 0.0009 | 1 | 0.093 | 0.760 |
| Climate | 76.31 | 1 | 7755.70 | <0.001 |
| Gender × Climate | 0.021 | 1 | 2.13 | 0.145 |
| Residual | 103.43 | 10512 | — | — |

### 13.2 Interruption Means by Climate and Gender

| Gender | Calm Climate | Hostile Climate |
|--------|---------------|--------------|
| Female | 0.0% | 44.32% |
| Male | 0.0% | 42.86% |

---

## 14. STICKY FLOOR

### 14.1 Time to First Intervention

| Gender | Mean (s) | Median (s) | SD |
|--------|-----------|-------------|-----|
| Female | 16.79 | 6.0 | 21.01 |
| Male | 19.33 | 10.0 | 20.76 |

### 14.2 Difference Test

| Metric | Value |
|---------|-------|
| Mann-Whitney U p-value | 0.0495 |
| Hedges' g | 0.122 |

### 14.3 First Intervention Characteristics

| Metric | Mean M | Mean F | p-value |
|---------|---------|---------|---------|
| Duration (s) | 16.88 | 17.95 | 0.515 |
| Words | 51.79 | 55.62 | 0.448 |

---

## 15. EXPLAINING PATTERN (MANSPLAINING/WOMANSPLAINING)

### 15.1 Rates by Gender

| Pattern | Rate |
|--------|------|
| Mansplaining (M→F) | 1.47% |
| Womansplaining (F→M) | 0.0% |
| Explaining Pattern (M) | 7.91% |
| Explaining Pattern (F) | 8.81% |

### 15.2 Difference Test

| Metric | Value |
|---------|-------|
| χ² | 0.880 |
| p-value | 0.348 |
| Cramér's V | 0.009 |
| Magnitude | Negligible |

---

## 16. TEMPORAL TRENDS

### 16.1 Regression by Session Quartiles

| Metric | Gender | Slope | p-value | R² |
|---------|--------|-----------|---------|-----|
| duration | M | -0.311 | 0.054 | 0.0005 |
| duration | F | -0.153 | 0.506 | 0.0001 |
| wpm | M | 4.008 | 0.001 | 0.0014 |
| wpm | F | 5.737 | <0.001 | 0.0027 |
| conflict_score | M | -0.003 | 0.735 | 0.00001 |
| conflict_score | F | 0.021 | 0.091 | 0.0006 |
| interrupts_previous | M | 0.006 | <0.001 | 0.0017 |
| interrupts_previous | F | 0.007 | 0.005 | 0.0018 |

### 16.2 Gender × Time Interaction Test

| Metric | p Interaction |
|---------|---------------|
| duration | 0.565 |
| wpm | 0.394 |
| conflict_score | 0.116 |
| interrupts_previous | 0.809 |

---

## 17. SENTIMENT AND EMOTION ANALYSIS

### 17.1 Sentiment Distribution (%)

| Gender | Negative | Neutral | Positive |
|--------|----------|--------|----------|
| Female | 10.99% | 72.14% | 16.86% |
| Male | 11.83% | 72.33% | 15.84% |

### 17.2 Emotion Distribution (%)

| Gender | Anger | Disgust | Fear | Joy | Others | Sadness | Surprise |
|--------|-------|---------|------|-----|--------|---------|----------|
| Female | 0.09% | 2.27% | 1.57% | 3.30% | 92.33% | 0.31% | 0.11% |
| Male | 0.27% | 2.28% | 1.33% | 2.76% | 92.86% | 0.33% | 0.18% |

---

## 18. PARTIAL CORRELATIONS

| Variable Pair | Global r | Global p | Male r | Female r | p Difference |
|------------------|----------|----------|-------------|------------|--------------|
| has_hedge × interrupted_by_next | 0.026 | 0.0047 | 0.028 | 0.021 | 0.680 |
| echoing_score × has_attribution | 0.079 | <0.001 | 0.076 | 0.083 | 0.705 |
| latency_s × interruption_success | -0.081 | <0.001 | -0.073 | -0.093 | 0.294 |
| lexical_diversity × interrupts_previous | 0.076 | <0.001 | 0.079 | 0.074 | 0.818 |
| duration × interruption_success | 0.015 | 0.096 | 0.015 | 0.015 | 0.993 |
| wpm × interruption_success | 0.004 | 0.653 | 0.012 | -0.009 | 0.274 |

---

## 19. POWER ANALYSIS

| Variable | Observed Hedges' g | Achieved Power | Min Detectable g | Interpretation |
|----------|---------------------|-------------------|---------------------|----------------|
| interrupts_previous | -0.036 | 0.484 | 0.053 | Poor power |
| interrupted_by_next | -0.033 | 0.415 | 0.053 | Poor power |

---

## 20. SENSITIVITY ANALYSIS (APPROPRIATION THRESHOLD)

| Echoing Threshold | Appropriation Rate M→F |
|----------------|---------------------|
| 0.1 | 16.90% |
| 0.2 | 8.94% |
| 0.3 | 5.58% |
| 0.4 | 3.98% |
| 0.5 | 2.12% |

---

## 21. EQUITY PROFILE BY SESSION

### 21.1 Most Equitable Sessions (Top 5)

| Session | % Time M | % Time F | Parity Index | N Interventions |
|--------|------------|------------|----------------|------------------|
| 46_How_promote_inclusion_disability | 51.01% | 48.99% | 0.980 | 134 |
| 34_Fluid_accumulation_critically_ill | 48.59% | 51.41% | 0.972 | 163 |
| 47_Debate_admission_organ_donation | 48.19% | 51.81% | 0.964 | 223 |
| Video_10_Young_voices | 52.18% | 47.82% | 0.956 | 75 |
| 38_parte_1 | 46.79% | 53.21% | 0.936 | 56 |

### 21.2 Most Unequal Sessions (Top 5)

| Session | % Time M | % Time F | Parity Index | N Interventions |
|--------|------------|------------|----------------|------------------|
| Video_12_Interactive | 5.32% | 94.68% | 0.106 | 177 |
| 2.15_InteractiveLecture_RespiratoryMonitoring | 3.79% | 96.21% | 0.076 | 240 |
| 45_report | 100.0% | 0.0% | 0.0 | 208 |
| Video_8_3G_Giant | 100.0% | 0.0% | 0.0 | 69 |
| 9_Families_in_ICU | 0.0% | 100.0% | 0.0 | 62 |

---

## 22. ESICM NEXT MEMBERS ANALYSIS

### 22.1 Context

The **ESICM Next** program is aimed at young researchers in intensive care medicine. A list of 2,339 Next members with names and surnames is available. We evaluated whether these members participated in the recorded sessions and, if so, whether they presented differentiated behavioral patterns.

### 22.2 Procedure

The Next Members list was cross-referenced against the 94 nominally identified speakers using:
1. **Token-subset**: all words in the speaker's name must be contained in the Next Member's name.
2. **Complementary fuzzy**: `token_set_ratio >= 85` (RapidFuzz).

### 22.3 Results

| Metric | Value |
|---------|-------|
| Total Next Members in list | 2,339 |
| Next Members identified in sessions | 11 (0.5%) |
| Of which, moderators | 9 (81.8%) |
| Of which, speakers | 2 (18.2%) |
| Of which, audience | 0 (0%) |

**Table 22. Identified Next Members**

| Person | Role | Session | Identification method |
|---------|-----|--------|-----------------------|
| David Pérez-Torres | Moderator / Public | 39 / 47 | Automated (fuzzy matching) |
| Ahmad El Ouweini | Moderator | 4 | Automated (fuzzy matching) |
| Hannah Wozniak | Moderator | 42 | Automated (fuzzy matching) |
| Andrea Ortiz | Moderator | 50 | Automated (fuzzy matching) |
| Gaetano Scaramuzzo | Moderator | Video 10 | Automated (fuzzy matching) |
| Adam Woodman-Bailey | Moderator | Video 27 | Automated (fuzzy matching) |
| Beatrice Brunoni | Moderator | Video 35 | Automated (fuzzy matching) |
| Mohamed Alebsawy | Moderator | 13 | Manual patch (self-intro: *"My name is Muhammad Al-Fsaoui from UK"*) |
| Margarita Borislavova | Speaker | 46 | Manual patch (self-intro: *"I'm Margarita…working in the ICU in France"*) |
| Stephan Katzenschlager | Speaker | Video 3 | Manual patch (self-intro: *"Stefan Kacznerschleuer from Germany"*) |
| Kevin Roedl | Moderator | 53 | Manual patch (introduced by co-moderator as *"Kevin…from Hamburg, Germany"*) |

*Note: 4 of the 11 identifications were made via manual review of transcription self-introductions, after automated fuzzy matching failed due to Whisper ASR transcription errors in the speakers' names.*

### 22.4 Behavioral Comparison

A Mann-Whitney U comparison was conducted between Next Members (n=11 persons, 195 interventions) and the remaining identified moderators (n=11 persons, 765 interventions) across four behavioral metrics:

| Metric | Next mean | Mod mean | p-value | r | sig |
|--------|----------:|----------:|--------:|----:|-----|
| Duration (s) | 13.74 | 9.81 | 0.980 | 0.001 | ns |
| Interruption rate (emitted) | 0.015 | 0.041 | 0.090 | 0.055 | ns |
| Interruption rate (received) | 0.031 | 0.044 | 0.394 | 0.028 | ns |
| Overlap rate | 0.031 | 0.044 | 0.394 | 0.028 | ns |

No statistically significant differences were found. Next Members showed a trend toward longer interventions and slightly lower interruption rates, but neither reached significance (all p > 0.05, all r < 0.06).

### 22.5 Conclusion

With n=11 identified Next Members (0.5% of the 2,339-strong list), the corpus does not provide sufficient statistical power for confirmatory analyses. The behavioral comparison with non-Next moderators yields no significant differences. Stratified analysis by Next program membership remains exploratory.

---

## 23. MAIN CORRELATIONS

### 23.1 Positive Correlations (r > 0.3)

| Variable A | Variable B | Pearson r |
|------------|------------|-----------|
| has_title | has_attribution | 0.975 |
| duration | word_count | 0.974 |
| has_overlap | interrupted_by_next | 0.973 |
| interrupts_previous | interruption_success | 0.665 |
| word_count | has_disagreement | 0.528 |
| duration | has_disagreement | 0.510 |
| overlap_duration | has_overlap | 0.502 |
| word_count | has_hedge | 0.495 |
| overlap_duration | interrupted_by_next | 0.484 |
| duration | has_hedge | 0.483 |
| has_hedge | has_disagreement | 0.327 |
| word_count | has_agreement | 0.324 |
| duration | has_agreement | 0.308 |

### 23.2 Negative Correlations (r < -0.3)

| Variable A | Variable B | Pearson r |
|------------|------------|-----------|
| lexical_diversity | has_hedge | -0.301 |
| latency_s | interrupts_previous | -0.326 |
| duration | lexical_diversity | -0.334 |
| lexical_diversity | word_count | -0.354 |

---

## 24. DATASET GENERAL STATISTICS

| Variable | N | Mean | SD | Min | Q1 | Median | Q3 | Max |
|----------|---|-------|-----|-----|-----|-----|-----|-----|
| duration | 12,138 | 12.33 | 16.27 | 0.3 | 2.28 | 6.32 | 15.75 | 167.42 |
| gender_confidence | 12,138 | 0.997 | 0.022 | 0.736 | 0.999 | 0.999 | 0.999 | 0.999 |
| overlap_duration | 12,138 | 0.27 | 2.52 | 0.0 | 0.0 | 0.0 | 0.0 | 71.7 |
| num_imperatives | 12,138 | 0.042 | 0.21 | 0.0 | 0.0 | 0.0 | 0.0 | 4.0 |
| latency_s | 12,138 | 1.19 | 3.53 | -71.7 | 0.56 | 0.78 | 1.29 | 72.21 |
| echoing_score | 12,138 | 0.068 | 0.16 | 0.0 | 0.0 | 0.0 | 0.059 | 1.0 |
| wpm | 12,138 | 211.30 | 120.22 | 16.71 | 159.44 | 192.42 | 235.66 | 5052.63 |
| lexical_diversity | 12,138 | 0.415 | 0.27 | 0.0 | 0.25 | 0.31 | 0.42 | 1.0 |
| word_count | 12,138 | 38.11 | 49.30 | 1 | 8 | 20 | 48 | 493 |

---

## 25. AMPLIFICATION MATRIX

| Previous Gender \ Current Gender | Female | Male |
|-------------------------------|----------|-----------|
| Female | 0.753 | 0.885 |
| Male | 1.062 | 0.671 |

---

## 26. CHARACTERISTICS AT POWER POSITIONS

### 26.1 First Speaker of the Session

| Metric | Mean M | Mean F | p-value |
|---------|---------|---------|---------|
| Duration (s) | 13.71 | 19.02 | 0.114 |
| Words | 40.57 | 56.21 | 0.114 |
| Conflict score | 0.277 | -0.214 | 0.013 |
| Assertiveness score | 1.94 | 1.86 | 0.620 |

### 26.2 Last Speaker of the Session

| Metric | Mean M | Mean F | p-value |
|---------|---------|---------|---------|
| Duration (s) | 11.90 | 9.34 | 0.496 |
| Words | 40.02 | 34.68 | 0.627 |
| Conflict score | -0.50 | -0.32 | 0.487 |
| Assertiveness score | 2.05 | 1.90 | 0.588 |

---

## 27. SUBGROUP ANALYSIS BY CONFLICT LEVEL

| Conflict Level | Mean Duration F (s) | Mean Duration M (s) |
|-----------------|---------------------|---------------------|
| Low | 10.90 | 10.45 |
| Medium | 12.90 | 10.86 |
| High | 15.29 | 14.49 |

---

## 28. ASSERTIVENESS BACKLASH

| Metric | Value |
|---------|-------|
| Women Slope (Backlash) | -0.0043 |
| Men Slope (Backlash) | -0.0001 |
| Interaction term | 0.0041 |
| Interaction p-value | 0.390 |

---

## 29. ABSOLUTE QUESTION-RESPONSE COUNTS

| Questioner \ Responder | Female | Male | Total |
|---------------------------|----------|-----------|-------|
| Female | 615 | 223 | 838 |
| Male | 243 | 1,191 | 1,434 |
| **Total** | **858** | **1,414** | **2,272** |

---

## 30. EXTREME CASES OF IGNORED QUESTIONS (Sample)

| Session | Question Speaker | Response Echoing |
|--------|-----------------|-------------------|
| LLM for ICU | SPEAKER_03 | 0.0 |
| LLM for ICU | SPEAKER_00 | 0.0 |
| LLM for ICU | SPEAKER_04 | 0.0 |
| LLM for ICU | SPEAKER_03 | 0.0 |
| LLM for ICU | SPEAKER_03 | 0.0 |
| LLM for ICU | SPEAKER_00 | 0.0 |
| LLM for ICU | SPEAKER_00 | 0.0 |
| LLM for ICU | SPEAKER_04 | 0.0 |
| LLM for ICU | SPEAKER_03 | 0.0 |
| LLM for ICU | SPEAKER_00 | 0.038 |

---

## 31. SIMPLE MODEL ODDS RATIOS (WITHOUT INTERVALS)

| Variable | Odds Ratio |
|----------|------------|
| word_count | 18.93 |
| duration | 13.70 |
| latency_s | 5.14 |
| gender_male | 1.27 |
| num_imperatives | 1.18 |
| conflict_score | 1.12 |
| has_hedge | 1.04 |
| assertiveness_score | 0.99 |
| lexical_diversity | 0.24 |
| wpm | 0.12 |

---

## 32. SPEAKER DISTRIBUTION AT BOUNDARIES (DETAIL)

| Gender | % First Speaker | % Last Speaker | Binomial p (First) | Binomial p (Last) |
|--------|-----------------|-----------------|----------------------|---------------------|
| Male | 62.67% | 58.67% | 0.037 | 0.165 |
| Female | 37.33% | 41.33% | — | — |

---

## 33. QUESTION RESPONSE STATISTICS (DETAIL)

| Questioner Gender | Mean Resp. Duration | Median Resp. Duration | SD Duration | Mean Words | Median Words | SD Words | Mean Echoing | SD Echoing |
|--------------------|---------------------|------------------------|-------------|----------------|------------------|-------------|---------------|------------|
| Female | 14.17 | 6.31 | 19.55 | 44.46 | 20.0 | 60.94 | 0.081 | 0.163 |
| Male | 13.30 | 6.29 | 17.99 | 41.47 | 20.0 | 54.98 | 0.082 | 0.170 |

---

## 34. ADDITIONAL PARTIAL CORRELATIONS

| Pair | Male r | Female r | p Difference |
|-----|-------------|------------|--------------|
| assertiveness_score × interrupted_by_next | 0.002 | -0.015 | 0.378 |
| conflict_score × interrupted_by_next | 0.025 | 0.026 | 0.970 |

---

## 35. MIXED LOGISTIC MODELS (STATUS)

| Variable | Status |
|----------|--------|
| has_overlap | Model failed |
| interrupts_previous | Model failed |
| interrupted_by_next | Model failed |
| interruption_success | Model failed |
| has_hedge | Model failed |
| has_disagreement | Model failed |
| has_agreement | Model failed |
| has_apology | Model failed |
| has_courtesy | Model failed |
| is_question | Model failed |

*Note: GLMM models for binary variables did not converge due to event sparsity in some categories.*

---

## 36. COMPLETE CORRELATION MATRICES

### 36.1 Global Correlations (Main Numeric Variables)

| | duration | word_count | lexical_div | wpm | latency_s | overlap_dur | echoing | imperatives |
|--|----------|------------|-------------|-----|-----------|-------------|---------|-------------|
| duration | 1.00 | 0.97 | -0.33 | -0.16 | 0.01 | 0.12 | 0.28 | 0.05 |
| word_count | 0.97 | 1.00 | -0.35 | -0.08 | 0.01 | 0.13 | 0.28 | 0.05 |
| lexical_diversity | -0.33 | -0.35 | 1.00 | 0.00 | -0.04 | -0.06 | -0.15 | 0.01 |
| wpm | -0.16 | -0.08 | 0.00 | 1.00 | -0.04 | -0.01 | -0.07 | -0.01 |
| latency_s | 0.01 | 0.01 | -0.04 | -0.04 | 1.00 | 0.00 | -0.02 | -0.02 |
| overlap_duration | 0.12 | 0.13 | -0.06 | -0.01 | 0.00 | 1.00 | 0.02 | 0.03 |
| echoing_score | 0.28 | 0.28 | -0.15 | -0.07 | -0.02 | 0.02 | 1.00 | -0.02 |
| num_imperatives | 0.05 | 0.05 | 0.01 | -0.01 | -0.02 | 0.03 | -0.02 | 1.00 |

### 36.2 Male Correlations

| | duration | word_count | lexical_div | wpm | latency_s | overlap_dur | echoing | imperatives |
|--|----------|------------|-------------|-----|-----------|-------------|---------|-------------|
| duration | 1.00 | 0.97 | -0.34 | -0.16 | 0.01 | 0.12 | 0.28 | 0.05 |
| word_count | 0.97 | 1.00 | -0.36 | -0.08 | 0.01 | 0.13 | 0.28 | 0.05 |
| lexical_diversity | -0.34 | -0.36 | 1.00 | 0.00 | -0.04 | -0.06 | -0.15 | 0.01 |
| wpm | -0.16 | -0.08 | 0.00 | 1.00 | -0.04 | -0.01 | -0.07 | -0.01 |
| latency_s | 0.01 | 0.01 | -0.04 | -0.04 | 1.00 | 0.00 | -0.02 | -0.02 |
| overlap_duration | 0.12 | 0.13 | -0.06 | -0.01 | 0.00 | 1.00 | 0.02 | 0.03 |
| echoing_score | 0.28 | 0.28 | -0.15 | -0.07 | -0.02 | 0.02 | 1.00 | -0.02 |
| num_imperatives | 0.05 | 0.05 | 0.01 | -0.01 | -0.02 | 0.03 | -0.02 | 1.00 |

### 36.3 Female Correlations

| | duration | word_count | lexical_div | wpm | latency_s | overlap_dur | echoing | imperatives |
|--|----------|------------|-------------|-----|-----------|-------------|---------|-------------|
| duration | 1.00 | 0.98 | -0.33 | -0.17 | 0.04 | 0.15 | 0.26 | 0.01 |
| word_count | 0.98 | 1.00 | -0.35 | -0.11 | 0.04 | 0.16 | 0.27 | 0.02 |
| lexical_diversity | -0.33 | -0.35 | 1.00 | 0.01 | -0.06 | -0.06 | -0.15 | 0.03 |
| wpm | -0.17 | -0.11 | 0.01 | 1.00 | -0.05 | -0.01 | -0.07 | 0.03 |
| latency_s | 0.04 | 0.04 | -0.06 | -0.05 | 1.00 | -0.02 | -0.02 | -0.00 |
| overlap_duration | 0.15 | 0.16 | -0.06 | -0.01 | -0.02 | 1.00 | 0.03 | 0.00 |
| echoing_score | 0.26 | 0.27 | -0.15 | -0.07 | -0.02 | 0.03 | 1.00 | -0.03 |
| num_imperatives | 0.01 | 0.02 | 0.03 | 0.03 | -0.00 | 0.00 | -0.03 | 1.00 |

---

## 37. APPROPRIATION EXAMPLES (QUALITATIVE SAMPLE)

| Previous Gender | Current Gender | Echoing | Attribution | Appropriation | Text (excerpt) |
|---------------|---------------|---------|------------|-------------|------------------|
| female | male | 1.0 | No | Yes | "Okay, I'm gonna start from the last question..." |
| male | male | 1.0 | No | Yes | "A high-risk patient, but the patient is doing quite okay..." |
| male | male | 1.0 | No | Yes | "The critical issue states in Pigeons we at least..." |
| male | male | 1.0 | No | Yes | "We fool the brain. This is exactly what happens..." |

---

## 38. QUESTION-RESPONSE PAIR SUMMARY

| Metric | Value |
|---------|-------|
| Total Q-A pairs | 2,272 |
| Pairs F asks → F answers | 615 |
| Pairs F asks → M answers | 223 |
| Pairs M asks → F answers | 243 |
| Pairs M asks → M answers | 1,191 |

---

## 39. COMPLETE INFERENTIAL ANALYSIS SUMMARY

| Variable | Type | Mean/Rate M | Mean/Rate F | p-value | Significant | Level |
|----------|------|--------------|--------------|---------|---------------|-------|
| duration | Numeric | 11.90 | 13.08 | 0.000165 | Yes | *** |
| word_count | Numeric | 36.85 | 40.29 | 0.000278 | Yes | *** |
| lexical_diversity | Numeric | 0.421 | 0.404 | 0.00122 | Yes | ** |
| has_disagreement | Categorical | 23.68% | 26.66% | 0.000265 | Yes | *** |
| has_overlap | Categorical | 4.16% | 4.99% | 0.0369 | Yes | * |
| overlap_duration | Numeric | 0.246 | 0.320 | 0.133 | No | ns |
| num_imperatives | Numeric | 0.044 | 0.040 | 0.293 | No | ns |
| latency_s | Numeric | 1.159 | 1.232 | 0.269 | No | ns |
| echoing_score | Numeric | 0.066 | 0.071 | 0.098 | No | ns |
| wpm | Numeric | 210.36 | 212.93 | 0.263 | No | ns |
| interrupts_previous | Categorical | 2.90% | 3.53% | 0.062 | No | ns |
| interrupted_by_next | Categorical | 3.99% | 4.65% | 0.090 | No | ns |
| is_question | Categorical | 18.65% | 18.84% | 0.812 | No | ns |
| has_hedge | Categorical | 33.89% | 35.12% | 0.176 | No | ns |
| has_apology | Categorical | 1.21% | 1.35% | 0.562 | No | ns |
| has_courtesy | Categorical | 6.80% | 6.45% | 0.481 | No | ns |
| has_agreement | Categorical | 10.26% | 10.61% | 0.562 | No | ns |
| has_title | Categorical | 5.47% | 6.09% | 0.169 | No | ns |
| has_vulnerability | Categorical | 2.00% | 2.23% | 0.445 | No | ns |
| is_backchannel | Categorical | 0.90% | 0.92% | 0.970 | No | ns |
| interruption_success | Categorical | 1.35% | 1.51% | 0.540 | No | ns |
| has_attribution | Categorical | 5.84% | 6.23% | 0.406 | No | ns |
| Explaining Pattern | Categorical | 7.91% | 8.81% | 0.086 | No | ns |
| sentiment | Categorical | Multicat | Multicat | 0.169 | No | ns |
| emotion | Categorical | Multicat | Multicat | 0.141 | No | ns |

*Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant*

---

## 40. EQUITY PROFILE - ALL SESSIONS (n=75)

| # | Session | % M | % F | Parity Index | N Interventions |
|---|--------|-----|-----|--------------|------------------|
| 1 | 46_How_promote_inclusion_disability | 51.01 | 48.99 | 0.9797 | 134 |
| 2 | 34 Fluid accumulation critically ill | 48.59 | 51.41 | 0.9718 | 163 |
| 3 | 47.Debate_admission_organ_donation | 48.19 | 51.81 | 0.9637 | 223 |
| 4 | Video 10 Young voices | 52.18 | 47.82 | 0.9563 | 75 |
| 5 | 38 parte 1 | 46.79 | 53.21 | 0.9358 | 56 |
| 6 | Grabación 2025-12-09 22.02.09 | 46.33 | 53.67 | 0.9267 | 57 |
| 7 | Grabación 2025-12-09 22.02.09 (1) | 46.33 | 53.67 | 0.9267 | 57 |
| 8 | Video 7 independent pharmacist | 53.69 | 46.31 | 0.9262 | 175 |
| 9 | 6 Immunomodulation severe infections | 54.61 | 45.39 | 0.9078 | 200 |
| 10 | 36.2 Intensive Care pharmacy skills | 55.01 | 44.99 | 0.8998 | 72 |
| 11 | Video 9 How can I decide | 44.92 | 55.08 | 0.8983 | 192 |
| 12 | Video 37 Post-ICU syndrome | 55.63 | 44.37 | 0.8874 | 183 |
| 13 | 30 | 56.78 | 43.22 | 0.8644 | 185 |
| 14 | 2.14.Defining_goal_frail_patient | 57.41 | 42.59 | 0.8517 | 232 |
| 15 | 41 | 42.57 | 57.43 | 0.8515 | 171 |
| 16 | 5 bridging planetary health | 58.47 | 41.53 | 0.8306 | 99 |
| 17 | Video 25 Joint with SMAAR | 59.49 | 40.51 | 0.8102 | 178 |
| 18 | Video 33 Dynamic medication | 59.64 | 40.36 | 0.8072 | 194 |
| 19 | 2 | 60.92 | 39.08 | 0.7815 | 46 |
| 20 | 38 parte 2 | 38.94 | 61.06 | 0.7788 | 76 |
| 21 | Video 27 Can AI help | 61.51 | 38.49 | 0.7699 | 163 |
| 22 | 22 Blood purification sepsis | 62.62 | 37.38 | 0.7476 | 118 |
| 23 | Video 23 A young man | 37.35 | 62.65 | 0.7469 | 156 |
| 24 | Video 29 How to cope | 63.06 | 36.94 | 0.7389 | 272 |
| 25 | 42 | 36.51 | 63.49 | 0.7303 | 217 |
| 26 | 49_Patient_fungi_respiratory | 64.12 | 35.88 | 0.7177 | 181 |
| 27 | 44 | 64.24 | 35.76 | 0.7151 | 216 |
| 28 | Video 15 debate | 64.38 | 35.62 | 0.7123 | 308 |
| 29 | 10 Large Language Model ICU | 64.55 | 35.45 | 0.7091 | 122 |
| 30 | 12 Perioperative anticoagulation | 64.67 | 35.33 | 0.7067 | 295 |
| 31 | Video 1 Therapeutic challenges | 65.55 | 34.45 | 0.6889 | 265 |
| 32 | 36.1 Intensive Care pharmacy | 66.33 | 33.67 | 0.6733 | 158 |
| 33 | Video 13 How to tackle | 33.53 | 66.47 | 0.6705 | 171 |
| 34 | 43 | 66.86 | 33.14 | 0.6628 | 186 |
| 35 | 28 Can we trust microbiological | 67.00 | 33.00 | 0.6600 | 119 |
| 36 | 2.16.ICM_Review_1 | 31.86 | 68.14 | 0.6372 | 79 |
| 37 | Video 31 Critical illness is bad | 69.41 | 30.59 | 0.6117 | 108 |
| 38 | 3 | 70.00 | 30.00 | 0.6000 | 48 |
| 39 | 4 | 71.02 | 28.98 | 0.5795 | 62 |
| 40 | Video 21 Should we monitorize | 71.61 | 28.39 | 0.5679 | 152 |
| 41 | Video 19 To feed | 72.80 | 27.20 | 0.5441 | 133 |
| 42 | Video 11 When intubating | 72.88 | 27.12 | 0.5424 | 275 |
| 43 | 26 When does organ inflammation | 72.96 | 27.04 | 0.5408 | 220 |
| 44 | Video 3 emergency care workers | 73.21 | 26.79 | 0.5359 | 180 |
| 45 | 8 How should ICU adapt aging | 73.66 | 26.34 | 0.5269 | 298 |
| 46 | 40 | 73.78 | 26.22 | 0.5245 | 217 |
| 47 | Video 6. Global collab | 75.73 | 24.27 | 0.4854 | 78 |
| 48 | 18 Joint SCCM ICU providers | 23.95 | 76.05 | 0.4790 | 228 |
| 49 | 39 | 77.35 | 22.65 | 0.4530 | 157 |
| 50 | 48.Balancing_PEEP_Elderly_ARDS | 77.39 | 22.61 | 0.4522 | 187 |
| 51 | 53_Should_intensivist_be_ED | 77.67 | 22.33 | 0.4466 | 114 |
| 52 | 50.Reconciliating_needs_families | 22.25 | 77.75 | 0.4449 | 163 |
| 53 | Video 5 avoiding polypharmacy | 77.80 | 22.20 | 0.4441 | 193 |
| 54 | 51. Dose_duration_RRT | 78.03 | 21.97 | 0.4394 | 114 |
| 55 | 11 Honorary Members 2025 | 79.23 | 20.77 | 0.4155 | 71 |
| 56 | 1 | 20.51 | 79.49 | 0.4102 | 85 |
| 57 | 32 Debate Post-ICU outpatient | 20.05 | 79.95 | 0.4010 | 181 |
| 58 | 7 global challenges intensive care | 80.02 | 19.98 | 0.3997 | 106 |
| 59 | 24 SBT Controversies | 80.08 | 19.92 | 0.3984 | 192 |
| 60 | 13 Interactive session armcuff | 80.20 | 19.80 | 0.3960 | 196 |
| 61 | 14 Joint JSICM hemoadsorption | 80.31 | 19.69 | 0.3938 | 207 |
| 62 | 20 consensus guideline shock | 80.56 | 19.44 | 0.3889 | 198 |
| 63 | 52.Organ_failure_assessment | 81.17 | 18.83 | 0.3766 | 184 |
| 64 | 4 Clinical conundrums BLING III | 81.32 | 18.68 | 0.3736 | 101 |
| 65 | 18 Joint SCCM ICU providers | 23.95 | 76.05 | 0.4790 | 228 |
| 66 | Video 35 Pro_con debate | 82.47 | 17.53 | 0.3506 | 234 |
| 67 | 2.15_InteractiveLecture_Respiratory | 3.79 | 96.21 | 0.0758 | 240 |
| 68 | Video 12 Interactive | 5.32 | 94.68 | 0.1064 | 177 |
| 69 | 45 | 100.0 | 0.0 | 0.0 | 208 |
| 70 | Video 8 3G Giant | 100.0 | 0.0 | 0.0 | 69 |
| 71 | 9 Families in ICU | 0.0 | 100.0 | 0.0 | 62 |
| 72 | 2.17.ICM_Review_2 | 86.02 | 13.98 | 0.2796 | 93 |
| 73 | 2.18.ICM_QA | 87.50 | 12.50 | 0.2500 | 112 |
| 74 | Video 17 Controversies | 88.01 | 11.99 | 0.2397 | 292 |
| 75 | Segunda tanda Video 12 Interactive | 5.32 | 94.68 | 0.1064 | 177 |
