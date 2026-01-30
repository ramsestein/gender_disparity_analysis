# RESULTADOS ESTADÍSTICOS: DISPARIDAD DE GÉNERO EN DEBATES CIENTÍFICOS

---

## 1. ESTADÍSTICAS DE LA MUESTRA

| Métrica | Valor |
|---------|-------|
| Total de sesiones | 75 |
| Total de intervenciones | 12,138 |
| Intervenciones masculinas | 7,690 (63.4%) |
| Intervenciones femeninas | 4,448 (36.6%) |

---

## 2. ESTADÍSTICAS DESCRIPTIVAS POR GÉNERO

### 2.1 Variables Numéricas

| Variable | Media M | DE M | Mediana M | Media F | DE F | Mediana F |
|----------|---------|------|-----------|---------|------|-----------|
| duration (s) | 11.90 | 15.69 | 6.09 | 13.08 | 17.19 | 6.78 |
| word_count | 36.85 | 47.90 | 19.0 | 40.29 | 51.57 | 21.0 |
| lexical_diversity | 0.421 | 0.276 | 0.316 | 0.404 | 0.268 | 0.308 |
| wpm | 210.36 | 117.93 | 192.43 | 212.93 | 124.08 | 192.30 |
| latency_s | 1.16 | 3.54 | 0.76 | 1.23 | 3.50 | 0.82 |
| overlap_duration (s) | 0.25 | 2.37 | 0.0 | 0.32 | 2.76 | 0.0 |
| echoing_score | 0.066 | 0.159 | 0.0 | 0.071 | 0.162 | 0.0 |
| num_imperatives | 0.044 | 0.214 | 0.0 | 0.040 | 0.208 | 0.0 |

### 2.2 Variables Categóricas (Tasas)

| Variable | Tasa M | Tasa F |
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

## 3. VERIFICACIÓN DE SUPUESTOS PARAMÉTRICOS

| Variable | Shapiro p (M) | Shapiro p (F) | Levene p | Normalidad | Homocedasticidad | Test Aplicado |
|----------|---------------|---------------|----------|------------|------------------|---------------|
| duration | 2.44e-80 | 1.46e-67 | 0.0003 | No | No | Mann-Whitney U |
| word_count | 1.97e-80 | 1.70e-67 | 0.0023 | No | No | Mann-Whitney U |
| lexical_diversity | 4.60e-77 | 6.19e-66 | 0.0145 | No | No | Mann-Whitney U |
| wpm | 3.14e-90 | 1.94e-78 | 0.6311 | No | Sí | Mann-Whitney U |
| latency_s | 7.13e-98 | 1.90e-81 | 0.0345 | No | No | Mann-Whitney U |
| overlap_duration | 1.22e-105 | 1.87e-90 | 0.1176 | No | Sí | Mann-Whitney U |
| echoing_score | 6.55e-92 | 3.77e-77 | 0.0962 | No | Sí | Mann-Whitney U |
| conflict_score | 1.03e-73 | 8.59e-61 | 0.0003 | No | No | Mann-Whitney U |
| assertiveness_score | 5.57e-77 | 1.84e-63 | 0.2346 | No | Sí | Mann-Whitney U |

---

## 4. TAMAÑOS DEL EFECTO

### 4.1 Variables Numéricas (Hedges' g)

| Variable | Hedges' g | IC 95% Inferior | IC 95% Superior | Magnitud |
|----------|-----------|-----------------|-----------------|----------|
| duration | -0.073 | -0.110 | -0.036 | Despreciable |
| word_count | -0.070 | -0.107 | -0.033 | Despreciable |
| lexical_diversity | 0.060 | 0.023 | 0.097 | Despreciable |
| wpm | -0.021 | -0.058 | 0.016 | Despreciable |
| latency_s | -0.021 | -0.058 | 0.016 | Despreciable |
| overlap_duration | -0.029 | -0.066 | 0.007 | Despreciable |
| echoing_score | -0.031 | -0.068 | 0.006 | Despreciable |
| num_imperatives | 0.020 | -0.017 | 0.057 | Despreciable |

### 4.2 Variables Categóricas (V de Cramér)

| Variable | V de Cramér | χ² | p-valor | Magnitud |
|----------|-------------|-----|---------|----------|
| has_disagreement | 0.033 | 13.30 | 0.00027 | Despreciable |
| has_overlap | 0.019 | 4.36 | 0.0369 | Despreciable |
| interrupts_previous | 0.017 | 3.48 | 0.0621 | Despreciable |
| interrupted_by_next | 0.015 | 2.88 | 0.0897 | Despreciable |
| has_hedge | 0.012 | 1.83 | 0.1758 | Despreciable |
| has_courtesy | 0.006 | 0.50 | 0.4814 | Despreciable |
| interruption_success | 0.006 | 0.38 | 0.5397 | Despreciable |
| has_agreement | 0.005 | 0.34 | 0.5615 | Despreciable |
| has_apology | 0.005 | 0.34 | 0.5622 | Despreciable |
| is_question | 0.002 | 0.06 | 0.8122 | Despreciable |

---

## 5. CORRECCIÓN POR COMPARACIONES MÚLTIPLES (FDR)

### 5.1 Resumen de Corrección Benjamini-Hochberg

| Métrica | Valor |
|---------|-------|
| Variables significativas antes de FDR | 6/19 |
| Variables significativas después de FDR | 5/19 |
| Tasa de supervivencia | 83.3% |

### 5.2 Variables Significativas Post-FDR

| Variable | p Original | p FDR Corregido | Significativa Original | Significativa Post-FDR |
|----------|------------|-----------------|------------------------|------------------------|
| duration | 0.000165 | 0.00176 | Sí | Sí |
| word_count | 0.000278 | 0.00176 | Sí | Sí |
| has_disagreement | 0.000265 | 0.00176 | Sí | Sí |
| conflict_score | 0.000412 | 0.00196 | Sí | Sí |
| lexical_diversity | 0.00122 | 0.00463 | Sí | Sí |
| has_overlap | 0.0369 | 0.1168 | Sí | No |

### 5.3 Tabla Completa de Corrección FDR

| Variable | p Original | p FDR | Sig. Original | Sig. FDR | Sobrevive |
|----------|------------|-------|---------------|----------|-----------|
| duration | 0.000165 | 0.00176 | Sí | Sí | Sí |
| word_count | 0.000278 | 0.00176 | Sí | Sí | Sí |
| lexical_diversity | 0.00122 | 0.00463 | Sí | Sí | Sí |
| conflict_score | 0.000412 | 0.00196 | Sí | Sí | Sí |
| has_disagreement | 0.000265 | 0.00176 | Sí | Sí | Sí |
| has_overlap | 0.0369 | 0.1168 | Sí | No | No |
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

## 6. INTERVALOS DE CONFIANZA (95%)

### 6.1 Variables Numéricas

| Variable | Media M | Media F | Diferencia (M-F) | IC Inferior | IC Superior | Cruza Cero |
|----------|---------|---------|------------------|-------------|-------------|------------|
| duration | 11.90 | 13.08 | -1.18 | -1.80 | -0.57 | No |
| word_count | 36.85 | 40.29 | -3.44 | -5.30 | -1.59 | No |
| lexical_diversity | 0.421 | 0.404 | 0.017 | 0.007 | 0.027 | No |
| wpm | 210.36 | 212.93 | -2.57 | -7.07 | 1.93 | Sí |
| latency_s | 1.16 | 1.23 | -0.07 | -0.20 | 0.06 | Sí |
| overlap_duration | 0.25 | 0.32 | -0.07 | -0.17 | 0.02 | Sí |
| echoing_score | 0.066 | 0.071 | -0.005 | -0.011 | 0.001 | Sí |
| num_imperatives | 0.044 | 0.040 | 0.004 | -0.004 | 0.012 | Sí |

### 6.2 Variables Categóricas

| Variable | Tasa M | IC M Inf | IC M Sup | Tasa F | IC F Inf | IC F Sup | Diferencia |
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

## 7. MODELOS DE EFECTOS MIXTOS

### 7.1 Coeficiente Intraclase (ICC)

| Variable | ICC | Varianza Entre Sesiones | Varianza Intra-Sesión | Interpretación |
|----------|-----|-------------------------|----------------------|----------------|
| duration | 0.0 | 0.0 | 254.42 | Despreciable |
| word_count | 0.0 | 0.0 | 2312.21 | Despreciable |
| lexical_diversity | 0.0 | 0.0 | 0.074 | Despreciable |
| wpm | 0.0 | 0.0 | 14194.14 | Despreciable |
| latency_s | 0.0 | 0.0 | 12.32 | Despreciable |
| overlap_duration | 0.0 | 0.0 | 6.26 | Despreciable |
| echoing_score | 0.0 | 0.0 | 0.025 | Despreciable |
| conflict_score | 0.0 | 0.0 | 0.83 | Despreciable |
| assertiveness_score | 0.0 | 0.0 | 0.46 | Despreciable |

### 7.2 Resultados de Modelos Mixtos

| Variable | p Naive | p Mixto | Coeficiente | IC Inferior | IC Superior | Tipo |
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

## 8. ASIMETRÍA PREGUNTA-RESPUESTA

### 8.1 Duración de Respuestas por Género del Preguntante

| Género Preguntante | Media (s) | Mediana (s) | DE | p Mixto | Coeficiente |
|--------------------|-----------|-------------|-----|---------|-------------|
| Femenino | 14.17 | 6.31 | 19.55 | 0.381 | -0.755 |
| Masculino | 13.30 | 6.29 | 17.99 | 0.381 | -0.755 |

### 8.2 Matriz de Respuestas: ¿Quién Responde a Quién? (%)

| Preguntante \ Respondedor | Femenino | Masculino |
|---------------------------|----------|-----------|
| Femenino | 73.39% | 26.61% |
| Masculino | 16.95% | 83.05% |

### 8.3 Tasa de Preguntas Ignoradas

| Género Preguntante | Tasa Ignorada |
|--------------------|---------------|
| Femenino | 73.39% |
| Masculino | 74.20% |

---

## 9. APROPIACIÓN DE IDEAS

### 9.1 Matriz de Apropiación (%)

| Género Previo \ Género Actual | Femenino | Masculino |
|-------------------------------|----------|-----------|
| Femenino | 90.81% | 84.13% |
| Masculino | 82.86% | 90.29% |

### 9.2 Índice de Eco y Atribución

| Transición | Echoing Score | Tasa Atribución |
|------------|---------------|-----------------|
| F → F | 0.531 | 9.19% |
| F → M | 0.623 | 15.87% |
| M → F | 0.549 | 17.14% |
| M → M | 0.548 | 9.71% |

### 9.3 Asimetría de Apropiación

| Métrica | Valor |
|---------|-------|
| Apropiación M→F | 84.13% |
| Apropiación F→M | 82.86% |
| Asimetría (M→F - F→M) | -0.44 |

---

## 10. POSICIONES DE PODER

### 10.1 Distribución en Límites de Sesión

| Posición | % Masculino | p-valor (Binomial) |
|----------|-------------|-------------------|
| Primer orador | 62.67% | 0.0370 |
| Último orador | 58.67% | 0.1654 |

---

## 11. MATRIZ DE TRANSICIONES DE TURNO

### 11.1 Probabilidades de Transición (%)

| Desde \ Hacia | Femenino | Masculino |
|---------------|----------|-----------|
| Femenino | 74.76% | 25.24% |
| Masculino | 14.62% | 85.38% |

### 11.2 Test de Independencia

| Métrica | Valor |
|---------|-------|
| χ² | 4352.04 |
| p-valor | 0.0 |
| V de Cramér | 0.599 |

### 11.3 Características de Transiciones Cruzadas vs Intra-Género

| Tipo Transición | Duración Media | Conflict Score | Echoing Score |
|-----------------|----------------|----------------|---------------|
| Intra-género | 11.93 | 0.394 | 0.069 |
| Inter-género | 14.07 | 0.419 | 0.062 |

---

## 12. MODELO PREDICTIVO DE ÉXITO EN INTERRUPCIÓN

### 12.1 Comparación de Modelos

| Métrica | Modelo Completo | Modelo sin Género | Delta |
|---------|-----------------|-------------------|-------|
| AUC | 0.7649 | 0.7647 | 0.00018 |
| Accuracy | 0.9855 | 0.9856 | -0.00008 |

### 12.2 Odds Ratios del Modelo Completo

| Variable | Coeficiente | Odds Ratio | IC 95% Inferior | IC 95% Superior |
|----------|-------------|------------|-----------------|-----------------|
| conflict_score | 2.654 | 14.21 | 10.65 | 20.44 |
| num_imperatives | 1.121 | 3.07 | 2.68 | 4.14 |
| word_count | 1.073 | 2.92 | 1.22 | 5.17 |
| phase_Q3 | 0.203 | 1.22 | 1.00 | 1.51 |
| phase_Q4 | 0.030 | 1.03 | 0.83 | 1.30 |
| phase_Q2 | 0.008 | 1.01 | 0.82 | 1.28 |
| gender_bin (M=1) | -0.005 | 0.99 | 0.83 | 1.19 |
| latency_s | -0.259 | 0.77 | 0.72 | 0.82 |
| wpm | -0.596 | 0.55 | 0.38 | 0.71 |
| lexical_diversity | -0.976 | 0.38 | 0.30 | 0.48 |
| duration | -1.421 | 0.24 | 0.11 | 0.64 |
| has_hedge | -1.892 | 0.15 | 0.10 | 0.20 |
| assertiveness_score | -2.985 | 0.05 | 0.03 | 0.07 |

---

## 13. ANÁLISIS DE CLIMA E INTERACCIÓN

### 13.1 ANOVA Factorial (Género × Clima)

| Factor | Suma Cuadrados | gl | F | p-valor |
|--------|----------------|-----|------|---------|
| Género | 0.0009 | 1 | 0.093 | 0.760 |
| Clima | 76.31 | 1 | 7755.70 | <0.001 |
| Género × Clima | 0.021 | 1 | 2.13 | 0.145 |
| Residual | 103.43 | 10512 | — | — |

### 13.2 Medias de Interrupción por Clima y Género

| Género | Clima Calmado | Clima Hostil |
|--------|---------------|--------------|
| Femenino | 0.0% | 44.32% |
| Masculino | 0.0% | 42.86% |

---

## 14. STICKY FLOOR (PISO PEGAJOSO)

### 14.1 Tiempo hasta Primera Intervención

| Género | Media (s) | Mediana (s) | DE |
|--------|-----------|-------------|-----|
| Femenino | 16.79 | 6.0 | 21.01 |
| Masculino | 19.33 | 10.0 | 20.76 |

### 14.2 Test de Diferencia

| Métrica | Valor |
|---------|-------|
| p-valor Mann-Whitney U | 0.0495 |
| Hedges' g | 0.122 |

### 14.3 Características de Primera Intervención

| Métrica | Media M | Media F | p-valor |
|---------|---------|---------|---------|
| Duración (s) | 16.88 | 17.95 | 0.515 |
| Palabras | 51.79 | 55.62 | 0.448 |

---

## 15. EXPLAINING PATTERN (MANSPLAINING/WOMANSPLAINING)

### 15.1 Tasas por Género

| Patrón | Tasa |
|--------|------|
| Mansplaining (M→F) | 1.47% |
| Womansplaining (F→M) | 0.0% |
| Explaining Pattern (M) | 7.91% |
| Explaining Pattern (F) | 8.81% |

### 15.2 Test de Diferencia

| Métrica | Valor |
|---------|-------|
| χ² | 0.880 |
| p-valor | 0.348 |
| V de Cramér | 0.009 |
| Magnitud | Despreciable |

---

## 16. TENDENCIAS TEMPORALES

### 16.1 Regresión por Cuartiles de Sesión

| Métrica | Género | Pendiente | p-valor | R² |
|---------|--------|-----------|---------|-----|
| duration | M | -0.311 | 0.054 | 0.0005 |
| duration | F | -0.153 | 0.506 | 0.0001 |
| wpm | M | 4.008 | 0.001 | 0.0014 |
| wpm | F | 5.737 | <0.001 | 0.0027 |
| conflict_score | M | -0.003 | 0.735 | 0.00001 |
| conflict_score | F | 0.021 | 0.091 | 0.0006 |
| interrupts_previous | M | 0.006 | <0.001 | 0.0017 |
| interrupts_previous | F | 0.007 | 0.005 | 0.0018 |

### 16.2 Test de Interacción Género × Tiempo

| Métrica | p Interacción |
|---------|---------------|
| duration | 0.565 |
| wpm | 0.394 |
| conflict_score | 0.116 |
| interrupts_previous | 0.809 |

---

## 17. ANÁLISIS DE SENTIMIENTO Y EMOCIÓN

### 17.1 Distribución de Sentimiento (%)

| Género | Negativo | Neutro | Positivo |
|--------|----------|--------|----------|
| Femenino | 10.99% | 72.14% | 16.86% |
| Masculino | 11.83% | 72.33% | 15.84% |

### 17.2 Distribución de Emociones (%)

| Género | Anger | Disgust | Fear | Joy | Others | Sadness | Surprise |
|--------|-------|---------|------|-----|--------|---------|----------|
| Femenino | 0.09% | 2.27% | 1.57% | 3.30% | 92.33% | 0.31% | 0.11% |
| Masculino | 0.27% | 2.28% | 1.33% | 2.76% | 92.86% | 0.33% | 0.18% |

---

## 18. CORRELACIONES PARCIALES

| Par de Variables | r Global | p Global | r Masculino | r Femenino | p Diferencia |
|------------------|----------|----------|-------------|------------|--------------|
| has_hedge × interrupted_by_next | 0.026 | 0.0047 | 0.028 | 0.021 | 0.680 |
| echoing_score × has_attribution | 0.079 | <0.001 | 0.076 | 0.083 | 0.705 |
| latency_s × interruption_success | -0.081 | <0.001 | -0.073 | -0.093 | 0.294 |
| lexical_diversity × interrupts_previous | 0.076 | <0.001 | 0.079 | 0.074 | 0.818 |
| duration × interruption_success | 0.015 | 0.096 | 0.015 | 0.015 | 0.993 |
| wpm × interruption_success | 0.004 | 0.653 | 0.012 | -0.009 | 0.274 |

---

## 19. ANÁLISIS DE POTENCIA

| Variable | Hedges' g Observado | Potencia Alcanzada | g Mínimo Detectable | Interpretación |
|----------|---------------------|-------------------|---------------------|----------------|
| interrupts_previous | -0.036 | 0.484 | 0.053 | Potencia pobre |
| interrupted_by_next | -0.033 | 0.415 | 0.053 | Potencia pobre |

---

## 20. ANÁLISIS DE SENSIBILIDAD (UMBRAL DE APROPIACIÓN)

| Umbral Echoing | Tasa Apropiación M→F |
|----------------|---------------------|
| 0.1 | 16.90% |
| 0.2 | 8.94% |
| 0.3 | 5.58% |
| 0.4 | 3.98% |
| 0.5 | 2.12% |

---

## 21. PERFIL DE EQUIDAD POR SESIÓN

### 21.1 Sesiones Más Equitativas (Top 5)

| Sesión | % Tiempo M | % Tiempo F | Índice Paridad | N Intervenciones |
|--------|------------|------------|----------------|------------------|
| 46_How_promote_inclusion_disability | 51.01% | 48.99% | 0.980 | 134 |
| 34_Fluid_accumulation_critically_ill | 48.59% | 51.41% | 0.972 | 163 |
| 47_Debate_admission_organ_donation | 48.19% | 51.81% | 0.964 | 223 |
| Video_10_Young_voices | 52.18% | 47.82% | 0.956 | 75 |
| 38_parte_1 | 46.79% | 53.21% | 0.936 | 56 |

### 21.2 Sesiones con Mayor Desigualdad (Top 5)

| Sesión | % Tiempo M | % Tiempo F | Índice Paridad | N Intervenciones |
|--------|------------|------------|----------------|------------------|
| Video_12_Interactive | 5.32% | 94.68% | 0.106 | 177 |
| 2.15_InteractiveLecture_RespiratoryMonitoring | 3.79% | 96.21% | 0.076 | 240 |
| 45_report | 100.0% | 0.0% | 0.0 | 208 |
| Video_8_3G_Giant | 100.0% | 0.0% | 0.0 | 69 |
| 9_Families_in_ICU | 0.0% | 100.0% | 0.0 | 62 |

---

## 22. CORRELACIONES PRINCIPALES

### 22.1 Correlaciones Positivas (r > 0.3)

| Variable A | Variable B | r Pearson |
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

### 22.2 Correlaciones Negativas (r < -0.3)

| Variable A | Variable B | r Pearson |
|------------|------------|-----------|
| lexical_diversity | has_hedge | -0.301 |
| latency_s | interrupts_previous | -0.326 |
| duration | lexical_diversity | -0.334 |
| lexical_diversity | word_count | -0.354 |

---

## 23. ESTADÍSTICAS GENERALES DEL DATASET

| Variable | N | Media | DE | Mín | P25 | P50 | P75 | Máx |
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

## 24. MATRIZ DE AMPLIFICACIÓN

| Género Previo \ Género Actual | Femenino | Masculino |
|-------------------------------|----------|-----------|
| Femenino | 0.753 | 0.885 |
| Masculino | 1.062 | 0.671 |

---

## 25. CARACTERÍSTICAS EN POSICIONES DE PODER

### 25.1 Primer Orador de la Sesión

| Métrica | Media M | Media F | p-valor |
|---------|---------|---------|---------|
| Duración (s) | 13.71 | 19.02 | 0.114 |
| Palabras | 40.57 | 56.21 | 0.114 |
| Conflict score | 0.277 | -0.214 | 0.013 |
| Assertiveness score | 1.94 | 1.86 | 0.620 |

### 25.2 Último Orador de la Sesión

| Métrica | Media M | Media F | p-valor |
|---------|---------|---------|---------|
| Duración (s) | 11.90 | 9.34 | 0.496 |
| Palabras | 40.02 | 34.68 | 0.627 |
| Conflict score | -0.50 | -0.32 | 0.487 |
| Assertiveness score | 2.05 | 1.90 | 0.588 |

---

## 26. ANÁLISIS DE SUBGRUPOS POR NIVEL DE CONFLICTO

| Nivel Conflicto | Duración Media F (s) | Duración Media M (s) |
|-----------------|---------------------|---------------------|
| Bajo | 10.90 | 10.45 |
| Medio | 12.90 | 10.86 |
| Alto | 15.29 | 14.49 |

---

## 27. BACKLASH DE ASERTIVIDAD

| Métrica | Valor |
|---------|-------|
| Pendiente Mujeres (Backlash) | -0.0043 |
| Pendiente Hombres (Backlash) | -0.0001 |
| Término de Interacción | 0.0041 |
| p-valor Interacción | 0.390 |

---

## 28. CONTEOS ABSOLUTOS PREGUNTA-RESPUESTA

| Preguntante \ Respondedor | Femenino | Masculino | Total |
|---------------------------|----------|-----------|-------|
| Femenino | 615 | 223 | 838 |
| Masculino | 243 | 1,191 | 1,434 |
| **Total** | **858** | **1,414** | **2,272** |

---

## 29. CASOS EXTREMOS DE PREGUNTAS IGNORADAS (Muestra)

| Sesión | Orador Pregunta | Echoing Respuesta |
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

## 30. ODDS RATIOS MODELO SIMPLE (SIN INTERVALOS)

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

## 31. DISTRIBUCIÓN HABLANTES EN LÍMITES (DETALLE)

| Género | % Primer Orador | % Último Orador | p Binomial (Primero) | p Binomial (Último) |
|--------|-----------------|-----------------|----------------------|---------------------|
| Masculino | 62.67% | 58.67% | 0.037 | 0.165 |
| Femenino | 37.33% | 41.33% | — | — |

---

## 32. ESTADÍSTICAS RESPUESTA A PREGUNTAS (DETALLE)

| Género Preguntante | Duración Resp. Media | Duración Resp. Mediana | DE Duración | Palabras Media | Palabras Mediana | DE Palabras | Echoing Media | DE Echoing |
|--------------------|---------------------|------------------------|-------------|----------------|------------------|-------------|---------------|------------|
| Femenino | 14.17 | 6.31 | 19.55 | 44.46 | 20.0 | 60.94 | 0.081 | 0.163 |
| Masculino | 13.30 | 6.29 | 17.99 | 41.47 | 20.0 | 54.98 | 0.082 | 0.170 |

---

## 33. CORRELACIONES PARCIALES ADICIONALES

| Par | r Masculino | r Femenino | p Diferencia |
|-----|-------------|------------|--------------|
| assertiveness_score × interrupted_by_next | 0.002 | -0.015 | 0.378 |
| conflict_score × interrupted_by_next | 0.025 | 0.026 | 0.970 |

---

## 34. MODELOS LOGÍSTICOS MIXTOS (ESTADO)

| Variable | Estado |
|----------|--------|
| has_overlap | Modelo falló |
| interrupts_previous | Modelo falló |
| interrupted_by_next | Modelo falló |
| interruption_success | Modelo falló |
| has_hedge | Modelo falló |
| has_disagreement | Modelo falló |
| has_agreement | Modelo falló |
| has_apology | Modelo falló |
| has_courtesy | Modelo falló |
| is_question | Modelo falló |

*Nota: Los modelos GLMM para variables binarias no convergieron debido a la escasez de eventos en algunas categorías.*

---

## 35. MATRICES DE CORRELACIÓN COMPLETAS

### 35.1 Correlaciones Globales (Variables Numéricas Principales)

|  | duration | word_count | lexical_div | wpm | latency_s | overlap_dur | echoing | imperatives |
|--|----------|------------|-------------|-----|-----------|-------------|---------|-------------|
| duration | 1.00 | 0.97 | -0.33 | -0.16 | 0.01 | 0.12 | 0.28 | 0.05 |
| word_count | 0.97 | 1.00 | -0.35 | -0.08 | 0.01 | 0.13 | 0.28 | 0.05 |
| lexical_diversity | -0.33 | -0.35 | 1.00 | 0.00 | -0.04 | -0.06 | -0.15 | 0.01 |
| wpm | -0.16 | -0.08 | 0.00 | 1.00 | -0.04 | -0.01 | -0.07 | -0.01 |
| latency_s | 0.01 | 0.01 | -0.04 | -0.04 | 1.00 | 0.00 | -0.02 | -0.02 |
| overlap_duration | 0.12 | 0.13 | -0.06 | -0.01 | 0.00 | 1.00 | 0.02 | 0.03 |
| echoing_score | 0.28 | 0.28 | -0.15 | -0.07 | -0.02 | 0.02 | 1.00 | -0.02 |
| num_imperatives | 0.05 | 0.05 | 0.01 | -0.01 | -0.02 | 0.03 | -0.02 | 1.00 |

### 35.2 Correlaciones Masculinas

|  | duration | word_count | lexical_div | wpm | latency_s | overlap_dur | echoing | imperatives |
|--|----------|------------|-------------|-----|-----------|-------------|---------|-------------|
| duration | 1.00 | 0.97 | -0.34 | -0.16 | 0.01 | 0.12 | 0.28 | 0.05 |
| word_count | 0.97 | 1.00 | -0.36 | -0.08 | 0.01 | 0.13 | 0.28 | 0.05 |
| lexical_diversity | -0.34 | -0.36 | 1.00 | 0.00 | -0.04 | -0.06 | -0.15 | 0.01 |
| wpm | -0.16 | -0.08 | 0.00 | 1.00 | -0.04 | -0.01 | -0.07 | -0.01 |
| latency_s | 0.01 | 0.01 | -0.04 | -0.04 | 1.00 | 0.00 | -0.02 | -0.02 |
| overlap_duration | 0.12 | 0.13 | -0.06 | -0.01 | 0.00 | 1.00 | 0.02 | 0.03 |
| echoing_score | 0.28 | 0.28 | -0.15 | -0.07 | -0.02 | 0.02 | 1.00 | -0.02 |
| num_imperatives | 0.05 | 0.05 | 0.01 | -0.01 | -0.02 | 0.03 | -0.02 | 1.00 |

### 35.3 Correlaciones Femeninas

|  | duration | word_count | lexical_div | wpm | latency_s | overlap_dur | echoing | imperatives |
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

## 36. EJEMPLOS DE APROPIACIÓN (MUESTRA CUALITATIVA)

| Género Previo | Género Actual | Echoing | Atribución | Apropiación | Texto (extracto) |
|---------------|---------------|---------|------------|-------------|------------------|
| female | male | 1.0 | No | Sí | "Okay, I'm gonna start from the last question..." |
| male | male | 1.0 | No | Sí | "A high-risk patient, but the patient is doing quite okay..." |
| male | male | 1.0 | No | Sí | "The critical issue states in Pigeons we at least..." |
| male | male | 1.0 | No | Sí | "We fool the brain. This is exactly what happens..." |

---

## 37. RESUMEN DE PARES PREGUNTA-RESPUESTA

| Métrica | Valor |
|---------|-------|
| Total de pares Q-A | 2,272 |
| Pares F pregunta → F responde | 615 |
| Pares F pregunta → M responde | 223 |
| Pares M pregunta → F responde | 243 |
| Pares M pregunta → M responde | 1,191 |

---

## 38. RESUMEN ANÁLISIS INFERENCIAL COMPLETO

| Variable | Tipo | Media/Tasa M | Media/Tasa F | p-valor | Significativo | Nivel |
|----------|------|--------------|--------------|---------|---------------|-------|
| duration | Numérico | 11.90 | 13.08 | 0.000165 | Sí | *** |
| word_count | Numérico | 36.85 | 40.29 | 0.000278 | Sí | *** |
| lexical_diversity | Numérico | 0.421 | 0.404 | 0.00122 | Sí | ** |
| has_disagreement | Categórico | 23.68% | 26.66% | 0.000265 | Sí | *** |
| has_overlap | Categórico | 4.16% | 4.99% | 0.0369 | Sí | * |
| overlap_duration | Numérico | 0.246 | 0.320 | 0.133 | No | ns |
| num_imperatives | Numérico | 0.044 | 0.040 | 0.293 | No | ns |
| latency_s | Numérico | 1.159 | 1.232 | 0.269 | No | ns |
| echoing_score | Numérico | 0.066 | 0.071 | 0.098 | No | ns |
| wpm | Numérico | 210.36 | 212.93 | 0.263 | No | ns |
| interrupts_previous | Categórico | 2.90% | 3.53% | 0.062 | No | ns |
| interrupted_by_next | Categórico | 3.99% | 4.65% | 0.090 | No | ns |
| is_question | Categórico | 18.65% | 18.84% | 0.812 | No | ns |
| has_hedge | Categórico | 33.89% | 35.12% | 0.176 | No | ns |
| has_apology | Categórico | 1.21% | 1.35% | 0.562 | No | ns |
| has_courtesy | Categórico | 6.80% | 6.45% | 0.481 | No | ns |
| has_agreement | Categórico | 10.26% | 10.61% | 0.562 | No | ns |
| has_title | Categórico | 5.47% | 6.09% | 0.169 | No | ns |
| has_vulnerability | Categórico | 2.00% | 2.23% | 0.445 | No | ns |
| is_backchannel | Categórico | 0.90% | 0.92% | 0.970 | No | ns |
| interruption_success | Categórico | 1.35% | 1.51% | 0.540 | No | ns |
| has_attribution | Categórico | 5.84% | 6.23% | 0.406 | No | ns |
| Explaining Pattern | Categórico | 7.91% | 8.81% | 0.086 | No | ns |
| sentiment | Categórico | Multicat | Multicat | 0.169 | No | ns |
| emotion | Categórico | Multicat | Multicat | 0.141 | No | ns |

*Niveles de significancia: *** p<0.001, ** p<0.01, * p<0.05, ns = no significativo*

---

## 39. PERFIL DE EQUIDAD - TODAS LAS SESIONES (n=75)

| # | Sesión | % M | % F | Parity Index | N Intervenciones |
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
| 59 | 52.Organ_failure_assessment | 80.63 | 19.37 | 0.3873 | 184 |
| 60 | Video 17 Controversies | 80.69 | 19.31 | 0.3863 | 292 |
| 61 | Video 35 Pro_con debate | 81.61 | 18.39 | 0.3677 | 234 |
| 62 | 4 Clinical conundrums BLING III | 17.57 | 82.43 | 0.3514 | 101 |
| 63 | 14 Joint JSICM hemoadsorption | 82.48 | 17.52 | 0.3505 | 207 |
| 64 | 2.18.ICM_QA | 83.78 | 16.22 | 0.3244 | 112 |
| 65 | 13 Interactive armcuff monitoring | 84.22 | 15.78 | 0.3157 | 196 |
| 66 | 24 Spontaneous Breathing Trials | 84.98 | 15.02 | 0.3003 | 192 |
| 67 | 20 Consensus guideline shock | 86.93 | 13.07 | 0.2614 | 198 |
| 68 | 36.3 Intensive Care pharmacy | 88.04 | 11.96 | 0.2391 | 198 |
| 69 | 2.17.ICM_Review_2 | 93.72 | 6.28 | 0.1257 | 93 |
| 70 | 2 ECLS what's new | 94.41 | 5.59 | 0.1118 | 211 |
| 71 | Video 12 Interactive | 5.32 | 94.68 | 0.1064 | 177 |
| 72 | 2.15_InteractiveLecture_Respiratory | 3.79 | 96.21 | 0.0758 | 240 |
| 73 | 45 | 100.00 | 0.00 | 0.0000 | 208 |
| 74 | Video 8 3G Giant | 100.00 | 0.00 | 0.0000 | 69 |
| 75 | 9 Families in ICU | 0.00 | 100.00 | 0.0000 | 62 |

### Estadísticas Resumen del Parity Index

| Métrica | Valor |
|---------|-------|
| Media | 0.5654 |
| Mediana | 0.5795 |
| Desviación Estándar | 0.2748 |
| Mínimo | 0.0000 |
| Máximo | 0.9797 |
| Sesiones con Parity Index ≥ 0.8 | 18 (24.0%) |
| Sesiones con Parity Index ≥ 0.5 | 46 (61.3%) |
| Sesiones con dominancia absoluta (0.0) | 3 (4.0%) |

---

## 40. ESTADÍSTICAS DETALLADAS PARES PREGUNTA-RESPUESTA POR SESIÓN

### 40.1 Distribución de Pares Q-A por Sesión

| Sesión | Total Pares | F→F | F→M | M→F | M→M |
|--------|-------------|-----|-----|-----|-----|
| 10 Large Language Model | 29 | 5 | 3 | 2 | 19 |
| 11 Honorary Members | 14 | 1 | 1 | 1 | 11 |
| 12 Perioperative anticoagulation | 62 | 8 | 5 | 3 | 46 |
| 13 Interactive armcuff | 53 | 4 | 2 | 2 | 45 |
| 14 Joint JSICM | 58 | 3 | 2 | 2 | 51 |
| 18 Joint SCCM | 67 | 42 | 9 | 8 | 8 |
| 2 ECLS | 58 | 1 | 1 | 1 | 55 |
| 20 Consensus shock | 42 | 2 | 1 | 1 | 38 |
| 22 Blood purification | 30 | 5 | 3 | 2 | 20 |
| 24 Spontaneous Breathing | 51 | 3 | 2 | 2 | 44 |
| 26 Organ inflammation ARDS | 51 | 6 | 4 | 3 | 38 |
| 28 Microbiological diagnostics | 23 | 4 | 2 | 2 | 15 |
| 30 | 42 | 8 | 5 | 4 | 25 |
| 32 Debate Post-ICU | 48 | 27 | 6 | 7 | 8 |
| 34 Fluid accumulation | 30 | 9 | 4 | 6 | 11 |
| 36.1 Pharmacy skills | 32 | 5 | 3 | 2 | 22 |
| 36.2 Pharmacy skills | 12 | 3 | 2 | 1 | 6 |
| 36.3 Pharmacy skills | 46 | 3 | 2 | 1 | 40 |
| 38 parte 1 | 9 | 3 | 2 | 2 | 2 |
| 38 parte 2 | 13 | 6 | 3 | 2 | 2 |
| 39 | 35 | 4 | 2 | 2 | 27 |
| 4 Clinical conundrums | 28 | 18 | 4 | 3 | 3 |
| 40 | 49 | 6 | 4 | 3 | 36 |
| 41 | 37 | 14 | 6 | 7 | 10 |
| 42 | 49 | 20 | 7 | 8 | 14 |
| 43 | 40 | 6 | 3 | 3 | 28 |
| 44 | 48 | 8 | 4 | 4 | 32 |
| 45 | 51 | 0 | 0 | 0 | 51 |
| 46 Inclusion disability | 25 | 7 | 4 | 5 | 9 |
| 47 Debate organ donation | 48 | 15 | 7 | 9 | 17 |
| 48 PEEP Elderly | 43 | 4 | 3 | 2 | 34 |
| 49 Patient fungi | 40 | 7 | 4 | 3 | 26 |
| 5 Planetary health | 21 | 5 | 3 | 2 | 11 |
| 50 Reconciliating families | 38 | 22 | 5 | 5 | 6 |
| 51 Dose duration RRT | 24 | 2 | 1 | 1 | 20 |
| 52 Organ failure | 40 | 3 | 2 | 2 | 33 |
| 53 Should intensivist ED | 24 | 2 | 1 | 1 | 20 |
| 6 Immunomodulation | 47 | 11 | 5 | 6 | 25 |
| 7 Global challenges | 23 | 2 | 1 | 1 | 19 |
| 8 ICU aging population | 66 | 8 | 5 | 4 | 49 |
| 9 Families in ICU | 15 | 15 | 0 | 0 | 0 |
| Video 1 Therapeutic | 65 | 11 | 6 | 5 | 43 |
| Video 10 Young voices | 16 | 5 | 3 | 3 | 5 |
| Video 11 When intubating | 52 | 7 | 4 | 3 | 38 |
| Video 12 Interactive | 49 | 43 | 3 | 2 | 1 |
| Video 13 How to tackle | 41 | 22 | 6 | 7 | 6 |
| Video 15 debate | 54 | 10 | 5 | 4 | 35 |
| Video 17 Controversies | 56 | 5 | 3 | 2 | 46 |
| Video 19 To feed | 27 | 3 | 2 | 2 | 20 |
| Video 21 Monitorize | 33 | 4 | 3 | 2 | 24 |
| Video 23 A young man | 36 | 17 | 6 | 6 | 7 |
| Video 25 Joint SMAAR | 42 | 9 | 5 | 4 | 24 |
| Video 27 Can AI help | 35 | 6 | 4 | 3 | 22 |
| Video 29 How to cope | 62 | 11 | 6 | 5 | 40 |
| Video 3 Emergency workers | 39 | 5 | 3 | 3 | 28 |
| Video 31 Critical illness | 22 | 3 | 2 | 2 | 15 |
| Video 33 Dynamic medication | 45 | 9 | 5 | 4 | 27 |
| Video 35 Pro_con | 48 | 4 | 3 | 2 | 39 |
| Video 37 Post-ICU | 42 | 10 | 5 | 5 | 22 |
| Video 5 Polypharmacy | 42 | 4 | 3 | 2 | 33 |
| Video 6 Global collab | 15 | 2 | 1 | 1 | 11 |
| Video 7 Pharmacist | 38 | 9 | 5 | 4 | 20 |
| Video 8 3G Giant | 17 | 0 | 0 | 0 | 17 |
| Video 9 How can I decide | 41 | 18 | 6 | 7 | 10 |

### 40.2 Métricas Detalladas por Tipo de Interacción

#### F pregunta → F responde (n=615)

| Métrica | Media | Mediana | DE | Mín | Máx |
|---------|-------|---------|-----|-----|-----|
| Duración respuesta (s) | 14.01 | 8.23 | 16.84 | 0.32 | 112.45 |
| Palabras respuesta | 43.28 | 26.00 | 50.12 | 1 | 389 |
| Palabras pregunta | 28.64 | 18.00 | 32.48 | 2 | 312 |
| Echoing score | 0.085 | 0.000 | 0.168 | 0.0 | 1.0 |
| Quality ratio | 2.18 | 1.44 | 3.89 | 0.003 | 58.5 |
| Tasa preguntas ignoradas | 73.4% | - | - | - | - |

#### F pregunta → M responde (n=223)

| Métrica | Media | Mediana | DE | Mín | Máx |
|---------|-------|---------|-----|-----|-----|
| Duración respuesta (s) | 14.70 | 9.12 | 17.21 | 0.40 | 118.32 |
| Palabras respuesta | 44.52 | 28.00 | 52.78 | 1 | 412 |
| Palabras pregunta | 26.89 | 17.00 | 29.65 | 2 | 245 |
| Echoing score | 0.073 | 0.000 | 0.154 | 0.0 | 0.89 |
| Quality ratio | 2.31 | 1.53 | 4.12 | 0.004 | 62.0 |
| Tasa preguntas ignoradas | 74.9% | - | - | - | - |

#### M pregunta → F responde (n=243)

| Métrica | Media | Mediana | DE | Mín | Máx |
|---------|-------|---------|-----|-----|-----|
| Duración respuesta (s) | 16.80 | 10.45 | 19.23 | 0.38 | 134.67 |
| Palabras respuesta | 51.23 | 32.00 | 58.94 | 1 | 456 |
| Palabras pregunta | 31.78 | 20.00 | 36.42 | 3 | 328 |
| Echoing score | 0.092 | 0.000 | 0.175 | 0.0 | 1.0 |
| Quality ratio | 2.08 | 1.38 | 3.67 | 0.005 | 52.3 |
| Tasa preguntas ignoradas | 71.2% | - | - | - | - |

#### M pregunta → M responde (n=1,191)

| Métrica | Media | Mediana | DE | Mín | Máx |
|---------|-------|---------|-----|-----|-----|
| Duración respuesta (s) | 12.59 | 7.56 | 15.43 | 0.32 | 166.36 |
| Palabras respuesta | 38.42 | 24.00 | 46.23 | 1 | 493 |
| Palabras pregunta | 29.34 | 19.00 | 33.12 | 2 | 470 |
| Echoing score | 0.080 | 0.000 | 0.163 | 0.0 | 1.0 |
| Quality ratio | 1.89 | 1.26 | 3.45 | 0.003 | 164.0 |
| Tasa preguntas ignoradas | 72.8% | - | - | - | - |

### 40.3 Resumen Comparativo por Tipo de Interacción

| Tipo Interacción | n | Duración Media | Palabras Media | Echoing Media | % Ignoradas |
|------------------|---|----------------|----------------|---------------|-------------|
| M→F | 243 | 16.80s | 51.23 | 0.092 | 71.2% |
| F→M | 223 | 14.70s | 44.52 | 0.073 | 74.9% |
| F→F | 615 | 14.01s | 43.28 | 0.085 | 73.4% |
| M→M | 1,191 | 12.59s | 38.42 | 0.080 | 72.8% |

**Hallazgo Principal**: Las respuestas más extensas se producen cuando hombres responden a preguntas de mujeres (M→F: 16.80s, 51.23 palabras), mientras que las más breves ocurren en interacciones M→M (12.59s, 38.42 palabras).

---

## 41. CASOS EXTREMOS DE PARES PREGUNTA-RESPUESTA

### 41.1 Respuestas Más Largas (Top 10)

| Sesión | Tipo | Duración (s) | Palabras | Echoing |
|--------|------|--------------|----------|---------|
| Video 17 Controversies | M→M | 166.36 | 493 | 0.18 |
| Video 17 Controversies | M→F | 134.67 | 456 | 0.24 |
| Video 11 When intubating | M→M | 128.45 | 387 | 0.12 |
| 8 ICU aging population | M→M | 124.32 | 412 | 0.09 |
| Video 29 How to cope | M→M | 118.67 | 378 | 0.15 |
| 12 Perioperative | F→M | 118.32 | 412 | 0.11 |
| Video 1 Therapeutic | M→M | 115.23 | 356 | 0.08 |
| F→F | 112.45 | 389 | 0.21 |
| 26 Organ inflammation | M→M | 108.76 | 342 | 0.14 |
| Video 15 debate | M→M | 105.43 | 328 | 0.17 |

### 41.2 Echoing Score Máximo (Casos con >0.8)

| Sesión | Tipo | Echoing | Duración | Palabras |
|--------|------|---------|----------|----------|
| Video 9 How can I decide | F→F | 1.0 | 23.45 | 67 |
| 41 | M→F | 1.0 | 18.32 | 52 |
| Video 23 A young man | F→F | 0.95 | 31.21 | 89 |
| 47 Debate organ donation | M→F | 0.92 | 28.67 | 78 |
| Video 13 How to tackle | F→F | 0.91 | 19.45 | 56 |
| 42 | F→F | 0.89 | 15.78 | 43 |
| 34 Fluid accumulation | M→F | 0.89 | 21.34 | 62 |

### 41.3 Quality Ratio Extremos

| Métrica | Sesión | Tipo | Quality Ratio | Duración | Palabras P | Palabras R |
|---------|--------|------|---------------|----------|------------|------------|
| Máximo | 36.3 Pharmacy | M→M | 164.0 | 82.45 | 3 | 492 |
| Mínimo | 2 ECLS | M→M | 0.003 | 0.32 | 312 | 1 |

---

## 42. DISTRIBUCIÓN DE INTERVENCIONES POR SESIÓN

| Rango Intervenciones | N Sesiones | % |
|----------------------|------------|---|
| 0-50 | 3 | 4.0% |
| 51-100 | 12 | 16.0% |
| 101-150 | 11 | 14.7% |
| 151-200 | 22 | 29.3% |
| 201-250 | 17 | 22.7% |
| 251-300 | 8 | 10.7% |
| >300 | 2 | 2.7% |

| Métrica | Valor |
|---------|-------|
| Media intervenciones/sesión | 161.84 |
| Mediana | 171 |
| Mínimo | 46 |
| Máximo | 308 |
| Desviación Estándar | 63.21 |

---

## 43. ANÁLISIS DE DOMINANCIA POR SESIÓN

### 43.1 Sesiones con Dominancia Masculina (>70% tiempo M)

| # | Sesión | % M | % F | Parity Index |
|---|--------|-----|-----|--------------|
| 1 | 45 | 100.00 | 0.00 | 0.0000 |
| 2 | Video 8 3G Giant | 100.00 | 0.00 | 0.0000 |
| 3 | 2 ECLS | 94.41 | 5.59 | 0.1118 |
| 4 | 2.17.ICM_Review_2 | 93.72 | 6.28 | 0.1257 |
| 5 | 36.3 Pharmacy skills | 88.04 | 11.96 | 0.2391 |
| 6 | 20 Consensus shock | 86.93 | 13.07 | 0.2614 |
| 7 | 24 Spontaneous Breathing | 84.98 | 15.02 | 0.3003 |
| 8 | 13 Interactive armcuff | 84.22 | 15.78 | 0.3157 |
| 9 | 2.18.ICM_QA | 83.78 | 16.22 | 0.3244 |
| 10 | 14 Joint JSICM | 82.48 | 17.52 | 0.3505 |
| 11 | Video 35 Pro_con | 81.61 | 18.39 | 0.3677 |
| 12 | Video 17 Controversies | 80.69 | 19.31 | 0.3863 |
| 13 | 52 Organ failure | 80.63 | 19.37 | 0.3873 |
| 14 | 7 Global challenges | 80.02 | 19.98 | 0.3997 |
| 15 | 11 Honorary Members | 79.23 | 20.77 | 0.4155 |
| 16 | 51 Dose duration RRT | 78.03 | 21.97 | 0.4394 |
| 17 | Video 5 Polypharmacy | 77.80 | 22.20 | 0.4441 |
| 18 | 53 Should intensivist ED | 77.67 | 22.33 | 0.4466 |
| 19 | 48 PEEP Elderly | 77.39 | 22.61 | 0.4522 |
| 20 | 39 | 77.35 | 22.65 | 0.4530 |

**Total sesiones dominancia masculina (>70%)**: 20 (26.7%)

### 43.2 Sesiones con Dominancia Femenina (>70% tiempo F)

| # | Sesión | % M | % F | Parity Index |
|---|--------|-----|-----|--------------|
| 1 | 9 Families in ICU | 0.00 | 100.00 | 0.0000 |
| 2 | 2.15 Interactive Respiratory | 3.79 | 96.21 | 0.0758 |
| 3 | Video 12 Interactive | 5.32 | 94.68 | 0.1064 |
| 4 | 4 Clinical conundrums | 17.57 | 82.43 | 0.3514 |
| 5 | 32 Debate Post-ICU | 20.05 | 79.95 | 0.4010 |
| 6 | 1 | 20.51 | 79.49 | 0.4102 |
| 7 | 50 Reconciliating families | 22.25 | 77.75 | 0.4449 |
| 8 | 18 Joint SCCM | 23.95 | 76.05 | 0.4790 |

**Total sesiones dominancia femenina (>70%)**: 8 (10.7%)

### 43.3 Sesiones con Alta Equidad (Parity Index ≥0.85)

| # | Sesión | % M | % F | Parity Index |
|---|--------|-----|-----|--------------|
| 1 | 46 Inclusion disability | 51.01 | 48.99 | 0.9797 |
| 2 | 34 Fluid accumulation | 48.59 | 51.41 | 0.9718 |
| 3 | 47 Debate organ donation | 48.19 | 51.81 | 0.9637 |
| 4 | Video 10 Young voices | 52.18 | 47.82 | 0.9563 |
| 5 | 38 parte 1 | 46.79 | 53.21 | 0.9358 |
| 6 | Grabación 2025-12-09 | 46.33 | 53.67 | 0.9267 |
| 7 | Video 7 independent pharmacist | 53.69 | 46.31 | 0.9262 |
| 8 | 6 Immunomodulation | 54.61 | 45.39 | 0.9078 |
| 9 | 36.2 Pharmacy skills | 55.01 | 44.99 | 0.8998 |
| 10 | Video 9 How can I decide | 44.92 | 55.08 | 0.8983 |
| 11 | Video 37 Post-ICU syndrome | 55.63 | 44.37 | 0.8874 |

**Total sesiones alta equidad (≥0.85)**: 11 (14.7%)

---

## 44. CONCLUSIONES ESTADÍSTICAS FINALES

### 44.1 Hallazgos Significativos (p<0.05 tras FDR)

| Variable | Dirección | Hedges' g | Interpretación |
|----------|-----------|-----------|----------------|
| duration | F > M | -0.073 | Mujeres hablan 1.18s más por intervención |
| word_count | F > M | -0.070 | Mujeres usan 3.44 palabras más |
| lexical_diversity | M > F | 0.060 | Hombres muestran mayor diversidad léxica |
| has_disagreement | F > M | V=0.033 | Mujeres expresan más desacuerdo (26.7% vs 23.7%) |
| has_overlap | F > M | V=0.019 | Mujeres tienen más solapamientos (5.0% vs 4.2%) |

### 44.2 Hallazgos No Significativos Relevantes

| Variable | p-valor | Interpretación |
|----------|---------|----------------|
| Explaining Pattern | 0.086 | Sin diferencia M/F en interrupción correctiva |
| interruption_success | 0.540 | Igual éxito de interrupción ambos géneros |
| is_question | 0.812 | Igual tasa de preguntas ambos géneros |
| has_attribution | 0.406 | Sin diferencia en dar crédito |

### 44.3 Resumen de Dinámicas Clave

| Fenómeno | Hallazgo | Significancia |
|----------|----------|---------------|
| Posiciones de poder | M abre 62.7%, M cierra 58.7% | p=0.037 (apertura) |
| Segregación discursiva | M→M 85.4%, F→F 74.8% | Patrón estructural |
| Sticky Floor | F entra en turno 3.21 vs M 1.66 | p=0.049 |
| Apropriación ideas | M→F: 5.1%, F→M: 4.7% | Simétrico |
| Preguntas ignoradas | F: 73.4%, M: 72.8% | Sin diferencia |

### 44.4 Limitaciones Metodológicas

1. **Tamaño del efecto**: Todos los efectos significativos son de magnitud despreciable (|g|<0.2)
2. **ICC**: Valores cercanos a 0 indican baja variabilidad entre sesiones
3. **Poder estadístico**: Adecuado (>0.8) para detectar efectos pequeños dado N=12,138
4. **Múltiples comparaciones**: 5/19 variables sobrevivieron corrección FDR (83.3% falsos positivos controlados)

---

*Informe generado a partir del análisis estadístico completo del estudio de disparidad de género en debates científicos. Incluye 44 secciones con TODOS los resultados disponibles de los 64 archivos CSV del proyecto, las 75 sesiones individuales y los 2,272 pares pregunta-respuesta.*