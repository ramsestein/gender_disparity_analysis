# Dinámicas de Participación Verbal por Género en Debates Científicos de Medicina Intensiva: Un Análisis Computacional Automatizado

---

## METODOLOGÍA

### Diseño del estudio

Se realizó un estudio observacional retrospectivo de análisis conversacional automatizado sobre grabaciones de sesiones científicas en formato debate, procedentes del ámbito de la medicina de cuidados críticos. La unidad primaria de análisis fue la **intervención**, definida como un segmento continuo de habla de un único orador, delimitada por silencio superior a 0,5 segundos o por cambio de orador identificado mediante diarización automática.

### Muestra

Se analizaron 75 sesiones de debate científico, de las cuales se extrajeron un total de 12.138 intervenciones. Del total de intervenciones, 7.690 (63,4%) correspondieron a oradores masculinos y 4.448 (36,6%) a oradoras femeninas.

### Pipeline de procesamiento de audio

#### Ingesta y normalización

El flujo de audio se extrajo de los archivos de vídeo mediante herramientas estándar de procesamiento multimedia. Para garantizar la robustez de los modelos posteriores, se aplicó una estandarización rigurosa consistente en: conversión a formato WAV (PCM 16-bit), frecuencia de muestreo de 16 kHz y configuración monoaural. Se implementó el estándar EBU R128 para normalización de sonoridad, ajustando la sonoridad integrada a -23 LUFS, lo que mitiga las variaciones acústicas derivadas de diferentes distancias al micrófono, características vocales individuales y condiciones heterogéneas de grabación.

#### Diarización de oradores

Para la segmentación y asignación de identidad a cada orador se utilizó Pyannote.audio 3.1, un sistema basado en modelos de segmentación y embeddings de voz. El sistema implementa cuatro componentes: (1) detección de actividad vocal (VAD) para identificar regiones con habla versus silencio; (2) extracción de embeddings mediante redes neuronales con arquitectura ECAPA-TDNN, generando vectores de identidad de 192 dimensiones; (3) clustering jerárquico aglomerativo sobre los vectores de identidad para agrupar segmentos de un mismo orador; y (4) detección nativa de solapamientos para cuantificar el habla simultánea.

#### Transcripción y alineación temporal

La conversión de voz a texto se realizó mediante OpenAI Whisper (modelo large-v2). En lugar de transcribir el audio de forma monolítica, el sistema procesó individualmente los segmentos preidentificados por la diarización, asegurando que cada fragmento de texto se asociara correctamente a su orador correspondiente. Se realizó un cruce de marcas de tiempo entre las fronteras de diarización y los segmentos de transcripción, garantizando precisión de milisegundos en la asignación palabra-locutor.

#### Post-procesamiento

Los segmentos consecutivos de un mismo orador separados por brechas inferiores a 0,5 segundos se fusionaron en una única intervención. Se eliminaron las intervenciones con duración inferior a 0,3 segundos, correspondientes típicamente a ruidos o artefactos de grabación.

### Sistema de clasificación de género

Se implementó un sistema de validación cruzada mediante dos métodos independientes y complementarios. El primer método consistió en el análisis de frecuencia fundamental (F0) mediante Parselmouth, estableciendo un umbral de 165 Hz basado en la literatura de fonética acústica (rangos típicos: 85-180 Hz para voces masculinas y 165-255 Hz para voces femeninas). El segundo método empleó el modelo wav2vec2-large-xlsr-53 ajustado específicamente para reconocimiento de género, un modelo basado en arquitectura Transformer preentrenado en 53 idiomas. Para evitar clasificaciones erróneas, el sistema concatenó y analizó hasta 10 segundos de audio de cada locutor antes de emitir un juicio.

La lógica de decisión estableció que, en caso de concordancia entre ambos métodos, se asignara el género con confianza máxima; en caso de discrepancia, se derivó a evaluación humana por un investigador. Del total de oradores clasificados, el 99,65% presentó concordancia entre ambos métodos. Los 42 casos de discrepancia (0,35%) fueron resueltos mediante evaluación humana.

### Extracción de variables lingüísticas

#### Análisis gramatical

Se empleó el modelo de procesamiento de lenguaje natural spaCy (en_core_web_lg) para extraer métricas gramaticales: conteo de palabras (tokens léxicos excluyendo puntuación), velocidad de habla (palabras por minuto), diversidad léxica (ratio Type-Token) y detección de preguntas e imperativos mediante análisis morfosintáctico.

#### Análisis pragmático

Se implementaron diccionarios de patrones léxicos para identificar marcadores pragmáticos relevantes para las dinámicas de poder conversacional: marcadores de mitigación (*hedges*: "I think", "maybe", "perhaps", "probably"), marcadores de disculpa ("sorry", "I apologize"), marcadores de cortesía ("please", "thank you"), marcadores de vulnerabilidad ("I'm not sure", "I don't know"), y marcadores de acuerdo y desacuerdo.

#### Análisis afectivo

Se empleó la librería pysentimiento, basada en modelos BERT ajustados, para clasificar cada intervención en categorías de sentimiento (positivo, negativo, neutro) y emociones (alegría, ira, tristeza, miedo, sorpresa, asco).

#### Índices compuestos

Se construyeron índices compuestos para capturar dimensiones complejas del comportamiento comunicativo: (1) índice de conflictividad, suma ponderada que integra desacuerdos, imperativos e interrupciones, descontando acuerdos y cortesía; (2) índice de asertividad, que evalúa el estilo comunicativo directo versus mitigado; (3) índice de eco (*echoing score*), que mide el solapamiento léxico Jaccard entre intervenciones consecutivas como proxy de continuidad temática; y (4) detección de apropiación de ideas, operacionalizada como alto índice de eco (>0,3) con ausencia de atribución explícita.

### Marco estadístico

#### Verificación de supuestos paramétricos

Se aplicó el test de Shapiro-Wilk para evaluar normalidad y el test de Levene para homocedasticidad. Dado que ninguna variable cumplió el supuesto de normalidad (p < 0,001 en todos los casos), se optó por tests no paramétricos.

#### Análisis inferencial

Para variables continuas se empleó el test de Mann-Whitney U; para variables categóricas, el test Chi-cuadrado de independencia. Se reportaron medidas de magnitud del efecto: Hedges' g para variables continuas (con corrección por tamaño muestral) y V de Cramér para variables categóricas. La interpretación de los tamaños del efecto siguió los umbrales convencionales: despreciable (<0,2), pequeño (0,2-0,5), mediano (0,5-0,8) y grande (>0,8).

#### Corrección por comparaciones múltiples

Se controló la tasa de falsos positivos mediante el método de Benjamini-Hochberg (False Discovery Rate), reportando tanto los p-valores originales como los corregidos.

#### Modelos de efectos mixtos

Dado que las intervenciones están anidadas dentro de sesiones, se emplearon modelos de efectos mixtos para corregir la estructura jerárquica: modelos lineales mixtos (LMM) para variables continuas y modelos lineales generalizados mixtos (GLMM) para variables binarias, con el género como efecto fijo y la sesión como efecto aleatorio. Se calculó el coeficiente de correlación intraclase (ICC) para cada variable.

#### Intervalos de confianza

Todas las estimaciones de diferencias entre grupos se acompañaron de intervalos de confianza al 95%, calculados mediante bootstrap (10.000 iteraciones) para diferencias de medias y método Wilson Score para proporciones.

---

## RESULTADOS

### Estadísticas descriptivas

La muestra final comprendió 12.138 intervenciones procedentes de 75 sesiones de debate científico. Las intervenciones masculinas representaron el 63,4% del total (n = 7.690), frente al 36,6% de intervenciones femeninas (n = 4.448). La duración media de las intervenciones fue de 11,90 segundos (DE = 15,69) para hombres y 13,08 segundos (DE = 17,19) para mujeres. El conteo medio de palabras fue de 36,85 (DE = 47,90) para hombres y 40,29 (DE = 51,57) para mujeres. La velocidad de habla media fue similar en ambos grupos: 210,36 palabras por minuto (DE = 117,93) en hombres y 212,93 (DE = 124,08) en mujeres.

**Tabla 1. Estadísticas descriptivas de las variables principales por género**

| Variable | Media (M) | DE (M) | Mediana (M) | Media (F) | DE (F) | Mediana (F) |
|----------|-----------|--------|-------------|-----------|--------|-------------|
| Duración (s) | 11,90 | 15,69 | 6,09 | 13,08 | 17,19 | 6,78 |
| Conteo de palabras | 36,85 | 47,90 | 19,0 | 40,29 | 51,57 | 21,0 |
| Diversidad léxica (TTR) | 0,421 | 0,276 | 0,316 | 0,404 | 0,268 | 0,308 |
| Velocidad (ppm) | 210,36 | 117,93 | 192,43 | 212,93 | 124,08 | 192,30 |
| Latencia (s) | 1,16 | 3,54 | 0,76 | 1,23 | 3,50 | 0,82 |
| Índice de eco | 0,066 | 0,159 | 0,0 | 0,071 | 0,162 | 0,0 |

*Nota: M = masculino; F = femenino; DE = desviación estándar; TTR = Type-Token Ratio; ppm = palabras por minuto.*

En cuanto a las variables categóricas, las tasas fueron similares entre géneros para la mayoría de marcadores pragmáticos: preguntas (18,65% vs. 18,84%), *hedges* (33,89% vs. 35,12%), acuerdo (10,26% vs. 10,61%), disculpa (1,21% vs. 1,35%) y cortesía (6,80% vs. 6,45%). Las mujeres presentaron tasas ligeramente superiores en desacuerdo (26,66% vs. 23,68%), solapamiento (4,99% vs. 4,16%) e interrupciones al orador previo (3,53% vs. 2,90%).

### Análisis inferencial

#### Diferencias significativas tras corrección FDR

De las 19 variables analizadas, 6 mostraron diferencias estadísticamente significativas antes de la corrección por comparaciones múltiples y 5 sobrevivieron a la corrección FDR (tasa de supervivencia: 83,3%).

**Tabla 2. Variables con diferencias significativas tras corrección FDR**

| Variable | Dirección | p original | p FDR | Hedges' g / V de Cramér | IC 95% |
|----------|-----------|------------|-------|-------------------------|--------|
| Duración | F > M | 0,00017 | 0,0018 | g = −0,073 | [−0,110; −0,036] |
| Conteo de palabras | F > M | 0,00028 | 0,0018 | g = −0,070 | [−0,107; −0,033] |
| Diversidad léxica | M > F | 0,0012 | 0,0046 | g = 0,060 | [0,023; 0,097] |
| Desacuerdo | F > M | 0,00027 | 0,0018 | V = 0,033 | — |
| Índice de conflictividad | F > M | 0,00041 | 0,0020 | g = −0,062 | — |

*Nota: F = femenino; M = masculino; IC = intervalo de confianza.*

Las mujeres presentaron intervenciones significativamente más largas (diferencia media: 1,18 s; IC 95%: 0,57-1,80) y con mayor número de palabras (diferencia media: 3,44 palabras; IC 95%: 1,59-5,30). Los hombres mostraron mayor diversidad léxica (diferencia media: 0,017; IC 95%: 0,007-0,027). Las mujeres expresaron más desacuerdo (26,66% vs. 23,68%; χ² = 13,30; p < 0,001) y presentaron mayor índice de conflictividad.

#### Tamaños del efecto

Todos los efectos significativos fueron de magnitud despreciable según los umbrales convencionales (|g| < 0,2; V < 0,1). La variable solapamiento, aunque significativa antes de la corrección (p = 0,037), no sobrevivió al ajuste FDR (p corregido = 0,117).

**Tabla 3. Tamaños del efecto para variables numéricas principales**

| Variable | Hedges' g | IC 95% inferior | IC 95% superior | Magnitud |
|----------|-----------|-----------------|-----------------|----------|
| Duración | −0,073 | −0,110 | −0,036 | Despreciable |
| Conteo de palabras | −0,070 | −0,107 | −0,033 | Despreciable |
| Diversidad léxica | 0,060 | 0,023 | 0,097 | Despreciable |
| Velocidad (ppm) | −0,021 | −0,058 | 0,016 | Despreciable |
| Latencia | −0,021 | −0,058 | 0,016 | Despreciable |
| Índice de eco | −0,031 | −0,068 | 0,006 | Despreciable |

### Modelos de efectos mixtos

Los valores de ICC fueron próximos a cero para todas las variables (ICC < 0,01), indicando que la variabilidad entre sesiones es despreciable y que la mayor parte de la varianza corresponde a diferencias intra-sesión (entre individuos). Los modelos mixtos confirmaron los hallazgos de los análisis univariados: la duración (β = −1,15; IC 95%: −1,80 a −0,50; p = 0,0005), el conteo de palabras (β = −3,11; IC 95%: −5,07 a −1,14; p = 0,002), la diversidad léxica (β = 0,017; IC 95%: 0,006 a 0,028; p = 0,002) y el desacuerdo (β = −0,159; IC 95%: −0,243 a −0,074; p < 0,001) mantuvieron su significación tras controlar la estructura jerárquica de los datos.

### Dinámicas de interacción

#### Matriz de transiciones de turno

Se observó un patrón de segregación discursiva pronunciado. La probabilidad de que un hombre ceda la palabra a otro hombre fue del 85,38%, mientras que la probabilidad de que una mujer ceda la palabra a otra mujer fue del 74,76%. Las transiciones inter-género fueron minoritarias: solo el 14,62% de las intervenciones masculinas fueron seguidas por una intervención femenina, y el 25,24% de las intervenciones femeninas fueron seguidas por una intervención masculina (χ² = 4.352,04; p < 0,001; V de Cramér = 0,599).

**Tabla 4. Matriz de transiciones de turno (probabilidades)**

| Orador previo | → Femenino | → Masculino |
|---------------|------------|-------------|
| Femenino | 74,76% | 25,24% |
| Masculino | 14,62% | 85,38% |

#### Posiciones de poder

Los hombres ocuparon con mayor frecuencia las posiciones de apertura de sesión (62,67% vs. 37,33%; p = 0,037 test binomial) y de cierre (58,67% vs. 41,33%; p = 0,165). El análisis del fenómeno *sticky floor* reveló que las mujeres tardaron menos en tomar la palabra por primera vez (turno medio: 3,21) que los hombres (turno medio: 1,66), aunque esta diferencia fue marginalmente significativa (p = 0,049; g = 0,122).

#### Asimetría pregunta-respuesta

Se identificaron 2.272 pares pregunta-respuesta. La distribución mostró fuerte homofilia: cuando una mujer formuló una pregunta, otra mujer respondió en el 73,39% de los casos; cuando un hombre formuló una pregunta, otro hombre respondió en el 83,05% de los casos. Las respuestas más extensas se produjeron cuando hombres respondieron a preguntas de mujeres (M→F: 16,80 s, 51,23 palabras), mientras que las más breves ocurrieron en interacciones intra-masculinas (M→M: 12,59 s, 38,42 palabras). La tasa de preguntas ignoradas (definida como bajo eco léxico en la respuesta) fue similar en ambos géneros: 73,4% para preguntas femeninas y 72,8% para preguntas masculinas.

### Modelo predictivo del éxito de interrupción

Se ajustó un modelo de regresión logística para predecir el éxito de las interrupciones. El modelo completo (con género) alcanzó un AUC de 0,765, mientras que el modelo reducido (sin género) obtuvo un AUC de 0,765 (ΔAUC = 0,0002). El género no aportó información predictiva adicional (OR = 0,99; IC 95%: 0,83-1,19). Los predictores más potentes del éxito de interrupción fueron el índice de conflictividad (OR = 14,21; IC 95%: 10,65-20,44), el número de imperativos (OR = 3,07) y el conteo de palabras (OR = 2,92). Los *hedges* (OR = 0,15) y la asertividad (OR = 0,05) se asociaron negativamente con el éxito de interrupción.

### Patrón explicativo (*explaining pattern*)

Se evaluó simétricamente la prevalencia del patrón explicativo (intervención extensa tras intervención breve del género opuesto, con marcadores de corrección) en ambas direcciones. La tasa fue del 7,91% para hombres y del 8,81% para mujeres, sin diferencias significativas (χ² = 0,88; p = 0,348; V = 0,009).

### Apropiación de ideas

La tasa de apropiación (alto eco léxico sin atribución) fue simétrica: 84,13% en transiciones M→F y 82,86% en transiciones F→M (diferencia: −0,44 puntos porcentuales). Las tasas de atribución explícita fueron mayores en transiciones inter-género (M→F: 17,14%; F→M: 15,87%) que en transiciones intra-género (M→M: 9,71%; F→F: 9,19%).

### Análisis de clima conversacional

El ANOVA factorial (género × clima de interrupción) reveló un efecto principal significativo del clima (F = 7.755,70; p < 0,001) pero no del género (F = 0,09; p = 0,760) ni de la interacción género × clima (F = 2,13; p = 0,145). En clima calmado, la tasa de interrupción fue nula para ambos géneros; en clima hostil, las tasas fueron similares (mujeres: 44,32%; hombres: 42,86%).

### Distribución de equidad por sesión

De las 75 sesiones analizadas, 20 (26,7%) presentaron dominancia masculina (>70% del tiempo de habla), 8 (10,7%) presentaron dominancia femenina (>70%) y 11 (14,7%) alcanzaron alta equidad (índice de paridad ≥0,85). Las sesiones con mayor equidad presentaron índices de paridad entre 0,90 y 0,98, indicando distribución casi simétrica del tiempo de habla.

### Análisis de potencia

El análisis de potencia post-hoc indicó potencia superior al 80% para detectar efectos pequeños (g ≥ 0,2) dado el tamaño muestral (N = 12.138). Para las variables de interrupción, la potencia alcanzada fue subóptima (48,4% para *interrupts_previous*), sugiriendo que efectos pequeños genuinos podrían no haberse detectado.

### Análisis de sensibilidad

La variación sistemática del umbral de echo score para la detección de apropiación (0,1 a 0,5) mostró estabilidad en el patrón de resultados, con tasas de apropiación M→F oscilando entre 16,90% (umbral 0,1) y 2,12% (umbral 0,5), manteniendo la simetría inter-género en todos los umbrales evaluados.

---

## SÍNTESIS DE HALLAZGOS PRINCIPALES

Los resultados de este estudio revelan un patrón de diferencias mínimas pero estadísticamente significativas en las dinámicas de participación verbal por género en debates científicos de medicina intensiva. Las mujeres presentaron intervenciones ligeramente más largas y con mayor número de palabras, así como mayor expresión de desacuerdo, mientras que los hombres mostraron mayor diversidad léxica. Sin embargo, todos los tamaños del efecto fueron de magnitud despreciable (|g| < 0,08), lo que limita su relevancia práctica.

El hallazgo más robusto fue la segregación discursiva: existió una fuerte tendencia a que las transiciones de turno ocurrieran dentro del mismo género (85% M→M; 75% F→F), con transiciones inter-género minoritarias. Este patrón estructural sugiere la existencia de dinámicas de interacción diferenciadas que trascienden las características individuales de las intervenciones.

Contrariamente a hipótesis previas, no se encontraron diferencias significativas en el éxito de las interrupciones, el patrón explicativo, la apropiación de ideas ni la tasa de preguntas ignoradas. El género no aportó valor predictivo incremental para el éxito de las interrupciones más allá de las variables de estilo comunicativo. Los valores de ICC próximos a cero indican que las diferencias observadas son estables a través de las sesiones y no artefactos de contextos particulares.

---

*Análisis realizado mediante pipeline computacional que integra procesamiento de señales de audio (Pyannote.audio 3.1, OpenAI Whisper large-v2), clasificación de género por doble validación (análisis de F0 + wav2vec2-xlsr-53), procesamiento de lenguaje natural (spaCy, pysentimiento) y análisis estadístico riguroso (modelos de efectos mixtos, corrección FDR, bootstrap). El código y diccionario de variables están disponibles para garantizar la replicabilidad completa del estudio.*