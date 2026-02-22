# Diccionario de Variables y Glosario Técnico

## Estudio de Disparidad de Género en Debates Científicos

Este documento proporciona definiciones completas de todas las variables extraídas y calculadas, así como un glosario de términos técnicos empleados en la metodología y análisis estadístico.

---

# PARTE I: VARIABLES DEL DATASET

---

## 1. Identificadores y Metadatos

### `intervention_id`
- **Tipo:** Entero
- **Rango:** 1 a N (secuencial por sesión)
- **Definición:** Identificador único que numera cada intervención de forma secuencial dentro de una sesión. Permite la trazabilidad y referencia unívoca de cada unidad de análisis.

### `session`
- **Tipo:** Cadena de texto
- **Definición:** Identificador único de la sesión o debate del que proviene la intervención. Corresponde al nombre del archivo de video/audio original procesado.

### `speaker`
- **Tipo:** Cadena de texto
- **Formato:** SPEAKER_XX (ej. SPEAKER_01, SPEAKER_02)
- **Definición:** Identificador anónimo asignado automáticamente por el algoritmo de diarización a cada orador único detectado en la sesión.

### `gender`
- **Tipo:** Categórico
- **Valores:** `male`, `female`
- **Definición:** Género del orador, clasificado mediante el sistema de doble validación (análisis de pitch + modelo de deep learning) con arbitraje humano en casos de discrepancia.

### `gender_bin`
- **Tipo:** Binario
- **Valores:** 1 (masculino), 0 (femenino)
- **Definición:** Codificación numérica del género para su uso en modelos de regresión y análisis estadístico.

### `turn_number`
- **Tipo:** Entero
- **Rango:** 1 a N
- **Definición:** Posición de la intervención en la secuencia lógica de turnos de habla. Se incrementa cada vez que hay un cambio de orador, independientemente de solapamientos.

---

## 2. Variables Temporales

### `start_time`
- **Tipo:** Flotante
- **Unidad:** Segundos
- **Definición:** Marca de tiempo del inicio de la intervención, medida desde el comienzo del audio de la sesión.

### `end_time`
- **Tipo:** Flotante
- **Unidad:** Segundos
- **Definición:** Marca de tiempo del fin de la intervención, medida desde el comienzo del audio de la sesión.

### `duration`
- **Tipo:** Flotante
- **Unidad:** Segundos
- **Rango:** >0.3 (intervenciones menores fueron filtradas)
- **Definición:** Duración neta de la intervención, calculada como `end_time - start_time`. Representa el tiempo total que el orador mantuvo el uso de la palabra.

### `latency_s`
- **Tipo:** Flotante
- **Unidad:** Segundos
- **Rango:** Puede ser negativo
- **Definición:** Tiempo de silencio (o solapamiento) entre el fin del orador anterior y el inicio del orador actual. Valores negativos indican que el orador comenzó a hablar antes de que el anterior terminara (solapamiento/interrupción).

### `session_phase`
- **Tipo:** Flotante
- **Rango:** 0.0 a 1.0
- **Definición:** Posición relativa de la intervención dentro de la duración total de la sesión. Un valor de 0.0 indica el inicio de la sesión; 1.0 indica el final. Permite análisis de evolución temporal.

### `phase_quartile`
- **Tipo:** Categórico
- **Valores:** Q1, Q2, Q3, Q4
- **Definición:** Cuartil de la sesión en el que ocurre la intervención:
  - **Q1:** Primer 25% de la sesión (apertura)
  - **Q2:** 25-50% de la sesión
  - **Q3:** 50-75% de la sesión
  - **Q4:** Último 25% de la sesión (cierre)

---

## 3. Dinámicas de Turno e Interrupción

### `has_overlap`
- **Tipo:** Booleano
- **Valores:** True/False (o 1/0)
- **Definición:** Indica si en algún momento de la intervención hubo habla simultánea con otro orador. Se detecta cuando los rangos temporales de dos intervenciones se superponen.

### `overlap_duration`
- **Tipo:** Flotante
- **Unidad:** Segundos
- **Rango:** ≥0
- **Definición:** Duración total del tiempo en que la intervención se solapó con el habla de otro orador. Si no hubo solapamiento, el valor es 0.

### `interrupts_previous`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica si el orador actual comenzó a hablar antes de que el orador anterior terminara su intervención. Captura la "entrada intrusiva" en el turno ajeno. Se determina cuando `latency_s < 0`.

### `interrupted_by_next`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica si la intervención actual fue cortada por el siguiente orador (es decir, si el siguiente orador comenzó antes de que este terminara). Es el complemento de `interrupts_previous` desde la perspectiva del orador interrumpido.

### `interruption_success`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica si un orador que interrumpió al anterior logró consolidar el turno de habla, es decir, si consiguió silenciar al orador previo y mantener la palabra. Se marca como exitosa cuando el orador interrumpido cesa de hablar tras el solapamiento.

### `is_backchannel`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Identifica intervenciones cortas de "habla de apoyo" o retroalimentación verbal que no tienen intención de tomar el turno de palabra. Ejemplos: "mhm", "yeah", "I see", "okay", "right". Se detecta por duración breve (<2 segundos) y patrones léxicos específicos.

---

## 4. Métricas Lingüísticas Básicas

### `text`
- **Tipo:** Cadena de texto
- **Definición:** Transcripción literal del contenido verbal de la intervención, obtenida mediante el modelo Whisper.

### `word_count`
- **Tipo:** Entero
- **Rango:** ≥0
- **Definición:** Número total de palabras (tokens léxicos) en la intervención, excluyendo signos de puntuación.

### `wpm` (Words Per Minute)
- **Tipo:** Flotante
- **Unidad:** Palabras por minuto
- **Rango típico:** 100-250
- **Definición:** Velocidad de habla del orador, calculada como `(word_count / duration) × 60`. Mide la fluidez y ritmo del discurso.

### `lexical_diversity`
- **Tipo:** Flotante
- **Rango:** 0.0 a 1.0
- **Sinónimo:** TTR (Type-Token Ratio)
- **Definición:** Proporción de palabras únicas (types) sobre el total de palabras (tokens) en la intervención. Un valor cercano a 1 indica alta variedad léxica (pocas repeticiones); valores bajos indican vocabulario repetitivo. Se calcula sobre lemas para normalizar variaciones morfológicas.

---

## 5. Variables Gramaticales y Sintácticas

### `is_question`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica si la intervención constituye una pregunta. Se detecta mediante:
  - Presencia de signo de interrogación final
  - Palabras interrogativas (what, who, when, where, why, how) en posición inicial
  - Estructura sintáctica interrogativa (inversión sujeto-verbo)

### `num_imperatives`
- **Tipo:** Entero
- **Rango:** ≥0
- **Definición:** Número de verbos en modo imperativo detectados en la intervención. Los imperativos representan órdenes, instrucciones directas o exhortaciones, y son indicadores de un estilo comunicativo directivo.

---

## 6. Marcadores Pragmáticos

### `has_hedge`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica el uso de lenguaje mitigador o dubitativo que reduce la fuerza asertiva del enunciado. Los hedges transmiten incertidumbre, tentativeness o cortesía negativa.
- **Ejemplos detectados:** "I think", "maybe", "perhaps", "probably", "possibly", "sort of", "kind of", "seems", "appears", "actually", "just", "a bit", "somewhat"

### `has_apology`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica la presencia de expresiones de disculpa explícita.
- **Ejemplos detectados:** "sorry", "I apologize", "excuse me", "pardon", "forgive me", "my apologies"

### `has_courtesy`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica el uso de marcadores de cortesía positiva o deferencia.
- **Ejemplos detectados:** "please", "thank you", "thanks", "kindly", "I appreciate", "if you don't mind"

### `has_vulnerability`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica la admisión explícita de falta de conocimiento, incertidumbre o limitación propia.
- **Ejemplos detectados:** "I'm not sure", "I don't know", "I'm uncertain", "I'm confused", "I need help", "I struggle with"

### `has_agreement`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica la expresión de conformidad o acuerdo con el orador anterior.
- **Ejemplos detectados:** "I agree", "absolutely", "exactly", "that's right", "good point", "you're right", "yes"

### `has_disagreement`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica la expresión de discrepancia o desacuerdo con el orador anterior.
- **Ejemplos detectados:** "I disagree", "I don't think", "however", "but", "on the contrary", "not necessarily", "actually...no"

---

## 7. Autoridad y Crédito Académico

### `has_title`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica el uso de títulos académicos o profesionales al referirse a otros participantes o a uno mismo.
- **Ejemplos detectados:** "Doctor", "Dr.", "Professor", "Prof.", "PhD", "colleague", "expert"

### `has_attribution`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica el reconocimiento explícito de la autoría de ideas, dando crédito a la fuente original.
- **Ejemplos detectados:** "As Dr. X said", "According to", "X mentioned", "X pointed out", "Building on what X said", "As my colleague noted"

---

## 8. Índice de Eco y Apropiación

### `echoing_score`
- **Tipo:** Flotante
- **Rango:** 0.0 a 1.0
- **Definición:** Índice de similitud léxica entre la intervención actual y la inmediatamente anterior. Mide qué proporción del vocabulario de contenido (sustantivos, adjetivos, verbos principales) del orador anterior es retomada por el orador actual. Se calcula como el coeficiente de Jaccard sobre los conjuntos de lemas.
- **Interpretación:**
  - **Bajo (<0.1):** Poca continuidad temática, cambio de tema
  - **Medio (0.1-0.3):** Continuidad temática normal
  - **Alto (>0.3):** Fuerte retoma del contenido anterior

### `appropriation`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica si una intervención presenta apropiación de ideas, operacionalizada como la combinación de:
  - Alto índice de eco (echoing_score > 0.3): el orador retoma contenido del anterior
  - Ausencia de atribución (has_attribution = False): no da crédito a la fuente
- **Interpretación:** Captura situaciones donde un orador repite o reformula ideas de otro sin reconocer su origen.

### `ignored`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Aplica solo a intervenciones que son preguntas. Indica si la respuesta subsecuente mostró bajo eco léxico (echoing_score < 0.1), sugiriendo que la pregunta no fue abordada o fue ignorada temáticamente.

---

## 9. Índices Compuestos

### `conflict_score`
- **Tipo:** Entero
- **Rango típico:** -2 a +5
- **Definición:** Índice de conflictividad de la intervención. Suma ponderada que captura el nivel de confrontación:
  - **Suma:** has_disagreement (+1), num_imperatives (+1), interrupts_previous (+1)
  - **Resta:** has_agreement (-1), has_courtesy (-1)
- **Interpretación:** Valores positivos altos indican intervenciones confrontativas; valores negativos indican intervenciones conciliadoras.

### `peak_conflict`
- **Tipo:** Booleano
- **Valores:** True/False
- **Definición:** Indica si la intervención se encuentra en el percentil 90 superior de conflictividad del dataset completo. Permite identificar los momentos de máxima tensión.

### `assertiveness_score`
- **Tipo:** Entero
- **Rango típico:** 0 a 5
- **Definición:** Índice de asertividad comunicativa. Evalúa el estilo directo versus mitigado:
  - **Suma:** num_imperatives (+1), has_disagreement (+1), baseline (+2)
  - **Resta:** has_hedge (-1), has_apology (-1)
- **Interpretación:** Valores altos indican estilo directo y asertivo; valores bajos indican estilo tentativo y mitigado.

### `interruption_climate`
- **Tipo:** Flotante
- **Rango:** 0.0 a 1.0
- **Definición:** Densidad de interrupciones en el contexto inmediato, calculada como la proporción de intervenciones con `interrupts_previous = True` en una ventana rodante de los 5 turnos anteriores.
- **Interpretación:** Captura si el momento de la sesión está caracterizado por alta o baja frecuencia de interrupciones.

### `climate_type`
- **Tipo:** Categórico
- **Valores:** `calm`, `neutral`, `hostile`
- **Definición:** Clasificación del clima conversacional basada en `interruption_climate`:
  - **calm:** interruption_climate < 0.1 (pocas interrupciones)
  - **neutral:** 0.1 ≤ interruption_climate ≤ 0.3
  - **hostile:** interruption_climate > 0.3 (muchas interrupciones)

---

## 10. Análisis Afectivo

### `sentiment`
- **Tipo:** Categórico
- **Valores:** `POS` (positivo), `NEG` (negativo), `NEU` (neutro)
- **Definición:** Clasificación del tono afectivo general de la intervención, determinada mediante modelo BERT de análisis de sentimiento (pysentimiento).

### `emotion`
- **Tipo:** Categórico
- **Valores:** `joy`, `anger`, `sadness`, `fear`, `surprise`, `disgust`, `others`
- **Definición:** Emoción predominante detectada en la intervención, clasificada mediante modelo BERT de análisis emocional. La categoría "others" incluye emociones no claramente clasificables en las seis básicas.

---

## 11. Variables Derivadas de Análisis

### `parity_index`
- **Tipo:** Flotante
- **Rango:** 0.0 a 1.0
- **Nivel:** Sesión
- **Definición:** Índice de equidad de género en la distribución del tiempo de habla de una sesión. Se calcula como el ratio entre el porcentaje menor y el porcentaje mayor de tiempo de habla por género. Un valor de 1.0 indica paridad perfecta (50%-50%); valores cercanos a 0 indican dominancia extrema de un género.

### `influence_ratio`
- **Tipo:** Flotante
- **Rango:** >0
- **Nivel:** Orador/Género
- **Definición:** Ratio entre palabras introducidas por primera vez en la sesión y palabras que repiten vocabulario de otros. Valores >1 indican tendencia a introducir vocabulario nuevo (liderazgo temático); valores <1 indican tendencia a seguir el vocabulario ajeno.

---

## 12. Variables de Análisis de Clusters (Nivel Usuario)

### `user_id`
- **Tipo:** Texto
- **Formato:** `{session}__{speaker}`
- **Nivel:** Usuario
- **Definición:** Identificador único de cada participante dentro de una sesión. Combina sesión y etiqueta de hablante. Un mismo orador en diferentes sesiones se trata como usuarios distintos.

### `is_top3_speaker`
- **Tipo:** Binario
- **Valores:** 0, 1
- **Nivel:** Usuario
- **Definición:** Indica si el usuario figura entre los 3 primeros hablantes distintos de la sesión, ordenados por `start_time`. Un valor de 1 define a los participantes que inician la conversación, típicamente moderadores o panelistas.

### `min_turn_number`
- **Tipo:** Entero
- **Rango:** ≥1
- **Nivel:** Usuario
- **Definición:** Turno más temprano en el que el usuario interviene por primera vez en la sesión. Valores bajos indican participación temprana (roles de liderazgo); valores altos indican incorporación tardía (audiencia).

### `cluster_kmeans`
- **Tipo:** Entero
- **Valores:** 0, 1
- **Nivel:** Usuario
- **Definición:** Asignación de cluster obtenida mediante K-Means (k=2) sobre 45 features estandarizadas. La interpretación empírica de los clusters es:
  - **Cluster 0 — Moderadores/Panelistas:** Usuarios que hablan primero (turno mediano=3), producen más intervenciones, duración y palabras. 58.9% son top-3 speakers.
  - **Cluster 1 — Público/Audiencia:** Usuarios que intervienen tarde (turno mediano=35), con menor volumen pero más disculpas y expresiones de tristeza. Solo 0.7% son top-3 speakers.

### `pct_female_mods` / `pct_female_auds`
- **Tipo:** Flotante
- **Rango:** 0.0 a 100.0
- **Nivel:** Sesión
- **Definición:** Porcentaje de mujeres entre moderadores (Cluster 0) o audiencia (Cluster 1) de cada sesión. Utilizado para cuantificar el efecto llamada de género.

---

# PARTE II: GLOSARIO DE TÉRMINOS TÉCNICOS

---

## Procesamiento de Audio

### Diarización (Speaker Diarization)
Proceso automático de segmentar una grabación de audio según "quién habla cuándo". El sistema identifica los cambios de orador y agrupa los segmentos correspondientes a cada persona, asignando etiquetas anónimas (SPEAKER_01, SPEAKER_02, etc.).

### Embedding de Voz (Speaker Embedding)
Representación vectorial de las características acústicas distintivas de un orador, extraída mediante redes neuronales. Estos vectores de dimensión fija (típicamente 192-512 dimensiones) capturan la "huella vocal" que permite distinguir a diferentes personas.

### ECAPA-TDNN
Arquitectura de red neuronal especializada en extracción de embeddings de voz. Combina capas convolucionales 1D con mecanismos de atención para capturar características espectrales a múltiples escalas temporales.

### EBU R128
Estándar de la Unión Europea de Radiodifusión para la normalización de sonoridad (loudness). Define métodos para medir y ajustar el nivel percibido de audio, utilizando la unidad LUFS (Loudness Units relative to Full Scale).

### LUFS (Loudness Units Full Scale)
Unidad de medida de sonoridad que considera la percepción humana del volumen, no solo la amplitud física de la señal. El estándar EBU R128 especifica -23 LUFS como nivel objetivo para contenido broadcast.

### VAD (Voice Activity Detection)
Técnica para determinar automáticamente qué segmentos de una señal de audio contienen habla humana versus silencio, ruido de fondo u otros sonidos no vocales.

### Frecuencia Fundamental (F0) / Pitch
Frecuencia de vibración de las cuerdas vocales durante la fonación, medida en Hercios (Hz). Es el correlato acústico de la percepción de tono grave/agudo. Presenta dimorfismo sexual: típicamente 85-180 Hz en varones adultos y 165-255 Hz en mujeres adultas.

### Whisper
Modelo de reconocimiento automático del habla (ASR) desarrollado por OpenAI. Entrenado en datos multilingües masivos, destaca por su robustez ante variaciones de acento, ruido y calidad de audio.

### Pyannote.audio
Librería de código abierto para análisis de audio centrado en el hablante, que incluye modelos preentrenados para diarización, detección de actividad vocal y verificación de locutor.

### wav2vec2
Arquitectura de red neuronal tipo Transformer desarrollada por Meta/Facebook para aprendizaje de representaciones de audio. Preentrenada de forma auto-supervisada, puede ajustarse para tareas específicas como reconocimiento de habla o clasificación de género.

---

## Procesamiento del Lenguaje Natural (NLP)

### Tokenización
Proceso de dividir un texto en unidades mínimas (tokens), típicamente palabras y signos de puntuación.

### Lematización
Proceso de reducir las palabras a su forma base o lema (ej. "running", "ran", "runs" → "run"). Permite agrupar variantes morfológicas de una misma palabra.

### POS Tagging (Part-of-Speech Tagging)
Etiquetado gramatical automático que asigna a cada palabra su categoría morfosintáctica (sustantivo, verbo, adjetivo, etc.).

### Análisis de Dependencias (Dependency Parsing)
Análisis sintáctico que identifica las relaciones gramaticales entre palabras (sujeto, objeto, modificador, etc.) en forma de árbol de dependencias.

### spaCy
Librería de NLP de código abierto optimizada para producción, que ofrece tokenización, lematización, POS tagging, reconocimiento de entidades y análisis de dependencias.

### Type-Token Ratio (TTR)
Medida de diversidad léxica calculada como la proporción de palabras únicas (types) sobre el total de palabras (tokens). Sensible a la longitud del texto: textos más largos tienden a tener TTR más bajo.

### Coeficiente de Jaccard
Medida de similitud entre dos conjuntos, calculada como el tamaño de la intersección dividido por el tamaño de la unión. Rango: 0 (conjuntos disjuntos) a 1 (conjuntos idénticos).

### Hedge (Mitigador)
En pragmática, expresión lingüística que reduce la fuerza o certeza de un enunciado. Función de cortesía negativa (proteger la imagen del interlocutor) o de expresión de incertidumbre epistémica.

### pysentimiento
Librería de análisis de sentimiento y emociones basada en modelos Transformer (BERT) ajustados para español e inglés. Proporciona clasificación de polaridad (positivo/negativo/neutro) y emociones discretas.

### BERT (Bidirectional Encoder Representations from Transformers)
Arquitectura de modelo de lenguaje que procesa texto de forma bidireccional, capturando contexto tanto anterior como posterior a cada palabra. Base de numerosos modelos de NLP de última generación.

---

## Estadística Inferencial

### Test de Mann-Whitney U
Prueba no paramétrica para comparar dos grupos independientes. Evalúa si una muestra tiende a tener valores mayores que la otra, sin asumir distribución normal. Equivalente no paramétrico del t-test.

### Test de Shapiro-Wilk
Prueba de normalidad que evalúa si una muestra proviene de una distribución normal. Un p-valor bajo (típicamente <0.05) indica desviación significativa de la normalidad.

### Test de Levene
Prueba de homocedasticidad que evalúa si dos o más grupos tienen varianzas iguales. Más robusto que el test de Bartlett ante desviaciones de normalidad.

### Test Chi-cuadrado (χ²)
Prueba para evaluar la asociación entre dos variables categóricas. Compara las frecuencias observadas con las esperadas bajo independencia.

### FDR (False Discovery Rate)
Proporción esperada de falsos positivos entre todos los resultados declarados significativos. Alternativa al control del error tipo I familia (FWER) que ofrece mayor potencia.

### Corrección de Benjamini-Hochberg
Procedimiento para controlar el FDR en múltiples comparaciones. Ordena los p-valores y aplica umbrales adaptativos que mantienen la tasa de falsos descubrimientos bajo control.

### Hedges' g
Medida de tamaño del efecto para diferencias entre medias, similar a la d de Cohen pero con corrección por sesgo en muestras pequeñas. Interpretación convencional: <0.2 despreciable, 0.2-0.5 pequeño, 0.5-0.8 mediano, >0.8 grande.

### V de Cramér
Medida de asociación para tablas de contingencia, normalizada entre 0 y 1 independientemente del tamaño de la tabla. Basada en el estadístico chi-cuadrado.

### Cohen's d
Medida de tamaño del efecto que expresa la diferencia entre dos medias en unidades de desviación estándar combinada. Valores: |d|<0.2 despreciable, 0.2-0.5 pequeño, 0.5-0.8 moderado, >0.8 grande. Se utiliza en el cruce cluster × género para cuantificar la magnitud de las diferencias entre hombres y mujeres dentro de cada cluster.

### Silhouette Score
Medida de calidad de clustering que evalúa cuán bien asignado está cada punto a su cluster. Rango: -1 a +1. Valores cercanos a 1 indican clusters densos y bien separados; valores cercanos a 0 indican solapamiento; valores negativos indican asignación incorrecta.

### Intervalo de Confianza (IC)
Rango de valores que, con cierto nivel de confianza (típicamente 95%), se espera contenga el verdadero valor del parámetro poblacional. Un IC que cruza el cero para una diferencia indica no significación estadística.

### Bootstrap
Método de remuestreo que estima la distribución muestral de un estadístico mediante la generación de múltiples muestras con reemplazo del dataset original. Útil para construir intervalos de confianza sin supuestos paramétricos.

### Intervalo de Wilson
Método para calcular intervalos de confianza de proporciones que ofrece mejor cobertura que el intervalo de Wald, especialmente con proporciones cercanas a 0 o 1 o muestras pequeñas.

---

## Modelos de Efectos Mixtos

### Modelo de Efectos Mixtos (Mixed-Effects Model)
Modelo estadístico que incluye tanto efectos fijos (factores de interés cuyo efecto se estima) como efectos aleatorios (factores que representan una muestra de una población más amplia). Apropiado para datos con estructura jerárquica o medidas repetidas.

### Efecto Fijo
En modelos mixtos, parámetro que representa el efecto promedio de una variable predictora en la población. Es el coeficiente de interés que se reporta e interpreta.

### Efecto Aleatorio
En modelos mixtos, parámetro que captura la variabilidad entre unidades de agrupamiento (ej. sesiones, sujetos). Permite que el intercepto o pendiente varíe entre grupos.

### ICC (Intraclass Correlation Coefficient)
Proporción de la varianza total explicada por las diferencias entre grupos (ej. sesiones). ICC = σ²_between / (σ²_between + σ²_within). Valores cercanos a 0 indican que la mayor variabilidad es intra-grupo; valores cercanos a 1 indican alta homogeneidad intra-grupo.

### LMM (Linear Mixed Model)
Modelo de efectos mixtos para variables dependientes continuas. Asume errores normales y relación lineal entre predictores y resultado.

### GLMM (Generalized Linear Mixed Model)
Extensión del LMM para variables dependientes no normales (binarias, conteos, etc.) mediante funciones de enlace apropiadas (logit para binarias, log para conteos).

### Función de Enlace (Link Function)
En modelos lineales generalizados, función que conecta el predictor lineal con la media de la distribución de respuesta. Para datos binarios, la función logit transforma probabilidades en log-odds.

---

## Modelado Predictivo

### Regresión Logística
Modelo para predecir la probabilidad de un resultado binario. Estima log-odds como función lineal de los predictores. Los coeficientes se interpretan en términos de odds ratios.

### Odds Ratio (OR)
Razón de probabilidades. OR = 1 indica sin efecto; OR > 1 indica que el predictor aumenta la probabilidad del resultado; OR < 1 indica que la disminuye. Se interpreta como el cambio multiplicativo en los odds por unidad de cambio en el predictor.

### AUC (Area Under the Curve)
Área bajo la curva ROC (Receiver Operating Characteristic). Mide la capacidad discriminativa de un modelo de clasificación. Rango: 0.5 (azar) a 1.0 (discriminación perfecta). Valores >0.7 se consideran aceptables; >0.8 buenos.

### Curva ROC
Gráfico que representa la tasa de verdaderos positivos (sensibilidad) versus la tasa de falsos positivos (1-especificidad) para diferentes umbrales de clasificación.

### Validación Cruzada (Cross-Validation)
Técnica para evaluar la capacidad de generalización de un modelo dividiendo los datos en subconjuntos de entrenamiento y prueba de forma sistemática.

---

## Análisis de Clusters (Aprendizaje No Supervisado)

### K-Means
Algoritmo de clustering particional que divide N observaciones en K grupos, minimizando la varianza intra-cluster (inercia). Requiere especificar K de antemano. En este estudio, K=2 fue seleccionado como óptimo por el criterio de Silhouette.

### DBSCAN (Density-Based Spatial Clustering)
Algoritmo de clustering basado en densidad que identifica grupos como regiones densas separadas por regiones de baja densidad. No requiere especificar el número de clusters y puede detectar outliers (puntos no asignados a ningún cluster).

### Clustering Jerárquico (Hierarchical Clustering)
Método aglomerativo que construye una jerarquía de clusters fusionando iterativamente los pares más cercanos. Produce un dendrograma que visualiza la estructura de agrupamiento a múltiples niveles de granularidad.

### PCA (Análisis de Componentes Principales)
Técnica de reducción de dimensionalidad que proyecta datos de alta dimensión en un espacio de menor dimensión, preservando la máxima varianza posible. Cada componente principal es una combinación lineal de las variables originales.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
Técnica de reducción de dimensionalidad no lineal optimizada para la visualización de datos de alta dimensión en 2D o 3D. Preserva la estructura local (vecindarios) a costa de distorsionar las distancias globales.

### ARI (Adjusted Rand Index)
Medida de concordancia entre dos particiones de datos, ajustada por azar. Rango: -1 a 1. Valores cercanos a 1 indican que las dos agrupaciones son muy similares; 0 indica concordancia al nivel del azar.

### Efecto Llamada de Género (Gender Attraction Effect)
Fenómeno observado en este estudio por el cual la composición de género de los moderadores/panelistas (Cluster 0) de una sesión correlaciona positivamente con la composición de género de la audiencia (Cluster 1). Sesiones con moderadoras mayoritariamente femeninas atraen un 51.6% de audiencia femenina, frente al 27.2% en sesiones con moderadores mayoritariamente masculinos (Spearman ρ=0.361, p=0.004).

---

## Análisis Conversacional

### Turno de Habla (Turn)
Período durante el cual un participante tiene el derecho reconocido de hablar. El sistema de turnos organiza la alternancia de participantes en la conversación.

### Solapamiento (Overlap)
Habla simultánea de dos o más participantes. Puede ser competitivo (lucha por el turno) o colaborativo (completar enunciados, mostrar acuerdo).

### Interrupción
Inicio de turno por un hablante mientras otro aún tiene la palabra, típicamente con intención de tomar el turno. Se distingue del solapamiento colaborativo por su carácter intrusivo.

### Backchannel
Señales verbales breves del oyente que indican atención y comprensión sin intención de tomar el turno. Ejemplos: "mhm", "yeah", "right", "I see".

### Latencia de Respuesta
Tiempo entre el fin del turno de un hablante y el inicio del turno del siguiente. Latencias muy cortas o negativas pueden indicar anticipación o interrupción.

### Apropiación de Ideas
Fenómeno donde un participante retoma o reformula ideas expresadas por otro sin atribuir su origen. En el estudio, operacionalizada como alto eco léxico sin atribución explícita.

### Mansplaining
Término coloquial para describir un patrón donde un hombre explica algo a una mujer de manera condescendiente, asumiendo falta de conocimiento. En el estudio, operacionalizado simétricamente como "explaining pattern" aplicable a ambas direcciones de género.

---

## Términos del Dominio de Estudio

### Debate Científico
Formato de sesión académica donde múltiples expertos discuten un tema, típicamente con posiciones contrastantes o complementarias, moderado por un facilitador.

### Medicina de Cuidados Críticos
Especialidad médica dedicada al diagnóstico y tratamiento de condiciones potencialmente mortales que requieren soporte vital y monitorización intensiva (UCI).

### Paridad de Género
Representación equitativa de hombres y mujeres, típicamente expresada como proporción 50%-50% en composición de paneles, tiempo de habla u otras métricas de participación.

### Sticky Floor (Piso Pegajoso)
Metáfora del ámbito laboral que describe barreras invisibles que dificultan el avance inicial desde posiciones de entrada. En el estudio, adaptada para referirse al tiempo que tarda cada género en tomar la palabra por primera vez en una sesión.

### Efecto Llamada (Attraction Effect)
Efecto por el cual la representación de un género en posiciones de liderazgo o visibilidad (moderadores, panelistas) influye en la composición de género de la participación subsiguiente (audiencia). Documentado empíricamente en este estudio con OR=3.16.

### Backlash
Reacción negativa o penalización social que pueden enfrentar individuos (especialmente mujeres) por comportamientos que violan expectativas de género tradicionales, como la asertividad o directividad.

---

*Documento preparado para garantizar la replicabilidad y comprensión completa del estudio.*