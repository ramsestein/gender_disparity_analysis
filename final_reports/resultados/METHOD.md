# Metodología: Análisis Automatizado de Disparidad de Género en Debates Científicos

---

## 1. Resumen del Estudio

### 1.1 Objetivo

Cuantificar de forma objetiva y automatizada las dinámicas de participación verbal por género en debates científicos del ámbito de medicina de cuidados críticos, implementando un pipeline computacional que integra procesamiento de señales de audio, aprendizaje profundo y análisis estadístico riguroso.

### 1.2 Diseño

Estudio observacional retrospectivo de análisis conversacional automatizado sobre grabaciones de sesiones científicas en formato debate.

### 1.3 Unidad de Análisis

La unidad primaria es la **intervención**, definida como un segmento continuo de habla de un único orador, delimitada por silencio superior a 0.5 segundos o por cambio de orador identificado mediante diarización automática.

---

## 2. Pipeline de Procesamiento de Audio

### 2.1 Ingesta y Normalización

El proceso comienza con la extracción del flujo de audio de los archivos de video mediante herramientas estándar de procesamiento multimedia. Para garantizar la robustez de los modelos posteriores, se aplica una estandarización rigurosa:

**Parámetros de estandarización:**
- Formato: WAV (PCM 16-bit)
- Frecuencia de muestreo: 16 kHz
- Canales: Mono

**Normalización de sonoridad:** Se implementa el estándar EBU R128, ajustando la sonoridad integrada a -23 LUFS. Esta normalización mitiga las variaciones acústicas derivadas de diferentes distancias al micrófono, características vocales de los ponentes y condiciones de grabación heterogéneas. Estudios previos indican que esta normalización mejora la precisión de la diarización entre un 15-20% en entornos con variabilidad acústica.

### 2.2 Diarización de Oradores

Para la segmentación y asignación de identidad a cada orador se utiliza **Pyannote.audio 3.1**, un sistema basado en modelos de segmentación y embeddings de voz.

**Arquitectura del sistema:**

1. **Detección de Actividad Vocal (VAD):** Identificación de regiones con habla versus silencio o ruido de fondo.

2. **Extracción de Embeddings:** Se emplean redes neuronales (arquitectura ECAPA-TDNN) para extraer vectores de identidad de 192 dimensiones de cada segmento de habla. Estos vectores capturan las características acústicas distintivas de cada orador.

3. **Clustering:** Se aplica Agglomerative Hierarchical Clustering sobre los vectores de identidad para agrupar los segmentos correspondientes a un mismo orador, asignando identificadores únicos (ej. SPEAKER_01, SPEAKER_02).

4. **Detección de Solapamientos:** El sistema detecta de forma nativa los momentos en que dos o más personas hablan simultáneamente, permitiendo cuantificar tanto la presencia como la duración del habla superpuesta.

### 2.3 Transcripción y Alineación Temporal

Para la conversión de voz a texto se utiliza **OpenAI Whisper** (modelo large-v2), reconocido por su robustez ante variaciones de acento y calidad de audio.

**Estrategia de segmentación dinámica:** En lugar de transcribir el audio completo de forma monolítica, el sistema procesa individualmente los segmentos pre-identificados por la diarización. Esto asegura que cada fragmento de texto se asocie correctamente a su orador correspondiente.

**Alineación temporal:** Se realiza un cruce de marcas de tiempo entre las fronteras de diarización y los segmentos de transcripción, garantizando que cada palabra se asigne al locutor correcto con precisión de milisegundos.

### 2.4 Post-procesamiento y Limpieza

**Fusión de intervenciones:** Los segmentos consecutivos de un mismo orador separados por brechas inferiores a 0.5 segundos se fusionan en una única intervención, evitando la fragmentación artificial del discurso.

**Filtrado de micro-segmentos:** Se eliminan intervenciones con duración inferior a 0.3 segundos, que típicamente corresponden a ruidos, toses o artefactos de grabación.

**Normalización textual:** Se corrigen artefactos de codificación y se estandariza el formato del texto a UTF-8.

---

## 3. Sistema de Clasificación de Género

### 3.1 Arquitectura de Doble Validación

Para maximizar la precisión en la clasificación de género, se implementa un sistema de validación cruzada mediante dos métodos independientes y complementarios:

**Método 1 - Análisis de Frecuencia Fundamental (F0):**

Se utiliza la interfaz Parselmouth (wrapper de Python para Praat) para extraer la frecuencia fundamental mediana de cada orador. El pitch o F0 es el correlato acústico de la vibración de las cuerdas vocales y presenta diferencias sistemáticas entre voces masculinas y femeninas.

Se establece un umbral de 165 Hz basado en la literatura de fonética acústica, donde los rangos típicos son:
- Voces masculinas: 85-180 Hz
- Voces femeninas: 165-255 Hz

**Método 2 - Aprendizaje Profundo:**

Se emplea el modelo wav2vec2-large-xlsr-53 ajustado específicamente para reconocimiento de género. Este modelo basado en arquitectura Transformer fue preentrenado en 53 idiomas y posteriormente fine-tuned para la tarea de clasificación de género, lo que le confiere robustez ante variaciones de acento y características vocales atípicas.

**Robustez por acumulación:** Para evitar clasificaciones erróneas basadas en fragmentos no representativos, el sistema concatena y analiza hasta 10 segundos de audio de cada locutor antes de emitir un juicio.

### 3.2 Lógica de Decisión

- **Concordancia:** Si ambos métodos coinciden en su clasificación, se asigna el género con confianza máxima.
- **Discrepancia:** En los casos donde los métodos difieren, se deriva a evaluación humana por un investigador que escucha el audio del orador y determina la clasificación correcta.

### 3.3 Validación

De la totalidad de oradores clasificados, el 99.65% presentó concordancia entre ambos métodos. Los 42 casos de discrepancia (0.35%) fueron resueltos mediante evaluación humana, garantizando una clasificación correcta en la totalidad de la muestra.

---

## 4. Extracción de Variables Lingüísticas

### 4.1 Análisis Gramatical

Se emplea el modelo de procesamiento de lenguaje natural spaCy (en_core_web_lg) para extraer métricas gramaticales de cada intervención:

**Métricas básicas:**
- **Conteo de palabras:** Número total de tokens léxicos, excluyendo puntuación.
- **Velocidad de habla (WPM):** Palabras por minuto, calculada como el cociente entre el conteo de palabras y la duración de la intervención.
- **Diversidad léxica (TTR):** Ratio Type-Token, que mide la riqueza del vocabulario como la proporción de palabras únicas sobre el total de palabras. Valores cercanos a 1 indican mayor variedad léxica.

**Detección de preguntas:** Se identifican intervenciones interrogativas mediante la presencia de signos de interrogación o de palabras interrogativas (what, who, when, where, why, how) en posición inicial.

**Conteo de imperativos:** Se detectan verbos en modo imperativo mediante análisis morfosintáctico, identificando órdenes o instrucciones directas.

### 4.2 Análisis Pragmático

Se implementan diccionarios de patrones léxicos para identificar marcadores pragmáticos relevantes para las dinámicas de poder conversacional:

**Marcadores de mitigación (Hedges):** Expresiones que reducen la fuerza asertiva del enunciado, como "I think", "maybe", "perhaps", "probably", "sort of", "kind of", "seems", "actually", "just".

**Marcadores de disculpa:** Expresiones de disculpa explícita como "sorry", "I apologize", "excuse me", "pardon".

**Marcadores de cortesía:** Expresiones de deferencia como "please", "thank you", "kindly", "I appreciate".

**Marcadores de vulnerabilidad:** Admisiones de incertidumbre o falta de conocimiento como "I'm not sure", "I don't know", "I'm uncertain".

**Marcadores de acuerdo:** Expresiones de conformidad como "I agree", "absolutely", "exactly", "that's right", "good point".

**Marcadores de desacuerdo:** Expresiones de discrepancia como "I disagree", "I don't think", "however", "on the contrary", "not necessarily".

### 4.3 Autoridad Académica

**Uso de títulos:** Se detecta la mención de títulos académicos (Doctor, Professor, PhD) al referirse a otros participantes o a uno mismo.

**Atribución de ideas:** Se identifica el reconocimiento explícito de la autoría de ideas ajenas mediante expresiones como "as Dr. X said", "according to", "building on what X mentioned".

### 4.4 Análisis Afectivo

Se emplea la librería pysentimiento, basada en modelos BERT ajustados para español e inglés, para realizar:

**Análisis de sentimiento:** Clasificación de cada intervención en categorías de Positivo, Negativo o Neutro.

**Análisis de emociones:** Clasificación en siete categorías emocionales: alegría, ira, tristeza, miedo, sorpresa, asco y otros.

### 4.5 Índice de Eco (Echoing Score)

Se calcula el solapamiento léxico entre intervenciones consecutivas como proxy de la continuidad temática o repetición de ideas. El índice mide la similitud Jaccard entre los conjuntos de lemas de sustantivos y adjetivos (palabras de contenido) de la intervención actual y la precedente.

Un valor alto indica que el orador está retomando vocabulario del orador anterior, lo que puede interpretarse como validación, elaboración o, en ausencia de atribución, potencial apropiación de ideas.

---

## 5. Construcción de Índices Compuestos

### 5.1 Índice de Conflictividad

Suma ponderada que captura el nivel de confrontación de una intervención, considerando positivamente la presencia de desacuerdos, imperativos e interrupciones, y negativamente la presencia de acuerdos y cortesía. Permite identificar intervenciones de alta conflictividad (percentil 90) para análisis de patrones extremos.

### 5.2 Índice de Asertividad

Evalúa el estilo comunicativo directo versus mitigado de cada intervención, ponderando positivamente el uso de imperativos y desacuerdos directos, y negativamente el uso de hedges y disculpas.

### 5.3 Clima de Interrupción

Métrica de ventana rodante que mide la densidad de interrupciones en los 5 turnos precedentes a cada intervención. Permite clasificar el clima conversacional del momento como:
- **Calmado:** Baja densidad de interrupciones (<10%)
- **Hostil:** Alta densidad de interrupciones (>30%)
- **Neutro:** Valores intermedios

### 5.4 Detección de Apropiación de Ideas

Se operacionaliza la apropiación como la combinación de alto índice de eco (>0.3, indicando repetición de contenido del orador anterior) con ausencia de atribución explícita. Esta definición captura situaciones donde un orador retoma ideas de otro sin reconocer su origen.

### 5.5 Detección del Patrón Explicativo (Explaining Pattern)

Se implementa una regla heurística simétrica para detectar tanto el denominado "mansplaining" (hombre explicando a mujer) como el patrón inverso "womansplaining" (mujer explicando a hombre). Las condiciones son:

1. Intervención anterior de corta duración (<10 segundos)
2. Intervención actual de larga duración (>15 segundos)
3. Presencia de marcadores de corrección o desacuerdo
4. Transición de género cruzado (M→F o F→M)

La aplicación simétrica de la misma regla a ambas direcciones permite una comparación no sesgada de la prevalencia de este patrón en cada género.

---

## 6. Marco Estadístico

### 6.1 Verificación de Supuestos Paramétricos

Antes de seleccionar los tests estadísticos apropiados, se verifica sistemáticamente el cumplimiento de supuestos:

**Normalidad:** Se aplica el test de Shapiro-Wilk a cada variable continua, segregada por género. Dado que ninguna variable cumple el supuesto de normalidad (p < 0.001 en todos los casos), se opta por tests no paramétricos.

**Homocedasticidad:** Se aplica el test de Levene para evaluar la igualdad de varianzas entre grupos. Las violaciones detectadas refuerzan la decisión de emplear métodos no paramétricos.

### 6.2 Tests de Comparación de Grupos

**Variables continuas:** Se emplea el test de Mann-Whitney U, equivalente no paramétrico del t-test, que compara las distribuciones de rangos entre grupos sin asumir normalidad.

**Variables categóricas:** Se emplea el test Chi-cuadrado de independencia para evaluar la asociación entre género y variables binarias o nominales.

### 6.3 Tamaños del Efecto

Más allá de la significación estadística, se reportan medidas de magnitud del efecto:

**Hedges' g (variables continuas):** Versión corregida de la d de Cohen que ajusta por tamaño muestral. Se interpreta según los umbrales convencionales:
- < 0.2: Despreciable
- 0.2-0.5: Pequeño
- 0.5-0.8: Mediano
- > 0.8: Grande

**V de Cramér (variables categóricas):** Medida de asociación para tablas de contingencia, normalizada entre 0 y 1. Se interpreta como:
- < 0.1: Despreciable
- 0.1-0.3: Pequeño
- 0.3-0.5: Mediano
- > 0.5: Grande

### 6.4 Corrección por Múltiples Comparaciones

Dado el elevado número de tests realizados, se controla la tasa de falsos positivos mediante el método de Benjamini-Hochberg (False Discovery Rate). Este procedimiento ordena los p-valores y aplica un umbral adaptativo que mantiene la proporción esperada de falsos positivos por debajo del nivel α especificado (0.05).

Se reportan tanto los p-valores originales como los corregidos, indicando qué hallazgos sobreviven a la corrección.

### 6.5 Modelos de Efectos Mixtos

Las intervenciones están anidadas dentro de sesiones, violando el supuesto de independencia de las observaciones. Para corregir esta estructura jerárquica, se emplean modelos de efectos mixtos:

**Modelos lineales mixtos (LMM):** Para variables dependientes continuas, con el género como efecto fijo y la sesión como efecto aleatorio (intercepto aleatorio).

**Modelos lineales generalizados mixtos (GLMM):** Para variables dependientes binarias, empleando función de enlace logit.

Estos modelos permiten obtener estimaciones del efecto del género que controlan apropiadamente la correlación intraclase.

### 6.6 Coeficiente de Correlación Intraclase (ICC)

Se calcula el ICC para cada variable como la proporción de varianza explicada por las diferencias entre sesiones respecto a la varianza total. Un ICC bajo (<0.05) indica que la mayor parte de la variabilidad es intra-sesión (entre individuos), validando que los comportamientos observados son rasgos estables y no artefactos de sesiones particulares.

### 6.7 Intervalos de Confianza

Todas las estimaciones de diferencias entre grupos se acompañan de intervalos de confianza al 95%, calculados mediante:

**Bootstrap:** Para diferencias de medias, mediante remuestreo con reemplazo (10,000 iteraciones).

**Wilson Score:** Para proporciones, método que proporciona mejor cobertura que el intervalo de Wald, especialmente con proporciones extremas.

---

## 7. Framework de Análisis Multidimensional

El análisis se estructura en un marco de 23 módulos independientes que abordan sistemáticamente diferentes dimensiones de las dinámicas de género:

### 7.1 Rigor Estadístico (Módulos P01-P03)

- Cálculo exhaustivo de tamaños del efecto
- Aplicación de corrección FDR
- Estimación mediante modelos de efectos mixtos

### 7.2 Dinámicas Relacionales (Módulos P04-P06)

- Análisis de asimetría pregunta-respuesta: ¿Reciben las preguntas de mujeres respuestas de diferente extensión o calidad?
- Matrices de apropiación y amplificación: ¿Quién retoma ideas de quién, con o sin atribución?
- Posiciones de poder: ¿Quién abre y cierra las sesiones?

### 7.3 Modelado Predictivo (Módulos P07-P08)

- Modelo de regresión logística para predecir el éxito de las interrupciones, evaluando la contribución incremental del género como predictor
- Análisis de ICC para validar la estructura de los datos

### 7.4 Validación de Supuestos (Módulos P09-P11)

- Verificación automática de supuestos paramétricos
- Cálculo de correlaciones parciales controlando por confusores
- Construcción de intervalos de confianza completos

### 7.5 Análisis Contextual (Módulos P12-P15)

- Efecto del clima de sesión sobre las dinámicas de género
- Tests de interacción género × clima
- Análisis del potencial "backlash" (penalización) a la asertividad
- Evaluación simétrica del patrón explicativo

### 7.6 Patrones Temporales (Módulos P16-P18)

- Análisis del "sticky floor": ¿Cuánto tarda cada género en tomar la palabra por primera vez?
- Matriz de transiciones de turno: ¿Con qué probabilidad un género cede la palabra al otro?
- Evolución de métricas a lo largo de los cuartiles de la sesión

### 7.7 Robustez y Validación (Módulos P19-P23)

- Identificación y análisis de casos extremos
- Análisis de subgrupos por nivel de conflictividad
- Análisis de potencia post-hoc
- Análisis de sensibilidad ante variación de umbrales
- Comparación de modelos con y sin género como predictor

---

## 8. Validación y Control de Calidad

### 8.1 Análisis de Potencia Post-hoc

Se calcula la potencia estadística alcanzada para cada variable, determinando la probabilidad de detectar efectos de diferentes magnitudes dado el tamaño muestral disponible. Esto permite interpretar los hallazgos nulos: una potencia alta (>80%) para detectar efectos pequeños indica que la ausencia de significación refleja genuinamente la inexistencia de diferencias, mientras que una potencia baja sugiere que el estudio podría no haber detectado efectos existentes.

### 8.2 Análisis de Sensibilidad

Se evalúa la robustez de las métricas operacionalizadas mediante umbrales (como el índice de eco para apropiación) variando sistemáticamente dichos umbrales y observando la estabilidad de los resultados.

### 8.3 Comparación de Modelos

Para evaluar la contribución específica del género como predictor, se comparan modelos completos (con género) versus modelos reducidos (sin género) mediante métricas de discriminación (AUC) y calibración (accuracy). Una diferencia mínima entre modelos indica que el género no aporta información predictiva adicional más allá de las variables de estilo comunicativo.

---

## 9. Consideraciones Éticas y Limitaciones

### 9.1 Clasificación de Género

El sistema implementa una clasificación binaria (masculino/femenino) basada en características acústicas. Esta aproximación:
- No captura identidades de género no binarias
- Asume correspondencia entre características vocales y género
- Los casos ambiguos fueron resueltos mediante evaluación humana

### 9.2 Limitaciones de Generalización

- **Contexto específico:** Los datos provienen de debates científicos en medicina de cuidados críticos, lo que limita la extrapolación a otros contextos profesionales o culturales.
- **Idioma:** El análisis se restringe a sesiones en inglés.
- **Formato:** Los debates analizados tienen formato estructurado con paridad de género forzada en la composición de paneles, lo que puede diferir de contextos menos regulados.

### 9.3 Limitaciones Interpretativas

- **Causalidad:** El diseño observacional no permite establecer relaciones causales entre género y comportamiento comunicativo.
- **Variables no medidas:** Factores como la seniority académica, especialidad médica o experiencia previa en debates no fueron controlados.

---

## 10. Reproducibilidad

### 10.1 Stack Tecnológico

El pipeline se implementa en Python 3.10+ utilizando las siguientes herramientas principales:

| Componente | Herramienta |
|------------|-------------|
| Procesamiento de audio | ffmpeg, librosa |
| Normalización | pyloudnorm (EBU R128) |
| Diarización | Pyannote.audio 3.1 |
| Transcripción | OpenAI Whisper (large-v2) |
| Clasificación de género | Parselmouth, wav2vec2-xlsr-53 |
| Análisis lingüístico | spaCy (en_core_web_lg) |
| Análisis afectivo | pysentimiento |
| Análisis estadístico | statsmodels, scipy, scikit-learn |

### 10.2 Diccionario de Variables

Se proporciona un diccionario exhaustivo de todas las variables extraídas y calculadas, especificando para cada una:
- Nombre técnico
- Tipo de dato
- Rango de valores posibles
- Definición operacional
- Método de cálculo

Este diccionario, junto con el código del pipeline, garantiza la replicabilidad completa del estudio.

---

*Documento metodológico preparado según estándares de publicación científica para garantizar transparencia y replicabilidad.*