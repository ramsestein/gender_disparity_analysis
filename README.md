# Gender Disparity Analysis Pipeline

**Análisis automatizado de disparidad de género en conferencias y congresos**

Este software procesa grabaciones de video de conferencias, congresos y eventos académicos para analizar automáticamente la participación por género, generando métricas sobre tiempo de habla, interrupciones y dinámicas conversacionales.

## 📜 Licencia

Este proyecto se distribuye bajo **Licencia MIT** (ver archivo [LICENSE](LICENSE)).

El código se libera públicamente para permitir:
- ✅ Auditorías independientes de metodologías
- ✅ Reproducibilidad de estudios
- ✅ Mejora continua por la comunidad
- ✅ Acceso democrático a herramientas de análisis de género

## 📋 Características

- **Extracción de audio** de archivos de video
- **Normalización** de niveles de audio
- **Diarización** de speakers (quién habla cuándo)
- **Transcripción** con Whisper (speech-to-text)
- **Clasificación de género** con enfoque híbrido (pitch + modelo pre-entrenado)
- **Reportes completos** en CSV y Excel con análisis de overlaps e interrupciones

## 🚀 Instalación Rápida (Para Usuarios Sin Experiencia)

### Requisitos Previos
- **Python 3.8 o superior** instalado en tu sistema
  - Windows: Descarga desde [python.org](https://www.python.org/downloads/)
  - Mac: `brew install python3` o descarga desde python.org
  - Linux: `sudo apt-get install python3 python3-venv python3-pip`

### Instalación Automática

1. **Descarga o clona este proyecto**

2. **Coloca tus videos** en la carpeta `video/`

3. **Ejecuta el script de instalación y pipeline:**

**Windows:**
```bash
run_pipeline.bat
```

**Linux/Mac:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

El script automáticamente:
- ✅ Crea el entorno virtual de Python
- ✅ Instala todas las dependencias necesarias
- ✅ Te pedirá tu token de Hugging Face (te guiará en cómo obtenerlo)
- ✅ Ejecuta todo el pipeline completo

**¡Eso es todo!** El script te guiará paso a paso.

## 📁 Estructura del Proyecto

```
gender_diaparity/
├── video/                      # Videos de entrada
├── fuentes/                    # Carpetas de trabajo (generadas automáticamente)
│   ├── audio/                  # Audio extraído
│   ├── audio_normalized/       # Audio normalizado
│   ├── diarization/           # Segmentos de speakers
│   ├── transcription/         # Transcripciones
│   └── gender_classification/ # Clasificación de género
├── final_reports/             # Reportes finales
│   ├── csv/                   # Reportes en CSV
│   └── excel/                 # Reportes en Excel
├── logs/                      # Logs de procesamiento
├── src/                       # Scripts del pipeline
│   ├── 01_video_to_audio.py
│   ├── 02_normalize_audio.py
│   ├── 03_diarization.py
│   ├── 04_transcription.py
│   ├── 05_gender_classification.py
│   └── 06_final_report.py
├── run_pipeline.bat           # Ejecutar pipeline (Windows)
├── run_pipeline.sh            # Ejecutar pipeline (Linux/Mac)
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## 🎯 Uso

### Opción 1: Ejecutar Pipeline Completo (Recomendado)

**Windows:**
```bash
run_pipeline.bat
```

**Linux/Mac:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Opción 2: Ejecutar Scripts Individuales

```bash
# 1. Extraer audio de videos
python src/01_video_to_audio.py

# 2. Normalizar audio
python src/02_normalize_audio.py

# 3. Diarización de speakers
python src/03_diarization.py

# 4. Transcripción
python src/04_transcription.py

# 5. Clasificación de género
python src/05_gender_classification.py

# 6. Generar reportes finales
python src/06_final_report.py
```

## 📊 Formato del Reporte Final

Los reportes incluyen las siguientes columnas:

| Columna | Descripción |
|---------|-------------|
| `intervention_id` | ID único de la intervención |
| `start_time` | Tiempo de inicio (segundos) |
| `end_time` | Tiempo de fin (segundos) |
| `duration` | Duración de la intervención (segundos) |
| `speaker` | ID del speaker (SPEAKER_00, SPEAKER_01, etc.) |
| `gender` | Género clasificado (male/female) |
| `gender_confidence` | Confianza de la clasificación (0-1) |
| `text` | Transcripción del texto |
| `has_overlap` | Si hay overlap con otro speaker |
| `overlap_duration` | Duración del overlap (segundos) |
| `interrupts_previous` | Si interrumpe al speaker anterior |
| `interrupted_by_next` | Si es interrumpido por el siguiente |
| `turn_number` | Número de turno en la conversación |

## 🔧 Configuración Avanzada

### Cambiar modelo de Whisper

En `src/04_transcription.py`, línea ~443:
```python
model_size="base"  # Opciones: tiny, base, small, medium, large
```

- `tiny`: Más rápido, menor precisión
- `base`: Balance (por defecto)
- `small`: Buena precisión
- `medium`: Alta precisión
- `large`: Máxima precisión, más lento

### Desactivar reducción de ruido

En `src/02_normalize_audio.py`, línea ~273:
```python
apply_noise_reduction=False  # Cambiar a False
```

## 📈 Rendimiento

Tiempos aproximados por audio de 20 minutos (CPU):

| Script | Tiempo |
|--------|--------|
| 01 - Extracción | ~2 min |
| 02 - Normalización | ~50s |
| 03 - Diarización | ~20 min |
| 04 - Transcripción | ~30s |
| 05 - Género | ~2 min |
| 06 - Reportes | 1s |
| **Total** | **~25 min** |

## 🎓 Métodos de Clasificación de Género

El sistema usa un **enfoque híbrido**:

1. **Análisis de Pitch (F0)**
   - Rápido, baseline
   - Male: < 165 Hz
   - Female: ≥ 165 Hz

2. **Modelo Pre-entrenado (Wav2Vec2)**
   - Mayor precisión (~95%+)
   - Decisión final

**Lógica de decisión:**
- Si ambos coinciden → Alta confianza
- Si difieren → Usar modelo (más preciso)

## 🐛 Solución de Problemas

### Error: "HF_TOKEN not found"
- Verifica que el archivo `.env` existe en la raíz
- Verifica que contiene `HF_TOKEN=tu_token`

### Error: "No module named 'pyannote'"
```bash
pip install -r requirements.txt
```

### Error: "GPU not available"
- El pipeline funciona en CPU por defecto
- No es necesario GPU

## 📝 Licencia

Este proyecto es para uso académico/investigación.

## 👥 Contribuciones

Para reportar bugs o sugerir mejoras, contacta al equipo de desarrollo.
