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

2. **Coloca tus videos** en la carpeta `video/` dentro del proyecto. Debes crearla y llamarla así en el proyecto

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

## 📖 Cómo Citar Este Proyecto

Si utilizas este software en tu investigación o publicación académica, por favor cítalo de la siguiente manera:

### Formato APA (7ª edición)
```
[Autor/es]. (2026). Gender Disparity Analysis Pipeline (Versión 1.0) [Software]. 
GitHub. https://github.com/ramsestein/gender_disparity_analysis
```

### Formato BibTeX
```bibtex
@software{gender_disparity_pipeline_2026,
  author = {Ramsés Marrero Garcia},
  title = {Gender Disparity Analysis Pipeline: Automated Gender Disparity Analysis in Academic Conferences},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/[usuario]/gender_disparity_analysis},
  version = {1.0}
}
```

## 📚 Referencias Bibliográficas

Este proyecto utiliza las siguientes herramientas y bibliotecas de código abierto:

### Herramientas Principales

**Pyannote Audio - Diarización de Speakers**
```
Bredin, H., & Laurent, A. (2021). End-to-end speaker segmentation for overlap-aware 
resegmentation. In Proc. Interspeech 2021 (pp. 3111-3115).
DOI: 10.21437/Interspeech.2021-560
```
```bibtex
@inproceedings{Bredin2021,
  author = {Hervé Bredin and Antoine Laurent},
  title = {{End-to-end speaker segmentation for overlap-aware resegmentation}},
  booktitle = {Proc. Interspeech 2021},
  year = {2021},
  pages = {3111--3115},
  doi = {10.21437/Interspeech.2021-560}
}
```

**OpenAI Whisper - Transcripción Automática**
```
Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). 
Robust speech recognition via large-scale weak supervision. arXiv preprint arXiv:2212.04356.
```
```bibtex
@article{radford2022whisper,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```

**Mozilla Common Voice - Clasificación de Género**
```
Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler, M., Meyer, J., ... & Weber, G. (2020). 
Common Voice: A massively-multilingual speech corpus. In Proceedings of the 12th Language 
Resources and Evaluation Conference (pp. 4218-4222).
```
```bibtex
@inproceedings{ardila2020common,
  title={Common Voice: A massively-multilingual speech corpus},
  author={Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
  booktitle={Proceedings of the 12th Language Resources and Evaluation Conference},
  pages={4218--4222},
  year={2020}
}
```

**Modelo Pre-entrenado:** `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech`
```
Modelo basado en Wav2Vec2-XLSR-53 fine-tuneado para clasificación de género.
Disponible en: https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech
```

### Bibliotecas de Procesamiento de Audio

**librosa - Análisis de Audio**
```
McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015). 
librosa: Audio and music signal analysis in python. In Proceedings of the 14th Python in Science 
Conference (Vol. 8, pp. 18-25).
```
```bibtex
@inproceedings{mcfee2015librosa,
  title={librosa: Audio and music signal analysis in python},
  author={McFee, Brian and Raffel, Colin and Liang, Dawen and Ellis, Daniel PW and McVicar, Matt and Battenberg, Eric and Nieto, Oriol},
  booktitle={Proceedings of the 14th python in science conference},
  volume={8},
  pages={18--25},
  year={2015}
}
```

