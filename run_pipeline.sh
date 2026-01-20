#!/bin/bash
# ============================================================================
#  Pipeline de Análisis de Disparidad de Género
#  Instalación y Ejecución Automática
# ============================================================================

echo ""
echo "========================================================================"
echo "   PIPELINE DE ANÁLISIS DE DISPARIDAD DE GÉNERO"
echo "   Instalación y Configuración Automática"
echo "========================================================================"
echo ""

# ============================================================================
# PASO 1: Verificar Python
# ============================================================================
echo "[1/7] Verificando instalación de Python..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "ERROR: Python 3 no está instalado"
    echo ""
    echo "Por favor instala Python 3:"
    echo "  - Ubuntu/Debian: sudo apt-get install python3 python3-venv python3-pip"
    echo "  - macOS: brew install python3"
    echo "  - Fedora: sudo dnf install python3"
    echo ""
    exit 1
fi
echo "   Python encontrado correctamente"
echo ""

# ============================================================================
# PASO 2: Crear entorno virtual
# ============================================================================
echo "[2/7] Creando entorno virtual de Python..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: No se pudo crear el entorno virtual"
        exit 1
    fi
    echo "   Entorno virtual creado exitosamente"
else
    echo "   Entorno virtual ya existe"
fi
echo ""

# ============================================================================
# PASO 3: Activar entorno virtual
# ============================================================================
echo "[3/7] Activando entorno virtual..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: No se pudo activar el entorno virtual"
    exit 1
fi
echo "   Entorno virtual activado"
echo ""

# ============================================================================
# PASO 4: Instalar dependencias
# ============================================================================
echo "[4/7] Instalando dependencias (esto puede tardar varios minutos)..."
echo "   Por favor espera, descargando e instalando paquetes..."
echo ""
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "ERROR: No se pudieron instalar las dependencias"
    echo "Intenta ejecutar manualmente: pip install -r requirements.txt"
    exit 1
fi
echo "   Dependencias instaladas correctamente"
echo ""

# ============================================================================
# PASO 5: Configurar token de Hugging Face
# ============================================================================
echo "[5/7] Configurando token de Hugging Face..."
if [ ! -f ".env" ]; then
    echo ""
    echo "   No se encontró archivo .env"
    echo ""
    echo "   Necesitas un token de Hugging Face para usar el modelo de diarización."
    echo ""
    echo "   Pasos para obtener tu token:"
    echo "   1. Ve a: https://huggingface.co/settings/tokens"
    echo "   2. Inicia sesión o crea una cuenta gratuita"
    echo "   3. Crea un nuevo token (Read access es suficiente)"
    echo "   4. Copia el token"
    echo ""
    echo "   También necesitas aceptar la licencia del modelo:"
    echo "   5. Ve a: https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "   6. Haz clic en 'Agree and access repository'"
    echo ""
    read -p "   Pega tu token aquí y presiona Enter: " HF_TOKEN
    
    if [ -z "$HF_TOKEN" ]; then
        echo ""
        echo "ERROR: No se proporcionó un token"
        echo "El script no puede continuar sin un token de Hugging Face"
        exit 1
    fi
    
    echo "HF_TOKEN=$HF_TOKEN" > .env
    echo ""
    echo "   Token guardado en .env"
else
    echo "   Archivo .env ya existe"
fi
echo ""

# ============================================================================
# PASO 6: Verificar estructura de carpetas
# ============================================================================
echo "[6/7] Verificando estructura de carpetas..."
mkdir -p video
mkdir -p fuentes/audio
mkdir -p fuentes/audio_normalized
mkdir -p fuentes/diarization
mkdir -p fuentes/transcription
mkdir -p fuentes/gender_classification
mkdir -p final_reports/csv
mkdir -p final_reports/excel
mkdir -p logs
echo "   Estructura de carpetas verificada"
echo ""

# ============================================================================
# PASO 7: Verificar videos
# ============================================================================
echo "[7/7] Verificando videos de entrada..."
if [ -z "$(ls -A video/*.{mp4,avi,mov,mkv} 2>/dev/null)" ]; then
    echo ""
    echo "ADVERTENCIA: No se encontraron videos en la carpeta 'video/'"
    echo ""
    echo "Por favor:"
    echo "1. Coloca tus archivos de video en la carpeta 'video/'"
    echo "2. Formatos soportados: .mp4, .avi, .mov, .mkv"
    echo "3. Ejecuta de nuevo este script"
    echo ""
    exit 1
fi
echo "   Videos encontrados en carpeta 'video/'"
echo ""

# ============================================================================
# EJECUTAR PIPELINE
# ============================================================================
echo "========================================================================"
echo "   CONFIGURACIÓN COMPLETADA - INICIANDO PIPELINE"
echo "========================================================================"
echo ""
echo "El pipeline procesará todos los videos en la carpeta 'video/'"
echo "Esto puede tardar varias horas dependiendo del número y duración de videos."
echo ""
echo "Puedes cerrar esta ventana en cualquier momento con Ctrl+C"
echo "El progreso se guardará y podrás continuar después."
echo ""
read -p "Presiona Enter para continuar..."

echo ""
echo "========================================================================"
echo "   EJECUTANDO PIPELINE"
echo "========================================================================"
echo ""

echo "[1/6] Extrayendo audio de videos..."
python src/01_video_to_audio.py
if [ $? -ne 0 ]; then
    echo "ERROR en script 01 - Extracción de audio"
    exit 1
fi

echo ""
echo "[2/6] Normalizando audio..."
python src/02_normalize_audio.py
if [ $? -ne 0 ]; then
    echo "ERROR en script 02 - Normalización"
    exit 1
fi

echo ""
echo "[3/6] Realizando diarización de speakers..."
python src/03_diarization.py
if [ $? -ne 0 ]; then
    echo "ERROR en script 03 - Diarización"
    exit 1
fi

echo ""
echo "[4/6] Transcribiendo audio..."
python src/04_transcription.py
if [ $? -ne 0 ]; then
    echo "ERROR en script 04 - Transcripción"
    exit 1
fi

echo ""
echo "[5/6] Clasificando género de speakers..."
python src/05_gender_classification.py
if [ $? -ne 0 ]; then
    echo "ERROR en script 05 - Clasificación de género"
    exit 1
fi

echo ""
echo "[6/6] Generando reportes finales..."
python src/06_final_report.py
if [ $? -ne 0 ]; then
    echo "ERROR en script 06 - Generación de reportes"
    exit 1
fi

echo ""
echo "========================================================================"
echo "   PIPELINE COMPLETADO EXITOSAMENTE"
echo "========================================================================"
echo ""
echo "Reportes generados en:"
echo "  - CSV:   final_reports/csv/"
echo "  - Excel: final_reports/excel/"
echo ""
echo "Logs detallados en: logs/"
echo ""
