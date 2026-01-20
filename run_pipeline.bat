@echo off
REM ============================================================================
REM  Pipeline de Análisis de Disparidad de Género
REM  Instalación y Ejecución Automática
REM ============================================================================

echo.
echo ========================================================================
echo    PIPELINE DE ANALISIS DE DISPARIDAD DE GENERO
echo    Instalacion y Configuracion Automatica
echo ========================================================================
echo.

REM ============================================================================
REM PASO 1: Verificar Python
REM ============================================================================
echo [1/7] Verificando instalacion de Python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo.
    echo Por favor:
    echo 1. Descarga Python desde: https://www.python.org/downloads/
    echo 2. Durante la instalacion, marca "Add Python to PATH"
    echo 3. Reinicia esta ventana y ejecuta de nuevo este script
    echo.
    pause
    exit /b 1
)
echo    Python encontrado correctamente
echo.

REM ============================================================================
REM PASO 2: Crear entorno virtual
REM ============================================================================
echo [2/7] Creando entorno virtual de Python...
if not exist "venv" (
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: No se pudo crear el entorno virtual
        pause
        exit /b 1
    )
    echo    Entorno virtual creado exitosamente
) else (
    echo    Entorno virtual ya existe
)
echo.

REM ============================================================================
REM PASO 3: Activar entorno virtual
REM ============================================================================
echo [3/7] Activando entorno virtual...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: No se pudo activar el entorno virtual
    pause
    exit /b 1
)
echo    Entorno virtual activado
echo.

REM ============================================================================
REM PASO 4: Instalar dependencias
REM ============================================================================
echo [4/7] Instalando dependencias (esto puede tardar varios minutos)...
echo    Por favor espera, descargando e instalando paquetes...
echo.
pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: No se pudieron instalar las dependencias
    echo Intenta ejecutar manualmente: pip install -r requirements.txt
    pause
    exit /b 1
)
echo    Dependencias instaladas correctamente
echo.

REM ============================================================================
REM PASO 5: Configurar token de Hugging Face
REM ============================================================================
echo [5/7] Configurando token de Hugging Face...
if not exist ".env" (
    echo.
    echo    No se encontro archivo .env
    echo.
    echo    Necesitas un token de Hugging Face para usar el modelo de diarizacion.
    echo.
    echo    Pasos para obtener tu token:
    echo    1. Ve a: https://huggingface.co/settings/tokens
    echo    2. Inicia sesion o crea una cuenta gratuita
    echo    3. Crea un nuevo token (Read access es suficiente^)
    echo    4. Copia el token
    echo.
    echo    Tambien necesitas aceptar la licencia del modelo:
    echo    5. Ve a: https://huggingface.co/pyannote/speaker-diarization-3.1
    echo    6. Haz clic en "Agree and access repository"
    echo.
    set /p HF_TOKEN="    Pega tu token aqui y presiona Enter: "
    
    if "!HF_TOKEN!"=="" (
        echo.
        echo ERROR: No se proporciono un token
        echo El script no puede continuar sin un token de Hugging Face
        pause
        exit /b 1
    )
    
    echo HF_TOKEN=!HF_TOKEN! > .env
    echo.
    echo    Token guardado en .env
) else (
    echo    Archivo .env ya existe
)
echo.

REM ============================================================================
REM PASO 6: Verificar estructura de carpetas
REM ============================================================================
echo [6/7] Verificando estructura de carpetas...
if not exist "video" mkdir video
if not exist "fuentes" mkdir fuentes
if not exist "fuentes\audio" mkdir fuentes\audio
if not exist "fuentes\audio_normalized" mkdir fuentes\audio_normalized
if not exist "fuentes\diarization" mkdir fuentes\diarization
if not exist "fuentes\transcription" mkdir fuentes\transcription
if not exist "fuentes\gender_classification" mkdir fuentes\gender_classification
if not exist "final_reports" mkdir final_reports
if not exist "final_reports\csv" mkdir final_reports\csv
if not exist "final_reports\excel" mkdir final_reports\excel
if not exist "logs" mkdir logs
echo    Estructura de carpetas verificada
echo.

REM ============================================================================
REM PASO 7: Verificar videos
REM ============================================================================
echo [7/7] Verificando videos de entrada...
dir /b video\*.mp4 video\*.avi video\*.mov video\*.mkv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ADVERTENCIA: No se encontraron videos en la carpeta 'video\'
    echo.
    echo Por favor:
    echo 1. Coloca tus archivos de video en la carpeta 'video\'
    echo 2. Formatos soportados: .mp4, .avi, .mov, .mkv
    echo 3. Ejecuta de nuevo este script
    echo.
    pause
    exit /b 1
)
echo    Videos encontrados en carpeta 'video\'
echo.

REM ============================================================================
REM EJECUTAR PIPELINE
REM ============================================================================
echo ========================================================================
echo    CONFIGURACION COMPLETADA - INICIANDO PIPELINE
echo ========================================================================
echo.
echo El pipeline procesara todos los videos en la carpeta 'video\'
echo Esto puede tardar varias horas dependiendo del numero y duracion de videos.
echo.
echo Puedes cerrar esta ventana en cualquier momento con Ctrl+C
echo El progreso se guardara y podras continuar despues.
echo.
pause

echo.
echo ========================================================================
echo    EJECUTANDO PIPELINE
echo ========================================================================
echo.

echo [1/6] Extrayendo audio de videos...
python src\01_video_to_audio.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en script 01 - Extraccion de audio
    pause
    exit /b 1
)

echo.
echo [2/6] Normalizando audio...
python src\02_normalize_audio.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en script 02 - Normalizacion
    pause
    exit /b 1
)

echo.
echo [3/6] Realizando diarizacion de speakers...
python src\03_diarization.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en script 03 - Diarizacion
    pause
    exit /b 1
)

echo.
echo [4/6] Transcribiendo audio...
python src\04_transcription.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en script 04 - Transcripcion
    pause
    exit /b 1
)

echo.
echo [5/6] Clasificando genero de speakers...
python src\05_gender_classification.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en script 05 - Clasificacion de genero
    pause
    exit /b 1
)

echo.
echo [6/6] Generando reportes finales...
python src\06_final_report.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en script 06 - Generacion de reportes
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo    PIPELINE COMPLETADO EXITOSAMENTE
echo ========================================================================
echo.
echo Reportes generados en:
echo   - CSV:   final_reports\csv\
echo   - Excel: final_reports\excel\
echo.
echo Logs detallados en: logs\
echo.
pause
