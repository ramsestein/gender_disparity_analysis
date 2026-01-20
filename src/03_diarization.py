#!/usr/bin/env python3
"""
Script 3: Diarización con PyAnnote Audio 3.1
Identifica quién habla cuándo en cada audio
Detecta overlaps (múltiples personas hablando simultáneamente)
Usa audios normalizados de audio_normalized/
"""

import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import torch
from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np
import tempfile
import warnings
warnings.filterwarnings('ignore')

class Diarizer:
    def __init__(self, audio_dir="fuentes/audio_normalized", output_dir="fuentes/diarization", logs_dir="logs", hf_token=None):
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)
        
        # Crear directorios
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Verificar audio_dir
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"No existe el directorio: {self.audio_dir}")
        
        # Token de Hugging Face
        if hf_token is None:
            hf_token = os.getenv('HF_TOKEN')
            if hf_token is None:
                raise ValueError(
                    "Token de Hugging Face no encontrado.\n"
                    "Crea un archivo .env con: HF_TOKEN=tu_token\n"
                    "O pasa el token como argumento: hf_token='tu_token'"
                )
        self.hf_token = hf_token
        
        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"03_diarization_{timestamp}.log"
        
        # Pipeline (se cargará después)
        self.pipeline = None
        self.device = None
    
    def log(self, message):
        """Escribe en consola y archivo de log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def setup_gpu(self):
        """Configura GPU si está disponible"""
        # Forzar CPU debido a detección incorrecta de RTX 5060
        self.device = torch.device("cpu")
        self.log("🖥️  Usando CPU (forzado)")
        return self.device
    
    def load_pipeline(self):
        """Carga el modelo de diarización"""
        self.log("Cargando modelo PyAnnote 3.1...")
        
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token
            )
            
            # Siempre en CPU
            self.log("✓ Modelo cargado en CPU")
            return True
            
        except Exception as e:
            self.log(f"✗ Error cargando modelo: {str(e)}")
            self.log("  Verifica:")
            self.log("  1. Token de Hugging Face correcto")
            self.log("  2. Aceptaste licencia en: https://huggingface.co/pyannote/speaker-diarization-3.1")
            return False
    
    def get_audio_files(self):
        """Obtiene lista de audios WAV"""
        audio_files = list(self.audio_dir.glob("*.wav"))
        return sorted(audio_files)
    
    def load_audio(self, audio_path):
        """
        Carga audio usando soundfile y lo convierte al formato de PyAnnote
        
        Returns:
            dict con waveform (tensor) y sample_rate
        """
        try:
            # Cargar audio con soundfile
            waveform, sample_rate = sf.read(str(audio_path), dtype='float32')
            
            # Convertir a formato PyAnnote (canal, tiempo)
            if len(waveform.shape) == 1:
                # Mono: añadir dimensión de canal
                waveform = waveform.reshape(1, -1)
            else:
                # Stereo o multi-canal: transponer
                waveform = waveform.T
            
            # Convertir a tensor de PyTorch
            waveform_tensor = torch.from_numpy(waveform).float()
            
            return {
                'waveform': waveform_tensor,
                'sample_rate': sample_rate
            }
            
        except Exception as e:
            raise Exception(f"Error cargando audio: {str(e)}")
    
    def extract_segments_from_rttm(self, diarization, audio_name):
        """
        Extrae segmentos del objeto de diarización de PyAnnote
        
        Returns:
            list de dicts con start, end, speaker
        """
        segments = []
        
        try:
            # DiarizeOutput tiene un atributo speaker_diarization que contiene la Annotation
            if hasattr(diarization, 'speaker_diarization'):
                annotation = diarization.speaker_diarization
                self.log(f"  Accediendo a speaker_diarization...")
            else:
                annotation = diarization
            
            # Iterar sobre la anotación
            for segment, track, speaker in annotation.itertracks(yield_label=True):
                segments.append({
                    'start': round(segment.start, 2),
                    'end': round(segment.end, 2),
                    'speaker': speaker
                })
            
            self.log(f"  ✓ Extraídos {len(segments)} segmentos")
            return segments
            
        except Exception as e:
            self.log(f"  ✗ Error extrayendo segmentos: {str(e)}")
            self.log(f"  Tipo de objeto: {type(diarization).__name__}")
            if hasattr(diarization, 'speaker_diarization'):
                self.log(f"  Tipo de speaker_diarization: {type(diarization.speaker_diarization).__name__}")
            raise
    
    def diarize_audio(self, audio_path):
        """
        Diariza un audio individual
        
        Returns:
            dict con resultado
        """
        audio_name = audio_path.stem
        output_csv = self.output_dir / f"{audio_name}.csv"
        
        # Si ya existe, saltar
        if output_csv.exists():
            df = pd.read_csv(output_csv)
            n_speakers = df['speaker'].nunique()
            self.log(f"⊘ Ya existe: {audio_name}.csv ({n_speakers} speakers) - Saltando")
            return {
                'audio': audio_name,
                'status': 'skipped',
                'n_speakers': n_speakers,
                'n_segments': len(df)
            }
        
        try:
            # Cargar audio pre-procesado
            self.log(f"  Cargando audio...")
            audio_dict = self.load_audio(audio_path)
            
            # Ejecutar diarización
            self.log(f"  Diarizando...")
            diarization = self.pipeline(audio_dict)
            
            # Debug: mostrar tipo de objeto
            self.log(f"  Tipo diarization: {type(diarization).__name__}")
            
            # Extraer segmentos usando RTTM directamente (método más robusto)
            self.log(f"  Extrayendo segmentos...")
            segments = self.extract_segments_from_rttm(diarization, audio_name)
            
            # Crear DataFrame
            df = pd.DataFrame(segments)
            
            if len(df) == 0:
                raise Exception("No se detectaron speakers")
            
            # Ordenar por tiempo
            df = df.sort_values('start').reset_index(drop=True)
            
            # Guardar
            df.to_csv(output_csv, index=False)
            
            # Estadísticas
            n_speakers = df['speaker'].nunique()
            n_segments = len(df)
            duration = df['end'].max()
            
            # Detectar overlaps
            overlaps = []
            for i in range(len(df) - 1):
                if df.iloc[i]['end'] > df.iloc[i + 1]['start']:
                    overlap_duration = df.iloc[i]['end'] - df.iloc[i + 1]['start']
                    overlaps.append(overlap_duration)
            
            n_overlaps = len(overlaps)
            total_overlap = sum(overlaps) if overlaps else 0
            
            self.log(f"✓ Guardado: {audio_name}.csv")
            self.log(f"  Speakers: {n_speakers} | Segmentos: {n_segments} | Overlaps: {n_overlaps}")
            self.log(f"  Duración: {duration/60:.1f} min | Overlap total: {total_overlap:.1f}s")
            
            return {
                'audio': audio_name,
                'status': 'success',
                'n_speakers': n_speakers,
                'n_segments': n_segments,
                'n_overlaps': n_overlaps,
                'total_overlap_seconds': round(total_overlap, 2),
                'duration_seconds': round(duration, 2)
            }
            
        except Exception as e:
            self.log(f"✗ Error: {audio_name} - {str(e)}")
            
            # Debug adicional
            import traceback
            self.log(f"  Traceback completo:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self.log(f"    {line}")
            
            return {
                'audio': audio_name,
                'status': 'error',
                'error': str(e)
            }
        
        finally:
            # Limpiar cache de GPU (aunque usemos CPU)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def process_all(self):
        """Procesa todos los audios"""
        self.log("="*70)
        self.log("SCRIPT 3: DIARIZACIÓN CON PYANNOTE")
        self.log("="*70)
        
        # Setup GPU
        device = self.setup_gpu()
        
        # Cargar modelo
        if not self.load_pipeline():
            self.log("\n❌ ABORTANDO: No se pudo cargar el modelo")
            return None
        
        # Obtener audios
        audio_files = self.get_audio_files()
        n_audios = len(audio_files)
        
        if n_audios == 0:
            self.log(f"\n✗ No se encontraron audios WAV en: {self.audio_dir.absolute()}")
            self.log("Ejecuta primero los Scripts 1 y 2:")
            self.log("  1. 01_extract_audio.py")
            self.log("  2. 02_normalize_audio.py")
            return None
        
        self.log(f"\n📁 Directorio entrada: {self.audio_dir.absolute()}")
        self.log(f"📁 Directorio salida: {self.output_dir.absolute()}")
        self.log(f"📊 Audios encontrados: {n_audios}")
        self.log(f"🎮 Dispositivo: {device.type.upper()}")
        self.log("-"*70)
        
        # Procesar cada audio
        results = []
        start_time = datetime.now()
        
        for i, audio_path in enumerate(audio_files, 1):
            self.log(f"\n[{i}/{n_audios}] Procesando: {audio_path.name}")
            
            audio_start = datetime.now()
            result = self.diarize_audio(audio_path)
            audio_elapsed = (datetime.now() - audio_start).total_seconds()
            
            result['processing_time_seconds'] = round(audio_elapsed, 2)
            results.append(result)
            
            # Estimación tiempo restante
            if result['status'] != 'skipped':
                elapsed = (datetime.now() - start_time).total_seconds()
                completed = sum(1 for r in results if r['status'] in ['success', 'error'])
                if completed > 0:
                    avg_time = elapsed / completed
                    remaining = (n_audios - i) * avg_time
                    eta_min = remaining / 60
                    
                    self.log(f"⏱️  Tiempo este audio: {audio_elapsed/60:.1f} min")
                    if i < n_audios:
                        self.log(f"⏱️  ETA restante: {eta_min:.1f} minutos ({n_audios - i} audios)")
        
        # Resumen final
        total_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        # Estadísticas agregadas
        total_speakers = sum(r.get('n_speakers', 0) for r in results if r['status'] in ['success', 'skipped'])
        total_overlaps = sum(r.get('n_overlaps', 0) for r in results if r['status'] in ['success', 'skipped'])
        
        self.log("\n" + "="*70)
        self.log("RESUMEN FINAL")
        self.log("="*70)
        self.log(f"📊 Total audios: {n_audios}")
        self.log(f"✅ Exitosos: {successful}")
        self.log(f"⊘  Saltados: {skipped}")
        self.log(f"❌ Errores: {failed}")
        self.log(f"⏱️  Tiempo total: {total_time/60:.1f} minutos")
        
        if successful > 0:
            avg_time = total_time / successful
            self.log(f"⏱️  Promedio: {avg_time/60:.1f} minutos/audio")
        
        self.log(f"\n📈 Estadísticas:")
        self.log(f"   Speakers totales detectados: {total_speakers}")
        self.log(f"   Overlaps totales detectados: {total_overlaps}")
        
        if failed > 0:
            self.log(f"\n⚠️  Hubo {failed} errores. Revisa el log para detalles.")
        
        self.log("="*70)
        
        # Guardar resultados
        results_file = self.logs_dir / "03_diarization_results.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'device': device.type,
            'total_audios': n_audios,
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'total_time_seconds': round(total_time, 2),
            'avg_time_per_audio': round(total_time / max(successful, 1), 2),
            'total_speakers_detected': total_speakers,
            'total_overlaps_detected': total_overlaps,
            'results': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.log(f"\n💾 Resultados guardados en: {results_file.absolute()}")
        self.log(f"📄 Log completo en: {self.log_file.absolute()}")
        
        return summary


def load_env_file():
    """Carga variables de entorno desde archivo .env"""
    env_file = Path(".env")
    
    if not env_file.exists():
        return False
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remover comillas si existen
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ[key] = value
        return True
    except Exception as e:
        print(f"⚠️  Error leyendo .env: {e}")
        return False


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("   SCRIPT 3: DIARIZACIÓN CON PYANNOTE")
    print("="*70 + "\n")
    
    # Cargar variables de entorno desde .env
    print("🔍 Verificando configuración...")
    
    env_file = Path(".env")
    print(f"   Directorio actual: {Path.cwd()}")
    print(f"   Archivo .env existe: {env_file.exists()}")
    
    if env_file.exists():
        print(f"   Ubicación .env: {env_file.absolute()}")
        if load_env_file():
            token = os.getenv('HF_TOKEN')
            if token:
                print(f"   ✓ Token cargado: {token[:10]}...")
            else:
                print("   ✗ Token no encontrado en .env")
                print("\n❌ Verifica que .env contenga:")
                print("   HF_TOKEN=tu_token_aqui")
                return
        else:
            print("   ✗ Error cargando .env")
            return
    else:
        print("   ✗ Archivo .env NO encontrado")
        print("\n❌ Crea archivo .env en la raíz del proyecto con:")
        print("   HF_TOKEN=tu_token_aqui")
        print("\nObtén tu token en: https://huggingface.co/settings/tokens")
        return
    
    print()
    
    try:
        diarizer = Diarizer(
            audio_dir="fuentes/audio_normalized",
            output_dir="fuentes/diarization",
            logs_dir="logs"
        )
        
        summary = diarizer.process_all()
        
        if summary is None:
            print("\n❌ El script no pudo completarse. Revisa los errores arriba.")
            return
        
        print("\n✅ Script completado exitosamente!")
        
        if summary['failed'] > 0:
            print(f"⚠️  Atención: {summary['failed']} audios fallaron")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrumpido por el usuario (Ctrl+C)")
    
    except Exception as e:
        print(f"\n❌ Error fatal: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()