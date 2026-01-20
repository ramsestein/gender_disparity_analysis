#!/usr/bin/env python3
"""
Script 4: Transcripción con Speaker Labels
Transcribe cada segmento de audio usando Whisper
Asocia las transcripciones con los speakers de la diarización
"""

import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import soundfile as sf
import whisper
import warnings
warnings.filterwarnings('ignore')


class Transcriber:
    def __init__(
        self, 
        audio_dir="fuentes/audio_normalized", 
        diarization_dir="fuentes/diarization",
        output_dir="fuentes/transcription", 
        logs_dir="logs",
        model_size="base"
    ):
        self.audio_dir = Path(audio_dir)
        self.diarization_dir = Path(diarization_dir)
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)
        self.model_size = model_size
        
        # Crear directorios
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Verificar directorios
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"No existe el directorio: {self.audio_dir}")
        if not self.diarization_dir.exists():
            raise FileNotFoundError(f"No existe el directorio: {self.diarization_dir}")
        
        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"04_transcription_{timestamp}.log"
        
        # Modelo Whisper (se cargará después)
        self.model = None
    
    def log(self, message):
        """Escribe en consola y archivo de log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def load_whisper_model(self):
        """Carga el modelo Whisper"""
        self.log(f"Cargando modelo Whisper '{self.model_size}'...")
        self.log("  (Primera vez puede tardar varios minutos descargando el modelo)")
        
        try:
            self.model = whisper.load_model(self.model_size)
            self.log(f"✓ Modelo Whisper '{self.model_size}' cargado")
            return True
        except Exception as e:
            self.log(f"✗ Error cargando modelo: {str(e)}")
            return False
    
    def get_audio_files(self):
        """Obtiene lista de audios WAV que tienen diarización"""
        audio_files = []
        
        for diar_file in sorted(self.diarization_dir.glob("*.csv")):
            audio_name = diar_file.stem
            audio_path = self.audio_dir / f"{audio_name}.wav"
            
            if audio_path.exists():
                audio_files.append({
                    'audio_path': audio_path,
                    'diar_path': diar_file,
                    'name': audio_name
                })
        
        return audio_files
    
    def extract_audio_segment(self, audio_path, start_time, end_time):
        """
        Extrae un segmento de audio entre start_time y end_time (en segundos)
        
        Returns:
            numpy array con el audio del segmento
        """
        try:
            # Cargar audio completo
            waveform, sample_rate = sf.read(str(audio_path), dtype='float32')
            
            # Convertir tiempos a samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Asegurar que no nos pasamos del final
            end_sample = min(end_sample, len(waveform))
            
            # Extraer segmento
            if len(waveform.shape) == 1:
                # Mono
                segment = waveform[start_sample:end_sample]
            else:
                # Stereo: convertir a mono promediando canales
                segment = waveform[start_sample:end_sample].mean(axis=1)
            
            return segment
            
        except Exception as e:
            raise Exception(f"Error extrayendo segmento: {str(e)}")
    
    def transcribe_segment(self, audio_segment, language="en"):
        """
        Transcribe un segmento de audio usando Whisper
        
        Returns:
            str con el texto transcrito
        """
        try:
            # Whisper espera audio en formato float32 entre -1 y 1
            # Ya está en ese formato por soundfile
            
            # Transcribir
            result = self.model.transcribe(
                audio_segment,
                language=language,
                task="transcribe",
                fp16=False  # Usar float32 en CPU
            )
            
            return result['text'].strip()
            
        except Exception as e:
            raise Exception(f"Error transcribiendo: {str(e)}")
    
    def transcribe_audio(self, audio_info):
        """
        Transcribe un audio completo usando su diarización
        
        Returns:
            dict con resultado
        """
        audio_name = audio_info['name']
        audio_path = audio_info['audio_path']
        diar_path = audio_info['diar_path']
        
        output_csv = self.output_dir / f"{audio_name}.csv"
        
        # Si ya existe, saltar
        if output_csv.exists():
            df = pd.read_csv(output_csv)
            n_segments = len(df)
            self.log(f"⊘ Ya existe: {audio_name}.csv ({n_segments} segmentos) - Saltando")
            return {
                'audio': audio_name,
                'status': 'skipped',
                'n_segments': n_segments
            }
        
        try:
            # Cargar diarización
            self.log(f"  Cargando diarización...")
            diar_df = pd.read_csv(diar_path)
            n_segments = len(diar_df)
            self.log(f"  {n_segments} segmentos a transcribir")
            
            # Transcribir cada segmento
            transcriptions = []
            
            for idx, row in diar_df.iterrows():
                start = row['start']
                end = row['end']
                speaker = row['speaker']
                
                # Mostrar progreso cada 10 segmentos
                if (idx + 1) % 10 == 0:
                    self.log(f"  Progreso: {idx + 1}/{n_segments} segmentos")
                
                try:
                    # Extraer segmento de audio
                    audio_segment = self.extract_audio_segment(audio_path, start, end)
                    
                    # Transcribir
                    text = self.transcribe_segment(audio_segment)
                    
                    transcriptions.append({
                        'start': start,
                        'end': end,
                        'speaker': speaker,
                        'text': text
                    })
                    
                except Exception as e:
                    # Si falla un segmento, continuar con el siguiente
                    self.log(f"  ⚠️  Error en segmento {idx+1} ({start:.2f}-{end:.2f}s): {str(e)}")
                    transcriptions.append({
                        'start': start,
                        'end': end,
                        'speaker': speaker,
                        'text': '[ERROR]'
                    })
            
            # Crear DataFrame
            df = pd.DataFrame(transcriptions)
            
            # Guardar
            df.to_csv(output_csv, index=False)
            
            # Estadísticas
            n_errors = len(df[df['text'] == '[ERROR]'])
            n_success = len(df) - n_errors
            total_duration = df['end'].max()
            
            self.log(f"✓ Guardado: {audio_name}.csv")
            self.log(f"  Segmentos: {n_success} exitosos, {n_errors} errores")
            self.log(f"  Duración total: {total_duration/60:.1f} min")
            
            return {
                'audio': audio_name,
                'status': 'success',
                'n_segments': len(df),
                'n_success': n_success,
                'n_errors': n_errors,
                'duration_seconds': round(total_duration, 2)
            }
            
        except Exception as e:
            self.log(f"✗ Error: {audio_name} - {str(e)}")
            
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
    
    def process_all(self):
        """Procesa todos los audios"""
        self.log("="*70)
        self.log("SCRIPT 4: TRANSCRIPCIÓN CON SPEAKER LABELS")
        self.log("="*70)
        
        # Cargar modelo
        if not self.load_whisper_model():
            self.log("\n❌ ABORTANDO: No se pudo cargar el modelo Whisper")
            return None
        
        # Obtener audios
        audio_files = self.get_audio_files()
        n_audios = len(audio_files)
        
        if n_audios == 0:
            self.log(f"\n✗ No se encontraron audios con diarización")
            self.log("Ejecuta primero los Scripts 1, 2 y 3:")
            self.log("  1. 01_video_to_audio.py")
            self.log("  2. 02_normalize_audio.py")
            self.log("  3. 03_diarization.py")
            return None
        
        self.log(f"\n📁 Directorio audios: {self.audio_dir.absolute()}")
        self.log(f"📁 Directorio diarización: {self.diarization_dir.absolute()}")
        self.log(f"📁 Directorio salida: {self.output_dir.absolute()}")
        self.log(f"📊 Audios a procesar: {n_audios}")
        self.log(f"🎤 Modelo Whisper: {self.model_size}")
        self.log("-"*70)
        
        # Procesar cada audio
        results = []
        start_time = datetime.now()
        
        for i, audio_info in enumerate(audio_files, 1):
            self.log(f"\n[{i}/{n_audios}] Procesando: {audio_info['name']}")
            
            audio_start = datetime.now()
            result = self.transcribe_audio(audio_info)
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
        
        total_segments = sum(r.get('n_segments', 0) for r in results if r['status'] in ['success', 'skipped'])
        
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
        self.log(f"   Segmentos totales transcritos: {total_segments}")
        
        if failed > 0:
            self.log(f"\n⚠️  Hubo {failed} errores. Revisa el log para detalles.")
        
        self.log("="*70)
        
        # Guardar resultados
        results_file = self.logs_dir / "04_transcription_results.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_size,
            'total_audios': n_audios,
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'total_time_seconds': round(total_time, 2),
            'avg_time_per_audio': round(total_time / max(successful, 1), 2),
            'total_segments_transcribed': total_segments,
            'results': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.log(f"\n💾 Resultados guardados en: {results_file.absolute()}")
        self.log(f"📄 Log completo en: {self.log_file.absolute()}")
        
        return summary


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("   SCRIPT 4: TRANSCRIPCIÓN CON SPEAKER LABELS")
    print("="*70 + "\n")
    
    print("🔍 Configurando transcriptor...")
    print("   Modelo Whisper: base (puedes cambiar a: tiny, small, medium, large)")
    print()
    
    try:
        transcriber = Transcriber(
            audio_dir="fuentes/audio_normalized",
            diarization_dir="fuentes/diarization",
            output_dir="fuentes/transcription",
            logs_dir="logs",
            model_size="base"  # Opciones: tiny, base, small, medium, large
        )
        
        summary = transcriber.process_all()
        
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
