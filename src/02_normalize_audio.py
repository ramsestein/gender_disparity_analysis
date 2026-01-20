#!/usr/bin/env python3
"""
Script 1B: Normalización de audio
Mejora calidad para diarización en grabaciones de pantalla
- Normaliza volumen (loudnorm)
- Reduce ruido de fondo (opcional)
- Estandariza niveles entre archivos
"""

import subprocess
from pathlib import Path
from datetime import datetime
import json
import os

class AudioNormalizer:
    def __init__(self, 
                 audio_dir="audio", 
                 audio_normalized_dir="audio_normalized", 
                 logs_dir="logs",
                 apply_noise_reduction=True):
        
        self.audio_dir = Path(audio_dir)
        self.audio_normalized_dir = Path(audio_normalized_dir)
        self.logs_dir = Path(logs_dir)
        self.apply_noise_reduction = apply_noise_reduction
        
        # Crear directorios
        self.audio_normalized_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Verificar audio_dir
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"No existe el directorio: {self.audio_dir}")
        
        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"01b_normalize_{timestamp}.log"
    
    def log(self, message):
        """Escribe en consola y archivo de log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def check_ffmpeg(self):
        """Verifica FFmpeg"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                shell=True
            )
            if result.returncode == 0:
                self.log("✓ FFmpeg encontrado")
                return True
        except:
            self.log("✗ FFmpeg NO encontrado")
            return False
    
    def get_audio_files(self):
        """Obtiene lista de audios WAV"""
        return sorted(self.audio_dir.glob("*.wav"))
    
    def normalize_audio(self, audio_path):
        """
        Normaliza audio con loudnorm y opcionalmente reduce ruido
        
        Returns:
            dict con resultado
        """
        output_path = self.audio_normalized_dir / audio_path.name
        
        # Si ya existe, saltar
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)
            self.log(f"⊘ Ya existe: {audio_path.name} ({file_size:.1f} MB) - Saltando")
            return {
                'audio': audio_path.name,
                'status': 'skipped',
                'size_mb': round(file_size, 2)
            }
        
        try:
            # Construir filtros de audio
            audio_filters = []
            
            # 1. Reducción de ruido (highpass + lowpass)
            if self.apply_noise_reduction:
                audio_filters.append('highpass=f=200')      # Elimina rumble bajo
                audio_filters.append('lowpass=f=3000')      # Elimina ruido agudo
            
            # 2. Normalización de volumen (EBU R128 loudnorm)
            # I=-16: Integrated loudness target (dB)
            # LRA=11: Loudness range target
            # TP=-1.5: True peak (evita clipping)
            audio_filters.append('loudnorm=I=-16:LRA=11:TP=-1.5')
            
            # 3. Compresión dinámica (reduce diferencias extremas)
            audio_filters.append('acompressor=threshold=-20dB:ratio=4:attack=5:release=50')
            
            # Unir filtros
            filter_chain = ','.join(audio_filters)
            
            # Comando FFmpeg
            cmd = [
                'ffmpeg',
                '-i', str(audio_path.absolute()),
                '-af', filter_chain,
                '-ar', '16000',                    # 16kHz (necesario para PyAnnote)
                '-ac', '1',                        # Mono
                '-acodec', 'pcm_s16le',           # WAV formato
                '-y',
                str(output_path.absolute()),
                '-loglevel', 'error'
            ]
            
            # Ejecutar
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
                shell=True
            )
            
            if result.returncode == 0:
                if output_path.exists():
                    file_size = output_path.stat().st_size / (1024 * 1024)
                    self.log(f"✓ Normalizado: {audio_path.name} ({file_size:.1f} MB)")
                    
                    return {
                        'audio': audio_path.name,
                        'status': 'success',
                        'size_mb': round(file_size, 2)
                    }
                else:
                    raise Exception("Audio normalizado no se creó")
            else:
                error_msg = result.stderr if result.stderr else "Error desconocido"
                raise Exception(f"FFmpeg error: {error_msg}")
                
        except subprocess.TimeoutExpired:
            self.log(f"✗ Timeout: {audio_path.name}")
            return {
                'audio': audio_path.name,
                'status': 'error',
                'error': 'Timeout (>5 min)'
            }
        
        except Exception as e:
            self.log(f"✗ Error: {audio_path.name} - {str(e)}")
            return {
                'audio': audio_path.name,
                'status': 'error',
                'error': str(e)
            }
    
    def process_all(self):
        """Procesa todos los audios"""
        self.log("="*70)
        self.log("SCRIPT 1B: NORMALIZACIÓN DE AUDIO")
        self.log("="*70)
        
        # Verificar FFmpeg
        if not self.check_ffmpeg():
            self.log("\n❌ ABORTANDO: FFmpeg no encontrado")
            return None
        
        # Obtener audios
        audio_files = self.get_audio_files()
        n_audios = len(audio_files)
        
        if n_audios == 0:
            self.log(f"\n✗ No se encontraron audios en: {self.audio_dir.absolute()}")
            self.log("Ejecuta primero el Script 1: 01_extract_audio.py")
            return None
        
        self.log(f"\n📁 Directorio entrada: {self.audio_dir.absolute()}")
        self.log(f"📁 Directorio salida: {self.audio_normalized_dir.absolute()}")
        self.log(f"📊 Audios encontrados: {n_audios}")
        self.log(f"🎛️  Reducción de ruido: {'✓ Activada' if self.apply_noise_reduction else '✗ Desactivada'}")
        self.log("-"*70)
        
        # Procesar cada audio
        results = []
        start_time = datetime.now()
        
        for i, audio_path in enumerate(audio_files, 1):
            self.log(f"\n[{i}/{n_audios}] Procesando: {audio_path.name}")
            
            audio_start = datetime.now()
            result = self.normalize_audio(audio_path)
            audio_elapsed = (datetime.now() - audio_start).total_seconds()
            
            results.append(result)
            
            # Estimación tiempo restante
            if result['status'] != 'skipped':
                elapsed = (datetime.now() - start_time).total_seconds()
                completed = sum(1 for r in results if r['status'] in ['success', 'error'])
                if completed > 0:
                    avg_time = elapsed / completed
                    remaining = (n_audios - i) * avg_time
                    eta_min = remaining / 60
                    
                    self.log(f"⏱️  Tiempo este audio: {audio_elapsed:.1f}s")
                    if i < n_audios:
                        self.log(f"⏱️  ETA restante: {eta_min:.1f} minutos ({n_audios - i} audios)")
        
        # Resumen final
        total_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        failed = sum(1 for r in results if r['status'] == 'error')
        
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
            self.log(f"⏱️  Promedio: {avg_time:.1f} segundos/audio")
        
        if failed > 0:
            self.log(f"\n⚠️  Hubo {failed} errores. Revisa el log para detalles.")
        
        self.log("="*70)
        
        # Guardar resultados
        results_file = self.logs_dir / "01b_normalize_results.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'noise_reduction_enabled': self.apply_noise_reduction,
            'total_audios': n_audios,
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'total_time_seconds': round(total_time, 2),
            'avg_time_per_audio': round(total_time / max(successful, 1), 2),
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
    print("   SCRIPT 1B: NORMALIZACIÓN DE AUDIO")
    print("="*70 + "\n")
    
    try:
        normalizer = AudioNormalizer(
            audio_dir="fuentes/audio",
            audio_normalized_dir="fuentes/audio_normalized",
            logs_dir="logs",
            apply_noise_reduction=True  # ← Cambia a False si no quieres reducción de ruido
        )
        
        summary = normalizer.process_all()
        
        if summary is None:
            print("\n❌ El script no pudo completarse. Revisa los errores arriba.")
            return
        
        print("\n✅ Script completado exitosamente!")
        print(f"\n📁 Audios normalizados guardados en: audio_normalized/")
        print(f"ℹ️  Usa esta carpeta en el Script 2 (diarización)")
        
        if summary['failed'] > 0:
            print(f"⚠️  Atención: {summary['failed']} audios fallaron")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrumpido por el usuario (Ctrl+C)")
    
    except Exception as e:
        print(f"\n❌ Error fatal: {str(e)}")


if __name__ == "__main__":
    main()