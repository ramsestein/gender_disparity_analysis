#!/usr/bin/env python3
"""
Script 1: Extracción de audio desde videos
Convierte todos los .mov de /videos a WAV 16kHz mono en /audio
Compatible con Windows
"""

import subprocess
import os
from pathlib import Path
from datetime import datetime
import json
import sys

class AudioExtractor:
    def __init__(self, videos_dir="videos", audio_dir="fuentes/audio", logs_dir="logs"):
        self.videos_dir = Path(videos_dir)
        self.audio_dir = Path(audio_dir)
        self.logs_dir = Path(logs_dir)
        
        # Crear directorios si no existen
        self.audio_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Verificar que existe carpeta de videos
        if not self.videos_dir.exists():
            raise FileNotFoundError(f"No existe el directorio: {self.videos_dir}")
        
        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"01_extract_audio_{timestamp}.log"
        
    def log(self, message):
        """Escribe en consola y en archivo de log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def check_ffmpeg(self):
        """Verifica que FFmpeg esté instalado"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                shell=True  # Necesario en Windows
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                self.log(f"✓ FFmpeg encontrado: {version_line}")
                return True
        except FileNotFoundError:
            self.log("✗ FFmpeg NO encontrado")
            self.log("  Instalar con Chocolatey: choco install ffmpeg")
            self.log("  O descargar de: https://www.gyan.dev/ffmpeg/builds/")
            return False
        except Exception as e:
            self.log(f"✗ Error verificando FFmpeg: {str(e)}")
            return False
    
    def get_video_files(self):
        """Obtiene lista de todos los videos .mov y .mp4"""
        video_files = []
    
        # Buscar .mov
        video_files.extend(self.videos_dir.glob("*.mov"))
        video_files.extend(self.videos_dir.glob("*.MOV"))
    
        # Buscar .mp4
        video_files.extend(self.videos_dir.glob("*.mp4"))
        video_files.extend(self.videos_dir.glob("*.MP4"))
    
        return sorted(video_files)
    
    def extract_audio(self, video_path):
        """
        Extrae audio de un video y lo convierte a WAV 16kHz mono
        
        Returns:
            dict con info del resultado
        """
        # Nombre del archivo de salida
        audio_filename = video_path.stem + ".wav"
        audio_path = self.audio_dir / audio_filename
        
        # Si ya existe, saltar
        if audio_path.exists():
            file_size = audio_path.stat().st_size / (1024 * 1024)  # MB
            self.log(f"⊘ Ya existe: {audio_filename} ({file_size:.1f} MB) - Saltando")
            return {
                'video': video_path.name,
                'audio': audio_filename,
                'status': 'skipped',
                'size_mb': round(file_size, 2),
                'message': 'Already exists'
            }
        
        try:
            # Comando FFmpeg (usando rutas absolutas en Windows)
            cmd = [
                'ffmpeg',
                '-i', str(video_path.absolute()),   # Input (ruta absoluta)
                '-vn',                               # Sin video
                '-ar', '16000',                      # Sample rate 16kHz
                '-ac', '1',                          # Mono
                '-acodec', 'pcm_s16le',             # Codec WAV
                '-y',                                # Sobrescribir si existe
                str(audio_path.absolute()),          # Output (ruta absoluta)
                '-loglevel', 'error'                 # Solo mostrar errores
            ]
            
            # Ejecutar
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # Timeout 10 minutos
                shell=True    # Necesario en Windows
            )
            
            if result.returncode == 0:
                # Verificar que el archivo se creó
                if audio_path.exists():
                    file_size = audio_path.stat().st_size / (1024 * 1024)  # MB
                    self.log(f"✓ Extraído: {audio_filename} ({file_size:.1f} MB)")
                    
                    return {
                        'video': video_path.name,
                        'audio': audio_filename,
                        'status': 'success',
                        'size_mb': round(file_size, 2)
                    }
                else:
                    raise Exception("Audio no se creó")
            else:
                error_msg = result.stderr if result.stderr else "Error desconocido"
                raise Exception(f"FFmpeg error: {error_msg}")
                
        except subprocess.TimeoutExpired:
            self.log(f"✗ Timeout: {video_path.name} (>10 minutos)")
            return {
                'video': video_path.name,
                'status': 'error',
                'error': 'Timeout (>10 min)'
            }
        
        except Exception as e:
            self.log(f"✗ Error: {video_path.name} - {str(e)}")
            return {
                'video': video_path.name,
                'status': 'error',
                'error': str(e)
            }
    
    def process_all(self):
        """Procesa todos los videos"""
        self.log("="*70)
        self.log("SCRIPT 1: EXTRACCIÓN DE AUDIO")
        self.log("="*70)
        
        # Verificar FFmpeg
        if not self.check_ffmpeg():
            self.log("\n❌ ABORTANDO: FFmpeg no está instalado o no está en PATH")
            self.log("Por favor instala FFmpeg y vuelve a ejecutar el script")
            return None
        
        # Obtener videos
        video_files = self.get_video_files()
        n_videos = len(video_files)
        
        if n_videos == 0:
            self.log(f"\n✗ No se encontraron videos .mov en: {self.videos_dir.absolute()}")
            self.log("Verifica que los videos estén en la carpeta correcta")
            return None
        
        self.log(f"\n📁 Directorio entrada: {self.videos_dir.absolute()}")
        self.log(f"📁 Directorio salida: {self.audio_dir.absolute()}")
        self.log(f"📊 Videos encontrados: {n_videos}")
        self.log("-"*70)
        
        # Procesar cada video
        results = []
        start_time = datetime.now()
        
        for i, video_path in enumerate(video_files, 1):
            self.log(f"\n[{i}/{n_videos}] Procesando: {video_path.name}")
            
            video_start = datetime.now()
            result = self.extract_audio(video_path)
            video_elapsed = (datetime.now() - video_start).total_seconds()
            
            results.append(result)
            
            # Estimación tiempo restante
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time = elapsed / i
            remaining = (n_videos - i) * avg_time
            eta_min = remaining / 60
            
            self.log(f"⏱️  Tiempo este video: {video_elapsed:.1f}s")
            if i < n_videos:
                self.log(f"⏱️  ETA restante: {eta_min:.1f} minutos ({n_videos - i} videos)")
        
        # Resumen final
        total_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        self.log("\n" + "="*70)
        self.log("RESUMEN FINAL")
        self.log("="*70)
        self.log(f"📊 Total videos: {n_videos}")
        self.log(f"✅ Exitosos: {successful}")
        self.log(f"⊘  Saltados (ya existían): {skipped}")
        self.log(f"❌ Errores: {failed}")
        self.log(f"⏱️  Tiempo total: {total_time/60:.1f} minutos")
        self.log(f"⏱️  Promedio: {total_time/n_videos:.1f} segundos/video")
        
        if failed > 0:
            self.log(f"\n⚠️  Hubo {failed} errores. Revisa el log para detalles.")
        
        self.log("="*70)
        
        # Guardar resultados en JSON
        results_file = self.logs_dir / "01_extract_audio_results.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': n_videos,
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'total_time_seconds': round(total_time, 2),
            'avg_time_per_video': round(total_time / n_videos, 2),
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
    print("   SCRIPT 1: EXTRACCIÓN DE AUDIO DE VIDEOS")
    print("="*70 + "\n")
    
    try:
        extractor = AudioExtractor(
            videos_dir="video",
            audio_dir="fuentes/audio",
            logs_dir="logs"
        )
        
        summary = extractor.process_all()
        
        if summary is None:
            print("\n❌ El script no pudo completarse. Revisa los errores arriba.")
            sys.exit(1)
        
        print("\n✅ Script completado exitosamente!")
        
        if summary['failed'] > 0:
            print(f"⚠️  Atención: {summary['failed']} videos fallaron")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrumpido por el usuario (Ctrl+C)")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error fatal: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()