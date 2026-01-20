#!/usr/bin/env python3
"""
Script Auxiliar: Revisión Manual de Género
Permite escuchar y corregir speakers donde hubo discrepancia entre métodos
"""

import os
import json
import pandas as pd
import soundfile as sf
import numpy as np
import winsound
import tempfile
from pathlib import Path
import sys

class GenderReviewer:
    def __init__(self):
        # Directorios
        self.base_dir = Path("fuentes")
        self.gender_dir = self.base_dir / "gender_classification"
        self.audio_dir = self.base_dir / "audio_normalized"
        self.diarization_dir = self.base_dir / "diarization"
        self.transcription_dir = self.base_dir / "transcription"
        
        # Verificar existencia
        if not self.gender_dir.exists():
            print(f"❌ No se encontró el directorio: {self.gender_dir}")
            sys.exit(1)

    def accumulate_audio(self, audio_path, speaker, target_seconds=5.0):
        """Acumula segmentos de audio para un speaker"""
        diar_path = self.diarization_dir / f"{audio_path.stem}.csv"
        
        if not diar_path.exists():
            return None, None
            
        try:
            # Leer segmentos del speaker
            df = pd.read_csv(diar_path)
            speaker_segments = df[df['speaker'] == speaker].copy()
            
            # Ordenar por duración (preferir segmentos largos)
            speaker_segments['duration'] = speaker_segments['end'] - speaker_segments['start']
            sorted_segments = speaker_segments.sort_values('duration', ascending=False)
            
            accumulated = []
            total_samples = 0
            sample_rate = None
            
            # Cargar audio completo (optimización: cargar solo partes necesarias sería mejor pero soundfile es rápido)
            # Para evitar cargar todo el archivo, usaremos sf.read con start/stop frames si es posible,
            # pero necesitamos saber el SR primero.
            
            # Primero obtener info del archivo
            info = sf.info(str(audio_path))
            sr = info.samplerate
            sample_rate = sr
            target_samples = int(target_seconds * sr)
            
            for _, row in sorted_segments.iterrows():
                start_frame = int(row['start'] * sr)
                stop_frame = int(row['end'] * sr)
                frames_to_read = stop_frame - start_frame
                
                # Calcular cuántos frames necesitamos para llegar al objetivo
                current_samples = total_samples
                remaining_samples = target_samples - current_samples
                
                if remaining_samples <= 0:
                    break
                    
                # Leer solo lo necesario
                frames_to_read = min(frames_to_read, remaining_samples)
                stop_frame = start_frame + frames_to_read
                
                # Leer segmento
                data, _ = sf.read(str(audio_path), start=start_frame, stop=stop_frame, dtype='float32')
                
                # Si es estéreo, convertir a mono
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                
                accumulated.append(data)
                total_samples += len(data)
                
                if total_samples >= target_samples:
                    break
            
            if not accumulated:
                return None, None
                
            full_audio = np.concatenate(accumulated)
            
            # Asegurar que no exceda el tiempo objetivo (por si acaso)
            if len(full_audio) > target_samples:
                full_audio = full_audio[:target_samples]
                
            return full_audio, sr
            
        except Exception as e:
            print(f"Error extrayendo audio: {e}")
            return None, None

    def play_audio(self, audio_data, sample_rate):
        """Reproduce audio usando winsound"""
        try:
            # Crear archivo temporal WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                temp_path = tf.name
            
            # Guardar audio
            sf.write(temp_path, audio_data, sample_rate)
            
            # Reproducir (bloqueante)
            winsound.PlaySound(temp_path, winsound.SND_FILENAME)
            
            # Limpiar
            try:
                os.unlink(temp_path)
            except:
                pass
                
        except Exception as e:
            print(f"Error reproduciendo audio: {e}")

    def update_transcription(self, audio_name, speaker, new_gender):
        """Actualiza el archivo de transcripción"""
        trans_path = self.transcription_dir / f"{audio_name}.csv"
        if not trans_path.exists():
            return
            
        try:
            df = pd.read_csv(trans_path)
            if 'gender' in df.columns:
                mask = df['speaker'] == speaker
                df.loc[mask, 'gender'] = new_gender
                df.to_csv(trans_path, index=False)
                print(f"   ✓ Transcripción actualizada")
        except Exception as e:
            print(f"   ✗ Error actualizando transcripción: {e}")

    def run(self):
        print("="*60)
        print("REVISOR MANUAL DE GÉNERO")
        print("="*60)
        print("Buscando discrepancias (agreement: false)...")
        
        json_files = sorted(list(self.gender_dir.glob("*.json")))
        
        discrepancies = []
        
        # 1. Encontrar discrepancias
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # La estructura puede ser directa o tener metadatos
                results = data.get('results', data)
                if isinstance(results, list): # Si es una lista, ignorar (formato antiguo o diferente)
                    continue
                    
                audio_name = json_file.stem
                
                for speaker, info in results.items():
                    # Verificar la estructura del resultado por speaker
                    if not isinstance(info, dict):
                        continue
                        
                    if info.get('agreement') is False and not info.get('manual_correction', False):
                        discrepancies.append({
                            'file': json_file,
                            'audio_name': audio_name,
                            'speaker': speaker,
                            'info': info
                        })
            except Exception as e:
                print(f"Error leyendo {json_file}: {e}")
                
        n_disc = len(discrepancies)
        print(f"🔍 Se encontraron {n_disc} casos para revisar.")
        print("-" * 60)
        
        if n_disc == 0:
            print("¡Todo perfecto! No hay discrepancias pendientes de revisar.")
            return

        # 2. Revisar caso por caso
        for i, case in enumerate(discrepancies, 1):
            audio_name = case['audio_name']
            speaker = case['speaker']
            info = case['info']
            json_file = case['file']
            
            p_gender = info['pitch_method'].get('gender', '?')
            m_gender = info['model_method'].get('gender', '?')
            curr_gender = info['final_gender']
            
            print(f"\n[{i}/{n_disc}] Archivo: {audio_name}")
            print(f"Speaker: {speaker}")
            print(f"🤖 Pitch dice: {p_gender} | Modelo dice: {m_gender}")
            print(f"👉 Asignado actualmente: {curr_gender.upper()}")
            
            # Obtener audio
            audio_path = self.audio_dir / f"{audio_name}.wav"
            print("   Cargando audio...", end='\r')
            audio_data, sr = self.accumulate_audio(audio_path, speaker)
            
            if audio_data is None:
                print("   ⚠️ No se pudo cargar audio para este speaker. Saltando.")
                continue
                
            # Loop de interacción
            while True:
                print("   🔊 Reproduciendo...", end='\r')
                self.play_audio(audio_data, sr)
                
                print("\n   ¿Qué género es? (M)ale / (F)emale / (R)eplay / (S)kip: ", end='')
                choice = input().strip().lower()
                
                if choice == 'r':
                    continue
                elif choice == 's':
                    print("   Saltado.")
                    break
                elif choice in ['m', 'f']:
                    new_gender = 'male' if choice == 'm' else 'female'
                    
                    # Actualizar JSON en memoria
                    with open(json_file, 'r', encoding='utf-8') as f:
                        full_data = json.load(f)
                    
                    # Acceder al speaker correcto
                    if 'results' in full_data:
                        target = full_data['results'][speaker]
                    else:
                        target = full_data[speaker]
                        
                    target['final_gender'] = new_gender
                    target['manual_correction'] = True
                    target['agreement'] = True # Forzamos agreement para que no salga de nuevo
                    
                    # Guardar JSON
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(full_data, f, indent=2, ensure_ascii=False)
                    
                    # Actualizar Transcripción CSV
                    self.update_transcription(audio_name, speaker, new_gender)
                    
                    print(f"   ✅ Guardado como: {new_gender.upper()}")
                    break
                else:
                    print("   Opción no válida.")

        print("\n" + "="*60)
        print("Revisión completada.")

if __name__ == "__main__":
    reviewer = GenderReviewer()
    reviewer.run()
