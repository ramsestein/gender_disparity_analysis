#!/usr/bin/env python3
"""
Script 5: Gender Classification for Speakers
Classifies each speaker as male or female using:
1. Pitch analysis (F0)
2. Mozilla's pre-trained gender classification model
If both agree → high confidence
If they disagree → use Mozilla model
"""

import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import warnings
warnings.filterwarnings('ignore')


class GenderClassifier:
    def __init__(
        self,
        audio_dir="fuentes/audio_normalized",
        diarization_dir="fuentes/diarization",
        transcription_dir="fuentes/transcription",
        output_dir="fuentes/gender_classification",
        logs_dir="logs"
    ):
        self.audio_dir = Path(audio_dir)
        self.diarization_dir = Path(diarization_dir)
        self.transcription_dir = Path(transcription_dir)
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)
        
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
        self.log_file = self.logs_dir / f"05_gender_classification_{timestamp}.log"
        
        # Modelos (se cargarán después)
        self.gender_model = None
        self.feature_extractor = None
        
        # Thresholds para pitch
        self.male_threshold = 165  # Hz
    
    def log(self, message):
        """Escribe en consola y archivo de log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def load_gender_model(self):
        """Carga el modelo de clasificación de género"""
        self.log("Cargando modelo de clasificación de género...")
        self.log("  Usando: alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")
        
        try:
            model_name = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
            
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            
            # Mover a CPU
            self.gender_model.eval()
            
            self.log("✓ Modelo de género cargado")
            return True
            
        except Exception as e:
            self.log(f"✗ Error cargando modelo: {str(e)}")
            self.log("  Intentando con modelo alternativo...")
            
            try:
                # Modelo alternativo
                model_name = "speechbrain/spkrec-xvect-voxceleb"
                self.log(f"  Usando: {model_name}")
                # Implementar carga alternativa si es necesario
                return False
            except:
                return False
    
    def extract_audio_segment(self, audio_path, start_time, end_time):
        """Extrae un segmento de audio"""
        try:
            waveform, sample_rate = sf.read(str(audio_path), dtype='float32')
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            end_sample = min(end_sample, len(waveform))
            
            if len(waveform.shape) == 1:
                segment = waveform[start_sample:end_sample]
            else:
                segment = waveform[start_sample:end_sample].mean(axis=1)
            
            return segment, sample_rate
            
        except Exception as e:
            raise Exception(f"Error extrayendo segmento: {str(e)}")
    
    def analyze_pitch(self, audio_segment, sample_rate):
        """
        Analiza el pitch (F0) de un segmento de audio
        
        Returns:
            float: median F0 en Hz, o None si no se puede calcular
        """
        try:
            # Crear objeto Sound de Praat
            sound = parselmouth.Sound(audio_segment, sampling_frequency=sample_rate)
            
            # Extraer pitch
            pitch = call(sound, "To Pitch", 0.0, 75, 600)  # 75-600 Hz range
            
            # Obtener valores de pitch
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]  # Filtrar valores no válidos
            
            if len(pitch_values) > 0:
                return np.median(pitch_values)
            else:
                return None
                
        except Exception as e:
            return None
    
    def classify_gender_by_pitch(self, median_f0):
        """
        Clasifica género basado en F0
        
        Returns:
            str: 'male' o 'female'
        """
        if median_f0 is None:
            return 'unknown'
        
        return 'male' if median_f0 < self.male_threshold else 'female'
    
    def classify_gender_by_model(self, audio_segment, sample_rate):
        """
        Clasifica género usando el modelo pre-entrenado
        
        Returns:
            tuple: (gender, confidence)
        """
        try:
            # Check segment length (must be at least 0.1s for the model to work properly)
            # 16000 Hz * 0.1s = 1600 samples
            min_samples = 1600
            
            # Resamplear a 16kHz si es necesario
            if sample_rate != 16000:
                audio_tensor = torch.from_numpy(audio_segment).float()
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_segment = resampler(audio_tensor).numpy()
            
            # Preparar input
            inputs = self.feature_extractor(
                audio_segment,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Check input size after feature extraction
            if inputs.input_values.shape[1] < 100:  # Safety check for very short inputs
                 return 'unknown', 0.0
            
            # Predicción
            with torch.no_grad():
                logits = self.gender_model(**inputs).logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
            
            # El modelo tiene labels: 0=female, 1=male
            gender = 'male' if predicted_id == 1 else 'female'
            
            return gender, confidence
            
        except Exception as e:
            self.log(f"  ⚠️  Error en clasificación por modelo: {str(e)}")
            return 'unknown', 0.0
    
    def classify_speaker(self, audio_path, speaker_segments, speaker_id):
        """
        Clasifica el género de un speaker usando concatenación de segmentos
        
        Returns:
            dict con resultados de clasificación
        """
        self.log(f"  Clasificando {speaker_id}...")
        
        # Select segments to analyze (prioritize longer ones)
        # Sort by duration descending
        speaker_segments['duration'] = speaker_segments['end'] - speaker_segments['start']
        sorted_segments = speaker_segments.sort_values('duration', ascending=False)
        
        accumulated_audio = []
        total_samples = 0
        target_accumulated_seconds = 10.0 # Accumulate up to 10 seconds for robust classification
        sample_rate_detected = None

        # 1. Accumulate audio segments
        for idx, row in sorted_segments.iterrows():
            start = row['start']
            end = row['end']
            
             # Skip extremely short segments (< 0.2s) even for accumulation
            if (end - start) < 0.2:
                continue

            try:
                segment, sr = self.extract_audio_segment(audio_path, start, end)
                if len(segment) > 0:
                    accumulated_audio.append(segment)
                    total_samples += len(segment)
                    sample_rate_detected = sr
                    
                    # Stop if we have enough audio
                    if total_samples >= (target_accumulated_seconds * sr):
                        break
            except Exception:
                continue
        
        # Prepare result structure
        result = {
            'speaker': speaker_id,
            'pitch_method': {},
            'model_method': {},
            'final_gender': 'unknown',
            'confidence': 0.0,
            'agreement': False
        }

        if not accumulated_audio or sample_rate_detected is None:
             self.log(f"    ✗ {speaker_id}: No se pudo extraer audio válido")
             return result

        # 2. Concatenate audio
        try:
             full_audio = np.concatenate(accumulated_audio)
             self.log(f"    Audio acumulado: {len(full_audio)/sample_rate_detected:.1f}s")
        except Exception as e:
             self.log(f"    ✗ Error concatenando audio: {e}")
             return result
             
        # 3. Classify ONE time on the concatenated audio
        
        # Method 1: Pitch
        median_f0 = self.analyze_pitch(full_audio, sample_rate_detected)
        if median_f0 is not None:
             pitch_gender = self.classify_gender_by_pitch(median_f0)
             result['pitch_method'] = {
                'gender': pitch_gender,
                'median_f0': round(float(median_f0), 2)
            }
            
        # Method 2: Model
        gender_model, confidence = self.classify_gender_by_model(full_audio, sample_rate_detected)
        if gender_model != 'unknown':
             result['model_method'] = {
                'gender': gender_model,
                'confidence': round(float(confidence), 3)
            }
            
        # 4. Final Decision
        pitch_gender = result['pitch_method'].get('gender', 'unknown')
        model_gender = result['model_method'].get('gender', 'unknown')
        
        if pitch_gender != 'unknown' and model_gender != 'unknown':
            if pitch_gender == model_gender:
                # Agreement
                result['final_gender'] = pitch_gender
                result['confidence'] = result['model_method']['confidence']
                result['agreement'] = True
                self.log(f"    ✓ {speaker_id}: {pitch_gender} (ambos coinciden, conf={confidence:.2f})")
            else:
                # Disagreement -> trust model
                result['final_gender'] = model_gender
                result['confidence'] = result['model_method']['confidence']
                result['agreement'] = False
                self.log(f"    ⚠️  {speaker_id}: {model_gender} (pitch={pitch_gender}, modelo={model_gender})")
                
        elif model_gender != 'unknown':
            result['final_gender'] = model_gender
            result['confidence'] = result['model_method']['confidence']
            self.log(f"    → {speaker_id}: {model_gender} (solo modelo, conf={confidence:.2f})")
            
        elif pitch_gender != 'unknown':
            result['final_gender'] = pitch_gender
            result['confidence'] = 0.7
            self.log(f"    → {speaker_id}: {pitch_gender} (solo pitch)")
            
        else:
            self.log(f"    ✗ {speaker_id}: No se pudo clasificar")
        
        return result
    
    def process_audio(self, audio_name):
        """Procesa un audio y clasifica todos sus speakers"""
        audio_path = self.audio_dir / f"{audio_name}.wav"
        diar_path = self.diarization_dir / f"{audio_name}.csv"
        output_json = self.output_dir / f"{audio_name}.json"
        
        # Si ya existe, saltar
        if output_json.exists():
            with open(output_json, 'r', encoding='utf-8') as f:
                results = json.load(f)
            n_speakers = len(results)
            self.log(f"⊘ Ya existe: {audio_name}.json ({n_speakers} speakers) - Saltando")
            return {
                'audio': audio_name,
                'status': 'skipped',
                'n_speakers': n_speakers
            }
        
        try:
            # Cargar diarización
            self.log(f"  Cargando diarización...")
            diar_df = pd.read_csv(diar_path)
            
            # Obtener speakers únicos
            speakers = diar_df['speaker'].unique()
            n_speakers = len(speakers)
            self.log(f"  {n_speakers} speakers encontrados")
            
            # Clasificar cada speaker
            results = {}
            for speaker in speakers:
                speaker_segments = diar_df[diar_df['speaker'] == speaker]
                result = self.classify_speaker(audio_path, speaker_segments, speaker)
                results[speaker] = result
            
            # Guardar resultados
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.log(f"✓ Guardado: {audio_name}.json")
            
            # Actualizar transcripción si existe
            trans_path = self.transcription_dir / f"{audio_name}.csv"
            if trans_path.exists():
                self.update_transcription_with_gender(trans_path, results)
            
            return {
                'audio': audio_name,
                'status': 'success',
                'n_speakers': n_speakers,
                'results': results
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
    
    def update_transcription_with_gender(self, trans_path, gender_results):
        """Actualiza el CSV de transcripción con la columna de género"""
        try:
            df = pd.read_csv(trans_path)
            
            # Crear mapeo speaker → gender
            gender_map = {
                speaker: data['final_gender']
                for speaker, data in gender_results.items()
            }
            
            # Añadir columna de género
            df['gender'] = df['speaker'].map(gender_map)
            
            # Guardar
            df.to_csv(trans_path, index=False)
            self.log(f"  ✓ Transcripción actualizada con género")
            
        except Exception as e:
            self.log(f"  ⚠️  No se pudo actualizar transcripción: {str(e)}")
    
    def process_all(self):
        """Procesa todos los audios"""
        self.log("="*70)
        self.log("SCRIPT 5: CLASIFICACIÓN DE GÉNERO DE SPEAKERS")
        self.log("="*70)
        
        # Cargar modelo
        if not self.load_gender_model():
            self.log("\n⚠️  ADVERTENCIA: No se pudo cargar el modelo de género")
            self.log("  Se usará solo análisis de pitch (menor precisión)")
        
        # Obtener audios
        audio_files = sorted([f.stem for f in self.diarization_dir.glob("*.csv")])
        n_audios = len(audio_files)
        
        if n_audios == 0:
            self.log(f"\n✗ No se encontraron archivos de diarización")
            return None
        
        self.log(f"\n📁 Directorio audios: {self.audio_dir.absolute()}")
        self.log(f"📁 Directorio diarización: {self.diarization_dir.absolute()}")
        self.log(f"📁 Directorio salida: {self.output_dir.absolute()}")
        self.log(f"📊 Audios a procesar: {n_audios}")
        self.log("-"*70)
        
        # Procesar cada audio
        results = []
        start_time = datetime.now()
        
        for i, audio_name in enumerate(audio_files, 1):
            self.log(f"\n[{i}/{n_audios}] Procesando: {audio_name}")
            
            audio_start = datetime.now()
            result = self.process_audio(audio_name)
            audio_elapsed = (datetime.now() - audio_start).total_seconds()
            
            result['processing_time_seconds'] = round(audio_elapsed, 2)
            results.append(result)
            
            if result['status'] != 'skipped':
                self.log(f"⏱️  Tiempo: {audio_elapsed:.1f}s")
        
        # Resumen final
        total_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        total_speakers = sum(r.get('n_speakers', 0) for r in results if r['status'] in ['success', 'skipped'])
        
        self.log("\n" + "="*70)
        self.log("RESUMEN FINAL")
        self.log("="*70)
        self.log(f"📊 Total audios: {n_audios}")
        self.log(f"✅ Exitosos: {successful}")
        self.log(f"⊘  Saltados: {skipped}")
        self.log(f"❌ Errores: {failed}")
        self.log(f"⏱️  Tiempo total: {total_time/60:.1f} minutos")
        self.log(f"\n📈 Estadísticas:")
        self.log(f"   Speakers totales clasificados: {total_speakers}")
        
        if failed > 0:
            self.log(f"\n⚠️  Hubo {failed} errores. Revisa el log para detalles.")
        
        self.log("="*70)
        
        # Guardar resultados
        results_file = self.logs_dir / "05_gender_classification_results.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_audios': n_audios,
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'total_time_seconds': round(total_time, 2),
            'total_speakers_classified': total_speakers,
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
    print("   SCRIPT 5: CLASIFICACIÓN DE GÉNERO DE SPEAKERS")
    print("="*70 + "\n")
    
    print("🔍 Configurando clasificador...")
    print("   Método 1: Análisis de pitch (F0)")
    print("   Método 2: Modelo pre-entrenado (Mozilla/Wav2Vec2)")
    print("   Decisión: Si coinciden → alta confianza, si difieren → usar modelo")
    print()
    
    try:
        classifier = GenderClassifier(
            audio_dir="fuentes/audio_normalized",
            diarization_dir="fuentes/diarization",
            transcription_dir="fuentes/transcription",
            output_dir="fuentes/gender_classification",
            logs_dir="logs"
        )
        
        summary = classifier.process_all()
        
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
