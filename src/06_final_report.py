#!/usr/bin/env python3
"""
Script 6: Generación de Reporte Final
Consolida toda la información en un CSV final por audio con:
- Timestamp de intervención (start, end, duration)
- Speaker ID
- Género (male/female)
- Texto transcrito
- Overlap (si se solapa con otro speaker)
- Interruption (si interrumpe o es interrumpido)
"""

import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class ReportGenerator:
    def __init__(
        self,
        transcription_dir="fuentes/transcription",
        gender_dir="fuentes/gender_classification",
        output_csv_dir="final_reports/csv",
        output_excel_dir="final_reports/excel",
        logs_dir="logs"
    ):
        self.transcription_dir = Path(transcription_dir)
        self.gender_dir = Path(gender_dir)
        self.output_csv_dir = Path(output_csv_dir)
        self.output_excel_dir = Path(output_excel_dir)
        self.logs_dir = Path(logs_dir)
        
        # Crear directorios
        self.output_csv_dir.mkdir(parents=True, exist_ok=True)
        self.output_excel_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Verificar directorios
        if not self.transcription_dir.exists():
            raise FileNotFoundError(f"No existe el directorio: {self.transcription_dir}")
        
        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"06_final_report_{timestamp}.log"
    
    def log(self, message):
        """Escribe en consola y archivo de log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def detect_overlaps(self, df):
        """
        Detecta overlaps entre segmentos consecutivos
        
        Returns:
            DataFrame con columna 'overlap' añadida
        """
        df = df.copy()
        df['overlap'] = False
        df['overlap_duration'] = 0.0
        
        for i in range(len(df) - 1):
            current_end = df.iloc[i]['end']
            next_start = df.iloc[i + 1]['start']
            
            if current_end > next_start:
                # Hay overlap
                overlap_duration = current_end - next_start
                df.at[i, 'overlap'] = True
                df.at[i, 'overlap_duration'] = round(overlap_duration, 2)
        
        return df
    
    def detect_interruptions(self, df):
        """
        Detecta interrupciones:
        - interrupts_previous: este speaker interrumpe al anterior
        - interrupted_by_next: este speaker es interrumpido por el siguiente
        
        Una interrupción se define como un overlap donde el speaker cambia
        """
        df = df.copy()
        df['interrupts_previous'] = False
        df['interrupted_by_next'] = False
        
        for i in range(len(df)):
            # Verificar si interrumpe al anterior
            if i > 0:
                prev_speaker = df.iloc[i - 1]['speaker']
                curr_speaker = df.iloc[i]['speaker']
                prev_end = df.iloc[i - 1]['end']
                curr_start = df.iloc[i]['start']
                
                # Si hay overlap y el speaker es diferente → interrupción
                if curr_start < prev_end and prev_speaker != curr_speaker:
                    df.at[i, 'interrupts_previous'] = True
                    df.at[i - 1, 'interrupted_by_next'] = True
        
        return df
    
    def generate_report(self, audio_name):
        """Genera el reporte final para un audio"""
        trans_path = self.transcription_dir / f"{audio_name}.csv"
        gender_path = self.gender_dir / f"{audio_name}.json"
        output_csv = self.output_csv_dir / f"{audio_name}_report.csv"
        
        # Si ya existe, saltar
        if output_csv.exists():
            df = pd.read_csv(output_csv)
            n_interventions = len(df)
            self.log(f"⊘ Ya existe: {audio_name}_report.csv ({n_interventions} intervenciones) - Saltando")
            return {
                'audio': audio_name,
                'status': 'skipped',
                'n_interventions': n_interventions
            }
        
        try:
            # Cargar transcripción
            if not trans_path.exists():
                raise FileNotFoundError(f"No existe transcripción: {trans_path}")
            
            df = pd.read_csv(trans_path)
            self.log(f"  Cargada transcripción: {len(df)} segmentos")
            
            # Cargar género si existe
            gender_map = {}
            confidence_map = {}
            if gender_path.exists():
                with open(gender_path, 'r', encoding='utf-8') as f:
                    gender_data = json.load(f)
                    gender_map = {
                        speaker: data['final_gender']
                        for speaker, data in gender_data.items()
                    }
                    confidence_map = {
                        speaker: data.get('confidence', 0.0)
                        for speaker, data in gender_data.items()
                    }
                self.log(f"  Cargada clasificación de género: {len(gender_map)} speakers")
            else:
                self.log(f"  ⚠️  No se encontró clasificación de género")
            
            # Preparar DataFrame
            report_df = pd.DataFrame()
            
            # Columnas básicas
            report_df['intervention_id'] = range(1, len(df) + 1)
            report_df['start_time'] = df['start']
            report_df['end_time'] = df['end']
            report_df['duration'] = (df['end'] - df['start']).round(2)
            report_df['speaker'] = df['speaker']
            
            # Género y confianza
            if 'gender' in df.columns:
                report_df['gender'] = df['gender']
            elif gender_map:
                report_df['gender'] = df['speaker'].map(gender_map)
            else:
                report_df['gender'] = 'unknown'
            
            # Confianza de género
            if confidence_map:
                report_df['gender_confidence'] = df['speaker'].map(confidence_map).round(3)
            else:
                report_df['gender_confidence'] = 0.0
            
            # Texto
            report_df['text'] = df['text']
            
            # Detectar overlaps
            df_with_overlap = self.detect_overlaps(df)
            report_df['has_overlap'] = df_with_overlap['overlap']
            report_df['overlap_duration'] = df_with_overlap['overlap_duration']
            
            # Detectar interrupciones
            df_with_interruptions = self.detect_interruptions(df)
            report_df['interrupts_previous'] = df_with_interruptions['interrupts_previous']
            report_df['interrupted_by_next'] = df_with_interruptions['interrupted_by_next']
            
            # Información adicional
            report_df['turn_number'] = 0
            current_speaker = None
            turn_count = 0
            
            for i, row in df.iterrows():
                if row['speaker'] != current_speaker:
                    turn_count += 1
                    current_speaker = row['speaker']
                report_df.at[i, 'turn_number'] = turn_count
            
            # Guardar CSV
            report_df.to_csv(output_csv, index=False)
            
            # Guardar Excel
            output_excel = self.output_excel_dir / f"{audio_name}_report.xlsx"
            report_df.to_excel(output_excel, index=False, sheet_name='Report')
            
            # Estadísticas
            n_interventions = len(report_df)
            n_overlaps = int(report_df['has_overlap'].sum())
            n_interruptions = int(report_df['interrupts_previous'].sum())
            n_turns = int(report_df['turn_number'].max())
            
            # Distribución por género
            gender_counts = report_df['gender'].value_counts().to_dict()
            
            # Duración total por género
            gender_duration = report_df.groupby('gender')['duration'].sum().to_dict()
            
            self.log(f"✓ Guardado: {audio_name}_report.csv y .xlsx")
            self.log(f"  Intervenciones: {n_interventions}")
            self.log(f"  Turnos de conversación: {n_turns}")
            self.log(f"  Overlaps: {n_overlaps}")
            self.log(f"  Interrupciones: {n_interruptions}")
            self.log(f"  Distribución por género: {gender_counts}")
            
            return {
                'audio': audio_name,
                'status': 'success',
                'n_interventions': n_interventions,
                'n_turns': n_turns,
                'n_overlaps': n_overlaps,
                'n_interruptions': n_interruptions,
                'gender_distribution': gender_counts,
                'gender_duration': {k: round(v, 2) for k, v in gender_duration.items()}
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
        self.log("SCRIPT 6: GENERACIÓN DE REPORTE FINAL")
        self.log("="*70)
        
        # Obtener audios
        audio_files = sorted([f.stem for f in self.transcription_dir.glob("*.csv")])
        n_audios = len(audio_files)
        
        if n_audios == 0:
            self.log(f"\n✗ No se encontraron archivos de transcripción")
            return None
        
        self.log(f"\n📁 Directorio transcripciones: {self.transcription_dir.absolute()}")
        self.log(f"📁 Directorio género: {self.gender_dir.absolute()}")
        self.log(f"📁 Directorio salida CSV: {self.output_csv_dir.absolute()}")
        self.log(f"📁 Directorio salida Excel: {self.output_excel_dir.absolute()}")
        self.log(f"📊 Audios a procesar: {n_audios}")
        self.log("-"*70)
        
        # Procesar cada audio
        results = []
        start_time = datetime.now()
        
        for i, audio_name in enumerate(audio_files, 1):
            self.log(f"\n[{i}/{n_audios}] Procesando: {audio_name}")
            
            result = self.generate_report(audio_name)
            results.append(result)
        
        # Resumen final
        total_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        # Estadísticas agregadas
        total_interventions = sum(r.get('n_interventions', 0) for r in results if r['status'] in ['success', 'skipped'])
        total_overlaps = sum(r.get('n_overlaps', 0) for r in results if r['status'] in ['success', 'skipped'])
        total_interruptions = sum(r.get('n_interruptions', 0) for r in results if r['status'] in ['success', 'skipped'])
        
        self.log("\n" + "="*70)
        self.log("RESUMEN FINAL")
        self.log("="*70)
        self.log(f"📊 Total audios: {n_audios}")
        self.log(f"✅ Exitosos: {successful}")
        self.log(f"⊘  Saltados: {skipped}")
        self.log(f"❌ Errores: {failed}")
        self.log(f"⏱️  Tiempo total: {total_time:.1f} segundos")
        
        self.log(f"\n📈 Estadísticas globales:")
        self.log(f"   Intervenciones totales: {total_interventions}")
        self.log(f"   Overlaps totales: {total_overlaps}")
        self.log(f"   Interrupciones totales: {total_interruptions}")
        
        if failed > 0:
            self.log(f"\n⚠️  Hubo {failed} errores. Revisa el log para detalles.")
        
        self.log("="*70)
        
        # Guardar resultados
        results_file = self.logs_dir / "06_final_report_results.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_audios': n_audios,
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'total_time_seconds': round(total_time, 2),
            'total_interventions': total_interventions,
            'total_overlaps': total_overlaps,
            'total_interruptions': total_interruptions,
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
    print("   SCRIPT 6: GENERACIÓN DE REPORTE FINAL")
    print("="*70 + "\n")
    
    print("📋 Generando reportes consolidados...")
    print("   Incluye: timestamps, speaker, género, texto, overlaps, interrupciones")
    print()
    
    try:
        generator = ReportGenerator(
            transcription_dir="fuentes/transcription",
            gender_dir="fuentes/gender_classification",
            output_csv_dir="final_reports/csv",
            output_excel_dir="final_reports/excel",
            logs_dir="logs"
        )
        
        summary = generator.process_all()
        
        if summary is None:
            print("\n❌ El script no pudo completarse. Revisa los errores arriba.")
            return
        
        print("\n✅ Script completado exitosamente!")
        print(f"\n📊 Reportes generados en: final_reports/")
        
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
