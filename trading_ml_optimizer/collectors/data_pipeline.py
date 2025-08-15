# trading_ml_optimizer/collectors/data_pipeline.py
"""
Data Pipeline - Integra el Log Watcher con la Base de Datos
Procesa logs en tiempo real y los almacena estructuradamente
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Agregar el path del proyecto
sys.path.append(str(Path(__file__).parent.parent.parent))

from trading_ml_optimizer.collectors.log_watcher import TradingLogWatcher
from trading_ml_optimizer.database.db_manager import TradingDatabase

class TradingDataPipeline:
    """
    Pipeline que conecta logs → base de datos → análisis
    """
    
    def __init__(self, log_path: str = "trend_strategy.log", db_path: str = None):
        """
        Inicializa el pipeline
        
        Args:
            log_path: Ruta al archivo de log
            db_path: Ruta a la base de datos (opcional)
        """
        print("🚀 Inicializando Data Pipeline...")
        
        # Inicializar componentes
        self.log_watcher = TradingLogWatcher(log_path)
        self.db = TradingDatabase(db_path) if db_path else TradingDatabase()
        
        # Estadísticas del pipeline
        self.events_processed = 0
        self.events_saved = 0
        self.errors = 0
        
        print("✅ Pipeline inicializado")
    
    def process_event(self, event: dict) -> bool:
        """
        Procesa un evento y lo guarda en la base de datos
        
        Args:
            event: Evento a procesar
            
        Returns:
            True si se procesó correctamente
        """
        try:
            # Guardar en la base de datos
            event_id = self.db.save_event(event)
            
            if event_id:
                self.events_saved += 1
                
                # Procesar según el tipo de evento
                event_type = event.get('event_type', '')
                
                if event_type == 'orden_colocada':
                    self._process_new_trade(event)
                elif event_type in ['resultado_ganancia', 'resultado_perdida']:
                    self._process_trade_result(event)
                elif event_type == 'ciclo_info':
                    self._update_session_stats(event)
                
                return True
            
        except Exception as e:
            print(f"❌ Error procesando evento: {e}")
            self.errors += 1
            return False
        
        return False
    
    def _process_new_trade(self, event: dict):
        """Procesa un nuevo trade"""
        print(f"  📈 Nuevo trade detectado: {event.get('activo', 'N/A')}")
    
    def _process_trade_result(self, event: dict):
        """Procesa el resultado de un trade"""
        result = event.get('resultado', '')
        amount = event.get('ganancia', 0) or event.get('perdida', 0)
        print(f"  💰 Resultado: {result} - ${amount:.2f}")
    
    def _update_session_stats(self, event: dict):
        """Actualiza estadísticas de sesión"""
        # Aquí actualizarías las estadísticas por sesión
        pass
    
    def process_historical_logs(self) -> dict:
        """
        Procesa todos los logs históricos
        
        Returns:
            Resumen del procesamiento
        """
        print("\n📖 Procesando logs históricos...")
        
        # Leer todos los eventos
        events = self.log_watcher.read_new_lines()
        
        if not events:
            print("⚠️ No hay eventos para procesar")
            return {
                'events_found': 0,
                'events_processed': 0,
                'events_saved': 0,
                'errors': 0
            }
        
        print(f"📊 {len(events)} eventos encontrados")
        
        # Procesar cada evento
        for i, event in enumerate(events, 1):
            self.events_processed += 1
            
            # Mostrar progreso cada 10 eventos
            if i % 10 == 0:
                print(f"  Procesando... {i}/{len(events)}")
            
            self.process_event(event)
        
        # Obtener estadísticas actualizadas
        stats = self.db.get_stats_summary()
        
        print(f"\n✅ Procesamiento completado:")
        print(f"  - Eventos procesados: {self.events_processed}")
        print(f"  - Eventos guardados: {self.events_saved}")
        print(f"  - Errores: {self.errors}")
        print(f"\n📊 Estadísticas de trading:")
        print(f"  - Total trades: {stats['total_trades']}")
        print(f"  - Wins: {stats['wins']}")
        print(f"  - Losses: {stats['losses']}")
        print(f"  - P&L Total: ${stats['total_pnl']:.2f}")
        
        return {
            'events_found': len(events),
            'events_processed': self.events_processed,
            'events_saved': self.events_saved,
            'errors': self.errors,
            'trading_stats': stats
        }
    
    def monitor_realtime(self, interval: int = 5):
        """
        Monitorea logs en tiempo real
        
        Args:
            interval: Segundos entre verificaciones
        """
        print("\n🔄 Iniciando monitoreo en tiempo real...")
        print("Presiona Ctrl+C para detener\n")
        
        try:
            while True:
                # Leer nuevos eventos
                new_events = self.log_watcher.read_new_lines()
                
                if new_events:
                    print(f"\n🔔 {len(new_events)} nuevos eventos detectados")
                    
                    for event in new_events:
                        self.events_processed += 1
                        
                        # Mostrar info del evento
                        event_type = event.get('event_type', 'unknown')
                        timestamp = event.get('timestamp', 'N/A')
                        print(f"  [{timestamp}] {event_type}")
                        
                        # Procesar y guardar
                        if self.process_event(event):
                            print(f"    ✅ Guardado en BD")
                    
                    # Mostrar resumen actualizado
                    self._show_current_stats()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n✋ Monitoreo detenido")
            self._show_final_summary()
    
    def _show_current_stats(self):
        """Muestra estadísticas actuales"""
        stats = self.db.get_stats_summary()
        summary = self.log_watcher.get_trading_summary()
        
        print(f"\n📈 Estado actual:")
        print(f"  Trades: {summary['ordenes_primarias']} | "
              f"Hedges: {summary['coberturas']} | "
              f"W/L: {summary['wins']}/{summary['losses']} | "
              f"P&L: ${summary['pnl_total']:.2f}")
    
    def _show_final_summary(self):
        """Muestra resumen final"""
        stats = self.db.get_stats_summary()
        
        print("\n" + "="*50)
        print("📊 RESUMEN FINAL DEL PIPELINE")
        print("="*50)
        print(f"\n📥 Eventos procesados: {self.events_processed}")
        print(f"💾 Eventos guardados: {self.events_saved}")
        print(f"❌ Errores: {self.errors}")
        
        print(f"\n📈 Estadísticas de Trading:")
        print(f"  - Total trades: {stats['total_trades']}")
        print(f"  - Wins: {stats['wins']}")
        print(f"  - Losses: {stats['losses']}")
        print(f"  - P&L Total: ${stats['total_pnl']:.2f}")
        
        if stats['best_asset']:
            print(f"  - Mejor activo: {stats['best_asset']}")
        if stats['best_session']:
            print(f"  - Mejor sesión: {stats['best_session']}")
        
        print("="*50)
    
    def generate_report(self) -> dict:
        """
        Genera un reporte completo
        
        Returns:
            Diccionario con el reporte
        """
        # Obtener DataFrames
        trades_df = self.db.get_trades_dataframe()
        perf_by_hour = self.db.get_performance_by_hour()
        perf_by_asset = self.db.get_performance_by_asset()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_stats': {
                'events_processed': self.events_processed,
                'events_saved': self.events_saved,
                'errors': self.errors
            },
            'trading_stats': self.db.get_stats_summary(),
            'performance_by_hour': perf_by_hour.to_dict() if not perf_by_hour.empty else {},
            'performance_by_asset': perf_by_asset.to_dict() if not perf_by_asset.empty else {},
            'recent_trades': trades_df.head(10).to_dict() if not trades_df.empty else {}
        }
        
        return report
    
    def close(self):
        """Cierra conexiones"""
        self.db.close()
        print("👋 Pipeline cerrado")


# Funciones de utilidad
def run_historical_processing():
    """Procesa todos los logs históricos"""
    pipeline = TradingDataPipeline()
    pipeline.process_historical_logs()
    pipeline.close()

def run_realtime_monitoring():
    """Inicia monitoreo en tiempo real"""
    pipeline = TradingDataPipeline()
    try:
        pipeline.monitor_realtime()
    finally:
        pipeline.close()

def generate_analysis_report():
    """Genera un reporte de análisis"""
    pipeline = TradingDataPipeline()
    report = pipeline.generate_report()
    
    # Guardar reporte
    report_path = Path("trading_ml_optimizer/data/analysis_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Reporte guardado en: {report_path}")
    pipeline.close()
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline de datos de trading')
    parser.add_argument('--mode', choices=['historical', 'realtime', 'report'], 
                       default='historical',
                       help='Modo de operación')
    
    args = parser.parse_args()
    
    if args.mode == 'historical':
        run_historical_processing()
    elif args.mode == 'realtime':
        run_realtime_monitoring()
    elif args.mode == 'report':
        report = generate_analysis_report()
        print("\n📊 Reporte generado:")
        print(json.dumps(report['trading_stats'], indent=2))