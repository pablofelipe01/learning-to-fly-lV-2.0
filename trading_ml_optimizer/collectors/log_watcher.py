# trading_ml_optimizer/collectors/log_watcher.py
"""
Log Watcher - Monitorea y extrae información de los logs del algoritmo de trading
"""

import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import time

class TradingLogWatcher:
    """
    Monitorea los logs del algoritmo y extrae eventos importantes
    """
    
    def __init__(self, log_file_path: str):
        """
        Inicializa el watcher
        
        Args:
            log_file_path: Ruta al archivo de log
        """
        self.log_file = Path(log_file_path)
        self.last_position = 0
        self.events = []
        
        # Patrones regex para extraer información
        self.patterns = {
            'capital_inicial': r'💰 Capital inicial: \$([0-9,]+\.\d{2})',
            'conexion': r'✅ Conexión exitosa',
            'activos_disponibles': r'📊 Total activos disponibles: (\d+)',
            'ventana_trading': r'⏰ VENTANA DE TRADING: (\d{2}:\d{2}) \[(\w+)\]',
            'orden_primaria': r'📈 Colocando (\w+) en (\w+)',
            'score_tendencia': r'Tendencia: (\w+) \| Score: ([\d.]+)',
            'tamano_posicion': r'Tamaño: \$([0-9,]+\.\d{2})',
            'orden_colocada': r'✅ Orden primaria colocada: ID (\d+)',
            'evaluacion_hedge': r'🔍 Evaluando cobertura para (\w+) \(ID: (\d+)\)',
            'precio_entrada': r'Precio entrada: ([\d.]+)',
            'precio_actual': r'Precio actual: ([\d.]+)',
            'diferencia_precio': r'Diferencia: ([-\d.]+)%',
            'estado_itm': r'Estado: (ITM ✅|OTM ❌)',
            'cobertura_colocada': r'✅ Cobertura colocada: ID (\d+)',
            'resultado_ganancia': r'✅ (\w+) GANADA: \+\$([0-9,]+\.\d{2})',
            'resultado_perdida': r'❌ (\w+) PERDIDA: -\$([0-9,]+\.\d{2})',
            'ciclo_info': r'🔄 Ciclo (\d+) \| Sesión (\w+) \((\w+)\) \| Primarias: (\d+) \| Coberturas: (\d+) \| P&L: \$([0-9,-]+\.\d{2})',
            'no_tendencias': r'⚠️ No hay tendencias válidas',
            'sesion_actual': r'📍 SESIÓN (\w+) - Ventana (\d{2}:\d{2})',
            'analisis_completado': r'✅ Análisis completado',
            'resumen_final': r'📈 Rendimiento: ([-\d.]+)%',
            'win_rate': r'🎯 Win Rate: ([\d.]+)%',
        }
        
    def parse_line(self, line: str) -> Optional[Dict]:
        """
        Parsea una línea del log y extrae información relevante
        
        Args:
            line: Línea del log
            
        Returns:
            Diccionario con la información extraída o None
        """
        # Extraer timestamp si existe
        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        timestamp = timestamp_match.group(1) if timestamp_match else None
        
        # Buscar patrones conocidos
        for pattern_name, pattern in self.patterns.items():
            match = re.search(pattern, line)
            if match:
                event = {
                    'timestamp': timestamp,
                    'event_type': pattern_name,
                    'raw_line': line.strip()
                }
                
                # Extraer datos específicos según el tipo de evento
                if pattern_name == 'capital_inicial':
                    event['capital'] = float(match.group(1).replace(',', ''))
                
                elif pattern_name == 'activos_disponibles':
                    event['num_activos'] = int(match.group(1))
                
                elif pattern_name == 'ventana_trading':
                    event['hora'] = match.group(1)
                    event['sesion'] = match.group(2)
                
                elif pattern_name == 'orden_primaria':
                    event['direccion'] = match.group(1)
                    event['activo'] = match.group(2)
                
                elif pattern_name == 'score_tendencia':
                    event['tendencia'] = match.group(1)
                    event['score'] = float(match.group(2))
                
                elif pattern_name == 'tamano_posicion':
                    event['monto'] = float(match.group(1).replace(',', ''))
                
                elif pattern_name == 'orden_colocada':
                    event['order_id'] = match.group(1)
                
                elif pattern_name == 'evaluacion_hedge':
                    event['activo'] = match.group(1)
                    event['order_id'] = match.group(2)
                
                elif pattern_name == 'precio_entrada':
                    event['precio'] = float(match.group(1))
                
                elif pattern_name == 'precio_actual':
                    event['precio'] = float(match.group(1))
                
                elif pattern_name == 'diferencia_precio':
                    event['diferencia'] = float(match.group(1))
                
                elif pattern_name == 'estado_itm':
                    event['itm'] = 'ITM' in match.group(1)
                
                elif pattern_name == 'cobertura_colocada':
                    event['hedge_id'] = match.group(1)
                
                elif pattern_name == 'resultado_ganancia':
                    event['tipo_orden'] = match.group(1)
                    event['ganancia'] = float(match.group(2).replace(',', ''))
                    event['resultado'] = 'WIN'
                
                elif pattern_name == 'resultado_perdida':
                    event['tipo_orden'] = match.group(1)
                    event['perdida'] = float(match.group(2).replace(',', ''))
                    event['resultado'] = 'LOSS'
                
                elif pattern_name == 'ciclo_info':
                    event['ciclo'] = int(match.group(1))
                    event['sesion'] = match.group(2)
                    event['estado_sesion'] = match.group(3)
                    event['num_primarias'] = int(match.group(4))
                    event['num_coberturas'] = int(match.group(5))
                    event['pnl'] = float(match.group(6).replace(',', ''))
                
                elif pattern_name == 'resumen_final':
                    event['rendimiento'] = float(match.group(1))
                
                elif pattern_name == 'win_rate':
                    event['win_rate'] = float(match.group(1))
                
                return event
        
        return None
    
    def read_new_lines(self) -> List[Dict]:
        """
        Lee nuevas líneas del archivo de log
        
        Returns:
            Lista de eventos extraídos
        """
        new_events = []
        
        if not self.log_file.exists():
            print(f"⚠️ Archivo de log no encontrado: {self.log_file}")
            return new_events
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            # Ir a la última posición conocida
            f.seek(self.last_position)
            
            # Leer nuevas líneas
            for line in f:
                event = self.parse_line(line)
                if event:
                    new_events.append(event)
                    self.events.append(event)
            
            # Actualizar posición
            self.last_position = f.tell()
        
        return new_events
    
    def get_trading_summary(self) -> Dict:
        """
        Genera un resumen de la actividad de trading
        
        Returns:
            Diccionario con estadísticas
        """
        summary = {
            'total_eventos': len(self.events),
            'ordenes_primarias': 0,
            'coberturas': 0,
            'wins': 0,
            'losses': 0,
            'pnl_total': 0.0,
            'sesiones': set(),
            'activos_operados': set()
        }
        
        for event in self.events:
            if event['event_type'] == 'orden_colocada':
                summary['ordenes_primarias'] += 1
            
            elif event['event_type'] == 'cobertura_colocada':
                summary['coberturas'] += 1
            
            elif event['event_type'] == 'resultado_ganancia':
                summary['wins'] += 1
                summary['pnl_total'] += event['ganancia']
            
            elif event['event_type'] == 'resultado_perdida':
                summary['losses'] += 1
                summary['pnl_total'] -= event['perdida']
            
            elif event['event_type'] == 'ventana_trading':
                summary['sesiones'].add(event['sesion'])
            
            elif event['event_type'] == 'orden_primaria':
                summary['activos_operados'].add(event['activo'])
        
        # Convertir sets a listas para JSON
        summary['sesiones'] = list(summary['sesiones'])
        summary['activos_operados'] = list(summary['activos_operados'])
        
        # Calcular win rate
        total_trades = summary['wins'] + summary['losses']
        summary['win_rate'] = (summary['wins'] / total_trades * 100) if total_trades > 0 else 0
        
        return summary
    
    def monitor_continuous(self, callback=None, interval=5):
        """
        Monitorea el archivo de log continuamente
        
        Args:
            callback: Función a llamar con nuevos eventos
            interval: Segundos entre verificaciones
        """
        print(f"📊 Monitoreando: {self.log_file}")
        print("Presiona Ctrl+C para detener\n")
        
        try:
            while True:
                new_events = self.read_new_lines()
                
                if new_events:
                    print(f"🔔 {len(new_events)} nuevos eventos detectados")
                    
                    for event in new_events:
                        # Mostrar evento
                        self._print_event(event)
                        
                        # Llamar callback si existe
                        if callback:
                            callback(event)
                    
                    # Mostrar resumen actualizado
                    summary = self.get_trading_summary()
                    print(f"\n📈 Resumen: Trades: {summary['ordenes_primarias']} | "
                          f"Wins: {summary['wins']} | Losses: {summary['losses']} | "
                          f"P&L: ${summary['pnl_total']:.2f}\n")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n✋ Monitoreo detenido")
            return self.get_trading_summary()
    
    def _print_event(self, event: Dict):
        """
        Imprime un evento de forma legible
        """
        event_type = event['event_type']
        timestamp = event.get('timestamp', 'N/A')
        
        if event_type == 'orden_primaria':
            print(f"  📍 [{timestamp}] ORDEN: {event['direccion']} en {event['activo']}")
        
        elif event_type == 'score_tendencia':
            print(f"     Score: {event['score']:.3f} | Tendencia: {event['tendencia']}")
        
        elif event_type == 'estado_itm':
            status = "✅ ITM" if event['itm'] else "❌ OTM"
            print(f"     Estado de hedge: {status}")
        
        elif event_type == 'resultado_ganancia':
            print(f"  💰 [{timestamp}] WIN: +${event['ganancia']:.2f} ({event['tipo_orden']})")
        
        elif event_type == 'resultado_perdida':
            print(f"  💸 [{timestamp}] LOSS: -${event['perdida']:.2f} ({event['tipo_orden']})")


# Ejemplo de uso
if __name__ == "__main__":
    # Ruta al archivo de log
    log_path = "trend_strategy.log"
    
    # Crear watcher
    watcher = TradingLogWatcher(log_path)
    
    # Opción 1: Leer logs existentes una vez
    print("📖 Leyendo logs existentes...\n")
    events = watcher.read_new_lines()
    
    if events:
        print(f"✅ {len(events)} eventos encontrados\n")
        summary = watcher.get_trading_summary()
        print("📊 RESUMEN DE TRADING:")
        print(json.dumps(summary, indent=2))
    else:
        print("⚠️ No se encontraron eventos en el log")
    
    # Opción 2: Monitorear continuamente (descomenta para usar)
    # print("\n🔄 Iniciando monitoreo continuo...")
    # final_summary = watcher.monitor_continuous(interval=5)
    # print("\n📊 RESUMEN FINAL:")
    # print(json.dumps(final_summary, indent=2))