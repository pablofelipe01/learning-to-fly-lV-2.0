# trading_ml_optimizer/database/db_manager.py
"""
Database Manager - Sistema de almacenamiento para datos de trading
Usa SQLite para simplicidad y portabilidad
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

class TradingDatabase:
    """
    Gestiona la base de datos de trading
    """
    
    def __init__(self, db_path: str = "trading_ml_optimizer/data/trading_data.db"):
        """
        Inicializa la conexión a la base de datos
        
        Args:
            db_path: Ruta al archivo de base de datos
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establece conexión con la base de datos"""
        self.conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        self.conn.row_factory = sqlite3.Row  # Para acceder por nombre de columna
        self.cursor = self.conn.cursor()
    
    def _create_tables(self):
        """Crea las tablas necesarias si no existen"""
        
        # Tabla de sesiones de trading
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                initial_capital REAL,
                final_capital REAL,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de trades
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                trade_id TEXT UNIQUE,
                asset TEXT,
                direction TEXT,
                trade_type TEXT,  -- PRIMARY o HEDGE
                amount REAL,
                entry_time TIMESTAMP,
                entry_price REAL,
                exit_time TIMESTAMP,
                exit_price REAL,
                result TEXT,  -- WIN, LOSS, TIE
                pnl REAL,
                score REAL,
                trend_direction TEXT,
                has_hedge BOOLEAN DEFAULT 0,
                hedge_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES trading_sessions(id)
            )
        """)
        
        # Tabla de análisis de tendencias
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trend_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                asset TEXT,
                direction TEXT,
                strength REAL,
                consistency REAL,
                movement REAL,
                retracements INTEGER,
                smoothness REAL,
                momentum REAL,
                score REAL,
                is_valid BOOLEAN,
                current_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de eventos del sistema
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                event_type TEXT,
                event_data TEXT,  -- JSON
                session TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de métricas por sesión de mercado
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_session_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,  -- NY_OPEN, OVERLAP, POWER, POST
                date DATE,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_score REAL,
                best_asset TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_name, date)
            )
        """)
        
        # Tabla de métricas por activo
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS asset_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT,
                date DATE,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_score REAL,
                best_session TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(asset, date)
            )
        """)
        
        self.conn.commit()
    
    def save_event(self, event: Dict) -> int:
        """
        Guarda un evento del log
        
        Args:
            event: Diccionario con datos del evento
            
        Returns:
            ID del evento guardado
        """
        # Determinar el tipo de evento y guardarlo apropiadamente
        event_type = event.get('event_type', '')
        
        if event_type == 'orden_colocada':
            return self._save_trade_start(event)
        elif event_type in ['resultado_ganancia', 'resultado_perdida']:
            return self._update_trade_result(event)
        elif event_type == 'score_tendencia':
            return self._save_trend_analysis(event)
        else:
            return self._save_system_event(event)
    
    def _save_trade_start(self, event: Dict) -> int:
        """Guarda el inicio de un trade"""
        self.cursor.execute("""
            INSERT OR IGNORE INTO trades (
                trade_id, asset, direction, trade_type, amount, 
                entry_time, score, trend_direction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.get('order_id'),
            event.get('activo'),
            event.get('direccion'),
            'PRIMARY',
            event.get('monto', 0),
            event.get('timestamp'),
            event.get('score', 0),
            event.get('tendencia')
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def _update_trade_result(self, event: Dict) -> int:
        """Actualiza el resultado de un trade"""
        # Aquí actualizarías el trade con el resultado
        # Por simplicidad, guardamos como evento
        return self._save_system_event(event)
    
    def _save_trend_analysis(self, event: Dict) -> int:
        """Guarda análisis de tendencia"""
        self.cursor.execute("""
            INSERT INTO trend_analysis (
                timestamp, asset, direction, score
            ) VALUES (?, ?, ?, ?)
        """, (
            event.get('timestamp'),
            event.get('activo'),
            event.get('tendencia'),
            event.get('score', 0)
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def _save_system_event(self, event: Dict) -> int:
        """Guarda un evento genérico del sistema"""
        self.cursor.execute("""
            INSERT INTO system_events (
                timestamp, event_type, event_data, session
            ) VALUES (?, ?, ?, ?)
        """, (
            event.get('timestamp'),
            event.get('event_type'),
            json.dumps(event),
            event.get('sesion')
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_stats_summary(self) -> Dict:
        """
        Obtiene un resumen de estadísticas
        
        Returns:
            Diccionario con estadísticas generales
        """
        # Total de trades
        self.cursor.execute("SELECT COUNT(*) as total FROM trades")
        total_trades = self.cursor.fetchone()['total']
        
        # Wins y losses
        self.cursor.execute("""
            SELECT 
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl
            FROM trades
        """)
        result = self.cursor.fetchone()
        
        # Mejor activo
        self.cursor.execute("""
            SELECT asset, COUNT(*) as trades, SUM(pnl) as total_pnl
            FROM trades
            WHERE asset IS NOT NULL
            GROUP BY asset
            ORDER BY total_pnl DESC
            LIMIT 1
        """)
        best_asset = self.cursor.fetchone()
        
        # Mejor sesión
        self.cursor.execute("""
            SELECT session_name, SUM(pnl) as total_pnl
            FROM market_session_stats
            GROUP BY session_name
            ORDER BY total_pnl DESC
            LIMIT 1
        """)
        best_session = self.cursor.fetchone()
        
        return {
            'total_trades': total_trades,
            'wins': result['wins'] or 0,
            'losses': result['losses'] or 0,
            'total_pnl': result['total_pnl'] or 0,
            'best_asset': best_asset['asset'] if best_asset else None,
            'best_session': best_session['session_name'] if best_session else None
        }
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Obtiene todos los trades como DataFrame de pandas
        
        Returns:
            DataFrame con los trades
        """
        query = """
            SELECT * FROM trades 
            ORDER BY entry_time DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_trend_analysis_dataframe(self) -> pd.DataFrame:
        """
        Obtiene análisis de tendencias como DataFrame
        
        Returns:
            DataFrame con análisis de tendencias
        """
        query = """
            SELECT * FROM trend_analysis 
            ORDER BY timestamp DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_performance_by_hour(self) -> pd.DataFrame:
        """
        Obtiene rendimiento por hora del día
        
        Returns:
            DataFrame con rendimiento por hora
        """
        query = """
            SELECT 
                strftime('%H', entry_time) as hour,
                COUNT(*) as trades,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as total_pnl
            FROM trades
            WHERE entry_time IS NOT NULL
            GROUP BY hour
            ORDER BY hour
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_performance_by_asset(self) -> pd.DataFrame:
        """
        Obtiene rendimiento por activo
        
        Returns:
            DataFrame con rendimiento por activo
        """
        query = """
            SELECT 
                asset,
                COUNT(*) as trades,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(score) as avg_score
            FROM trades
            WHERE asset IS NOT NULL
            GROUP BY asset
            ORDER BY total_pnl DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def close(self):
        """Cierra la conexión a la base de datos"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Para usar con context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cierra la conexión al salir del context"""
        self.close()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear/conectar a la base de datos
    db = TradingDatabase()
    
    print("📊 Base de datos creada/conectada")
    
    # Obtener estadísticas
    stats = db.get_stats_summary()
    print("\n📈 Estadísticas actuales:")
    print(json.dumps(stats, indent=2))
    
    # Guardar un evento de ejemplo
    test_event = {
        'timestamp': datetime.now().isoformat(),
        'event_type': 'test_event',
        'data': 'Prueba de base de datos'
    }
    event_id = db.save_event(test_event)
    print(f"\n✅ Evento de prueba guardado con ID: {event_id}")
    
    # Cerrar conexión
    db.close()
    print("\n👋 Base de datos cerrada")