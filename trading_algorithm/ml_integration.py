# trading_algorithm/ml_integration.py
"""
ML Integration - Asesor de Machine Learning para el Algoritmo de Trading
Proporciona predicciones y recomendaciones sin interferir con la operación
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

# Agregar path para importar el ML Optimizer
sys.path.append(str(Path(__file__).parent.parent))

try:
    from trading_ml_optimizer.analyzers.ml_analyzer import TradingMLAnalyzer
    from trading_ml_optimizer.database.db_manager import TradingDatabase
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML Optimizer no disponible: {e}")
    ML_AVAILABLE = False


class MLAdvisor:
    """
    Asesor de ML que proporciona predicciones y recomendaciones
    sin interferir con la operación del algoritmo
    """
    
    def __init__(self, mode: str = "advisory", log_decisions: bool = True):
        """
        Inicializa el ML Advisor
        
        Args:
            mode: Modo de operación ('advisory', 'protective', 'full')
            log_decisions: Si guardar las decisiones para análisis
        """
        self.mode = mode
        self.log_decisions = log_decisions
        self.enabled = ML_AVAILABLE
        
        # Configuración por modo
        self.config = self._get_mode_config(mode)
        
        # Inicializar ML Analyzer si está disponible
        if ML_AVAILABLE:
            try:
                self.analyzer = TradingMLAnalyzer()
                self.db = TradingDatabase()
                print(f"✅ ML Advisor inicializado en modo: {mode.upper()}")
            except Exception as e:
                print(f"❌ Error inicializando ML: {e}")
                self.enabled = False
        else:
            print("⚠️ ML Advisor deshabilitado (ML Optimizer no encontrado)")
            self.enabled = False
        
        # Logger para decisiones
        self.setup_logger()
        
        # Estadísticas de sesión
        self.session_stats = {
            'trades_analyzed': 0,
            'trades_approved': 0,
            'trades_rejected': 0,
            'ml_predictions': [],
            'start_time': datetime.now()
        }
    
    def _get_mode_config(self, mode: str) -> Dict:
        """
        Obtiene configuración según el modo
        
        Args:
            mode: Modo de operación
            
        Returns:
            Diccionario con configuración
        """
        configs = {
            'advisory': {
                'min_confidence': 0.0,  # No bloquea nada
                'position_sizing': False,  # No ajusta tamaños
                'hedge_control': False,  # No controla hedges
                'blocking': False  # Solo sugiere
            },
            'protective': {
                'min_confidence': 0.35,  # Bloquea si < 35% prob
                'position_sizing': False,
                'hedge_control': False,
                'blocking': True  # Puede bloquear trades peligrosos
            },
            'full': {
                'min_confidence': 0.55,  # Requiere > 55% prob
                'position_sizing': True,  # Ajusta tamaños
                'hedge_control': True,  # Controla hedges
                'blocking': True  # Control total
            }
        }
        
        return configs.get(mode, configs['advisory'])
    
    def setup_logger(self):
        """Configura logger para decisiones del ML"""
        log_file = Path("ml_decisions.log")
        
        self.logger = logging.getLogger('MLAdvisor')
        self.logger.setLevel(logging.INFO)
        
        # Handler para archivo
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Formato
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
    
    def should_take_trade(self, conditions: Dict) -> Tuple[bool, Dict]:
        """
        Evalúa si se debe tomar un trade
        
        Args:
            conditions: Condiciones del trade potencial
            
        Returns:
            (should_trade, analysis_details)
        """
        self.session_stats['trades_analyzed'] += 1
        
        # Si ML no está disponible, aprobar siempre
        if not self.enabled:
            return True, {'reason': 'ML no disponible', 'ml_used': False}
        
        try:
            # Preparar condiciones para el ML
            ml_conditions = self._prepare_ml_conditions(conditions)
            
            # Obtener predicción
            prediction = self.analyzer.predict_trade_outcome(ml_conditions)
            
            if not prediction:
                # No hay modelo entrenado aún
                analysis = {
                    'ml_used': False,
                    'reason': 'Modelo no entrenado',
                    'recommendation': 'NEUTRAL',
                    'confidence': 0.5
                }
                should_trade = True  # No bloquear si no hay modelo
            else:
                # Analizar predicción
                win_prob = prediction.get('probability_win', 0.5)
                confidence = prediction.get('confidence', 0.5)
                
                # Decisión según modo
                if self.config['blocking']:
                    should_trade = win_prob >= self.config['min_confidence']
                else:
                    should_trade = True  # Advisory no bloquea
                
                # Construir análisis
                analysis = {
                    'ml_used': True,
                    'prediction': prediction.get('prediction', 'UNKNOWN'),
                    'win_probability': win_prob,
                    'confidence': confidence,
                    'recommendation': self._get_recommendation(win_prob),
                    'reason': self._get_reason(win_prob, should_trade),
                    'mode': self.mode,
                    'blocked': not should_trade and self.config['blocking']
                }
                
                # Registrar estadísticas
                if should_trade:
                    self.session_stats['trades_approved'] += 1
                else:
                    self.session_stats['trades_rejected'] += 1
                
                self.session_stats['ml_predictions'].append({
                    'timestamp': datetime.now().isoformat(),
                    'conditions': conditions,
                    'prediction': analysis
                })
            
            # Loggear decisión
            if self.log_decisions:
                self._log_decision(conditions, analysis, should_trade)
            
            return should_trade, analysis
            
        except Exception as e:
            self.logger.error(f"Error en evaluación ML: {e}")
            return True, {'error': str(e), 'ml_used': False}
    
    def _prepare_ml_conditions(self, conditions: Dict) -> Dict:
        """
        Prepara las condiciones en el formato esperado por el ML
        
        Args:
            conditions: Condiciones raw del algoritmo
            
        Returns:
            Condiciones formateadas para ML
        """
        # Mapear hora a número si viene como string
        hour = conditions.get('hour', datetime.now().hour)
        if isinstance(hour, str):
            hour = int(hour.split(':')[0]) if ':' in hour else int(hour)
        
        # Preparar condiciones ML
        ml_conditions = {
            'hour': hour,
            'day_of_week': conditions.get('day_of_week', datetime.now().weekday()),
            'is_weekend': datetime.now().weekday() >= 5,
            'market_session': conditions.get('session', 'OFF_HOURS'),
            'asset': conditions.get('asset', 'UNKNOWN'),
            'score': conditions.get('score', 0.0),
            'amount': conditions.get('amount', 0.0),
            'has_hedge': conditions.get('has_hedge', 0),
            'is_call': 1 if conditions.get('direction', '').lower() == 'call' else 0,
            'trend_up': 1 if conditions.get('trend_direction', '').upper() == 'UP' else 0,
            'volatility_estimate': conditions.get('volatility', 2)
        }
        
        return ml_conditions
    
    def _get_recommendation(self, win_prob: float) -> str:
        """
        Obtiene recomendación basada en probabilidad
        
        Args:
            win_prob: Probabilidad de ganar
            
        Returns:
            Recomendación textual
        """
        if win_prob >= 0.75:
            return "HIGHLY_RECOMMENDED"
        elif win_prob >= 0.65:
            return "RECOMMENDED"
        elif win_prob >= 0.55:
            return "NEUTRAL_POSITIVE"
        elif win_prob >= 0.45:
            return "NEUTRAL"
        elif win_prob >= 0.35:
            return "NEUTRAL_NEGATIVE"
        elif win_prob >= 0.25:
            return "NOT_RECOMMENDED"
        else:
            return "STRONGLY_AVOID"
    
    def _get_reason(self, win_prob: float, should_trade: bool) -> str:
        """
        Genera razón de la decisión
        
        Args:
            win_prob: Probabilidad de ganar
            should_trade: Si se aprobó el trade
            
        Returns:
            Razón en texto
        """
        prob_text = f"Win prob: {win_prob:.1%}"
        
        if self.mode == 'advisory':
            return f"{prob_text} (Advisory - no bloquea)"
        elif should_trade:
            return f"{prob_text} - Aprobado"
        else:
            return f"{prob_text} - Bloqueado (< {self.config['min_confidence']:.0%})"
    
    def get_position_size_multiplier(self, confidence: float, 
                                    win_probability: float) -> float:
        """
        Calcula multiplicador para el tamaño de posición
        
        Args:
            confidence: Confianza del modelo
            win_probability: Probabilidad de ganar
            
        Returns:
            Multiplicador (0.5 a 1.5)
        """
        if not self.config['position_sizing']:
            return 1.0  # No ajustar si no está habilitado
        
        # Fórmula conservadora basada en Kelly Criterion
        if win_probability >= 0.75 and confidence >= 0.8:
            return 1.5  # 150% del tamaño base
        elif win_probability >= 0.65 and confidence >= 0.7:
            return 1.25  # 125%
        elif win_probability >= 0.55:
            return 1.0  # 100% (normal)
        elif win_probability >= 0.45:
            return 0.75  # 75%
        else:
            return 0.5  # 50% (mínimo)
    
    def should_place_hedge(self, conditions: Dict) -> Tuple[bool, Dict]:
        """
        Evalúa si se debe colocar hedge
        
        Args:
            conditions: Condiciones actuales del trade
            
        Returns:
            (should_hedge, analysis)
        """
        if not self.config['hedge_control']:
            return True, {'reason': 'Control de hedge deshabilitado'}
        
        # Aquí implementarías lógica ML para hedge
        # Por ahora, mantener comportamiento estándar
        return True, {'reason': 'Hedge estándar', 'confidence': 0.75}
    
    def _log_decision(self, conditions: Dict, analysis: Dict, decision: bool):
        """
        Registra la decisión en el log
        
        Args:
            conditions: Condiciones evaluadas
            analysis: Análisis realizado
            decision: Decisión tomada
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'asset': conditions.get('asset', 'N/A'),
            'score': conditions.get('score', 0),
            'ml_prediction': analysis.get('prediction', 'N/A'),
            'win_probability': analysis.get('win_probability', 0),
            'recommendation': analysis.get('recommendation', 'N/A'),
            'decision': 'APPROVED' if decision else 'REJECTED',
            'blocked_by_ml': analysis.get('blocked', False)
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def get_session_report(self) -> Dict:
        """
        Obtiene reporte de la sesión actual
        
        Returns:
            Diccionario con estadísticas
        """
        runtime = (datetime.now() - self.session_stats['start_time']).total_seconds() / 60
        
        report = {
            'mode': self.mode,
            'runtime_minutes': round(runtime, 1),
            'trades_analyzed': self.session_stats['trades_analyzed'],
            'trades_approved': self.session_stats['trades_approved'],
            'trades_rejected': self.session_stats['trades_rejected'],
            'approval_rate': (
                self.session_stats['trades_approved'] / 
                max(1, self.session_stats['trades_analyzed'])
            ),
            'ml_available': self.enabled
        }
        
        # Agregar análisis de predicciones si hay
        if self.session_stats['ml_predictions']:
            predictions = self.session_stats['ml_predictions']
            win_probs = [p['prediction']['win_probability'] 
                        for p in predictions 
                        if 'win_probability' in p.get('prediction', {})]
            
            if win_probs:
                report['avg_win_probability'] = sum(win_probs) / len(win_probs)
                report['max_win_probability'] = max(win_probs)
                report['min_win_probability'] = min(win_probs)
        
        return report
    
    def reset_session_stats(self):
        """Reinicia estadísticas de sesión"""
        self.session_stats = {
            'trades_analyzed': 0,
            'trades_approved': 0,
            'trades_rejected': 0,
            'ml_predictions': [],
            'start_time': datetime.now()
        }
        print("📊 Estadísticas de sesión reiniciadas")


# Función de prueba
def test_ml_advisor():
    """Prueba el ML Advisor con condiciones simuladas"""
    print("\n🧪 PROBANDO ML ADVISOR")
    print("="*50)
    
    # Crear advisor en modo advisory
    advisor = MLAdvisor(mode='advisory')
    
    # Condiciones de prueba
    test_conditions = [
        {
            'asset': 'EURUSD',
            'score': 0.72,
            'hour': 10,
            'session': 'OVERLAP',
            'direction': 'call',
            'trend_direction': 'UP',
            'amount': 200
        },
        {
            'asset': 'GBPUSD',
            'score': 0.58,
            'hour': 14,
            'session': 'POWER',
            'direction': 'put',
            'trend_direction': 'DOWN',
            'amount': 200
        },
        {
            'asset': 'BTCUSD',
            'score': 0.81,
            'hour': 17,
            'session': 'POST',
            'direction': 'call',
            'trend_direction': 'UP',
            'amount': 200
        }
    ]
    
    # Probar cada condición
    for i, conditions in enumerate(test_conditions, 1):
        print(f"\n📊 Test {i}: {conditions['asset']}")
        print(f"  Score: {conditions['score']}")
        print(f"  Sesión: {conditions['session']}")
        
        should_trade, analysis = advisor.should_take_trade(conditions)
        
        print(f"\n  Resultado ML:")
        print(f"    Decisión: {'✅ APROBAR' if should_trade else '❌ RECHAZAR'}")
        print(f"    ML usado: {analysis.get('ml_used', False)}")
        
        if analysis.get('ml_used'):
            print(f"    Predicción: {analysis.get('prediction', 'N/A')}")
            print(f"    Prob. WIN: {analysis.get('win_probability', 0):.1%}")
            print(f"    Recomendación: {analysis.get('recommendation', 'N/A')}")
        
        print(f"    Razón: {analysis.get('reason', 'N/A')}")
    
    # Mostrar reporte de sesión
    print("\n" + "="*50)
    print("📈 REPORTE DE SESIÓN")
    report = advisor.get_session_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Prueba completada")


if __name__ == "__main__":
    test_ml_advisor()