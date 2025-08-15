# trend_strategy_ml.py
# Estrategia de tendencia con protección dinámica para opciones binarias
# Identifica tendencias claras y coloca coberturas condicionales

import time
import json
import os
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import threading

from iqoptionapi.stable_api import IQ_Option

from trading_algorithm.config_trend import (
    IQ_EMAIL, IQ_PASSWORD, ACCOUNT_TYPE,
    TRADING_WINDOWS, TRADING_SESSIONS, EXPIRY_MINUTES, HEDGE_CHECK_MINUTE, HEDGE_WAIT_SECONDS,
    WARMUP_PERIOD, SCAN_INTERVAL, TREND_CANDLE_TIMEFRAME, TREND_LOOKBACK_CANDLES,
    MIN_TREND_STRENGTH, MIN_TREND_CONSISTENCY, MIN_PRICE_MOVEMENT, MAX_RETRACEMENTS,
    TREND_WEIGHTS, MAX_SIMULTANEOUS_PRIMARY, MAX_SIMULTANEOUS_HEDGE,
    POSITION_SIZE_PERCENT, HEDGE_SIZE_MULTIPLIER, MIN_POSITION_SIZE,
    DAILY_MAX_TRADES, DAILY_LOSS_LIMIT, DAILY_PROFIT_TARGET,
    TRADING_ASSETS, get_asset_group, get_trend_config_for_asset, get_current_session,
    LOG_LEVEL, LOG_FILE, STATE_FILE, SAVE_STATE_INTERVAL
)
from trading_algorithm.utils_trend import (
    setup_logger, calculate_trend_strength, calculate_trend_consistency,
    detect_trend_direction, format_currency, calculate_win_rate
)
# ============ ML ADVISOR INTEGRATION ============
try:
    from trading_algorithm.ml_integration import MLAdvisor
    ML_ADVISOR = MLAdvisor(mode='advisory', log_decisions=True)
    USE_ML_ADVISOR = True
    print("🤖 ML Advisor cargado en modo ADVISORY")
except:
    ML_ADVISOR = None
    USE_ML_ADVISOR = False
    print("⚠️ ML Advisor no disponible")
# ================================================
class TrendProtectionStrategy:
    def __init__(self, email, password, account_type="PRACTICE", skip_warmup=False, session_filter="TODAS"):
        """
        Inicializar estrategia de tendencia con protección dinámica
        
        Args:
            email: Email de IQ Option
            password: Contraseña
            account_type: Tipo de cuenta
            skip_warmup: Omitir período de análisis inicial
            session_filter: Filtro de sesión ("TODAS", "NY_OPEN", "OVERLAP", "POWER", "POST")
        """
        # Filtro de sesión
        self.session_filter = session_filter
        
        # Logger
        self.logger = setup_logger(__name__, LOG_FILE, getattr(logging, LOG_LEVEL))
        self.logger.info("="*60)
        self.logger.info("📈 INICIANDO ESTRATEGIA DE TENDENCIA CON PROTECCIÓN")
        self.logger.info("="*60)
        
        # Mostrar filtro de sesión si está activo
        if self.session_filter != "TODAS":
            self.logger.info(f"🎯 Filtro de sesión activo: Solo operar en {self.session_filter}")
        else:
            self.logger.info("🎯 Operando en TODAS las sesiones disponibles")
        
        # Período de calentamiento/análisis DINÁMICO
        from trading_algorithm.config_trend import get_dynamic_warmup_period
        self.warmup_period, warmup_type = get_dynamic_warmup_period()
        self.warmup_start_time = time.time()
        self.is_warming_up = not skip_warmup and self.warmup_period > 0
        self.trend_scores = defaultdict(dict)  # Scores de tendencia por activo
        self.trend_history = defaultdict(list)  # Historial de análisis
        
        if self.is_warming_up:
            self.logger.info(f"🔥 PERÍODO DE ANÁLISIS: {warmup_type}")
            self.logger.info(f"⏱️ Duración: {self.warmup_period//60} minutos")
            self.logger.info("📊 Objetivo: Identificar activos con tendencias claras")
            
            # Calcular a qué hora termina el calentamiento
            from datetime import datetime, timedelta
            end_time = datetime.now() + timedelta(seconds=self.warmup_period)
            self.logger.info(f"✅ Análisis terminará a las: {end_time.strftime('%H:%M')}")
            self.logger.info("="*60)
        elif skip_warmup:
            self.logger.info("⚠️ Período de análisis OMITIDO por solicitud del usuario")
        else:
            self.logger.info("⚡ Período de análisis OMITIDO - Sesión muy próxima")
            self.logger.info("   Comenzando operaciones directamente")
        
        # Conectar a IQ Option
        self._connect_to_iq_option(email, password, account_type)
        
        # Capital y gestión
        self.initial_capital = self.iqoption.get_balance()
        self.logger.info(f"💰 Capital inicial: {format_currency(self.initial_capital)}")
        
        # Control de operaciones
        self.primary_orders = {}  # Órdenes primarias activas
        self.hedge_orders = {}    # Coberturas activas
        self.pending_hedges = {}  # Coberturas pendientes de evaluación
        
        # Estadísticas
        self.daily_trades = 0
        self.daily_profit = 0.0
        self.wins = 0
        self.losses = 0
        self.hedges_placed = 0
        self.hedges_avoided = 0
        
        # Control de ventanas de trading
        self.last_window_traded = None
        self.current_window = None
        self.current_session = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Validar activos disponibles
        self.valid_assets = []
        self.asset_mapping = {}
        self.check_valid_assets()
        
        # Mostrar sesión actual/próxima
        session, status = get_current_session()
        if status == "ACTIVA":
            session_info = TRADING_SESSIONS.get(session, {})
            self.logger.info(f"📍 Sesión {session} actualmente ACTIVA")
            self.logger.info(f"   {session_info.get('description', '')}")
        else:
            session_info = TRADING_SESSIONS.get(session, {})
            self.logger.info(f"⏰ Próxima sesión: {session} ({session_info.get('start_hour', '')}:00)")
            self.logger.info(f"   {session_info.get('description', '')}")
        
        # Cargar estado previo
        self.load_state()
        
        self.logger.info("✅ Sistema inicializado correctamente")
        # ML Advisor stats
        self.ml_decisions = {
            'total_consultations': 0,
            'ml_approved': 0,
            'ml_rejected': 0,
            'ml_predictions': []
        }
        
        if USE_ML_ADVISOR:
            self.logger.info("🤖 ML Advisor activo - Modo: ADVISORY (solo observa)")
        else:
            self.logger.info("📊 ML Advisor no disponible - Operando normal")
    
    def _connect_to_iq_option(self, email, password, account_type):
        """Conectar a IQ Option"""
        self.logger.info("🔗 Conectando a IQ Option...")
        self.iqoption = IQ_Option(email, password)
        login_status, login_reason = self.iqoption.connect()
        
        if not login_status:
            self.logger.error(f"❌ Error al conectar: {login_reason}")
            raise Exception(f"Error al conectar: {login_reason}")
        
        self.logger.info("✅ Conexión exitosa")
        self.iqoption.change_balance(account_type)
    
    def check_valid_assets(self):
        """Verificar activos disponibles"""
        self.logger.info("🔍 Verificando activos disponibles...")
        
        try:
            self.iqoption.update_ACTIVES_OPCODE()
            all_assets = self.iqoption.get_all_open_time()
            
            if not all_assets:
                self.logger.error("❌ No se pudieron obtener activos")
                return
            
            self.valid_assets = []
            for asset in TRADING_ASSETS:
                found = False
                
                # Buscar en binary y turbo
                for option_type in ["binary", "turbo"]:
                    if option_type not in all_assets:
                        continue
                    
                    # Buscar variantes del activo
                    variants = [asset.upper(), f"{asset.upper()}-OTC"]
                    
                    for variant in variants:
                        if variant in all_assets[option_type]:
                            if all_assets[option_type][variant].get("open", False):
                                self.valid_assets.append(asset)
                                self.asset_mapping[asset] = {
                                    "name": variant,
                                    "type": option_type
                                }
                                self.logger.info(f"✅ {asset}: Disponible como {variant}")
                                found = True
                                break
                    
                    if found:
                        break
                
                if not found:
                    self.logger.debug(f"⚠️ {asset}: No disponible")
            
            self.logger.info(f"📊 Total activos disponibles: {len(self.valid_assets)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error verificando activos: {str(e)}")
    
    def analyze_trend(self, asset):
        """
        Analizar tendencia de un activo
        
        Returns:
            dict: Información de tendencia con score
        """
        try:
            # Obtener velas históricas
            candles = self.iqoption.get_candles(
                self.asset_mapping[asset]["name"],
                TREND_CANDLE_TIMEFRAME,
                TREND_LOOKBACK_CANDLES,
                time.time()
            )
            
            if not candles or len(candles) < 10:
                return None
            
            # Extraer precios
            closes = [float(c['close']) for c in candles]
            opens = [float(c['open']) for c in candles]
            highs = [float(c['max']) for c in candles]
            lows = [float(c['min']) for c in candles]
            
            # Calcular métricas de tendencia
            direction = detect_trend_direction(closes)
            strength = calculate_trend_strength(closes)
            consistency = calculate_trend_consistency(closes)
            
            # Calcular movimiento total
            price_movement = abs(closes[-1] - closes[0]) / closes[0]
            
            # Contar retrocesos
            retracements = 0
            for i in range(1, len(closes)):
                if direction == "UP" and closes[i] < closes[i-1]:
                    retracements += 1
                elif direction == "DOWN" and closes[i] > closes[i-1]:
                    retracements += 1
            
            # Calcular suavidad (menor variación = más suave)
            returns = np.diff(closes) / closes[:-1]
            smoothness = 1 - np.std(returns)
            
            # Calcular momentum actual (últimas 3 velas)
            recent_movement = (closes[-1] - closes[-3]) / closes[-3]
            momentum = abs(recent_movement)
            
            # Score final ponderado
            config = get_trend_config_for_asset(asset)
            
            score = (
                TREND_WEIGHTS["strength"] * strength +
                TREND_WEIGHTS["consistency"] * consistency +
                TREND_WEIGHTS["momentum"] * momentum * config["momentum_weight"] +
                TREND_WEIGHTS["smoothness"] * smoothness
            )
            
            # Validar criterios mínimos
            is_valid = (
                strength >= MIN_TREND_STRENGTH and
                consistency >= MIN_TREND_CONSISTENCY and
                price_movement >= config["min_movement"] and
                retracements <= MAX_RETRACEMENTS and
                smoothness >= config["smoothness_threshold"]
            )
            
            return {
                "asset": asset,
                "direction": direction,
                "strength": strength,
                "consistency": consistency,
                "movement": price_movement,
                "retracements": retracements,
                "smoothness": smoothness,
                "momentum": momentum,
                "score": score if is_valid else 0,
                "is_valid": is_valid,
                "current_price": closes[-1],
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analizando {asset}: {str(e)}")
            return None
    
    def warmup_analysis(self):
        """Realizar análisis durante el período de calentamiento"""
        self.logger.info("🔥 Iniciando análisis de tendencias...")
        start_time = time.time()
        
        while time.time() - start_time < self.warmup_period:
            elapsed = time.time() - start_time
            remaining = self.warmup_period - elapsed
            
            # Mostrar progreso cada minuto
            if int(elapsed) % 60 == 0:
                mins_remaining = int(remaining // 60)
                self.logger.info(f"⏱️ Análisis: {mins_remaining} minutos restantes")
            
            # Analizar todos los activos
            for asset in self.valid_assets:
                trend_data = self.analyze_trend(asset)
                if trend_data:
                    self.trend_scores[asset] = trend_data
                    self.trend_history[asset].append(trend_data)
            
            # Mostrar mejores tendencias actuales
            if int(elapsed) % 300 == 0:  # Cada 5 minutos
                self.show_top_trends()
            
            time.sleep(SCAN_INTERVAL)
        
        self.logger.info("✅ Análisis completado")
        self.show_analysis_summary()
        self.is_warming_up = False
    
    def show_top_trends(self):
        """Mostrar las mejores tendencias identificadas"""
        valid_trends = [
            t for t in self.trend_scores.values() 
            if t.get("is_valid", False)
        ]
        
        if not valid_trends:
            self.logger.info("📊 No hay tendencias válidas en este momento")
            return
        
        # Ordenar por score
        top_trends = sorted(valid_trends, key=lambda x: x["score"], reverse=True)[:5]
        
        self.logger.info("\n🏆 TOP TENDENCIAS ACTUALES:")
        for i, trend in enumerate(top_trends, 1):
            self.logger.info(
                f"  {i}. {trend['asset']}: {trend['direction']} | "
                f"Score: {trend['score']:.2f} | "
                f"Fuerza: {trend['strength']:.2%} | "
                f"Consistencia: {trend['consistency']:.2%}"
            )
    
    def show_analysis_summary(self):
        """Mostrar resumen del análisis de calentamiento"""
        self.logger.info("="*60)
        self.logger.info("📊 RESUMEN DEL ANÁLISIS DE TENDENCIAS")
        self.logger.info("="*60)
        
        valid_trends = [t for t in self.trend_scores.values() if t.get("is_valid", False)]
        
        if valid_trends:
            self.logger.info(f"✅ Tendencias válidas encontradas: {len(valid_trends)}")
            
            # Agrupar por dirección
            up_trends = [t for t in valid_trends if t["direction"] == "UP"]
            down_trends = [t for t in valid_trends if t["direction"] == "DOWN"]
            
            self.logger.info(f"📈 Tendencias alcistas: {len(up_trends)}")
            self.logger.info(f"📉 Tendencias bajistas: {len(down_trends)}")
            
            # Mejor tendencia
            best_trend = max(valid_trends, key=lambda x: x["score"])
            self.logger.info(f"\n🏆 MEJOR TENDENCIA:")
            self.logger.info(f"  Activo: {best_trend['asset']}")
            self.logger.info(f"  Dirección: {best_trend['direction']}")
            self.logger.info(f"  Score: {best_trend['score']:.3f}")
            self.logger.info(f"  Fuerza: {best_trend['strength']:.2%}")
            self.logger.info(f"  Movimiento: {best_trend['movement']:.2%}")
        else:
            self.logger.warning("⚠️ No se encontraron tendencias válidas")
            self.logger.info("💡 Continuando análisis durante el trading...")
        
        self.logger.info("="*60)
    
    def get_best_trending_asset(self):
        """Obtener el activo con mejor tendencia actual"""
        # Actualizar análisis de tendencias
        for asset in self.valid_assets:
            trend_data = self.analyze_trend(asset)
            if trend_data:
                self.trend_scores[asset] = trend_data
        
        # Filtrar tendencias válidas
        valid_trends = [
            t for t in self.trend_scores.values() 
            if t.get("is_valid", False)
        ]
        
        if not valid_trends:
            return None
        
        # Retornar la mejor tendencia
        return max(valid_trends, key=lambda x: x["score"])
    
    def place_primary_order(self, asset, direction, trend_data):
        """Colocar orden primaria según tendencia"""
        try:
            # ========== INICIO INTEGRACIÓN ML ==========
            ml_decision = None
            ml_should_trade = True  # Default: aprobar
            
            if USE_ML_ADVISOR and ML_ADVISOR:
                try:
                    # Preparar condiciones para ML
                    ml_conditions = {
                        'asset': asset,
                        'score': trend_data.get('score', 0),
                        'direction': direction.lower(),
                        'trend_direction': trend_data.get('direction', ''),
                        'hour': datetime.now().hour,
                        'session': self.current_session or 'OFF_HOURS',
                        'amount': self.calculate_position_size(),
                        'volatility': trend_data.get('volatility', 2),
                        'momentum': trend_data.get('momentum', 0),
                        'consistency': trend_data.get('consistency', 0)
                    }
                    
                    # Consultar ML Advisor
                    ml_should_trade, ml_analysis = ML_ADVISOR.should_take_trade(ml_conditions)
                    
                    # Registrar estadísticas
                    self.ml_decisions['total_consultations'] += 1
                    if ml_should_trade:
                        self.ml_decisions['ml_approved'] += 1
                    else:
                        self.ml_decisions['ml_rejected'] += 1
                    
                    # Loggear decisión ML (ADVISORY - no bloquea)
                    if ml_analysis.get('ml_used'):
                        win_prob = ml_analysis.get('win_probability', 0)
                        rec = ml_analysis.get('recommendation', 'NEUTRAL')
                        self.logger.info(f"  🤖 ML: {rec} (Win: {win_prob:.1%})")
                    else:
                        self.logger.info(f"  🤖 ML: {ml_analysis.get('reason', 'Sin modelo')}")
                    
                    # En modo ADVISORY, solo registramos pero NO bloqueamos
                    if not ml_should_trade:
                        self.logger.info(f"  ⚠️ ML habría rechazado (modo advisory - no bloquea)")
                        
                except Exception as e:
                    self.logger.error(f"  ❌ Error consultando ML: {e}")
                    # Continuar sin ML si hay error
            
            # ========== FIN INTEGRACIÓN ML ==========
            # Calcular tamaño de posición
            position_size = self.calculate_position_size()
            
            # Determinar dirección de la opción
            option_direction = "call" if direction == "UP" else "put"
            
            self.logger.info(f"📈 Colocando {option_direction.upper()} en {asset}")
            self.logger.info(f"  Tendencia: {direction} | Score: {trend_data['score']:.3f}")
            self.logger.info(f"  Tamaño: {format_currency(position_size)}")
            
            # Colocar orden
            asset_info = self.asset_mapping[asset]
            status, order_id = self.iqoption.buy(
                int(position_size),
                asset_info["name"],
                option_direction,
                EXPIRY_MINUTES
            )
            
            if status:
                # Registrar orden
                order_info = {
                    "id": order_id,
                    "asset": asset,
                    "direction": option_direction,
                    "amount": position_size,
                    "entry_time": datetime.now(),
                    "expiry_time": datetime.now() + timedelta(minutes=EXPIRY_MINUTES),
                    "hedge_check_time": datetime.now() + timedelta(minutes=HEDGE_CHECK_MINUTE),
                    "trend_score": trend_data["score"],
                    "trend_direction": direction,
                    "entry_price": trend_data["current_price"],
                    "has_hedge": False
                }
                
                self.primary_orders[order_id] = order_info
                self.pending_hedges[order_id] = order_info
                self.daily_trades += 1
                
                self.logger.info(f"✅ Orden primaria colocada: ID {order_id}")
                
                # Programar verificación de cobertura
                self.schedule_hedge_check(order_id)
                
                return order_id
            else:
                self.logger.error(f"❌ Error colocando orden: {order_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error en place_primary_order: {str(e)}")
            return None
    
    def schedule_hedge_check(self, order_id):
        """Programar verificación para cobertura"""
        def check_and_place_hedge():
            time.sleep(HEDGE_CHECK_MINUTE * 60 + HEDGE_WAIT_SECONDS)
            
            if order_id in self.pending_hedges:
                self.evaluate_hedge(order_id)
        
        # Ejecutar en thread separado
        self.executor.submit(check_and_place_hedge)
    
    def evaluate_hedge(self, order_id):
        """Evaluar si colocar cobertura"""
        try:
            if order_id not in self.primary_orders:
                return
            
            order = self.primary_orders[order_id]
            asset = order["asset"]
            
            self.logger.info(f"🔍 Evaluando cobertura para {asset} (ID: {order_id})")
            
            # Obtener precio actual
            current_candles = self.iqoption.get_candles(
                self.asset_mapping[asset]["name"],
                60,  # 1 minuto
                1,
                time.time()
            )
            
            if not current_candles:
                self.logger.warning("⚠️ No se pudo obtener precio actual")
                return
            
            current_price = float(current_candles[0]['close'])
            entry_price = order["entry_price"]
            
            # Verificar si está ITM
            is_itm = False
            if order["direction"] == "call":
                is_itm = current_price > entry_price
                price_diff = ((current_price - entry_price) / entry_price) * 100
            else:  # put
                is_itm = current_price < entry_price
                price_diff = ((entry_price - current_price) / entry_price) * 100
            
            self.logger.info(f"  Precio entrada: {entry_price:.5f}")
            self.logger.info(f"  Precio actual: {current_price:.5f}")
            self.logger.info(f"  Diferencia: {price_diff:.3f}%")
            self.logger.info(f"  Estado: {'ITM ✅' if is_itm else 'OTM ❌'}")
            
            if is_itm:
                # Colocar cobertura
                self.place_hedge_order(order_id, asset, current_price)
                self.hedges_placed += 1
            else:
                self.logger.info("📊 No se coloca cobertura (OTM)")
                self.hedges_avoided += 1
                order["has_hedge"] = False
            
            # Remover de pendientes
            if order_id in self.pending_hedges:
                del self.pending_hedges[order_id]
                
        except Exception as e:
            self.logger.error(f"❌ Error evaluando cobertura: {str(e)}")
    
    def place_hedge_order(self, primary_order_id, asset, current_price):
        """Colocar orden de cobertura"""
        try:
            primary_order = self.primary_orders[primary_order_id]
            
            # Calcular tamaño de cobertura
            hedge_size = primary_order["amount"] * HEDGE_SIZE_MULTIPLIER
            
            # Dirección opuesta
            hedge_direction = "put" if primary_order["direction"] == "call" else "call"
            
            self.logger.info(f"🛡️ Colocando COBERTURA {hedge_direction.upper()} en {asset}")
            self.logger.info(f"  Tamaño: {format_currency(hedge_size)}")
            
            # Calcular tiempo restante para expiración
            time_to_expiry = (primary_order["expiry_time"] - datetime.now()).total_seconds() / 60
            
            # Colocar orden con el mismo vencimiento que la primaria
            asset_info = self.asset_mapping[asset]
            status, hedge_id = self.iqoption.buy(
                int(hedge_size),
                asset_info["name"],
                hedge_direction,
                int(time_to_expiry)  # Mismo vencimiento que la primaria
            )
            
            if status:
                hedge_info = {
                    "id": hedge_id,
                    "primary_id": primary_order_id,
                    "asset": asset,
                    "direction": hedge_direction,
                    "amount": hedge_size,
                    "entry_time": datetime.now(),
                    "entry_price": current_price,
                    "expiry_time": primary_order["expiry_time"]  # Mismo vencimiento
                }
                
                self.hedge_orders[hedge_id] = hedge_info
                primary_order["has_hedge"] = True
                primary_order["hedge_id"] = hedge_id
                
                self.logger.info(f"✅ Cobertura colocada: ID {hedge_id}")
            else:
                self.logger.error(f"❌ Error colocando cobertura")
                
        except Exception as e:
            self.logger.error(f"❌ Error en place_hedge_order: {str(e)}")
    
    def calculate_position_size(self):
        """Calcular tamaño de posición"""
        current_balance = self.iqoption.get_balance()
        position_size = current_balance * POSITION_SIZE_PERCENT
        return max(MIN_POSITION_SIZE, round(position_size, 2))
    
    def check_trading_window(self):
        """Verificar si estamos en una ventana de trading"""
        now = datetime.now()
        current_time = now.replace(second=0, microsecond=0)
        
        for window in TRADING_WINDOWS:
            # Filtrar por sesión si se especificó
            if self.session_filter != "TODAS":
                if window.get("session") != self.session_filter:
                    continue
            
            window_time = now.replace(
                hour=window["hour"],
                minute=window["minute"],
                second=0,
                microsecond=0
            )
            
            # Ventana de 30 segundos para entrar
            if abs((current_time - window_time).total_seconds()) <= 30:
                window_key = f"{window['hour']:02d}:{window['minute']:02d}"
                session = window.get("session", "")
                
                # Verificar si ya operamos en esta ventana
                if self.last_window_traded != window_key:
                    self.current_window = window_key
                    self.current_session = session
                    
                    # Log de sesión
                    if session:
                        self.logger.info(f"📍 SESIÓN {session} - Ventana {window_key}")
                        if self.session_filter != "TODAS":
                            self.logger.info(f"   (Operando solo en sesión {self.session_filter})")
                    
                    return True
        
        return False
    
    def process_expired_orders(self):
        """Procesar órdenes expiradas"""
        current_time = datetime.now()
        
        # Procesar órdenes primarias
        for order_id, order in list(self.primary_orders.items()):
            if current_time > order["expiry_time"] + timedelta(seconds=30):
                self.check_order_result(order_id, "primary")
        
        # Procesar coberturas
        for hedge_id, hedge in list(self.hedge_orders.items()):
            if current_time > hedge["expiry_time"] + timedelta(seconds=30):
                self.check_order_result(hedge_id, "hedge")
    
    def check_order_result(self, order_id, order_type):
        """Verificar resultado de una orden"""
        try:
            # Intentar obtener resultado
            result = None
            
            # Verificar en order_binary
            if hasattr(self.iqoption.api, 'order_binary') and order_id in self.iqoption.api.order_binary:
                order_data = self.iqoption.api.order_binary[order_id]
                result = order_data.get('result', '').lower()
            
            if not result:
                # Intentar con get_async_order
                order_result = self.iqoption.get_async_order(order_id)
                if order_result:
                    result = str(order_result.get('win', '')).lower()
            
            if result:
                if order_type == "primary":
                    order = self.primary_orders[order_id]
                else:
                    order = self.hedge_orders[order_id]
                
                if result == 'win':
                    profit = order["amount"] * 0.85  # Asumiendo 85% payout
                    self.daily_profit += profit
                    self.wins += 1
                    self.logger.info(f"✅ {order_type.upper()} GANADA: +{format_currency(profit)}")
                elif result == 'loose':
                    loss = order["amount"]
                    self.daily_profit -= loss
                    self.losses += 1
                    self.logger.info(f"❌ {order_type.upper()} PERDIDA: -{format_currency(loss)}")
                else:  # equal/tie
                    self.logger.info(f"🟡 {order_type.upper()} EMPATE")
                
                # Limpiar orden procesada
                if order_type == "primary":
                    del self.primary_orders[order_id]
                else:
                    del self.hedge_orders[order_id]
                    
        except Exception as e:
            self.logger.error(f"❌ Error verificando resultado: {str(e)}")
    
    def check_daily_limits(self):
        """Verificar límites diarios"""
        # Verificar máximo de trades
        if self.daily_trades >= DAILY_MAX_TRADES:
            self.logger.info("📊 Máximo de trades diarios alcanzado")
            return False
        
        # Verificar límite de pérdida
        current_balance = self.iqoption.get_balance()
        daily_return = (current_balance - self.initial_capital) / self.initial_capital
        
        if daily_return <= -DAILY_LOSS_LIMIT:
            self.logger.warning("⛔ Límite de pérdida diaria alcanzado")
            return False
        
        # Verificar objetivo de ganancia
        if daily_return >= DAILY_PROFIT_TARGET:
            self.logger.info("🎯 Objetivo de ganancia diaria alcanzado")
            return False
        
        return True
    
    def save_state(self):
        """Guardar estado de la estrategia"""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "daily_trades": self.daily_trades,
                "daily_profit": self.daily_profit,
                "wins": self.wins,
                "losses": self.losses,
                "hedges_placed": self.hedges_placed,
                "hedges_avoided": self.hedges_avoided,
                "trend_scores": {
                    asset: {
                        "score": data.get("score", 0),
                        "direction": data.get("direction", ""),
                        "strength": data.get("strength", 0)
                    }
                    for asset, data in self.trend_scores.items()
                },
                "last_window_traded": self.last_window_traded
            }
            
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"❌ Error guardando estado: {str(e)}")
    
    def load_state(self):
        """Cargar estado previo"""
        try:
            if not os.path.exists(STATE_FILE):
                return
            
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            
            # Verificar si es del mismo día
            saved_date = datetime.fromisoformat(state["timestamp"]).date()
            if saved_date == datetime.now().date():
                self.daily_trades = state.get("daily_trades", 0)
                self.daily_profit = state.get("daily_profit", 0)
                self.wins = state.get("wins", 0)
                self.losses = state.get("losses", 0)
                self.hedges_placed = state.get("hedges_placed", 0)
                self.hedges_avoided = state.get("hedges_avoided", 0)
                
                self.logger.info(f"✅ Estado cargado del {saved_date}")
                
        except Exception as e:
            self.logger.error(f"❌ Error cargando estado: {str(e)}")
    
    def print_summary(self):
        """Imprimir resumen de la sesión"""
        current_balance = self.iqoption.get_balance()
        total_return = ((current_balance - self.initial_capital) / self.initial_capital) * 100
        
        self.logger.info("="*60)
        self.logger.info("📊 RESUMEN DE LA SESIÓN")
        self.logger.info("="*60)
        
        # Información de sesiones
        self.logger.info("📍 SESIONES DE TRADING (Hora Colombia UTC-5):")
        self.logger.info("  - NY_OPEN: 08:00-09:00 (Apertura Wall Street)")
        self.logger.info("  - OVERLAP: 10:00-11:00 (Solapamiento EU-NY) ⭐")
        self.logger.info("  - POWER: 14:00-15:00 (Power Hour NY)")
        self.logger.info("  - POST: 17:00-18:00 (Post-mercado OTC)")
        
        self.logger.info(f"\n💰 Capital inicial: {format_currency(self.initial_capital)}")
        self.logger.info(f"💰 Balance final: {format_currency(current_balance)}")
        self.logger.info(f"📈 Rendimiento: {total_return:.2f}%")
        self.logger.info(f"📊 Trades realizados: {self.daily_trades}/16 posibles")
        self.logger.info(f"✅ Victorias: {self.wins}")
        self.logger.info(f"❌ Derrotas: {self.losses}")
        
        if self.wins + self.losses > 0:
            win_rate = (self.wins / (self.wins + self.losses)) * 100
            self.logger.info(f"🎯 Win Rate: {win_rate:.1f}%")
        
        self.logger.info(f"\n🛡️ Coberturas colocadas: {self.hedges_placed}")
        self.logger.info(f"⏭️ Coberturas evitadas: {self.hedges_avoided}")
        
        if self.hedges_placed + self.hedges_avoided > 0:
            hedge_rate = (self.hedges_placed / (self.hedges_placed + self.hedges_avoided)) * 100
            self.logger.info(f"📊 Tasa de cobertura: {hedge_rate:.1f}%")
        # Estadísticas ML si está activo
        if USE_ML_ADVISOR:
            ml_stats = self.get_ml_stats()
            self.logger.info(f"\n🤖 ML ADVISOR (Modo Advisory):")
            self.logger.info(f"  Consultas: {ml_stats.get('consultations', 0)}")
            self.logger.info(f"  ML aprobaría: {ml_stats.get('ml_approved', 0)}")
            self.logger.info(f"  ML rechazaría: {ml_stats.get('ml_rejected', 0)}")
            
            if ml_stats.get('consultations', 0) > 0:
                approval_rate = ml_stats.get('approval_rate', 0) * 100
                self.logger.info(f"  Tasa aprobación ML: {approval_rate:.1f}%")
                
                if 'avg_win_prob' in ml_stats:
                    self.logger.info(f"  Prob. WIN promedio: {ml_stats['avg_win_prob']:.1%}")
        self.logger.info("="*60)
        
    def get_ml_stats(self) -> dict:
        """
        Obtiene estadísticas del ML Advisor
        
        Returns:
            Diccionario con estadísticas
        """
        if not USE_ML_ADVISOR:
            return {'ml_enabled': False}
        
        stats = {
            'ml_enabled': True,
            'mode': 'advisory',
            'consultations': self.ml_decisions['total_consultations'],
            'ml_approved': self.ml_decisions['ml_approved'],
            'ml_rejected': self.ml_decisions['ml_rejected'],
            'approval_rate': (
                self.ml_decisions['ml_approved'] / 
                max(1, self.ml_decisions['total_consultations'])
            )
        }
        
        # Análisis de predicciones recientes
        recent = self.ml_decisions['ml_predictions'][-10:]  # Últimas 10
        if recent:
            win_probs = []
            for pred in recent:
                if 'analysis' in pred and 'win_probability' in pred['analysis']:
                    win_probs.append(pred['analysis']['win_probability'])
            
            if win_probs:
                stats['avg_win_prob'] = sum(win_probs) / len(win_probs)
                stats['max_win_prob'] = max(win_probs)
                stats['min_win_prob'] = min(win_probs)
        
        return stats
    def run(self):
        """Ejecutar estrategia principal"""
        try:
            # Período de análisis inicial
            if self.is_warming_up:
                self.warmup_analysis()
            
            self.logger.info("🚀 Iniciando operaciones...")
            cycle_count = 0
            
            while True:
                cycle_count += 1
                
                # Verificar conexión
                if not self.iqoption.check_connect():
                    self.logger.warning("🔌 Reconectando...")
                    self._connect_to_iq_option(IQ_EMAIL, IQ_PASSWORD, ACCOUNT_TYPE)
                    time.sleep(5)
                    continue
                
                # Verificar límites diarios
                if not self.check_daily_limits():
                    self.logger.info("🛑 Límites diarios alcanzados. Esperando...")
                    time.sleep(300)
                    continue
                
                # Procesar órdenes expiradas
                self.process_expired_orders()
                
                # Verificar ventana de trading
                if self.check_trading_window():
                    session_info = ""
                    if hasattr(self, 'current_session') and self.current_session:
                        session_info = f" [{self.current_session}]"
                    
                    self.logger.info(f"\n⏰ VENTANA DE TRADING: {self.current_window}{session_info}")
                    
                    # Obtener mejor tendencia
                    best_trend = self.get_best_trending_asset()
                    
                    if best_trend:
                        # Verificar que no tengamos orden activa
                        if len(self.primary_orders) < MAX_SIMULTANEOUS_PRIMARY:
                            self.place_primary_order(
                                best_trend["asset"],
                                best_trend["direction"],
                                best_trend
                            )
                            self.last_window_traded = self.current_window
                        else:
                            self.logger.info("⏭️ Máximo de órdenes primarias activas")
                    else:
                        self.logger.warning("⚠️ No hay tendencias válidas en esta ventana")
                
                # Guardar estado periódicamente
                if cycle_count % SAVE_STATE_INTERVAL == 0:
                    self.save_state()
                
                # Log periódico
                if cycle_count % 20 == 0:
                    active_primary = len(self.primary_orders)
                    active_hedges = len(self.hedge_orders)
                    session, status = get_current_session()
                    
                    session_info = f"Sesión {session} ({status})"
                    
                    self.logger.info(
                        f"🔄 Ciclo {cycle_count} | {session_info} | "
                        f"Primarias: {active_primary} | "
                        f"Coberturas: {active_hedges} | "
                        f"P&L: {format_currency(self.daily_profit)}"
                    )
                
                time.sleep(15)  # Ciclo cada 15 segundos
                
        except KeyboardInterrupt:
            self.logger.info("\n⏹️ Estrategia detenida por el usuario")
        except Exception as e:
            self.logger.critical(f"🚨 Error crítico: {str(e)}")
            self.logger.critical(traceback.format_exc())
        finally:
            self.save_state()
            self.print_summary()
            self.executor.shutdown(wait=True)
            self.logger.info("👋 Estrategia finalizada")