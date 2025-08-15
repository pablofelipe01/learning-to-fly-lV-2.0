# config_trend.py
# Configuración para estrategia de tendencia con protección dinámica
# Basada en identificación de tendencias claras y cobertura condicional
# ACTUALIZADO: Vencimientos de 5 minutos para mantener momentum
# ACTUALIZADO: Soporte para variables de entorno para despliegue en servidor

from datetime import datetime
import os

# Credenciales IQ Option - AHORA DESDE VARIABLES DE ENTORNO
IQ_EMAIL = os.getenv('IQ_EMAIL', 'tu_email@example.com')
IQ_PASSWORD = os.getenv('IQ_PASSWORD', 'tu_password')
ACCOUNT_TYPE = os.getenv('ACCOUNT_TYPE', 'PRACTICE')

# Si las variables no están configuradas, intentar cargar desde .env
if IQ_EMAIL == 'tu_email@example.com':
    try:
        # Intentar cargar archivo .env si existe
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            
            # Recargar las variables
            IQ_EMAIL = os.getenv('IQ_EMAIL', 'tu_email@example.com')
            IQ_PASSWORD = os.getenv('IQ_PASSWORD', 'tu_password')
            ACCOUNT_TYPE = os.getenv('ACCOUNT_TYPE', 'PRACTICE')
    except:
        pass

# ===============================================
# CONFIGURACIÓN DE LA ESTRATEGIA DE TENDENCIA
# ===============================================

# VENTANAS DE TRADING - 4 Sesiones Optimizadas para Colombia (UTC-5)
TRADING_WINDOWS = [
    # SESIÓN APERTURA NY (8:00 - 9:00)
    {"hour": 8, "minute": 0,  "session": "NY_OPEN"},
    {"hour": 8, "minute": 15, "session": "NY_OPEN"},
    {"hour": 8, "minute": 30, "session": "NY_OPEN"},
    {"hour": 8, "minute": 45, "session": "NY_OPEN"},
    
    # SESIÓN SOLAPAMIENTO (10:00 - 11:00) - MEJOR MOMENTO
    {"hour": 10, "minute": 0,  "session": "OVERLAP"},
    {"hour": 10, "minute": 15, "session": "OVERLAP"},
    {"hour": 10, "minute": 30, "session": "OVERLAP"},
    {"hour": 10, "minute": 45, "session": "OVERLAP"},
    
    # SESIÓN POWER HOUR (14:00 - 15:00)
    {"hour": 14, "minute": 0,  "session": "POWER"},
    {"hour": 14, "minute": 15, "session": "POWER"},
    {"hour": 14, "minute": 30, "session": "POWER"},
    {"hour": 14, "minute": 45, "session": "POWER"},
    
    # SESIÓN POST-MERCADO (17:00 - 18:00)
    {"hour": 17, "minute": 0,  "session": "POST"},
    {"hour": 17, "minute": 15, "session": "POST"},
    {"hour": 17, "minute": 30, "session": "POST"},
    {"hour": 17, "minute": 45, "session": "POST"},
]

# CONFIGURACIÓN DE SESIONES (Hora Colombia UTC-5)
TRADING_SESSIONS = {
    "NY_OPEN": {
        "start_hour": 8,
        "end_hour": 9,
        "description": "Apertura Wall Street 🇺🇸",
        "volatility_expected": "ALTA",
        "markets_status": "NYSE/NASDAQ abriendo, gaps iniciales, noticias pre-mercado",
        "best_assets": ["US500", "USNDAQ100", "Acciones US", "USD pairs"],
        "strategy_note": "Cuidado con gaps y volatilidad inicial"
    },
    "OVERLAP": {
        "start_hour": 10,
        "end_hour": 11,
        "description": "Solapamiento Europa-NY 🌍🇺🇸 (MEJOR MOMENTO)",
        "volatility_expected": "MUY ALTA",
        "markets_status": "Londres y NY activos simultáneamente - Máxima liquidez",
        "best_assets": ["EURUSD", "GBPUSD", "Todos los majors", "Índices"],
        "strategy_note": "⭐ Mejor momento del día para trading"
    },
    "POWER": {
        "start_hour": 14,
        "end_hour": 15,
        "description": "Power Hour NY 🚀",
        "volatility_expected": "ALTA",
        "markets_status": "Últimas 2 horas Wall Street, movimientos institucionales",
        "best_assets": ["Acciones US", "Índices US", "USDJPY", "USDCAD"],
        "strategy_note": "Movimientos finales, posicionamiento para cierre"
    },
    "POST": {
        "start_hour": 17,
        "end_hour": 18,
        "description": "Post-Mercado / OTC 🌙",
        "volatility_expected": "BAJA-MEDIA",
        "markets_status": "Mercados principales cerrados, solo OTC disponible",
        "best_assets": ["Pares OTC", "Crypto", "Algunos índices sintéticos"],
        "strategy_note": "Menor liquidez, spreads más amplios, usar con precaución"
    }
}

# LÍMITES POR SESIÓN
MAX_TRADES_PER_SESSION = 4     # Máximo 4 trades por sesión (uno por ventana)
MAX_SESSIONS_PER_DAY = 4       # Las 4 sesiones configuradas

# AJUSTE DE LÍMITES DIARIOS
DAILY_MAX_TRADES = 16          # Máximo de operaciones primarias por día (4 sesiones × 4 ventanas)

# ===============================================
# CONFIGURACIÓN DE OPCIONES - MOMENTUM RÁPIDO
# ===============================================
EXPIRY_MINUTES = 5  # Tiempo de expiración REDUCIDO (de 15 a 5) para mantener momentum
HEDGE_CHECK_MINUTE = 3  # Minuto para verificar si colocar cobertura (3 de 5 = 60% del tiempo)
HEDGE_WAIT_SECONDS = 20  # Segundos adicionales de espera antes de cobertura (reducido de 30 a 20)

# PERÍODO DE ANÁLISIS - DINÁMICO
WARMUP_PERIOD_FULL = 7200  # 2 horas completas (ideal)
WARMUP_PERIOD_MEDIUM = 3600  # 1 hora (compromiso)
WARMUP_PERIOD_SHORT = 1800  # 30 minutos (mínimo útil)
WARMUP_PERIOD_EXPRESS = 900  # 15 minutos (emergencia)

# El período real se ajustará dinámicamente basado en la próxima sesión
WARMUP_PERIOD = WARMUP_PERIOD_FULL  # Default, se ajustará en runtime

def get_dynamic_warmup_period():
    """Calcular período de calentamiento dinámico basado en próxima sesión"""
    from datetime import datetime, timedelta
    
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    
    # Encontrar próxima sesión
    next_session_times = [
        (8, 0, "NY_OPEN"),
        (10, 0, "OVERLAP"),
        (14, 0, "POWER"),
        (17, 0, "POST")
    ]
    
    for hour, minute, session in next_session_times:
        session_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Si la sesión es hoy y no ha pasado
        if session_time > now:
            time_until_session = (session_time - now).total_seconds()
            
            # Ajustar período de calentamiento según tiempo disponible
            if time_until_session >= 7200:  # 2+ horas disponibles
                return WARMUP_PERIOD_FULL, "COMPLETO (2 horas)"
            elif time_until_session >= 3600:  # 1+ hora disponible
                return WARMUP_PERIOD_MEDIUM, "MEDIO (1 hora)"
            elif time_until_session >= 1800:  # 30+ minutos disponibles
                return WARMUP_PERIOD_SHORT, "CORTO (30 minutos)"
            elif time_until_session >= 900:  # 15+ minutos disponibles
                return WARMUP_PERIOD_EXPRESS, "EXPRESS (15 minutos)"
            else:
                return 0, "OMITIDO (sesión muy próxima)"
    
    # Si no hay más sesiones hoy, usar calentamiento completo
    return WARMUP_PERIOD_FULL, "COMPLETO (2 horas - para mañana)"

SCAN_INTERVAL = 60  # Intervalo de escaneo durante calentamiento (segundos)
TREND_CANDLE_TIMEFRAME = 300  # Timeframe para análisis de tendencia (5 minutos)
TREND_LOOKBACK_CANDLES = 20  # Velas hacia atrás para analizar tendencia

# CRITERIOS DE TENDENCIA
MIN_TREND_STRENGTH = 0.65  # Fuerza mínima de tendencia (0-1)
MIN_TREND_CONSISTENCY = 0.60  # Consistencia mínima requerida
MIN_PRICE_MOVEMENT = 0.0015  # Movimiento mínimo de precio (0.15%)
MAX_RETRACEMENTS = 3  # Máximo de retrocesos permitidos en la tendencia

# SCORING DE TENDENCIAS (pesos para evaluación)
TREND_WEIGHTS = {
    "strength": 0.35,      # Peso de la fuerza de tendencia
    "consistency": 0.25,   # Peso de la consistencia
    "momentum": 0.20,      # Peso del momentum actual
    "volume": 0.10,        # Peso del volumen (si disponible)
    "smoothness": 0.10     # Peso de la suavidad de la tendencia
}

# CONTROL DE OPERACIONES
MAX_SIMULTANEOUS_PRIMARY = 1  # Máximo de operaciones primarias simultáneas
MAX_SIMULTANEOUS_HEDGE = 1    # Máximo de coberturas simultáneas
POSITION_SIZE_PERCENT = 0.02   # 2% del capital por operación primaria
HEDGE_SIZE_MULTIPLIER = 1.0   # Multiplicador para el tamaño de la cobertura
MIN_POSITION_SIZE = 1          # Tamaño mínimo de posición

# GESTIÓN DE RIESGO BÁSICA
DAILY_LOSS_LIMIT = 0.10        # Límite de pérdida diaria (10%)
DAILY_PROFIT_TARGET = 0.15     # Objetivo de ganancia diaria (15%)

# ===============================================
# ACTIVOS A OPERAR
# ===============================================

FOREX_ASSETS = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF",
    "USDCAD", "AUDUSD", "NZDUSD"
]

INDEX_ASSETS = [
    "USNDAQ100", "US2000", "GER 30"
]

STOCK_ASSETS = [
    "TESLA", "Meta", "SNAP", "Amazon",
    "GOOGLE", "ALIBABA", "Apple", "MSFT"
]

CRYPTO_ASSETS = [
    "BTCUSD", "ETHUSD", "MATICUSD", "NEARUSD",
    "ATOMUSD", "DOTUSD", "ARBUSD", "LINKUSD"
]

PAIR_ASSETS = [
    "NVDA/AMD", "TESLA/FORD", "META/GOOGLE",
    "AMZN/ALIBABA", "MSFT/AAPL", "AMZN/EBAY",
    "NFLX/AMZN", "GOOGLE/MSFT", "INTEL/IBM"
]

# Lista consolidada
TRADING_ASSETS = FOREX_ASSETS + INDEX_ASSETS + STOCK_ASSETS + CRYPTO_ASSETS + PAIR_ASSETS

# ===============================================
# CONFIGURACIÓN DE TENDENCIA POR GRUPO
# ===============================================

TREND_CONFIG_BY_GROUP = {
    "FOREX": {
        "min_movement": 0.0010,      # 0.10% para Forex
        "smoothness_threshold": 0.7,  # Mayor suavidad requerida
        "momentum_weight": 1.2        # Mayor peso al momentum
    },
    "INDEX": {
        "min_movement": 0.0015,      # 0.15% para índices
        "smoothness_threshold": 0.6,
        "momentum_weight": 1.0
    },
    "STOCK": {
        "min_movement": 0.0020,      # 0.20% para acciones
        "smoothness_threshold": 0.5,  # Menos suavidad requerida
        "momentum_weight": 1.1
    },
    "CRYPTO": {
        "min_movement": 0.0025,      # 0.25% para crypto (más volátil)
        "smoothness_threshold": 0.4,
        "momentum_weight": 1.3
    },
    "PAIR": {
        "min_movement": 0.0012,      # 0.12% para pares
        "smoothness_threshold": 0.65,
        "momentum_weight": 0.9
    }
}

# ===============================================
# FUNCIONES AUXILIARES
# ===============================================

def get_current_session():
    """Obtener la sesión de trading actual o próxima"""
    now = datetime.now()
    current_hour = now.hour
    
    # Verificar sesión activa
    for session_name, session_info in TRADING_SESSIONS.items():
        if session_info["start_hour"] <= current_hour < session_info["end_hour"]:
            return session_name, "ACTIVA"
    
    # Si no hay sesión activa, buscar la próxima
    for session_name, session_info in TRADING_SESSIONS.items():
        if current_hour < session_info["start_hour"]:
            return session_name, "PRÓXIMA"
    
    # Si pasaron todas las sesiones del día
    return "NY_OPEN", "MAÑANA"

def get_asset_group(asset):
    """Determinar el grupo de un activo"""
    if asset in FOREX_ASSETS:
        return "FOREX"
    elif asset in INDEX_ASSETS:
        return "INDEX"
    elif asset in STOCK_ASSETS:
        return "STOCK"
    elif asset in CRYPTO_ASSETS:
        return "CRYPTO"
    elif asset in PAIR_ASSETS:
        return "PAIR"
    return "DEFAULT"

def get_trend_config_for_asset(asset):
    """Obtener configuración de tendencia para un activo"""
    group = get_asset_group(asset)
    return TREND_CONFIG_BY_GROUP.get(group, {
        "min_movement": MIN_PRICE_MOVEMENT,
        "smoothness_threshold": 0.5,
        "momentum_weight": 1.0
    })

# ===============================================
# CONFIGURACIÓN DE LOGGING Y ESTADO
# ===============================================

LOG_LEVEL = "INFO"
LOG_FILE = "trend_strategy.log"
STATE_FILE = "trend_strategy_state.json"
SAVE_STATE_INTERVAL = 30  # Guardar estado cada 30 ciclos

# ===============================================
# FUNCIONES DE DISPLAY
# ===============================================

def print_strategy_configuration():
    """Imprimir configuración de la estrategia"""
    print("\n" + "="*60)
    print("📈 CONFIGURACIÓN DE ESTRATEGIA DE TENDENCIA")
    print("="*60)
    print("\n⏰ SESIONES Y VENTANAS DE TRADING (Hora Colombia UTC-5):")
    
    # Mostrar por sesión
    for session_name, session_info in TRADING_SESSIONS.items():
        print(f"\n📍 SESIÓN {session_name} ({session_info['start_hour']:02d}:00 - {session_info['end_hour']:02d}:00)")
        print(f"   {session_info['description']}")
        print(f"   Volatilidad esperada: {session_info['volatility_expected']}")
        if 'best_assets' in session_info:
            print(f"   Mejores activos: {', '.join(session_info['best_assets'][:3])}...")
        print("   Ventanas:")
        
        # Mostrar ventanas de esta sesión
        session_windows = [w for w in TRADING_WINDOWS if w.get('session') == session_name]
        for window in session_windows:
            print(f"     - {window['hour']:02d}:{window['minute']:02d} → Vencimiento en {EXPIRY_MINUTES} min")
    
    print(f"\n📊 TOTAL: {len(TRADING_WINDOWS)} ventanas de trading al día")
    print(f"          ({MAX_SESSIONS_PER_DAY} sesiones × {MAX_TRADES_PER_SESSION} ventanas)")
    
    print(f"\n🎯 LÓGICA DE OPERACIÓN (MOMENTUM RÁPIDO):")
    print(f"  1. Identificar tendencia clara durante análisis inicial")
    print(f"  2. Colocar opción primaria según tendencia")
    print(f"  3. A los {HEDGE_CHECK_MINUTE} minutos: verificar si está ITM")
    print(f"  4. Si ITM → colocar cobertura (opción contraria)")
    print(f"  5. Vencimiento rápido de {EXPIRY_MINUTES} min mantiene el momentum")
    
    print(f"\n⚡ TIMING OPTIMIZADO:")
    print(f"  - Entrada: :00, :15, :30, :45")
    print(f"  - Check hedge: +{HEDGE_CHECK_MINUTE}:{HEDGE_WAIT_SECONDS} ({HEDGE_CHECK_MINUTE*60+HEDGE_WAIT_SECONDS} seg)")
    print(f"  - Vencimiento: +{EXPIRY_MINUTES}:00 ({EXPIRY_MINUTES*60} seg)")
    print(f"  - Hedge duration: ~{EXPIRY_MINUTES - HEDGE_CHECK_MINUTE} minutos")
    
    print(f"\n📊 CRITERIOS DE TENDENCIA:")
    print(f"  - Fuerza mínima: {MIN_TREND_STRENGTH*100:.0f}%")
    print(f"  - Consistencia mínima: {MIN_TREND_CONSISTENCY*100:.0f}%")
    print(f"  - Movimiento mínimo: {MIN_PRICE_MOVEMENT*100:.2f}%")
    print(f"  - Retrocesos máximos: {MAX_RETRACEMENTS}")
    
    print(f"\n💰 GESTIÓN DE CAPITAL:")
    print(f"  - Tamaño posición: {POSITION_SIZE_PERCENT*100}%")
    print(f"  - Multiplicador cobertura: {HEDGE_SIZE_MULTIPLIER}x")
    print(f"  - Trades máximos/día: {DAILY_MAX_TRADES}")
    
    print(f"\n📈 ACTIVOS CONFIGURADOS: {len(TRADING_ASSETS)}")
    print("="*60)

# Mostrar configuración al importar (solo si no está en servidor)
if __name__ != "__main__" and not os.getenv('DEPLOYMENT_ENV'):
    # Solo mostrar config si no estamos en producción
    if IQ_EMAIL != 'tu_email@example.com':
        print_strategy_configuration()