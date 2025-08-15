# utils_trend.py
# Funciones auxiliares para análisis de tendencias y estrategia de protección

import numpy as np
from datetime import datetime
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression

def setup_logger(name, log_file, level=logging.INFO):
    """Configurar logger con formato personalizado"""
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configurar logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def calculate_trend_strength(prices):
    """
    Calcular la fuerza de la tendencia usando regresión lineal
    
    Args:
        prices: Lista de precios de cierre
    
    Returns:
        float: Fuerza de la tendencia (0-1)
    """
    if len(prices) < 3:
        return 0
    
    try:
        # Crear índices temporales
        x = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices)
        
        # Regresión lineal
        model = LinearRegression()
        model.fit(x, y)
        
        # Calcular R²
        y_pred = model.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0
        
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calcular pendiente normalizada
        price_range = max(prices) - min(prices)
        if price_range > 0:
            normalized_slope = abs(model.coef_[0]) / price_range
        else:
            normalized_slope = 0
        
        # Combinar R² y pendiente para obtener fuerza
        strength = (r_squared * 0.7 + min(normalized_slope, 1) * 0.3)
        
        return max(0, min(1, strength))
        
    except Exception as e:
        logging.error(f"Error calculando fuerza de tendencia: {str(e)}")
        return 0

def calculate_trend_consistency(prices):
    """
    Calcular consistencia de la tendencia (qué tan uniforme es el movimiento)
    
    Args:
        prices: Lista de precios
    
    Returns:
        float: Consistencia (0-1)
    """
    if len(prices) < 3:
        return 0
    
    try:
        # Calcular cambios porcentuales
        returns = np.diff(prices) / prices[:-1]
        
        # Determinar dirección general
        total_change = prices[-1] - prices[0]
        is_uptrend = total_change > 0
        
        # Contar movimientos en la dirección correcta
        if is_uptrend:
            correct_moves = sum(r > 0 for r in returns)
        else:
            correct_moves = sum(r < 0 for r in returns)
        
        # Calcular consistencia
        consistency = correct_moves / len(returns) if len(returns) > 0 else 0
        
        return consistency
        
    except Exception as e:
        logging.error(f"Error calculando consistencia: {str(e)}")
        return 0

def detect_trend_direction(prices):
    """
    Detectar dirección de la tendencia
    
    Args:
        prices: Lista de precios
    
    Returns:
        str: "UP", "DOWN" o "LATERAL"
    """
    if len(prices) < 3:
        return "LATERAL"
    
    try:
        # Usar regresión lineal para determinar dirección
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Calcular cambio porcentual total
        price_change_pct = (prices[-1] - prices[0]) / prices[0]
        
        # Determinar dirección basada en pendiente y cambio total
        if slope > 0 and price_change_pct > 0.001:  # 0.1%
            return "UP"
        elif slope < 0 and price_change_pct < -0.001:
            return "DOWN"
        else:
            return "LATERAL"
            
    except Exception as e:
        logging.error(f"Error detectando dirección: {str(e)}")
        return "LATERAL"

def calculate_momentum(prices, period=5):
    """
    Calcular momentum del precio
    
    Args:
        prices: Lista de precios
        period: Período para calcular momentum
    
    Returns:
        float: Momentum normalizado
    """
    if len(prices) < period:
        return 0
    
    try:
        # Momentum simple: cambio en los últimos n períodos
        momentum = (prices[-1] - prices[-period]) / prices[-period]
        return momentum
        
    except Exception as e:
        logging.error(f"Error calculando momentum: {str(e)}")
        return 0

def detect_support_resistance(prices, window=5):
    """
    Detectar niveles de soporte y resistencia
    
    Args:
        prices: Lista de precios
        window: Ventana para detectar máximos/mínimos locales
    
    Returns:
        tuple: (soportes, resistencias)
    """
    if len(prices) < window * 2:
        return [], []
    
    try:
        supports = []
        resistances = []
        
        for i in range(window, len(prices) - window):
            # Detectar mínimos locales (soportes)
            if prices[i] == min(prices[i-window:i+window+1]):
                supports.append(prices[i])
            
            # Detectar máximos locales (resistencias)
            if prices[i] == max(prices[i-window:i+window+1]):
                resistances.append(prices[i])
        
        return supports, resistances
        
    except Exception as e:
        logging.error(f"Error detectando S/R: {str(e)}")
        return [], []

def calculate_volatility(prices):
    """
    Calcular volatilidad histórica
    
    Args:
        prices: Lista de precios
    
    Returns:
        float: Volatilidad (desviación estándar de retornos)
    """
    if len(prices) < 2:
        return 0
    
    try:
        # Calcular retornos logarítmicos
        log_returns = np.diff(np.log(prices))
        
        # Calcular volatilidad (desviación estándar)
        volatility = np.std(log_returns)
        
        return volatility
        
    except Exception as e:
        logging.error(f"Error calculando volatilidad: {str(e)}")
        return 0

def calculate_trend_score(strength, consistency, momentum, volatility):
    """
    Calcular score compuesto de tendencia
    
    Args:
        strength: Fuerza de la tendencia (0-1)
        consistency: Consistencia (0-1)
        momentum: Momentum actual
        volatility: Volatilidad del activo
    
    Returns:
        float: Score de tendencia (0-1)
    """
    try:
        # Normalizar momentum
        norm_momentum = min(abs(momentum) * 10, 1)  # Escalar a 0-1
        
        # Penalizar alta volatilidad
        volatility_factor = max(0, 1 - volatility * 5)
        
        # Score ponderado
        score = (
            strength * 0.35 +
            consistency * 0.30 +
            norm_momentum * 0.20 +
            volatility_factor * 0.15
        )
        
        return max(0, min(1, score))
        
    except Exception as e:
        logging.error(f"Error calculando score: {str(e)}")
        return 0

def identify_trend_pattern(prices, highs, lows):
    """
    Identificar patrones de tendencia (Higher Highs/Lower Lows)
    
    Args:
        prices: Precios de cierre
        highs: Precios máximos
        lows: Precios mínimos
    
    Returns:
        str: Tipo de patrón identificado
    """
    if len(highs) < 3 or len(lows) < 3:
        return "INDEFINIDO"
    
    try:
        # Verificar Higher Highs y Higher Lows (tendencia alcista)
        hh = all(highs[i] <= highs[i+1] for i in range(len(highs)-1))
        hl = all(lows[i] <= lows[i+1] for i in range(len(lows)-1))
        
        # Verificar Lower Highs y Lower Lows (tendencia bajista)
        lh = all(highs[i] >= highs[i+1] for i in range(len(highs)-1))
        ll = all(lows[i] >= lows[i+1] for i in range(len(lows)-1))
        
        if hh and hl:
            return "ALCISTA_FUERTE"
        elif lh and ll:
            return "BAJISTA_FUERTE"
        elif hh:
            return "ALCISTA_DEBIL"
        elif ll:
            return "BAJISTA_DEBIL"
        else:
            return "LATERAL"
            
    except Exception as e:
        logging.error(f"Error identificando patrón: {str(e)}")
        return "INDEFINIDO"

def calculate_price_position(current_price, support, resistance):
    """
    Calcular posición del precio respecto a soporte/resistencia
    
    Args:
        current_price: Precio actual
        support: Nivel de soporte más cercano
        resistance: Nivel de resistencia más cercano
    
    Returns:
        float: Posición relativa (0 = en soporte, 1 = en resistencia)
    """
    if resistance <= support:
        return 0.5
    
    try:
        position = (current_price - support) / (resistance - support)
        return max(0, min(1, position))
        
    except Exception as e:
        logging.error(f"Error calculando posición: {str(e)}")
        return 0.5

def format_currency(amount):
    """Formatear cantidad como moneda"""
    return f"${amount:,.2f}"

def calculate_win_rate(wins, losses):
    """Calcular tasa de éxito"""
    total = wins + losses
    if total == 0:
        return 0.0
    return (wins / total) * 100

def calculate_expected_value(win_rate, avg_win, avg_loss):
    """
    Calcular valor esperado de la estrategia
    
    Args:
        win_rate: Tasa de éxito (0-1)
        avg_win: Ganancia promedio
        avg_loss: Pérdida promedio
    
    Returns:
        float: Valor esperado
    """
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

def validate_trend_quality(trend_data):
    """
    Validar calidad de la tendencia para operar
    
    Args:
        trend_data: Diccionario con datos de tendencia
    
    Returns:
        bool: True si la tendencia es válida para operar
    """
    required_fields = ['strength', 'consistency', 'movement', 'retracements']
    
    # Verificar campos requeridos
    if not all(field in trend_data for field in required_fields):
        return False
    
    # Aplicar criterios de validación
    if trend_data['strength'] < 0.6:
        return False
    
    if trend_data['consistency'] < 0.5:
        return False
    
    if abs(trend_data['movement']) < 0.001:  # 0.1%
        return False
    
    if trend_data['retracements'] > 5:
        return False
    
    return True

def calculate_hedge_threshold(volatility, trend_strength):
    """
    Calcular umbral dinámico para cobertura basado en condiciones del mercado
    
    Args:
        volatility: Volatilidad actual
        trend_strength: Fuerza de la tendencia
    
    Returns:
        float: Umbral de precio para considerar ITM
    """
    # Base threshold
    base_threshold = 0.0005  # 0.05%
    
    # Ajustar por volatilidad
    volatility_adjustment = volatility * 0.5
    
    # Ajustar por fuerza de tendencia (tendencias fuertes = umbral menor)
    trend_adjustment = (1 - trend_strength) * 0.0003
    
    threshold = base_threshold + volatility_adjustment + trend_adjustment
    
    return min(threshold, 0.002)  # Máximo 0.2%