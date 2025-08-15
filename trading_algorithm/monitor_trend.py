#!/usr/bin/env python3
"""
Monitor en tiempo real para la estrategia de tendencia
Muestra estadísticas y estado actual sin interferir con la ejecución
"""

import json
import os
import time
from datetime import datetime, timedelta
import sys

# Configuración
STATE_FILE = "trend_strategy_state.json"
LOG_FILE = "trend_strategy.log"
REFRESH_INTERVAL = 5  # Segundos entre actualizaciones

# Colores para terminal (opcional)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    """Limpiar pantalla de manera multiplataforma"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_state():
    """Cargar el estado actual"""
    if not os.path.exists(STATE_FILE):
        return None
    
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

def get_current_session():
    """Determinar la sesión actual o próxima"""
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    
    sessions = [
        {"name": "NY_OPEN", "start": 8, "end": 9, "desc": "Apertura Wall Street"},
        {"name": "OVERLAP", "start": 10, "end": 11, "desc": "Solapamiento EU-NY ⭐"},
        {"name": "POWER", "start": 14, "end": 15, "desc": "Power Hour NY"},
        {"name": "POST", "start": 17, "end": 18, "desc": "Post-mercado OTC"}
    ]
    
    # Verificar sesión activa
    for session in sessions:
        if session["start"] <= current_hour < session["end"]:
            return session["name"], "ACTIVA", session
        elif current_hour == session["end"] and current_minute == 0:
            return session["name"], "FINALIZANDO", session
    
    # Buscar próxima sesión
    for session in sessions:
        if current_hour < session["start"]:
            time_to_session = timedelta(
                hours=session["start"] - current_hour,
                minutes=-current_minute
            )
            return session["name"], f"en {time_to_session}", session
    
    # Si pasaron todas las sesiones
    tomorrow_session = sessions[0]
    time_to_session = timedelta(
        hours=24 - current_hour + tomorrow_session["start"],
        minutes=-current_minute
    )
    return tomorrow_session["name"], f"MAÑANA a las {tomorrow_session['start']:02d}:00", tomorrow_session

def get_next_window():
    """Obtener la próxima ventana de trading"""
    now = datetime.now()
    windows = []
    
    # Ventanas de trading actualizadas (4 sesiones)
    for hour in [8, 10, 14, 17]:
        for minute in [0, 15, 30, 45]:
            windows.append((hour, minute))
    
    # Encontrar próxima ventana
    for hour, minute in windows:
        window_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if window_time > now:
            time_diff = window_time - now
            minutes_to_window = int(time_diff.total_seconds() / 60)
            return f"{hour:02d}:{minute:02d}", minutes_to_window
    
    # Si no hay más ventanas hoy
    return "08:00 (mañana)", None

def format_profit(amount):
    """Formatear cantidad con color según signo"""
    if amount > 0:
        return f"{Colors.GREEN}+${amount:,.2f}{Colors.RESET}"
    elif amount < 0:
        return f"{Colors.RED}-${abs(amount):,.2f}{Colors.RESET}"
    else:
        return f"${amount:,.2f}"

def display_dashboard(state):
    """Mostrar dashboard con información actual"""
    clear_screen()
    
    # Header
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}📊 MONITOR EN TIEMPO REAL - ESTRATEGIA DE TENDENCIA{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    # Hora actual
    now = datetime.now()
    print(f"\n⏰ Hora actual: {Colors.WHITE}{now.strftime('%H:%M:%S')}{Colors.RESET}")
    
    # Estado de sesión
    session_name, session_status, session_info = get_current_session()
    if "ACTIVA" in session_status:
        print(f"📍 Sesión: {Colors.GREEN}{session_name} - {session_status}{Colors.RESET}")
    else:
        print(f"📍 Próxima sesión: {Colors.YELLOW}{session_name} {session_status}{Colors.RESET}")
    
    # Próxima ventana
    next_window, minutes_to = get_next_window()
    if minutes_to:
        print(f"⏱️ Próxima ventana: {Colors.BLUE}{next_window}{Colors.RESET} (en {minutes_to} minutos)")
    else:
        print(f"⏱️ Próxima ventana: {Colors.BLUE}{next_window}{Colors.RESET}")
    
    print(f"\n{Colors.CYAN}{'─'*70}{Colors.RESET}")
    
    if not state:
        print(f"\n{Colors.YELLOW}⚠️ No hay datos de estado disponibles{Colors.RESET}")
        print("   La estrategia puede no estar ejecutándose")
        return
    
    # Estadísticas del día
    print(f"\n{Colors.BOLD}📈 ESTADÍSTICAS DEL DÍA:{Colors.RESET}")
    
    # Trades
    daily_trades = state.get('daily_trades', 0)
    print(f"  📊 Trades realizados: {Colors.WHITE}{daily_trades}/12{Colors.RESET}")
    
    # Win/Loss
    wins = state.get('wins', 0)
    losses = state.get('losses', 0)
    total = wins + losses
    
    if total > 0:
        win_rate = (wins / total) * 100
        color = Colors.GREEN if win_rate >= 55 else Colors.YELLOW if win_rate >= 50 else Colors.RED
        print(f"  ✅ Victorias: {Colors.GREEN}{wins}{Colors.RESET}")
        print(f"  ❌ Pérdidas: {Colors.RED}{losses}{Colors.RESET}")
        print(f"  🎯 Win Rate: {color}{win_rate:.1f}%{Colors.RESET}")
    else:
        print(f"  📊 Sin operaciones completadas aún")
    
    # Profit/Loss
    daily_profit = state.get('daily_profit', 0)
    print(f"  💰 P&L del día: {format_profit(daily_profit)}")
    
    print(f"\n{Colors.CYAN}{'─'*70}{Colors.RESET}")
    
    # Coberturas
    print(f"\n{Colors.BOLD}🛡️ GESTIÓN DE COBERTURAS:{Colors.RESET}")
    hedges_placed = state.get('hedges_placed', 0)
    hedges_avoided = state.get('hedges_avoided', 0)
    total_hedges = hedges_placed + hedges_avoided
    
    if total_hedges > 0:
        hedge_rate = (hedges_placed / total_hedges) * 100
        print(f"  📍 Colocadas: {Colors.GREEN}{hedges_placed}{Colors.RESET}")
        print(f"  ⏭️ Evitadas: {Colors.YELLOW}{hedges_avoided}{Colors.RESET}")
        print(f"  📊 Tasa: {hedge_rate:.1f}%")
    else:
        print(f"  📊 Sin decisiones de cobertura aún")
    
    print(f"\n{Colors.CYAN}{'─'*70}{Colors.RESET}")
    
    # Top tendencias
    trend_scores = state.get('trend_scores', {})
    if trend_scores:
        print(f"\n{Colors.BOLD}🏆 TOP TENDENCIAS ACTUALES:{Colors.RESET}")
        top_trends = sorted(
            [(asset, data) for asset, data in trend_scores.items() if 'score' in data],
            key=lambda x: x[1]['score'],
            reverse=True
        )[:5]
        
        for i, (asset, data) in enumerate(top_trends, 1):
            score = data.get('score', 0)
            direction = data.get('direction', '?')
            strength = data.get('strength', 0)
            
            if score > 0:
                dir_symbol = "📈" if direction == "UP" else "📉"
                color = Colors.GREEN if score > 0.7 else Colors.YELLOW if score > 0.5 else Colors.WHITE
                print(f"  {i}. {dir_symbol} {asset}: {color}Score {score:.3f}{Colors.RESET} (Fuerza: {strength:.1%})")
    
    # Última actualización
    timestamp = state.get('timestamp', 'N/A')
    if timestamp != 'N/A':
        last_update = datetime.fromisoformat(timestamp)
        time_ago = (datetime.now() - last_update).total_seconds()
        if time_ago < 60:
            update_text = f"{int(time_ago)} segundos"
        else:
            update_text = f"{int(time_ago/60)} minutos"
        print(f"\n📅 Última actualización: hace {update_text}")
    
    print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")

def tail_log(n=10):
    """Mostrar las últimas n líneas del log"""
    if not os.path.exists(LOG_FILE):
        return []
    
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except:
        return []

def monitor_mode():
    """Modo monitor continuo"""
    print(f"{Colors.CYAN}🔄 Iniciando monitor en tiempo real...{Colors.RESET}")
    print(f"   Actualización cada {REFRESH_INTERVAL} segundos")
    print(f"   Presiona Ctrl+C para salir\n")
    time.sleep(2)
    
    try:
        while True:
            state = load_state()
            display_dashboard(state)
            time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⏹️ Monitor detenido{Colors.RESET}")

def log_mode():
    """Modo seguimiento de logs"""
    print(f"{Colors.CYAN}📝 Mostrando últimas líneas del log...{Colors.RESET}\n")
    
    try:
        last_size = 0
        while True:
            if os.path.exists(LOG_FILE):
                current_size = os.path.getsize(LOG_FILE)
                if current_size != last_size:
                    with open(LOG_FILE, 'r') as f:
                        f.seek(last_size)
                        new_lines = f.read()
                        if new_lines:
                            print(new_lines, end='')
                    last_size = current_size
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⏹️ Seguimiento de logs detenido{Colors.RESET}")

def main():
    """Función principal"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--log':
            log_mode()
        elif sys.argv[1] == '--once':
            state = load_state()
            display_dashboard(state)
        else:
            print("Uso:")
            print("  python monitor_trend.py         # Monitor continuo")
            print("  python monitor_trend.py --once  # Mostrar una vez")
            print("  python monitor_trend.py --log   # Seguir logs")
    else:
        monitor_mode()

if __name__ == "__main__":
    main()