#!/usr/bin/env python3
"""
Script para reiniciar la estrategia de tendencia con protección
Borra todas las estadísticas acumuladas y permite empezar de cero
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path

# Configuración de archivos de la estrategia de tendencia
STATE_FILE = "trend_strategy_state.json"
LOG_FILE = "trend_strategy.log"
BACKUP_DIR = "backups_trend"

def load_current_state():
    """Cargar el estado actual si existe"""
    if not os.path.exists(STATE_FILE):
        return None
    
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error leyendo estado: {e}")
        return None

def show_current_statistics(state):
    """Mostrar las estadísticas actuales de la estrategia de tendencia"""
    if not state:
        print("📊 No hay estadísticas previas")
        return
    
    print("\n📊 ESTADÍSTICAS ACTUALES - ESTRATEGIA DE TENDENCIA:")
    print("=" * 60)
    
    # Información básica
    timestamp = state.get('timestamp', 'N/A')
    print(f"📅 Última actualización: {timestamp}")
    
    # Trades realizados
    daily_trades = state.get('daily_trades', 0)
    print(f"\n📈 Trades del día: {daily_trades}/12 posibles")
    
    # Wins/Losses
    wins = state.get('wins', 0)
    losses = state.get('losses', 0)
    total_trades = wins + losses
    
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        print(f"\n🎯 Resultados:")
        print(f"  ✅ Victorias: {wins} ({win_rate:.1f}%)")
        print(f"  ❌ Pérdidas: {losses} ({100-win_rate:.1f}%)")
        print(f"  📊 Win Rate: {win_rate:.1f}%")
    else:
        print("\n📊 Sin operaciones registradas aún")
    
    # Profit diario
    daily_profit = state.get('daily_profit', 0)
    print(f"\n💰 Profit/Loss del día: ${daily_profit:,.2f}")
    
    # Coberturas
    hedges_placed = state.get('hedges_placed', 0)
    hedges_avoided = state.get('hedges_avoided', 0)
    total_hedge_decisions = hedges_placed + hedges_avoided
    
    if total_hedge_decisions > 0:
        hedge_rate = (hedges_placed / total_hedge_decisions) * 100
        print(f"\n🛡️ Gestión de Coberturas:")
        print(f"  📍 Coberturas colocadas: {hedges_placed}")
        print(f"  ⏭️ Coberturas evitadas: {hedges_avoided}")
        print(f"  📊 Tasa de cobertura: {hedge_rate:.1f}%")
    
    # Scores de tendencia guardados
    trend_scores = state.get('trend_scores', {})
    if trend_scores:
        print(f"\n📈 Tendencias monitoreadas: {len(trend_scores)} activos")
        # Mostrar top 3 tendencias
        top_trends = sorted(
            [(asset, data['score']) for asset, data in trend_scores.items() if 'score' in data],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        if top_trends:
            print("🏆 Top 3 tendencias guardadas:")
            for i, (asset, score) in enumerate(top_trends, 1):
                print(f"   {i}. {asset}: Score {score:.3f}")
    
    # Última ventana operada
    last_window = state.get('last_window_traded', 'Ninguna')
    print(f"\n⏰ Última ventana operada: {last_window}")
    
    print("=" * 60)

def show_sesions_info():
    """Mostrar información sobre las sesiones de trading"""
    print("\n📍 SESIONES DE TRADING CONFIGURADAS (Hora Colombia UTC-5):")
    print("=" * 60)
    print("• NY_OPEN:  08:00 - 09:00 (Apertura Wall Street) 🇺🇸")
    print("• OVERLAP:  10:00 - 11:00 (Solapamiento EU-NY) ⭐ MEJOR")
    print("• POWER:    14:00 - 15:00 (Power Hour NY) 🚀")
    print("• POST:     17:00 - 18:00 (Post-mercado OTC) 🌙")
    print("=" * 60)
    print("📊 Total: 16 ventanas de trading al día (4 sesiones × 4 ventanas)")
    print("=" * 60)

def create_backup(state):
    """Crear backup del estado actual"""
    if not state:
        return None
    
    # Crear directorio de backups si no existe
    Path(BACKUP_DIR).mkdir(exist_ok=True)
    
    # Nombre del archivo de backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{BACKUP_DIR}/trend_state_backup_{timestamp}.json"
    
    # Guardar backup
    try:
        with open(backup_file, 'w') as f:
            json.dump(state, f, indent=4)
        print(f"\n💾 Backup creado: {backup_file}")
        return backup_file
    except Exception as e:
        print(f"❌ Error creando backup: {e}")
        return None

def reset_strategy():
    """Reiniciar la estrategia de tendencia"""
    print("\n🔄 REINICIANDO ESTRATEGIA DE TENDENCIA...")
    
    # Eliminar archivo de estado
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print(f"✅ Archivo {STATE_FILE} eliminado")
    else:
        print(f"📄 No se encontró {STATE_FILE}")
    
    # Opcional: Limpiar log
    response = input("\n¿Deseas también limpiar el archivo de log? (s/N): ").lower()
    if response == 's':
        if os.path.exists(LOG_FILE):
            # Hacer backup del log
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_backup = f"{BACKUP_DIR}/trend_log_{timestamp}.log"
            Path(BACKUP_DIR).mkdir(exist_ok=True)
            shutil.move(LOG_FILE, log_backup)
            print(f"✅ Log movido a: {log_backup}")
        else:
            print("📄 No se encontró archivo de log")
    
    # Opcional: Limpiar análisis de tendencias guardados
    response = input("\n¿Deseas reiniciar el análisis de tendencias? (s/N): ").lower()
    if response == 's':
        print("✅ El próximo análisis empezará desde cero")
    
    print("\n✨ ¡Estrategia de tendencia reiniciada exitosamente!")
    print("📌 La próxima vez que ejecutes la estrategia:")
    print("   - Comenzará con estadísticas en cero")
    print("   - Realizará nuevo análisis de tendencias (2 horas)")
    print("   - Mantendrá tu configuración actual")
    print("   - Usará tu balance actual como capital inicial")

def show_partial_reset_options():
    """Mostrar opciones de reinicio parcial"""
    print("\n🔧 OPCIONES DE REINICIO PARCIAL:")
    print("=" * 60)
    print("1. Reiniciar solo estadísticas diarias")
    print("2. Reiniciar solo análisis de tendencias")
    print("3. Reiniciar solo coberturas")
    print("4. Reiniciar todo (completo)")
    print("5. Cancelar")
    print("=" * 60)
    
    choice = input("\nSelecciona una opción (1-5): ")
    return choice

def partial_reset(state, option):
    """Realizar reinicio parcial según la opción seleccionada"""
    if not state:
        state = {}
    
    if option == "1":
        # Reiniciar solo estadísticas diarias
        state['daily_trades'] = 0
        state['daily_profit'] = 0
        state['last_window_traded'] = None
        print("✅ Estadísticas diarias reiniciadas")
        
    elif option == "2":
        # Reiniciar análisis de tendencias
        state['trend_scores'] = {}
        print("✅ Análisis de tendencias reiniciado")
        
    elif option == "3":
        # Reiniciar coberturas
        state['hedges_placed'] = 0
        state['hedges_avoided'] = 0
        print("✅ Estadísticas de coberturas reiniciadas")
        
    elif option == "4":
        # Reinicio completo
        return None
        
    else:
        print("❌ Opción inválida")
        return state
    
    # Guardar estado modificado
    if option != "4":
        state['timestamp'] = datetime.now().isoformat()
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        print(f"💾 Estado actualizado guardado en {STATE_FILE}")
    
    return state

def main():
    """Función principal"""
    print("\n🔄 SCRIPT DE REINICIO - ESTRATEGIA DE TENDENCIA")
    print("=" * 60)
    print("📈 Estrategia: Tendencia con Protección Dinámica")
    print("=" * 60)
    
    # Mostrar información de sesiones
    show_sesions_info()
    
    # Cargar estado actual
    current_state = load_current_state()
    
    # Mostrar estadísticas actuales
    show_current_statistics(current_state)
    
    # Menú de opciones
    print("\n¿Qué deseas hacer?")
    print("1. Reinicio completo (borra todo)")
    print("2. Reinicio parcial (mantener algunas estadísticas)")
    print("3. Solo ver estadísticas (no hacer cambios)")
    print("4. Cancelar")
    
    choice = input("\nSelecciona una opción (1-4): ")
    
    if choice == "1":
        # Reinicio completo
        print("\n⚠️  ADVERTENCIA: Esta acción borrará TODAS las estadísticas")
        response = input("¿Estás seguro? (s/N): ").lower()
        
        if response == 's':
            # Crear backup si hay estado
            if current_state:
                backup_file = create_backup(current_state)
                if backup_file:
                    print(f"💡 Puedes restaurar el estado anterior con:")
                    print(f"   cp {backup_file} {STATE_FILE}")
            
            # Reiniciar estrategia
            reset_strategy()
        else:
            print("❌ Operación cancelada")
            
    elif choice == "2":
        # Reinicio parcial
        option = show_partial_reset_options()
        if option != "5":
            if option == "4":
                # Reinicio completo desde menú parcial
                print("\n⚠️  ADVERTENCIA: Esta acción borrará TODAS las estadísticas")
                response = input("¿Estás seguro? (s/N): ").lower()
                if response == 's':
                    if current_state:
                        create_backup(current_state)
                    reset_strategy()
            else:
                partial_reset(current_state, option)
        else:
            print("❌ Operación cancelada")
            
    elif choice == "3":
        # Solo mostrar estadísticas
        print("\n✅ Solo se mostraron las estadísticas, no se realizaron cambios")
        
    else:
        print("❌ Operación cancelada")
    
    print("\n🚀 Script finalizado")
    print("📌 Ejecuta 'python main_trend.py' para iniciar la estrategia")

if __name__ == "__main__":
    main()