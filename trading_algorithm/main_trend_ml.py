#!/usr/bin/env python3
"""
main_trend_ml.py - Punto de entrada para la estrategia de tendencia con protección
Estrategia con 3 sesiones diarias de trading
"""

import sys
import argparse
import logging
from datetime import datetime

from trading_algorithm.config_trend import (
    IQ_EMAIL, IQ_PASSWORD, ACCOUNT_TYPE, 
    LOG_FILE, print_strategy_configuration, get_current_session,
    TRADING_SESSIONS
)
from trading_algorithm.trend_strategy_ml import TrendProtectionStrategy
from trading_algorithm.utils_trend import setup_logger

def main():
    """Función principal para ejecutar la estrategia de tendencia"""
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description='Estrategia de Tendencia con Protección Dinámica para IQ Option',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main_trend.py                    # Ejecutar con análisis de 2 horas
  python main_trend.py --skip-warmup      # Omitir análisis inicial
  python main_trend.py --test              # Modo prueba - verificar configuración
  python main_trend.py --show-config       # Mostrar configuración y salir
  python main_trend.py --session MAÑANA    # Operar solo en sesión específica
        """
    )
    
    # Argumentos de credenciales
    parser.add_argument('--email', type=str, help='Email de IQ Option')
    parser.add_argument('--password', type=str, help='Contraseña de IQ Option')
    parser.add_argument('--account', type=str, choices=['PRACTICE', 'REAL'], 
                       default=ACCOUNT_TYPE, help='Tipo de cuenta')
    
    # Argumentos de modo
    parser.add_argument('--test', action='store_true', 
                       help='Ejecutar en modo prueba')
    parser.add_argument('--skip-warmup', action='store_true',
                       help='Omitir período de análisis inicial')
    parser.add_argument('--show-config', action='store_true',
                       help='Mostrar configuración y salir')
    parser.add_argument('--session', type=str, choices=['NY_OPEN', 'OVERLAP', 'POWER', 'POST', 'TODAS'],
                       default='TODAS', help='Sesión específica para operar')
    
    args = parser.parse_args()
    
    # Usar credenciales de argumentos o de config
    email = args.email or IQ_EMAIL
    password = args.password or IQ_PASSWORD
    account_type = args.account
    
    # Verificar credenciales
    if not email or not password or email == "tu_email@example.com":
        print("❌ ERROR: Configura tus credenciales en config_trend.py")
        print("O pásalas como argumentos:")
        print("python main_trend.py --email tu_email --password tu_password")
        sys.exit(1)
    
    # Configurar logger
    logger = setup_logger('main', LOG_FILE)
    
    # Mostrar configuración si se solicita
    if args.show_config:
        print_strategy_configuration()
        return
    
    # Banner de inicio
    print("\n" + "="*60)
    print("   📈 ESTRATEGIA DE TENDENCIA CON PROTECCIÓN DINÁMICA")
    print("="*60)
    print(f"📅 Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📧 Usuario: {email[:3]}***@***")
    print(f"💼 Cuenta: {account_type}")
    
    # Mostrar sesión actual
    current_session, status = get_current_session()
    print(f"📍 Estado actual: Sesión {current_session} ({status})")
    
    if args.session != 'TODAS':
        print(f"🎯 Modo: Operar solo en sesión {args.session}")
    else:
        print("🎯 Modo: Operar en TODAS las sesiones")
    
    print("="*60)
    
    print("\n🎯 CONCEPTO DE LA ESTRATEGIA:")
    print("1️⃣ Identificar tendencias claras durante análisis inicial")
    print("2️⃣ Colocar opción primaria siguiendo la tendencia")
    print("3️⃣ A los 10 minutos: verificar si está ITM")
    print("4️⃣ Si ITM → colocar cobertura (opción contraria)")
    print("5️⃣ Objetivo: Ganar con tendencia o minimizar pérdidas")
    print("="*60)
    
    if not args.skip_warmup:
        print("\n🔥 Se realizará análisis inicial de 2 horas")
        print("📊 Objetivo: Identificar los mejores activos en tendencia")
        print("\n⏰ HORARIOS DE TRADING (Hora Colombia UTC-5):")
        print("  📍 NY_OPEN: 08:00 - 09:00 (Apertura Wall Street)")
        print("  📍 OVERLAP: 10:00 - 11:00 (Solapamiento EU-NY) ⭐ MEJOR")
        print("  📍 POWER: 14:00 - 15:00 (Power Hour NY)")
        print("  📍 POST: 17:00 - 18:00 (Post-mercado OTC)")
        print("  📊 Total: 16 oportunidades de trading al día")
        print("="*60)
    else:
        print("\n⚠️ PERÍODO DE ANÁLISIS OMITIDO")
        print("   El sistema comenzará a operar inmediatamente")
        print("   Esto puede resultar en señales de menor calidad")
        print("="*60)
    
    # Mostrar información de sesiones
    print("\n📊 INFORMACIÓN DE SESIONES (Hora Colombia UTC-5):")
    
    sessions_info = {
        "NY_OPEN": ("08:00-09:00", "Apertura Wall Street", "ALTA", "🇺🇸"),
        "OVERLAP": ("10:00-11:00", "Solapamiento EU-NY", "MUY ALTA", "🌍🇺🇸⭐"),
        "POWER": ("14:00-15:00", "Power Hour NY", "ALTA", "🚀"),
        "POST": ("17:00-18:00", "Post-mercado OTC", "BAJA-MEDIA", "🌙")
    }
    
    for session_key, (time_range, desc, vol, emoji) in sessions_info.items():
        print(f"\n📍 {session_key} {emoji}:")
        print(f"   Horario: {time_range}")
        print(f"   {desc}")
        print(f"   Volatilidad esperada: {vol}")
        if session_key == "OVERLAP":
            print(f"   ⭐ MEJOR MOMENTO DEL DÍA PARA TRADING")
    
    print("="*60)
    
    try:
        # Crear estrategia con configuración de sesión
        logger.info("🚀 Inicializando estrategia...")
        
        strategy = TrendProtectionStrategy(
            email=email,
            password=password,
            account_type=account_type,
            skip_warmup=args.skip_warmup,
            session_filter=args.session  # Pasar filtro de sesión
        )
        
        if args.test:
            # Modo prueba
            logger.info("🧪 MODO PRUEBA - Verificando configuración...")
            logger.info(f"✅ Conexión exitosa")
            logger.info(f"💰 Balance: ${strategy.initial_capital:,.2f}")
            logger.info(f"📊 Activos disponibles: {len(strategy.valid_assets)}")
            
            if strategy.valid_assets:
                logger.info("📋 Primeros 10 activos:")
                for asset in strategy.valid_assets[:10]:
                    logger.info(f"   - {asset}")
            
            # Hacer un análisis rápido
            logger.info("\n🔍 Analizando tendencias actuales...")
            best_trend = strategy.get_best_trending_asset()
            
            if best_trend:
                logger.info(f"\n🏆 MEJOR TENDENCIA ACTUAL:")
                logger.info(f"  Activo: {best_trend['asset']}")
                logger.info(f"  Dirección: {best_trend['direction']}")
                logger.info(f"  Fuerza: {best_trend['strength']:.2%}")
                logger.info(f"  Score: {best_trend['score']:.3f}")
                logger.info(f"  Movimiento: {best_trend['movement']:.3%}")
                logger.info(f"  Retrocesos: {best_trend['retracements']}")
            else:
                logger.info("⚠️ No se encontraron tendencias válidas")
            
            # Mostrar próximas ventanas de trading
            logger.info("\n⏰ PRÓXIMAS VENTANAS DE TRADING:")
            now = datetime.now()
            upcoming_windows = []
            
            from trading_algorithm.config_trend import TRADING_WINDOWS
            for window in TRADING_WINDOWS:
                window_time = now.replace(
                    hour=window["hour"],
                    minute=window["minute"],
                    second=0,
                    microsecond=0
                )
                if window_time > now:
                    upcoming_windows.append(window)
                    if len(upcoming_windows) <= 3:  # Mostrar próximas 3
                        session = window.get("session", "")
                        logger.info(f"  - {window['hour']:02d}:{window['minute']:02d} [{session}]")
            
            logger.info("\n✅ Prueba completada exitosamente")
            return
        
        # Ejecutar estrategia
        logger.info("🎯 Iniciando estrategia de tendencia...")
        logger.info("📍 Operando en sesión(es): " + 
                   (args.session if args.session != 'TODAS' else 'TODAS LAS SESIONES'))
        logger.info("ℹ️ Presiona Ctrl+C para detener")
        
        strategy.run()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Estrategia detenida por el usuario")
    except Exception as e:
        logger.error(f"❌ Error fatal: {str(e)}")
        logger.error("Traceback completo:", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("👋 Programa finalizado")
        
        # Mostrar resumen si existe la estrategia
        if 'strategy' in locals():
            try:
                strategy.print_summary()
            except:
                pass

if __name__ == "__main__":
    main()