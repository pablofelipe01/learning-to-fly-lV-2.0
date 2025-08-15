# 🚀 GUÍA RÁPIDA - Estrategia de Tendencia con Protección

## 📦 Instalación (5 minutos)

### 1. Instalar dependencias
```bash
pip install iqoptionapi numpy scipy scikit-learn
```

### 2. Configurar credenciales
Edita `config_trend.py`:
```python
IQ_EMAIL = "tu_email@ejemplo.com"
IQ_PASSWORD = "tu_contraseña"
ACCOUNT_TYPE = "PRACTICE"  # Empezar siempre en PRACTICE
```

### 3. Verificar instalación
```bash
python main_trend.py --test
```

## 🎯 Uso Rápido

### Opción 1: Usar el menú interactivo
```bash
# Linux/Mac
chmod +x run_trend.sh
./run_trend.sh

# Windows
run_trend.bat
```

### Opción 2: Comandos directos

#### Ejecutar estrategia completa (recomendado para iniciar)
```bash
python main_trend.py
```

#### Operar solo en una sesión específica
```bash
# Solo apertura NY (08:00-09:00)
python main_trend.py --session NY_OPEN

# Solo solapamiento EU-NY (10:00-11:00) ⭐ RECOMENDADO
python main_trend.py --session OVERLAP

# Solo Power Hour (14:00-15:00)
python main_trend.py --session POWER

# Solo post-mercado OTC (17:00-18:00)
python main_trend.py --session POST
```

#### Sin período de análisis (más rápido pero menos preciso)
```bash
python main_trend.py --skip-warmup
```

## 📊 Horarios de Trading (Hora Colombia UTC-5)

| Sesión | Horario | Ventanas | Contexto | Volatilidad |
|--------|---------|----------|----------|-------------|
| **NY_OPEN** | 08:00-09:00 | 08:00, 08:15, 08:30, 08:45 | Apertura Wall Street 🇺🇸 | Alta |
| **OVERLAP** | 10:00-11:00 | 10:00, 10:15, 10:30, 10:45 | Solapamiento EU-NY 🌍🇺🇸 | **MUY ALTA** ⭐ |
| **POWER** | 14:00-15:00 | 14:00, 14:15, 14:30, 14:45 | Power Hour NY 🚀 | Alta |
| **POST** | 17:00-18:00 | 17:00, 17:15, 17:30, 17:45 | Post-mercado OTC 🌙 | Baja-Media |

### ⭐ Mejores Momentos para Trading

1. **10:00-11:00 (OVERLAP)** - Máxima liquidez y volatilidad
2. **08:00-09:00 (NY_OPEN)** - Gaps y movimientos iniciales
3. **14:00-15:00 (POWER)** - Movimientos institucionales finales
4. **17:00-18:00 (POST)** - Solo OTC, menor liquidez

## 🎯 Cómo Funciona

```
1. ANÁLISIS (2 horas)
   ↓
   Identifica tendencias fuertes
   ↓
2. VENTANA DE TRADING (cada 15 min)
   ↓
   Coloca opción primaria (CALL o PUT)
   ↓
3. MINUTO 10
   ↓
   ¿Está ITM?
   ├─ SÍ → Coloca cobertura
   └─ NO → No hacer nada
   ↓
4. MINUTO 15
   ↓
   Vencimiento y resultado
```

## 📈 Ejemplos de Resultados

### Caso 1: Tendencia continúa
- Primaria: **GANA** ✅ (+85%)
- Cobertura: **PIERDE** ❌ (-100%)
- **Neto**: -15% (pérdida mínima)

### Caso 2: Reversión
- Primaria: **PIERDE** ❌ (-100%)
- Cobertura: **GANA** ✅ (+85%)
- **Neto**: -15% (pérdida mínima)

### Caso 3: Zona ganadora
- Primaria: **GANA** ✅ (+85%)
- Cobertura: **GANA** ✅ (+85%)
- **Neto**: +170% (¡mejor caso!)

## 🔍 Monitoreo

### Ver logs en tiempo real
```bash
tail -f trend_strategy.log
```

### Ver estado guardado
```bash
cat trend_strategy_state.json | python -m json.tool
```

## ⚙️ Configuración Rápida

### Cambiar tamaño de posición
En `config_trend.py`:
```python
POSITION_SIZE_PERCENT = 0.02  # 2% del capital
```

### Ajustar criterios de tendencia
```python
MIN_TREND_STRENGTH = 0.65    # Más alto = menos señales, más calidad
MIN_TREND_CONSISTENCY = 0.60  # Más alto = tendencias más limpias
```

### Modificar horarios
```python
TRADING_WINDOWS = [
    {"hour": 11, "minute": 0, "session": "MAÑANA"},
    # Agregar o modificar ventanas aquí
]
```

## 📊 Checklist Pre-Trading

- [ ] Cuenta en PRACTICE primero
- [ ] Período de análisis de 2 horas completado
- [ ] Al menos 5 tendencias válidas identificadas
- [ ] Capital suficiente para 12 operaciones
- [ ] Logs monitoreados
- [ ] Horarios de sesiones verificados

## 🚨 Comandos de Emergencia

### Detener inmediatamente
```
Ctrl + C
```

### Ver últimas operaciones
```bash
tail -n 50 trend_strategy.log | grep -E "(GANADA|PERDIDA)"
```

### Verificar balance actual
```bash
python main_trend.py --test | grep Balance
```

## 💡 Tips para Mejores Resultados

1. **Siempre hacer el análisis de 2 horas** - identifica mejores tendencias
2. **Empezar con una sola sesión** - sugiero TARDE (14:00-15:00) por mayor volatilidad
3. **Monitorear las primeras operaciones** - ajustar si necesario
4. **No omitir el warmup los primeros días** - necesitas datos de calidad
5. **Revisar logs después de cada sesión** - aprender patrones

## 📱 Notificaciones (Opcional)

Para recibir alertas de trades:
```python
# Agregar en config_trend.py
TELEGRAM_BOT_TOKEN = "tu_token"
TELEGRAM_CHAT_ID = "tu_chat_id"
```

## 🆘 Soporte Rápido

### La estrategia no encuentra tendencias
- Reducir `MIN_TREND_STRENGTH` a 0.55
- Verificar que los mercados estén abiertos
- Aumentar período de análisis

### No se colocan coberturas
- Verificar que las primarias estén ITM
- Revisar logs en el minuto 10
- Ajustar `HEDGE_CHECK_MINUTE` si necesario

### Error de conexión
- Verificar credenciales
- Probar con VPN si es necesario
- Usar cuenta PRACTICE primero

## 📈 Métricas de Éxito

- **Win Rate objetivo**: >55%
- **Tasa de cobertura ideal**: 40-60%
- **Profit diario esperado**: 5-15%
- **Máximo drawdown aceptable**: 10%

---

**⚠️ IMPORTANTE**: Siempre probar en PRACTICE al menos 1 semana antes de usar dinero real.