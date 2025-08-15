# 🔥 Guía del Período de Análisis (Calentamiento)

## 📊 ¿Qué es el Período de Análisis?

Es el tiempo que la estrategia dedica a:
- 📈 Analizar tendencias en todos los activos
- 🔍 Identificar los mejores patrones
- 📊 Construir historial de movimientos
- ⚠️ Evitar señales "gastadas"

## ⚡ NUEVO: Calentamiento Dinámico

El sistema ahora **ajusta automáticamente** el período de análisis según el tiempo disponible hasta la próxima sesión:

| Tiempo hasta sesión | Duración análisis | Tipo |
|-------------------|------------------|------|
| 2+ horas | 2 horas | COMPLETO (ideal) |
| 1-2 horas | 1 hora | MEDIO (bueno) |
| 30-60 min | 30 min | CORTO (aceptable) |
| 15-30 min | 15 min | EXPRESS (mínimo) |
| <15 min | 0 min | OMITIDO (directo) |

## 🕐 Ejemplos Prácticos (Hora Colombia)

### Ejemplo 1: Inicias a las 15:52 (3:52 PM)
```
Hora actual: 15:52
Próxima sesión: POST a las 17:00
Tiempo disponible: 68 minutos
→ Análisis MEDIO (1 hora)
→ Termina a las 16:52
→ ✅ Alcanzas todas las ventanas de POST
```

### Ejemplo 2: Inicias a las 16:30 (4:30 PM)
```
Hora actual: 16:30
Próxima sesión: POST a las 17:00
Tiempo disponible: 30 minutos
→ Análisis CORTO (30 min)
→ Termina a las 17:00
→ ✅ Justo a tiempo para POST
```

### Ejemplo 3: Inicias a las 09:30 AM
```
Hora actual: 09:30
Próxima sesión: OVERLAP a las 10:00
Tiempo disponible: 30 minutos
→ Análisis CORTO (30 min)
→ Termina a las 10:00
→ ✅ Listo para la MEJOR sesión
```

### Ejemplo 4: Inicias a las 16:50 (4:50 PM)
```
Hora actual: 16:50
Próxima sesión: POST a las 17:00
Tiempo disponible: 10 minutos
→ Análisis OMITIDO
→ Comienza directo a las 17:00
→ ⚠️ Menor precisión pero no pierdes sesión
```

## 🎯 Opciones de Control Manual

### 1. Omitir completamente el análisis
```bash
python3 main_trend.py --skip-warmup
```
- ✅ Ventaja: Operas inmediatamente
- ❌ Desventaja: Sin análisis previo, menor precisión

### 2. Especificar duración exacta
```bash
# Análisis de 15 minutos
python3 main_trend.py --warmup-minutes 15

# Análisis de 30 minutos
python3 main_trend.py --warmup-minutes 30

# Análisis de 1 hora
python3 main_trend.py --warmup-minutes 60

# Análisis completo de 2 horas
python3 main_trend.py --warmup-minutes 120
```

### 3. Dejar que el sistema decida (RECOMENDADO)
```bash
python3 main_trend.py
```
El sistema calculará automáticamente el mejor período según la próxima sesión.

## 📈 Calidad del Análisis por Duración

| Duración | Calidad | Tendencias detectadas | Recomendado para |
|----------|---------|----------------------|------------------|
| **2 horas** | ⭐⭐⭐⭐⭐ | Muy preciso, múltiples confirmaciones | Inicio del día |
| **1 hora** | ⭐⭐⭐⭐ | Bueno, tendencias principales | Entre sesiones |
| **30 min** | ⭐⭐⭐ | Aceptable, tendencias obvias | Tiempo limitado |
| **15 min** | ⭐⭐ | Básico, solo tendencias fuertes | Emergencia |
| **0 min** | ⭐ | Sin análisis, opera "a ciegas" | Último recurso |

## 💡 Mejores Prácticas

### Para máxima efectividad:
1. **Inicia 2 horas antes** de tu sesión objetivo
2. **Usa el análisis completo** al menos una vez al día
3. **No omitas** el análisis en la sesión OVERLAP (la mejor)

### Si tienes poco tiempo:
1. **Mínimo 30 minutos** de análisis es mejor que nada
2. **15 minutos EXPRESS** puede detectar tendencias obvias
3. **Omitir** solo si la sesión está por comenzar

## 🔧 Configuración Avanzada

Si quieres cambiar los períodos predefinidos, edita en `config_trend.py`:

```python
WARMUP_PERIOD_FULL = 7200     # 2 horas
WARMUP_PERIOD_MEDIUM = 3600   # 1 hora
WARMUP_PERIOD_SHORT = 1800    # 30 minutos
WARMUP_PERIOD_EXPRESS = 900   # 15 minutos
```

## 📊 Tabla de Decisión Rápida

| Si inicias a... | Para sesión de... | Usa... |
|----------------|-------------------|---------|
| 06:00 | NY_OPEN (08:00) | Análisis completo 2h |
| 07:30 | NY_OPEN (08:00) | Análisis corto 30min |
| 08:00 | OVERLAP (10:00) | Análisis completo 2h |
| 09:00 | OVERLAP (10:00) | Análisis medio 1h |
| 09:45 | OVERLAP (10:00) | Análisis express 15min |
| 12:00 | POWER (14:00) | Análisis completo 2h |
| 13:30 | POWER (14:00) | Análisis corto 30min |
| 15:00 | POST (17:00) | Análisis completo 2h |
| 16:00 | POST (17:00) | Análisis medio 1h |
| 16:45 | POST (17:00) | --skip-warmup |

## ❓ FAQ

**P: ¿Puedo cambiar de sesión después del análisis?**
R: Sí, el análisis sirve para todas las sesiones del día.

**P: ¿El análisis se guarda entre ejecuciones?**
R: Parcialmente. Los scores se guardan pero es mejor hacer análisis fresco cada día.

**P: ¿Qué pasa si inicio durante una sesión activa?**
R: El sistema esperará a la siguiente sesión después del análisis.

**P: ¿Puedo forzar análisis largo aunque la sesión esté cerca?**
R: Sí, usa `--warmup-minutes 120` pero perderás la sesión próxima.

## 🚀 Comando Recomendado para Tu Caso (15:52)

```bash
# Opción 1: Dejar que el sistema decida (1 hora de análisis)
python3 main_trend.py

# Opción 2: Análisis corto para alcanzar sesión POST
python3 main_trend.py --warmup-minutes 30

# Opción 3: Sin análisis para operar ya en POST
python3 main_trend.py --skip-warmup
```

Con el nuevo sistema dinámico, **nunca perderás una sesión** por el calentamiento, ya que se ajusta automáticamente al tiempo disponible.