# trading_ml_optimizer/analyzers/ml_analyzer.py
"""
ML Analyzer - Sistema de Machine Learning para análisis y predicción
Encuentra patrones, predice resultados y optimiza parámetros
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# Para análisis avanzado
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from trading_ml_optimizer.database.db_manager import TradingDatabase


class TradingMLAnalyzer:
    """
    Analizador de Machine Learning para trading
    """
    
    def __init__(self, db_path: str = None):
        """
        Inicializa el analizador ML
        
        Args:
            db_path: Ruta a la base de datos
        """
        print("🤖 Inicializando ML Analyzer...")
        
        # Conectar a la base de datos
        self.db = TradingDatabase(db_path) if db_path else TradingDatabase()
        
        # Modelos
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Paths para guardar modelos
        self.models_dir = Path("trading_ml_optimizer/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración
        self.min_samples_for_training = 30  # Mínimo de muestras para entrenar
        self.feature_importance_threshold = 0.01  # Umbral de importancia
        
        print("✅ ML Analyzer inicializado")
    
    def prepare_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepara features para el modelo ML
        
        Args:
            df: DataFrame con datos raw (opcional)
            
        Returns:
            DataFrame con features engineered
        """
        if df is None:
            # Obtener datos de la base de datos
            query = """
                SELECT 
                    t.*,
                    strftime('%H', entry_time) as hour,
                    strftime('%w', entry_time) as day_of_week,
                    strftime('%d', entry_time) as day_of_month,
                    CASE 
                        WHEN strftime('%H', entry_time) BETWEEN '08' AND '09' THEN 'NY_OPEN'
                        WHEN strftime('%H', entry_time) BETWEEN '10' AND '11' THEN 'OVERLAP'
                        WHEN strftime('%H', entry_time) BETWEEN '14' AND '15' THEN 'POWER'
                        WHEN strftime('%H', entry_time) BETWEEN '17' AND '18' THEN 'POST'
                        ELSE 'OFF_HOURS'
                    END as market_session
                FROM trades t
                WHERE t.result IS NOT NULL
            """
            df = pd.read_sql_query(query, self.db.conn)
        
        if df.empty:
            print("⚠️ No hay datos suficientes para preparar features")
            return pd.DataFrame()
        
        # Feature Engineering
        features = pd.DataFrame()
        
        # 1. Temporales
        features['hour'] = pd.to_numeric(df['hour'], errors='coerce')
        features['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce')
        features['is_weekend'] = features['day_of_week'].isin([0, 6]).astype(int)
        
        # 2. Sesión de mercado
        session_encoder = LabelEncoder()
        features['market_session_encoded'] = session_encoder.fit_transform(
            df['market_session'].fillna('OFF_HOURS')
        )
        self.encoders['market_session'] = session_encoder
        
        # 3. Activo
        if 'asset' in df.columns:
            asset_encoder = LabelEncoder()
            features['asset_encoded'] = asset_encoder.fit_transform(
                df['asset'].fillna('UNKNOWN')
            )
            self.encoders['asset'] = asset_encoder
        
        # 4. Características del trade
        features['score'] = pd.to_numeric(df.get('score', 0), errors='coerce').fillna(0)
        features['amount'] = pd.to_numeric(df.get('amount', 0), errors='coerce').fillna(0)
        features['has_hedge'] = pd.to_numeric(df.get('has_hedge', 0), errors='coerce').fillna(0)
        
        # 5. Dirección del trade
        if 'direction' in df.columns:
            features['is_call'] = (df['direction'] == 'call').astype(int)
        
        # 6. Tendencia
        if 'trend_direction' in df.columns:
            features['trend_up'] = (df['trend_direction'] == 'UP').astype(int)
        
        # 7. Volatilidad estimada por sesión
        volatility_map = {
            'NY_OPEN': 3,
            'OVERLAP': 4,  # Máxima volatilidad
            'POWER': 3,
            'POST': 2,
            'OFF_HOURS': 1
        }
        features['volatility_estimate'] = df['market_session'].map(volatility_map).fillna(1)
        
        # 8. Features derivadas
        features['hour_squared'] = features['hour'] ** 2
        features['score_x_volatility'] = features['score'] * features['volatility_estimate']
        features['is_major_session'] = features['market_session_encoded'].isin([1, 2]).astype(int)
        
        # 9. Rolling features (si hay suficientes datos)
        if len(df) > 10:
            # Win rate de las últimas 5 operaciones
            df['is_win'] = (df['result'] == 'WIN').astype(int)
            features['rolling_win_rate'] = df['is_win'].rolling(5, min_periods=1).mean()
            features['rolling_win_rate'].fillna(0.5, inplace=True)
        
        # Target variable
        if 'result' in df.columns:
            features['target'] = (df['result'] == 'WIN').astype(int)
        
        # Eliminar NaN
        features = features.fillna(0)
        
        print(f"📊 Features preparadas: {len(features.columns)} variables, {len(features)} muestras")
        
        return features
    
    def train_prediction_model(self, features: pd.DataFrame = None) -> Dict:
        """
        Entrena modelo para predecir WIN/LOSS
        
        Args:
            features: DataFrame con features (opcional)
            
        Returns:
            Métricas del modelo
        """
        print("\n🎯 Entrenando modelo de predicción...")
        
        # Preparar features si no se proporcionan
        if features is None:
            features = self.prepare_features()
        
        if features.empty or len(features) < self.min_samples_for_training:
            print(f"⚠️ Datos insuficientes. Necesitas al menos {self.min_samples_for_training} trades")
            return {}
        
        # Separar features y target
        X = features.drop(['target'], axis=1, errors='ignore')
        y = features['target'] if 'target' in features else None
        
        if y is None or len(y.unique()) < 2:
            print("⚠️ No hay suficiente variación en los resultados para entrenar")
            return {}
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['prediction'] = scaler
        
        # Modelo 1: Random Forest (robusto y interpretable)
        print("  🌲 Entrenando Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Modelo 2: XGBoost (si está disponible)
        xgb_model = None
        if HAS_XGBOOST and len(X_train) > 50:
            print("  🚀 Entrenando XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluar modelos
        rf_pred = rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_precision = precision_score(y_test, rf_pred, zero_division=0)
        rf_recall = recall_score(y_test, rf_pred, zero_division=0)
        rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
        
        print(f"\n📊 Resultados Random Forest:")
        print(f"  Accuracy: {rf_accuracy:.2%}")
        print(f"  Precision: {rf_precision:.2%}")
        print(f"  Recall: {rf_recall:.2%}")
        print(f"  F1-Score: {rf_f1:.2%}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔑 Top 5 Features más importantes:")
        for idx, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Guardar modelos
        self.models['rf_prediction'] = rf_model
        if xgb_model:
            self.models['xgb_prediction'] = xgb_model
            
            # Evaluar XGBoost
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            print(f"\n📊 Resultados XGBoost:")
            print(f"  Accuracy: {xgb_accuracy:.2%}")
        
        # Guardar modelo a disco
        self.save_model('rf_prediction', rf_model)
        
        return {
            'model': 'RandomForest',
            'accuracy': rf_accuracy,
            'precision': rf_precision,
            'recall': rf_recall,
            'f1_score': rf_f1,
            'samples_train': len(X_train),
            'samples_test': len(X_test),
            'feature_importance': feature_importance.to_dict()
        }
    
    def predict_trade_outcome(self, trade_conditions: Dict) -> Dict:
        """
        Predice el resultado de un trade dados las condiciones
        
        Args:
            trade_conditions: Diccionario con las condiciones del trade
            
        Returns:
            Predicción y probabilidad
        """
        if 'rf_prediction' not in self.models:
            # Intentar cargar modelo guardado
            loaded_model = self.load_model('rf_prediction')
            if loaded_model:
                self.models['rf_prediction'] = loaded_model
            else:
                print("⚠️ No hay modelo entrenado. Entrena primero con train_prediction_model()")
                return {}
        
        # Preparar features
        features = pd.DataFrame([trade_conditions])
        
        # Aplicar transformaciones necesarias
        if 'market_session' in features.columns and 'market_session' in self.encoders:
            features['market_session_encoded'] = self.encoders['market_session'].transform(
                features['market_session']
            )
        
        # Escalar
        if 'prediction' in self.scalers:
            features_scaled = self.scalers['prediction'].transform(features)
        else:
            features_scaled = features
        
        # Predecir
        model = self.models['rf_prediction']
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': 'WIN' if prediction == 1 else 'LOSS',
            'probability_win': probability[1],
            'probability_loss': probability[0],
            'confidence': max(probability)
        }
    
    def optimize_parameters(self) -> Dict:
        """
        Encuentra parámetros óptimos para la estrategia
        
        Returns:
            Parámetros optimizados
        """
        print("\n⚙️ Optimizando parámetros de estrategia...")
        
        # Obtener datos históricos
        query = """
            SELECT 
                score,
                has_hedge,
                amount,
                result,
                strftime('%H', entry_time) as hour,
                asset
            FROM trades
            WHERE result IS NOT NULL
        """
        df = pd.read_sql_query(query, self.db.conn)
        
        if df.empty or len(df) < 20:
            print("⚠️ Datos insuficientes para optimización")
            return {}
        
        # 1. Encontrar score threshold óptimo
        scores = df[df['result'] == 'WIN']['score'].dropna()
        if len(scores) > 0:
            optimal_score = scores.quantile(0.3)  # 30 percentil de scores ganadores
        else:
            optimal_score = 0.65
        
        # 2. Encontrar mejor horario
        win_by_hour = df[df['result'] == 'WIN'].groupby('hour').size()
        total_by_hour = df.groupby('hour').size()
        win_rate_by_hour = (win_by_hour / total_by_hour).fillna(0)
        best_hours = win_rate_by_hour.nlargest(4).index.tolist()
        
        # 3. Encontrar mejores activos
        win_by_asset = df[df['result'] == 'WIN'].groupby('asset').size()
        total_by_asset = df.groupby('asset').size()
        win_rate_by_asset = (win_by_asset / total_by_asset).fillna(0)
        best_assets = win_rate_by_asset.nlargest(5).index.tolist()
        
        # 4. Efectividad del hedge
        hedge_effectiveness = df[df['has_hedge'] == 1]['result'].value_counts(normalize=True)
        hedge_win_rate = hedge_effectiveness.get('WIN', 0)
        
        # 5. Tamaño óptimo de posición (basado en Kelly Criterion simplificado)
        overall_win_rate = (df['result'] == 'WIN').mean()
        avg_win = df[df['result'] == 'WIN']['amount'].mean() if len(df[df['result'] == 'WIN']) > 0 else 0
        avg_loss = df[df['result'] == 'LOSS']['amount'].mean() if len(df[df['result'] == 'LOSS']) > 0 else 0
        
        if avg_loss > 0:
            kelly_fraction = (overall_win_rate * avg_win - (1 - overall_win_rate) * avg_loss) / avg_win
            optimal_position_size = max(0.01, min(0.05, kelly_fraction * 0.25))  # 25% de Kelly, max 5%
        else:
            optimal_position_size = 0.02
        
        optimized_params = {
            'optimal_score_threshold': round(optimal_score, 3),
            'best_trading_hours': best_hours,
            'best_assets': best_assets,
            'hedge_effectiveness': round(hedge_win_rate, 2),
            'optimal_position_size': round(optimal_position_size, 3),
            'current_win_rate': round(overall_win_rate, 2),
            'recommendations': []
        }
        
        # Generar recomendaciones
        if optimal_score < 0.65:
            optimized_params['recommendations'].append(
                f"📉 Reducir score mínimo a {optimal_score:.3f}"
            )
        elif optimal_score > 0.65:
            optimized_params['recommendations'].append(
                f"📈 Aumentar score mínimo a {optimal_score:.3f}"
            )
        
        if best_hours:
            optimized_params['recommendations'].append(
                f"⏰ Enfocarse en horas: {', '.join(best_hours)}"
            )
        
        if best_assets:
            optimized_params['recommendations'].append(
                f"🎯 Priorizar activos: {', '.join(best_assets[:3])}"
            )
        
        if hedge_win_rate < 0.5:
            optimized_params['recommendations'].append(
                "⚠️ Revisar estrategia de hedge (efectividad < 50%)"
            )
        
        print("\n📊 Parámetros Optimizados:")
        for key, value in optimized_params.items():
            if key != 'recommendations':
                print(f"  {key}: {value}")
        
        if optimized_params['recommendations']:
            print("\n💡 Recomendaciones:")
            for rec in optimized_params['recommendations']:
                print(f"  {rec}")
        
        return optimized_params
    
    def analyze_patterns(self) -> Dict:
        """
        Encuentra patrones en los datos de trading
        
        Returns:
            Diccionario con patrones encontrados
        """
        print("\n🔍 Analizando patrones...")
        
        # Obtener todos los datos
        trades_df = self.db.get_trades_dataframe()
        
        if trades_df.empty:
            print("⚠️ No hay datos para analizar")
            return {}
        
        patterns = {}
        
        # 1. Patrón: Racha de victorias/derrotas
        if 'result' in trades_df.columns:
            trades_df['is_win'] = trades_df['result'] == 'WIN'
            trades_df['streak'] = trades_df['is_win'].groupby(
                (trades_df['is_win'] != trades_df['is_win'].shift()).cumsum()
            ).cumcount() + 1
            
            max_win_streak = trades_df[trades_df['is_win']]['streak'].max()
            max_loss_streak = trades_df[~trades_df['is_win']]['streak'].max()
            
            patterns['streaks'] = {
                'max_win_streak': int(max_win_streak) if pd.notna(max_win_streak) else 0,
                'max_loss_streak': int(max_loss_streak) if pd.notna(max_loss_streak) else 0
            }
        
        # 2. Patrón: Correlación score-resultado
        if 'score' in trades_df.columns and 'result' in trades_df.columns:
            winning_scores = trades_df[trades_df['result'] == 'WIN']['score'].dropna()
            losing_scores = trades_df[trades_df['result'] == 'LOSS']['score'].dropna()
            
            if len(winning_scores) > 0 and len(losing_scores) > 0:
                patterns['score_analysis'] = {
                    'avg_winning_score': round(winning_scores.mean(), 3),
                    'avg_losing_score': round(losing_scores.mean(), 3),
                    'score_difference': round(winning_scores.mean() - losing_scores.mean(), 3)
                }
        
        # 3. Patrón: Mejor momento del día
        if 'entry_time' in trades_df.columns:
            trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
            hourly_performance = trades_df.groupby('hour').agg({
                'result': lambda x: (x == 'WIN').mean() if len(x) > 0 else 0
            })
            
            if not hourly_performance.empty:
                best_hour = hourly_performance.idxmax()['result']
                worst_hour = hourly_performance.idxmin()['result']
                
                patterns['time_patterns'] = {
                    'best_hour': int(best_hour),
                    'worst_hour': int(worst_hour),
                    'best_hour_win_rate': round(hourly_performance.loc[best_hour, 'result'], 2)
                }
        
        # 4. Patrón: Efectividad por activo
        if 'asset' in trades_df.columns:
            asset_performance = trades_df.groupby('asset').agg({
                'result': lambda x: (x == 'WIN').mean() if len(x) > 0 else 0
            }).sort_values('result', ascending=False)
            
            if not asset_performance.empty:
                patterns['asset_patterns'] = {
                    'best_asset': asset_performance.index[0],
                    'best_asset_win_rate': round(asset_performance.iloc[0]['result'], 2),
                    'worst_asset': asset_performance.index[-1],
                    'worst_asset_win_rate': round(asset_performance.iloc[-1]['result'], 2)
                }
        
        print("\n📊 Patrones encontrados:")
        for category, data in patterns.items():
            print(f"\n  {category}:")
            for key, value in data.items():
                print(f"    {key}: {value}")
        
        return patterns
    
    def generate_insights(self) -> Dict:
        """
        Genera insights accionables basados en el análisis
        
        Returns:
            Diccionario con insights
        """
        print("\n💡 Generando insights...")
        
        # Recopilar todos los análisis
        patterns = self.analyze_patterns()
        optimized_params = self.optimize_parameters()
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'key_insights': [],
            'warnings': [],
            'opportunities': [],
            'action_items': []
        }
        
        # Generar insights basados en patrones
        if patterns:
            # Rachas
            if patterns.get('streaks', {}).get('max_loss_streak', 0) > 3:
                insights['warnings'].append(
                    f"⚠️ Racha máxima de pérdidas: {patterns['streaks']['max_loss_streak']} trades"
                )
                insights['action_items'].append(
                    "Implementar stop-loss después de 2 pérdidas consecutivas"
                )
            
            # Score
            if 'score_analysis' in patterns:
                if patterns['score_analysis']['score_difference'] > 0.05:
                    insights['key_insights'].append(
                        f"✅ Score es buen predictor (diff: {patterns['score_analysis']['score_difference']:.3f})"
                    )
                else:
                    insights['warnings'].append(
                        "⚠️ Score no está diferenciando bien entre WIN/LOSS"
                    )
            
            # Tiempo
            if 'time_patterns' in patterns:
                insights['opportunities'].append(
                    f"🎯 Mejor hora para operar: {patterns['time_patterns']['best_hour']}:00 "
                    f"(WR: {patterns['time_patterns']['best_hour_win_rate']:.0%})"
                )
            
            # Activos
            if 'asset_patterns' in patterns:
                insights['opportunities'].append(
                    f"💰 Mejor activo: {patterns['asset_patterns']['best_asset']} "
                    f"(WR: {patterns['asset_patterns']['best_asset_win_rate']:.0%})"
                )
                
                if patterns['asset_patterns']['worst_asset_win_rate'] < 0.3:
                    insights['action_items'].append(
                        f"Evitar operar: {patterns['asset_patterns']['worst_asset']}"
                    )
        
        # Agregar recomendaciones de optimización
        if optimized_params and 'recommendations' in optimized_params:
            insights['action_items'].extend(optimized_params['recommendations'])
        
        # Resumen general
        stats = self.db.get_stats_summary()
        total_trades = stats.get('total_trades', 0)
        
        if total_trades > 0:
            win_rate = stats.get('wins', 0) / total_trades
            
            if win_rate < 0.5:
                insights['warnings'].append(
                    f"📉 Win rate bajo: {win_rate:.1%}"
                )
                insights['action_items'].append(
                    "Revisar criterios de entrada y considerar ser más selectivo"
                )
            elif win_rate > 0.65:
                insights['key_insights'].append(
                    f"🎯 Excelente win rate: {win_rate:.1%}"
                )
        
        # Imprimir insights
        print("\n📋 INSIGHTS GENERADOS:")
        
        if insights['key_insights']:
            print("\n🔑 Insights Clave:")
            for insight in insights['key_insights']:
                print(f"  {insight}")
        
        if insights['warnings']:
            print("\n⚠️ Advertencias:")
            for warning in insights['warnings']:
                print(f"  {warning}")
        
        if insights['opportunities']:
            print("\n💰 Oportunidades:")
            for opp in insights['opportunities']:
                print(f"  {opp}")
        
        if insights['action_items']:
            print("\n📌 Acciones Recomendadas:")
            for action in insights['action_items']:
                print(f"  {action}")
        
        return insights
    
    def save_model(self, model_name: str, model):
        """
        Guarda un modelo a disco
        
        Args:
            model_name: Nombre del modelo
            model: Objeto del modelo
        """
        model_path = self.models_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Modelo guardado: {model_path}")
    
    def load_model(self, model_name: str):
        """
        Carga un modelo de disco
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Modelo cargado o None
        """
        model_path = self.models_dir / f"{model_name}.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"📂 Modelo cargado: {model_path}")
            return model
        return None
    
    def generate_report(self) -> Dict:
        """
        Genera un reporte completo de ML
        
        Returns:
            Reporte completo
        """
        print("\n📄 Generando reporte ML...")
        
        # Entrenar modelo si no existe
        features = self.prepare_features()
        model_metrics = {}
        
        if not features.empty and len(features) >= self.min_samples_for_training:
            model_metrics = self.train_prediction_model(features)
        
        # Obtener insights
        insights = self.generate_insights()
        
        # Obtener patrones
        patterns = self.analyze_patterns()
        
        # Obtener parámetros optimizados
        optimized_params = self.optimize_parameters()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': model_metrics,
            'insights': insights,
            'patterns': patterns,
            'optimized_parameters': optimized_params,
            'data_summary': {
                'total_features': len(features.columns) if not features.empty else 0,
                'total_samples': len(features) if not features.empty else 0,
                'model_ready': 'rf_prediction' in self.models
            }
        }
        
        # Guardar reporte
        report_path = Path("trading_ml_optimizer/data/ml_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Reporte guardado: {report_path}")
        
        return report


# Funciones de utilidad
def run_ml_analysis():
    """Ejecuta análisis ML completo"""
    analyzer = TradingMLAnalyzer()
    
    # Generar reporte completo
    report = analyzer.generate_report()
    
    print("\n" + "="*60)
    print("📊 RESUMEN DEL ANÁLISIS ML")
    print("="*60)
    
    if 'model_performance' in report and report['model_performance']:
        print(f"\n🤖 Modelo predictivo:")
        print(f"  Accuracy: {report['model_performance']['accuracy']:.2%}")
        print(f"  Samples: {report['model_performance']['samples_train']} train, "
              f"{report['model_performance']['samples_test']} test")
    
    if 'optimized_parameters' in report:
        print(f"\n⚙️ Parámetros optimizados:")
        print(f"  Score óptimo: {report['optimized_parameters'].get('optimal_score_threshold', 'N/A')}")
        print(f"  Position size: {report['optimized_parameters'].get('optimal_position_size', 'N/A')}")
    
    print("\n✅ Análisis completado")
    
    return report


def predict_next_trade(conditions: Dict):
    """
    Predice el resultado del próximo trade
    
    Args:
        conditions: Condiciones del trade
    """
    analyzer = TradingMLAnalyzer()
    
    # Hacer predicción
    prediction = analyzer.predict_trade_outcome(conditions)
    
    if prediction:
        print(f"\n🎯 Predicción para próximo trade:")
        print(f"  Resultado esperado: {prediction['prediction']}")
        print(f"  Probabilidad WIN: {prediction['probability_win']:.1%}")
        print(f"  Confianza: {prediction['confidence']:.1%}")
    else:
        print("⚠️ No se pudo hacer predicción (modelo no entrenado)")
    
    return prediction


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Analyzer para trading')
    parser.add_argument('--mode', choices=['analyze', 'train', 'predict', 'optimize'], 
                       default='analyze',
                       help='Modo de operación')
    parser.add_argument('--asset', type=str, help='Activo para predicción')
    parser.add_argument('--score', type=float, help='Score para predicción')
    parser.add_argument('--hour', type=int, help='Hora para predicción')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        run_ml_analysis()
    
    elif args.mode == 'train':
        analyzer = TradingMLAnalyzer()
        features = analyzer.prepare_features()
        if not features.empty:
            metrics = analyzer.train_prediction_model(features)
            print(f"\n✅ Modelo entrenado con accuracy: {metrics.get('accuracy', 0):.2%}")
    
    elif args.mode == 'predict':
        conditions = {
            'hour': args.hour or 10,
            'asset': args.asset or 'EURUSD',
            'score': args.score or 0.7,
            'market_session': 'OVERLAP',
            'is_call': 1,
            'has_hedge': 0
        }
        predict_next_trade(conditions)
    
    elif args.mode == 'optimize':
        analyzer = TradingMLAnalyzer()
        params = analyzer.optimize_parameters()
        print(f"\n✅ Optimización completada")