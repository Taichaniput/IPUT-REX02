# core/src/__init__.py
"""
ビジネスロジック層のパッケージ

このパッケージには以下のモジュールが含まれます：
- financial_utils: 財務指標計算
- ml_analytics: 機械学習・予測・クラスタリング  
- ai_analysis: AI分析・LLM統合
"""

from .financial_utils import calculate_financial_indicators
from .ml_analytics import PredictionService, ClusteringService, PositioningService
from .ai_analysis import AIAnalysisService

# 各サービスのインスタンス化
prediction_service = PredictionService()
clustering_service = ClusteringService()
positioning_service = PositioningService()
ai_analysis_service = AIAnalysisService()

__all__ = [
    'calculate_financial_indicators',
    'prediction_service',
    'clustering_service',
    'positioning_service',
    'ai_analysis_service',
]