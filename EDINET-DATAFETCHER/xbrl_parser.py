# xbrl_parser.py

import re
import logging
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import pandas as pd

logger = logging.getLogger(__name__)


class XbrlParser:
    """XBRLファイルから財務データを抽出するパーサー"""
    
    # タグマッピング定義
    TAG_MAPPINGS = {
        'net_sales': [
            "jppfs_cor:NetSales", "jppfs_cor:Sales", "jppfs_cor:OperatingRevenues",
            "jppfs_cor:Revenue", "jppfs_cor:Revenues", "jppfs_cor:SalesRevenue"
        ],
        'operating_income': [
            "jppfs_cor:OperatingIncome", "jppfs_cor:OperatingProfit",
            "jppfs_cor:OperatingProfitLoss", "jppfs_cor:ProfitLossFromOperatingActivities"
        ],
        'ordinary_income': [
            "jppfs_cor:OrdinaryIncome", "jppfs_cor:OrdinaryProfitLoss",
            "jppfs_cor:ProfitLossBeforeIncomeTaxes", "jppfs_cor:OrdinaryProfit"
        ],
        'net_income': [
            "jppfs_cor:ProfitLossAttributableToOwnersOfParent", "jppfs_cor:NetIncome",
            "jppfs_cor:ProfitLoss", "jppfs_cor:ProfitAttributableToOwnersOfParent"
        ],
        'net_assets': [
            "jppfs_cor:NetAssets", "jppfs_cor:TotalNetAssets",
            "jppfs_cor:ShareholdersEquity", "jppfs_cor:Equity"
        ],
        'total_assets': [
            "jppfs_cor:TotalAssets", "jppfs_cor:Assets", 
            "jppfs_cor:TotalAssetsIFRS"
        ],
        'total_liabilities': [
            "jppfs_cor:TotalLiabilities", "jppfs_cor:Liabilities",
            "jppfs_cor:TotalLiabilitiesIFRS"
        ],
        'operating_cash_flow': [
            "jppfs_cor:NetCashProvidedByUsedInOperatingActivities",
            "jppfs_cor:CashFlowsFromOperatingActivities",
            "jppfs_cor:OperatingCashFlow"
        ],
        'r_and_d_expenses': [
            "jppfs_cor:ResearchAndDevelopmentExpense",
            "jppfs_cor:ResearchAndDevelopmentExpenses",
            "jppfs_cor:ResearchAndDevelopmentCosts"
        ],
        'number_of_employees': [
            "jpcrp_cor:NumberOfEmployees", "jpcrp_cor:TotalNumberOfEmployees",
            "jppfs_cor:NumberOfEmployees"
        ]
    }
    
    # コンテキスト定義
    DURATION_CONTEXTS = [
        "CurrentYearDuration_ConsolidatedMember",
        "CurrentYearDuration_NonConsolidatedMember",
        "CurrentYearDuration"
    ]
    
    INSTANT_CONTEXTS = [
        "CurrentYearInstant_ConsolidatedMember",
        "CurrentYearInstant_NonConsolidatedMember",
        "CurrentYearInstant"
    ]
    
    # 期間データの項目
    DURATION_ITEMS = {
        'net_sales', 'operating_income', 'ordinary_income', 
        'net_income', 'operating_cash_flow', 'r_and_d_expenses'
    }
    
    def __init__(self, xbrl_content: bytes, doc_id: str = "N/A"):
        self.soup = BeautifulSoup(xbrl_content, "lxml-xml")
        self.doc_id = doc_id
    
    def parse(self) -> pd.Series:
        """XBRLから財務データを抽出"""
        data = {}
        
        for key, tags in self.TAG_MAPPINGS.items():
            contexts = self.DURATION_CONTEXTS if key in self.DURATION_ITEMS else self.INSTANT_CONTEXTS
            value = self._get_value(key, tags, contexts)
            data[key] = value
        
        return pd.Series(data, dtype='object')
    
    def _get_value(self, item_name: str, tag_names: List[str], contexts: List[str]) -> Optional[int]:
        """指定されたタグとコンテキストから値を取得"""
        # 優先順位順に検索
        for strategy in [self._exact_match, self._partial_match, self._fallback_match]:
            value = strategy(item_name, tag_names, contexts)
            if value is not None:
                return value
        
        return None
    
    def _exact_match(self, item_name: str, tag_names: List[str], contexts: List[str]) -> Optional[int]:
        """正確なコンテキストマッチング"""
        for tag in tag_names:
            for context in contexts:
                elements = self.soup.find_all(
                    name=tag, 
                    attrs={'contextRef': lambda x: x and context in x}
                )
                value = self._extract_value_from_elements(elements)
                if value is not None:
                    logger.debug(f"[{self.doc_id}] Found '{item_name}' with exact match")
                    return value
        return None
    
    def _partial_match(self, item_name: str, tag_names: List[str], contexts: List[str]) -> Optional[int]:
        """部分的なコンテキストマッチング"""
        for tag in tag_names:
            elements = self.soup.find_all(name=tag)
            for element in elements:
                context_ref = element.get('contextRef', '')
                if any(keyword in context_ref for keyword in ['Current', 'Instant', 'Duration']):
                    value = self._extract_value_from_element(element)
                    if value is not None:
                        logger.debug(f"[{self.doc_id}] Found '{item_name}' with partial match")
                        return value
        return None
    
    def _fallback_match(self, item_name: str, tag_names: List[str], contexts: List[str]) -> Optional[int]:
        """任意のコンテキストで検索"""
        for tag in tag_names:
            elements = self.soup.find_all(name=tag)
            value = self._extract_value_from_elements(elements)
            if value is not None:
                logger.debug(f"[{self.doc_id}] Found '{item_name}' with fallback search")
                return value
        return None
    
    def _extract_value_from_elements(self, elements: List) -> Optional[int]:
        """要素リストから最初の有効な値を抽出"""
        for element in elements:
            value = self._extract_value_from_element(element)
            if value is not None:
                return value
        return None
    
    def _extract_value_from_element(self, element) -> Optional[int]:
        """単一要素から値を抽出"""
        if not element or not element.text:
            return None
        
        try:
            text = element.text.strip()
            if not text:
                return None
            
            # 数値変換処理
            cleaned = text.replace(',', '').replace('△', '-').replace('(', '-').replace(')', '')
            return int(float(cleaned))
        except (ValueError, TypeError):
            return None


class DataProcessor:
    """財務データの補完と検証を行うプロセッサー"""
    
    # 業界平均値
    INDUSTRY_PARAMS = {
        'manufacturing': {
            'rd_ratio': 0.03,
            'asset_turnover': 0.8,
            'liability_ratio': 0.6,
        },
        'service': {
            'rd_ratio': 0.01,
            'asset_turnover': 1.2,
            'liability_ratio': 0.5,
        }
    }
    
    # デフォルト値
    DEFAULT_VALUES = {
        'net_assets': 0,
        'total_assets': 1000000,
        'net_sales': 0,
        'operating_income': 0,
        'ordinary_income': 0,
        'net_income': 0,
        'total_liabilities': 0,
        'operating_cash_flow': 0,
        'r_and_d_expenses': 0,
        'number_of_employees': 1
    }
    
    # 必須フィールド
    CRITICAL_FIELDS = ['total_assets', 'net_sales']
    
    # 最小信頼度スコア
    MIN_CONFIDENCE_SCORE = 30
    
    def process(self, data: pd.Series, doc_id: str, company_name: str = "") -> Optional[Dict]:
        """データベース保存用にデータを処理"""
        # 会計恒等式による補完
        data = self._apply_accounting_rules(data)
        
        # 業界平均による推定
        data = self._estimate_missing_values(data, company_name)
        
        # デフォルト値の適用
        data = self._apply_defaults(data)
        
        # データ検証
        validation = self._validate_data(data, doc_id)
        
        if not validation['is_valid'] or validation['confidence_score'] < self.MIN_CONFIDENCE_SCORE:
            logger.warning(f"[{doc_id}] Data quality insufficient (score: {validation['confidence_score']})")
            return None
        
        # 整数変換とDB用データ作成
        return self._prepare_db_data(data, validation['confidence_score'])
    
    def _apply_accounting_rules(self, data: pd.Series) -> pd.Series:
        """会計恒等式を使用してデータを補完"""
        s = data.copy()
        
        # 純資産 = 総資産 - 総負債
        if pd.isna(s.get('net_assets')) and pd.notna(s.get('total_assets')) and pd.notna(s.get('total_liabilities')):
            s['net_assets'] = s['total_assets'] - s['total_liabilities']
        elif pd.isna(s.get('total_assets')) and pd.notna(s.get('net_assets')) and pd.notna(s.get('total_liabilities')):
            s['total_assets'] = s['total_liabilities'] + s['net_assets']
        elif pd.isna(s.get('total_liabilities')) and pd.notna(s.get('total_assets')) and pd.notna(s.get('net_assets')):
            s['total_liabilities'] = s['total_assets'] - s['net_assets']
        
        return s
    
    def _estimate_missing_values(self, data: pd.Series, company_name: str) -> pd.Series:
        """業界平均を使用して欠損値を推定"""
        s = data.copy()
        
        # 業界判定
        industry = self._determine_industry(company_name)
        params = self.INDUSTRY_PARAMS[industry]
        
        # 研究開発費の推定
        if pd.isna(s.get('r_and_d_expenses')) and pd.notna(s.get('net_sales')) and s['net_sales'] > 0:
            s['r_and_d_expenses'] = int(s['net_sales'] * params['rd_ratio'])
        
        # 総資産の推定
        if pd.isna(s.get('total_assets')) and pd.notna(s.get('net_sales')) and s['net_sales'] > 0:
            s['total_assets'] = int(s['net_sales'] / params['asset_turnover'])
        
        # 負債の推定
        if pd.isna(s.get('total_liabilities')) and pd.notna(s.get('total_assets')) and s['total_assets'] > 0:
            s['total_liabilities'] = int(s['total_assets'] * params['liability_ratio'])
        
        # 営業CFの推定
        if pd.isna(s.get('operating_cash_flow')) and pd.notna(s.get('net_income')):
            s['operating_cash_flow'] = int(s['net_income'] * 1.1)
        
        return s
    
    def _determine_industry(self, company_name: str) -> str:
        """企業名から業界を推定"""
        service_keywords = ['サービス', 'コンサル', 'システム', 'ソフト', '銀行', '証券', '保険']
        return 'service' if any(kw in company_name for kw in service_keywords) else 'manufacturing'
    
    def _apply_defaults(self, data: pd.Series) -> pd.Series:
        """デフォルト値を適用"""
        s = data.copy()
        for field, default in self.DEFAULT_VALUES.items():
            if pd.isna(s.get(field)):
                s[field] = default
        return s
    
    def _validate_data(self, data: pd.Series, doc_id: str) -> Dict:
        """データ品質を検証"""
        result = {
            'is_valid': True,
            'confidence_score': 100,
            'issues': []
        }
        
        # 必須フィールドチェック
        for field in self.CRITICAL_FIELDS:
            if pd.isna(data.get(field)) or data.get(field, 0) <= 0:
                result['is_valid'] = False
                result['issues'].append(f"Critical field '{field}' is invalid")
                result['confidence_score'] -= 50
        
        # 会計恒等式チェック
        if all(pd.notna(data.get(f)) for f in ['total_assets', 'net_assets', 'total_liabilities']):
            balance_diff = abs(data['total_assets'] - (data['net_assets'] + data['total_liabilities']))
            tolerance = max(data['total_assets'] * 0.01, 1000000)
            if balance_diff > tolerance:
                result['confidence_score'] -= 20
        
        # NULL値カウント
        null_count = sum(1 for field in self.DEFAULT_VALUES if pd.isna(data.get(field)))
        result['confidence_score'] -= (null_count * 5)
        
        result['confidence_score'] = max(0, min(100, result['confidence_score']))
        return result
    
    def _prepare_db_data(self, data: pd.Series, confidence_score: int) -> Dict:
        """データベース保存用にデータを準備"""
        db_data = {}
        for field in self.DEFAULT_VALUES:
            value = data.get(field, self.DEFAULT_VALUES[field])
            db_data[field] = int(value) if pd.notna(value) else self.DEFAULT_VALUES[field]
        
        db_data['confidence_score'] = confidence_score
        return db_data