# pipeline.py

import requests
import pandas as pd
import zipfile
import io
import time
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import psycopg2
from psycopg2.extras import DictCursor

from config import EDINET_API_KEY, DB_CONFIG
from xbrl_parser import XbrlParser, DataProcessor

logger = logging.getLogger(__name__)


class EdinetClient:
    """EDINET APIクライアント"""
    
    API_BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"
    FORM_CODES = ["030000", "040000"]  # 有価証券報告書、四半期報告書
    DOC_TYPE_CODES = ["120", "130"]     # 通常、訂正
    
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.params = {"Subscription-Key": api_key}
    
    def fetch_document_list(self, date: str) -> List[Dict]:
        """指定日の書類一覧を取得"""
        try:
            response = self.session.get(
                f"{self.API_BASE_URL}/documents.json",
                params={"date": date, "type": "2"}
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.RequestException as e:
            logger.error(f"書類一覧取得エラー: {e}")
            return []
    
    def fetch_company_info(self, edinet_code: str) -> Optional[Dict]:
        """EDINETコードから企業情報を取得"""
        try:
            response = self.session.get(
                f"{self.API_BASE_URL}/metadata.json",
                params={"edinetCode": edinet_code}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"企業情報取得エラー [{edinet_code}]: {e}")
            return None
    
    def download_xbrl(self, doc_id: str) -> Optional[bytes]:
        """XBRLファイルをダウンロード"""
        try:
            response = self.session.get(
                f"{self.API_BASE_URL}/documents/{doc_id}",
                params={"type": 1}
            )
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                xbrl_path = next(
                    (f for f in z.namelist() if "PublicDoc" in f and f.endswith(".xbrl")),
                    None
                )
                if xbrl_path:
                    return z.read(xbrl_path)
                    
        except (requests.RequestException, zipfile.BadZipFile) as e:
            logger.error(f"XBRLダウンロードエラー [{doc_id}]: {e}")
        
        return None


class DocumentProcessor:
    """書類の取得と解析を管理"""
    
    # IT企業識別用キーワード
    IT_KEYWORDS = {
        'core_it': [
            'システム', 'ソフトウェア', 'ソフト', 'テクノロジー', 'テック',
            'IT', 'アイティー', 'コンピューター', 'コンピュータ', 'デジタル',
            'インフォメーション', 'データ', 'ネットワーク', 'クラウド',
            'AI', 'IoT', 'DX', 'セキュリティ', 'プログラミング'
        ],
        'industry_specific': [
            'SaaS', 'SIer', 'フィンテック', 'EdTech', 'マーケティングテック',
            'CRM', 'ERP', 'BI', 'RPA', 'API', 'SDK', 'プラットフォーム',
            'アプリケーション', 'ウェブ', 'モバイル', 'ゲーム', 'エンターテイメント'
        ],
        'business_model': [
            'サービス', 'ソリューション', 'コンサルティング', 'インテグレーション',
            'クリエイティブ', 'メディア', 'コンテンツ', 'EC', 'イーコマース'
        ]
    }
    
    # IT企業を強く示唆する会社名パターン
    IT_COMPANY_PATTERNS = [
        r'.*システム.*', r'.*ソフト.*', r'.*テクノロジー.*', r'.*テック.*',
        r'.*IT.*', r'.*アイティー.*', r'.*コンピューター.*', r'.*コンピュータ.*',
        r'.*デジタル.*', r'.*ネットワーク.*', r'.*クラウド.*', r'.*データ.*',
        r'.*セキュリティ.*', r'.*ゲーム.*', r'.*エンターテイメント.*',
        r'.*プラットフォーム.*', r'.*アプリケーション.*', r'.*ウェブ.*',
        r'.*モバイル.*', r'.*メディア.*', r'.*コンテンツ.*'
    ]
    
    def __init__(self, db_config: dict, api_key: str):
        self.db_config = db_config
        self.client = EdinetClient(api_key)
        self.data_processor = DataProcessor()
    
    def fetch_documents(self, days: int = 1) -> None:
        """指定日数分の書類メタデータを取得してDBに保存"""
        target_codes = self._load_target_companies()
        if not target_codes:
            return
        
        logger.info(f"過去{days}日分の書類を検索します...")
        
        with psycopg2.connect(**self.db_config) as conn:
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                self._process_date(conn, date, target_codes)
                time.sleep(0.1)
    
    def parse_documents(self, limit: int = 10, use_validation: bool = True) -> None:
        """未解析の書類を解析してDBに保存"""
        with psycopg2.connect(**self.db_config) as conn:
            documents = self._get_unparsed_documents(conn, limit)
            
            if not documents:
                logger.info("解析対象の書類はありません。")
                return
            
            logger.info(f"{len(documents)}件の書類を解析します...")
            
            for doc in documents:
                self._process_document(conn, doc, use_validation)
                time.sleep(0.1)
    
    def _is_it_company(self, company_name: str) -> bool:
        """会社名からIT企業かどうかを判定"""
        if not company_name:
            return False
        
        # パターンマッチング
        for pattern in self.IT_COMPANY_PATTERNS:
            if re.search(pattern, company_name, re.IGNORECASE):
                return True
        
        # キーワードマッチング (重み付き)
        score = 0
        for category, keywords in self.IT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in company_name:
                    if category == 'core_it':
                        score += 3
                    elif category == 'industry_specific':
                        score += 2
                    else:
                        score += 1
        
        return score >= 3
    
    def _load_target_companies(self) -> Optional[set]:
        """EDINETの全企業からIT企業を動的に抽出"""
        logger.info("out_dic.csvを使用せず、EDINETの全企業からIT企業を動的に抽出します。")
        logger.info("以下の条件でIT企業を判定します:")
        logger.info("  - 会社名に含まれるIT関連キーワード")
        logger.info("  - 業種コード（3050: 情報・通信業等）")
        logger.info("  - 事業内容に含まれるIT関連キーワード")
        
        # 空のセットを返して、全てを動的判定で処理
        return set()
    
    def _process_date(self, conn, date: str, target_codes: set) -> None:
        """指定日の書類を処理"""
        logger.info(f"  {date} を処理中...")
        
        documents = self.client.fetch_document_list(date)
        count = 0
        
        with conn.cursor() as cur:
            for doc in documents:
                if self._is_target_document(doc, target_codes):
                    self._save_document_metadata(cur, doc)
                    count += 1
            
            if count > 0:
                conn.commit()
                logger.info(f"    → {count} 件の書類を保存しました。")
    
    def _is_target_document(self, doc: Dict, target_codes: set) -> bool:
        """対象書類かどうかを判定（IT企業の動的判定のみ）"""
        # 基本的な書類タイプチェック
        if not (doc.get("formCode") in self.client.FORM_CODES and 
                doc.get("docTypeCode") in self.client.DOC_TYPE_CODES):
            return False
        
        # EDINETの業種分類とキーワードを使用したIT企業判定のみ
        return self._is_it_company_by_edinet_data(doc)
    
    def _is_it_company_by_edinet_data(self, doc: Dict) -> bool:
        """EDINET データを使用してIT企業かどうかを判定"""
        # 会社名による判定
        company_name = doc.get("filerName", "")
        if self._is_it_company(company_name):
            return True
        
        # 業種コードによる判定（情報・通信業関連）
        industry_code = doc.get("industryCode", "")
        it_industry_codes = [
            "3050",  # 情報・通信業
            "3900",  # サービス業（IT関連）
            "3800",  # 精密機器（IT関連）
            "3700",  # 電気機器（IT関連）
        ]
        
        if industry_code in it_industry_codes:
            return True
        
        # 事業内容による判定
        business_summary = doc.get("businessSummary", "")
        if business_summary and self._contains_it_business_keywords(business_summary):
            return True
        
        return False
    
    def _contains_it_business_keywords(self, business_summary: str) -> bool:
        """事業内容からIT関連キーワードを検出"""
        if not business_summary:
            return False
        
        it_business_keywords = [
            "ソフトウェア開発", "システム開発", "IT", "情報システム", "データ処理",
            "インターネット", "ウェブ", "アプリケーション", "プラットフォーム",
            "デジタル", "AI", "人工知能", "機械学習", "クラウド", "IoT",
            "セキュリティ", "ネットワーク", "データベース", "EC", "電子商取引",
            "フィンテック", "テクノロジー", "プログラミング", "アルゴリズム"
        ]
        
        score = 0
        for keyword in it_business_keywords:
            if keyword in business_summary:
                score += 1
                if score >= 2:  # 2つ以上のキーワードがあればIT企業とみなす
                    return True
        
        return False
    
    def _save_document_metadata(self, cursor, doc: Dict) -> None:
        """書類メタデータをDBに保存"""
        sql = """
            INSERT INTO edinet_documents (
                doc_id, edinet_code, filer_name, doc_type_code, 
                period_end, submit_datetime, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (doc_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
        """
        cursor.execute(sql, (
            doc['docID'], doc.get('edinetCode'), doc.get('filerName'),
            doc.get('docTypeCode'), doc.get('periodEnd'), doc.get('submitDateTime')
        ))
    
    def _get_unparsed_documents(self, conn, limit: int) -> List[Dict]:
        """未解析の書類を取得"""
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = "SELECT * FROM edinet_documents WHERE is_xbrl_parsed = FALSE"
            if limit > 0:
                query += f" LIMIT {limit}"
            cur.execute(query)
            return cur.fetchall()
    
    def _process_document(self, conn, doc: Dict, use_validation: bool) -> None:
        """個別の書類を処理"""
        logger.info(f"  解析中: {doc['filer_name']} ({doc['doc_id']})")
        
        try:
            # XBRLダウンロード
            xbrl_content = self.client.download_xbrl(doc['doc_id'])
            if not xbrl_content:
                self._mark_as_parsed(conn, doc['doc_id'])
                return
            
            # 解析
            parser = XbrlParser(xbrl_content, doc['doc_id'])
            raw_data = parser.parse()
            
            # データ処理
            if use_validation:
                processed_data = self.data_processor.process(
                    raw_data, doc['doc_id'], doc['filer_name']
                )
                if not processed_data:
                    self._mark_as_parsed(conn, doc['doc_id'])
                    return
            else:
                processed_data = self._simple_process(raw_data)
            
            # DB保存
            self._save_financial_data(conn, doc, processed_data, use_validation)
            self._mark_as_parsed(conn, doc['doc_id'])
            
            logger.info("    → 解析完了")
            
        except Exception as e:
            logger.error(f"    → エラー: {e}")
            conn.rollback()
    
    def _simple_process(self, data: pd.Series) -> Dict:
        """単純なデータ処理（検証なし）"""
        result = {}
        for field in DataProcessor.DEFAULT_VALUES:
            value = data.get(field)
            result[field] = int(value) if pd.notna(value) else None
        return result
    
    def _save_financial_data(self, conn, doc: Dict, data: Dict, use_validation: bool) -> None:
        """財務データをDBに保存"""
        fiscal_year = doc['period_end'].year if doc['period_end'] else None
        
        with conn.cursor() as cur:
            # 標準テーブルへの保存
            self._save_to_standard_table(cur, doc, data, fiscal_year)
            
            # 検証済みテーブルへの保存
            if use_validation and 'confidence_score' in data:
                self._save_to_validated_table(cur, doc, data, fiscal_year)
            
            conn.commit()
    
    def _save_to_standard_table(self, cursor, doc: Dict, data: Dict, fiscal_year: int) -> None:
        """標準テーブルに保存"""
        sql = """
            INSERT INTO financial_data (
                document_id, edinet_code, filer_name, fiscal_year,
                net_assets, total_assets, net_sales, operating_income,
                ordinary_income, net_income, operating_cash_flow,
                r_and_d_expenses, number_of_employees
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO NOTHING
        """
        
        values = (
            doc['doc_id'], doc['edinet_code'], doc['filer_name'], fiscal_year,
            data.get('net_assets'), data.get('total_assets'), data.get('net_sales'),
            data.get('operating_income'), data.get('ordinary_income'), data.get('net_income'),
            data.get('operating_cash_flow'), data.get('r_and_d_expenses'),
            data.get('number_of_employees')
        )
        
        cursor.execute(sql, values)
    
    def _save_to_validated_table(self, cursor, doc: Dict, data: Dict, fiscal_year: int) -> None:
        """検証済みテーブルに保存"""
        sql = """
            INSERT INTO financial_data_validated (
                document_id, edinet_code, filer_name, fiscal_year,
                net_assets, total_assets, net_sales, operating_income,
                ordinary_income, net_income, total_liabilities,
                operating_cash_flow, r_and_d_expenses, number_of_employees,
                confidence_score
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO NOTHING
        """
        
        values = (
            doc['doc_id'], doc['edinet_code'], doc['filer_name'], fiscal_year,
            data['net_assets'], data['total_assets'], data['net_sales'],
            data['operating_income'], data['ordinary_income'], data['net_income'],
            data.get('total_liabilities', 0), data['operating_cash_flow'],
            data['r_and_d_expenses'], data['number_of_employees'],
            data['confidence_score']
        )
        
        cursor.execute(sql, values)
    
    def _mark_as_parsed(self, conn, doc_id: str) -> None:
        """書類を解析済みとしてマーク"""
        with conn.cursor() as cur:
            sql = """
                UPDATE edinet_documents 
                SET is_xbrl_parsed = TRUE, updated_at = CURRENT_TIMESTAMP 
                WHERE doc_id = %s
            """
            cur.execute(sql, (doc_id,))
            conn.commit()


# メイン関数
def main():
    """コマンドライン実行用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EDINETデータパイプライン")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # fetch コマンド
    parser_fetch = subparsers.add_parser("fetch", help="書類メタデータを取得")
    parser_fetch.add_argument("--days", type=int, default=1, help="取得日数")
    
    # parse コマンド
    parser_parse = subparsers.add_parser("parse", help="XBRLを解析")
    parser_parse.add_argument("--limit", type=int, default=10, help="処理件数上限")
    parser_parse.add_argument("--no-validation", action="store_true", help="検証をスキップ")
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 処理実行
    processor = DocumentProcessor(DB_CONFIG, EDINET_API_KEY)
    
    try:
        if args.command == "fetch":
            processor.fetch_documents(args.days)
        elif args.command == "parse":
            processor.parse_documents(args.limit, not args.no_validation)
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        raise


if __name__ == "__main__":
    main()