# db_setup.py

import psycopg2
import logging
from typing import List, Optional
from config import DB_CONFIG

logger = logging.getLogger(__name__)


class DatabaseSetup:
    """データベースのセットアップを管理するクラス"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
    
    def create_tables(self) -> None:
        """必要なテーブルを作成"""
        commands = self._get_table_commands()
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    for command in commands:
                        cur.execute(command)
                conn.commit()
            logger.info("テーブルの準備が完了しました。")
        except psycopg2.DatabaseError as e:
            logger.error(f"テーブル作成エラー: {e}")
            raise
    
    def _get_table_commands(self) -> List[str]:
        """テーブル作成SQLを返す"""
        return [
            # edinet_documents テーブル
            """
            CREATE TABLE IF NOT EXISTS edinet_documents (
                doc_id VARCHAR(10) PRIMARY KEY,
                edinet_code VARCHAR(10),
                filer_name VARCHAR(255),
                doc_type_code VARCHAR(10),
                period_end DATE,
                submit_datetime TIMESTAMP,
                is_xbrl_parsed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """CREATE INDEX IF NOT EXISTS idx_edinet_code ON edinet_documents (edinet_code)""",
            """CREATE INDEX IF NOT EXISTS idx_submit_datetime ON edinet_documents (submit_datetime)""",
            """CREATE INDEX IF NOT EXISTS idx_is_xbrl_parsed ON edinet_documents (is_xbrl_parsed)""",
            
            # financial_data テーブル
            """
            CREATE TABLE IF NOT EXISTS financial_data (
                document_id VARCHAR(10) PRIMARY KEY REFERENCES edinet_documents(doc_id) ON DELETE CASCADE,
                edinet_code VARCHAR(10),
                filer_name VARCHAR(255),
                fiscal_year INTEGER,
                net_assets BIGINT,
                total_assets BIGINT,
                net_sales BIGINT,
                operating_income BIGINT,
                ordinary_income BIGINT,
                net_income BIGINT,
                operating_cash_flow BIGINT,
                r_and_d_expenses BIGINT,
                number_of_employees INTEGER
            )
            """,
            """CREATE INDEX IF NOT EXISTS idx_financial_data_edinet_code ON financial_data (edinet_code)""",
            """CREATE INDEX IF NOT EXISTS idx_financial_data_fiscal_year ON financial_data (fiscal_year)""",
            
            # financial_data_validated テーブル（旧enhanced）
            """
            CREATE TABLE IF NOT EXISTS financial_data_validated (
                document_id VARCHAR(10) PRIMARY KEY REFERENCES edinet_documents(doc_id) ON DELETE CASCADE,
                edinet_code VARCHAR(10) NOT NULL,
                filer_name VARCHAR(255) NOT NULL,
                fiscal_year INTEGER,
                net_assets BIGINT NOT NULL DEFAULT 0,
                total_assets BIGINT NOT NULL DEFAULT 0,
                net_sales BIGINT NOT NULL DEFAULT 0,
                operating_income BIGINT NOT NULL DEFAULT 0,
                ordinary_income BIGINT NOT NULL DEFAULT 0,
                net_income BIGINT NOT NULL DEFAULT 0,
                total_liabilities BIGINT NOT NULL DEFAULT 0,
                operating_cash_flow BIGINT NOT NULL DEFAULT 0,
                r_and_d_expenses BIGINT NOT NULL DEFAULT 0,
                number_of_employees INTEGER NOT NULL DEFAULT 1,
                confidence_score INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                CONSTRAINT chk_total_assets_positive CHECK (total_assets > 0),
                CONSTRAINT chk_net_sales_non_negative CHECK (net_sales >= 0),
                CONSTRAINT chk_employees_positive CHECK (number_of_employees > 0),
                CONSTRAINT chk_confidence_score_range CHECK (confidence_score BETWEEN 0 AND 100)
            )
            """,
            """CREATE INDEX IF NOT EXISTS idx_financial_validated_edinet_code ON financial_data_validated (edinet_code)""",
            """CREATE INDEX IF NOT EXISTS idx_financial_validated_fiscal_year ON financial_data_validated (fiscal_year)""",
            """CREATE INDEX IF NOT EXISTS idx_financial_validated_confidence ON financial_data_validated (confidence_score)"""
        ]


def create_tables():
    """後方互換性のための関数"""
    setup = DatabaseSetup(DB_CONFIG)
    setup.create_tables()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    create_tables()