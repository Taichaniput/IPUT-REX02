# cli.py

import os
import sys
import logging
from typing import Optional, Dict
import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor

from config import DB_CONFIG, EDINET_API_KEY
from db_setup import DatabaseSetup
from pipeline import DocumentProcessor
from test_output import get_financial_data, calculate_indicators, display_data

logger = logging.getLogger(__name__)


class EdinetCLI:
    """EDINETデータパイプラインのCLIインターフェース"""
    
    def __init__(self):
        self.db_config = DB_CONFIG
        self.ticker_map = self._load_ticker_map()
        self.processor = DocumentProcessor(DB_CONFIG, EDINET_API_KEY)
        self.db_setup = DatabaseSetup(DB_CONFIG)
        
    def run(self):
        """メインループ"""
        while True:
            choice = self._show_menu()
            
            if choice == '1':
                self._setup_database()
            elif choice == '2':
                self._fetch_documents()
            elif choice == '3':
                self._parse_documents()
            elif choice == '4':
                self._display_financial_data()
            elif choice == '5':
                self._show_data_statistics()
            elif choice == '6':
                print("プログラムを終了します。")
                break
            else:
                print("無効な選択です。1から6の番号を入力してください。")
    
    def _show_menu(self) -> str:
        """メニューを表示"""
        print("\n" + "="*60)
        print(" EDINET データパイプライン")
        print("="*60)
        print("[1] データベースのセットアップ")
        print("[2] 書類メタデータの取得")
        print("[3] XBRLデータの解析")
        print("[4] 保存済みデータの表示")
        print("[5] データ品質統計の確認")
        print("[6] 終了")
        print("-"*60)
        return input("実行したい操作の番号を入力してください: ")
    
    def _setup_database(self):
        """データベースをセットアップ"""
        confirm = input("データベースにテーブルを作成します。よろしいですか？ (y/n): ")
        if confirm.lower() == 'y':
            try:
                self.db_setup.create_tables()
                print("✅ データベースのセットアップが完了しました。")
            except Exception as e:
                logger.error(f"セットアップエラー: {e}")
        else:
            print("キャンセルされました。")
    
    def _fetch_documents(self):
        """書類メタデータを取得"""
        days = self._get_integer_input("何日前までの書類を取得しますか？", default=1)
        try:
            self.processor.fetch_documents(days)
        except Exception as e:
            logger.error(f"取得エラー: {e}")
    
    def _parse_documents(self):
        """XBRLを解析"""
        limit = self._get_integer_input("一度に何件の書類を解析しますか？", default=10)
        validation = self._get_yes_no_input("データ検証を行いますか？", default=True)
        
        try:
            self.processor.parse_documents(limit, validation)
        except Exception as e:
            logger.error(f"解析エラー: {e}")
    
    def _display_financial_data(self):
        """財務データを表示"""
        identifier = input("証券コードまたはEDINETコードを入力してください: ")
        edinet_code = self._resolve_edinet_code(identifier)
        
        if not edinet_code:
            print(f"'{identifier}' に対応する企業が見つかりませんでした。")
            return
        
        years = self._get_integer_input("何年分のデータを表示しますか？", default=5)
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                financial_df = get_financial_data(conn, edinet_code, limit=years + 1)
                if financial_df.empty:
                    print("データが見つかりませんでした。")
                    return
                
                analyzed_df = calculate_indicators(financial_df)
                display_data(analyzed_df, years)
        except Exception as e:
            logger.error(f"表示エラー: {e}")
    
    def _show_data_statistics(self):
        """データ品質統計を表示"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                self._display_table_stats(conn, 'financial_data', '標準')
                self._display_table_stats(conn, 'financial_data_validated', '検証済み')
        except Exception as e:
            logger.error(f"統計取得エラー: {e}")
    
    def _display_table_stats(self, conn, table_name: str, label: str):
        """テーブルの統計情報を表示"""
        with conn.cursor() as cur:
            # テーブルの存在確認
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            if not cur.fetchone()[0]:
                return
            
            # 基本統計
            cur.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(net_assets) as net_assets_count,
                    COUNT(total_assets) as total_assets_count,
                    COUNT(net_sales) as net_sales_count,
                    COUNT(r_and_d_expenses) as rd_count
                FROM {table_name}
            """)
            
            stats = cur.fetchone()
            if stats and stats[0] > 0:
                print(f"\n📊 {label}データ統計 ({table_name})")
                print(f"   総レコード数: {stats[0]:,}")
                print(f"   純資産: {stats[1]:,} ({stats[1]/stats[0]*100:.1f}%)")
                print(f"   総資産: {stats[2]:,} ({stats[2]/stats[0]*100:.1f}%)")
                print(f"   売上高: {stats[3]:,} ({stats[3]/stats[0]*100:.1f}%)")
                print(f"   研究開発費: {stats[4]:,} ({stats[4]/stats[0]*100:.1f}%)")
            
            # 検証済みテーブルの追加統計
            if table_name == 'financial_data_validated':
                cur.execute(f"""
                    SELECT 
                        AVG(confidence_score) as avg_score,
                        COUNT(CASE WHEN confidence_score >= 80 THEN 1 END) as high,
                        COUNT(CASE WHEN confidence_score < 50 THEN 1 END) as low
                    FROM {table_name}
                """)
                
                val_stats = cur.fetchone()
                if val_stats:
                    print(f"   平均信頼度: {val_stats[0]:.1f}/100")
                    print(f"   高品質 (80+): {val_stats[1]:,}")
                    print(f"   低品質 (<50): {val_stats[2]:,}")
    
    def _load_ticker_map(self) -> Optional[Dict[str, str]]:
        """証券コードマップを読み込み"""
        filepath = 'out_dic.csv'
        if not os.path.exists(filepath):
            logger.warning(f"'{filepath}' が見つかりません。")
            return None
        
        try:
            df = pd.read_csv(
                filepath,
                usecols=['証券コード', 'EDINETコード'],
                dtype={'証券コード': str, 'EDINETコード': str}
            )
            return dict(zip(df['証券コード'], df['EDINETコード']))
        except Exception as e:
            logger.error(f"証券コードマップ読み込みエラー: {e}")
            return None
    
    def _resolve_edinet_code(self, identifier: str) -> Optional[str]:
        """入力からEDINETコードを解決"""
        if not identifier:
            return None
        
        # EDINETコード形式の場合
        if identifier.upper().startswith('E'):
            return identifier.upper()
        
        # 証券コードの場合
        if self.ticker_map and identifier in self.ticker_map:
            edinet_code = self.ticker_map[identifier]
            logger.info(f"証券コード '{identifier}' → EDINETコード '{edinet_code}'")
            return edinet_code
        
        return None
    
    def _get_integer_input(self, prompt: str, default: int) -> int:
        """整数入力を取得"""
        value = input(f"{prompt} (デフォルト: {default}): ")
        try:
            return int(value) if value else default
        except ValueError:
            return default
    
    def _get_yes_no_input(self, prompt: str, default: bool) -> bool:
        """Yes/No入力を取得"""
        default_str = "y" if default else "n"
        value = input(f"{prompt} (y/n, デフォルト: {default_str}): ")
        if not value:
            return default
        return value.lower() == 'y'


def main():
    """エントリーポイント"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # CLI起動
    cli = EdinetCLI()
    
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\n操作が中断されました。")
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        raise


if __name__ == '__main__':
    main()