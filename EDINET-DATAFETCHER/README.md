# EDINET データパイプライン

EDINETから上場企業の財務データを取得・解析するPythonアプリケーションです。

## 必要な環境

- Python 3.8+
- PostgreSQL
- EDINET API Key

## セットアップ

### 1. 依存関係のインストール
```bash
pip install pandas psycopg2-binary beautifulsoup4 lxml requests
```

### 2. 設定ファイルの作成
`config.py`を作成：
```python
# EDINET APIキー
EDINET_API_KEY = "YOUR_API_KEY_HERE"

# データベース設定
DB_CONFIG = {
    "host": "localhost",
    "database": "edinet_db",
    "user": "your_user",
    "password": "your_password",
    "port": 5432
}
```

### 3. 対象企業リストの準備
`out_dic.csv`に証券コードとEDINETコードを記載：
```csv
証券コード,EDINETコード
7203,E02144
6758,E01706
```

## 使い方

### 方法1: 対話型CLI（推奨）
```bash
python cli.py
```

メニューから選択：
1. **データベースのセットアップ** - 初回のみ実行
2. **書類メタデータの取得** - 企業の提出書類一覧を取得
3. **XBRLデータの解析** - 財務データを抽出・保存
4. **保存済みデータの表示** - 財務指標を表示
5. **データ品質統計の確認** - 収集状況を確認
　解析に関しては書類の件数に「-1」を指定するとすべての未解析書類を処理してくれます
### 方法2: コマンドライン
```bash
# データベース初期化
python db_setup.py

# 過去7日分の書類を取得
python pipeline.py fetch --days 7

# 50件の書類を解析（データ検証あり）
python pipeline.py parse --limit 50

# 全件を解析（データ検証なし・高速）
python pipeline.py parse --limit 0 --no-validation
```

## 典型的な使用フロー

### 初回セットアップ
```bash
python cli.py
# → 1 (データベースセットアップ)＊＊実施済み＊＊　間違えて実行しても問題ありません。
# → 2 (書類取得、日数: 30)
# → 3 (解析、件数: 100)
```

### 日次更新
```bash
python pipeline.py fetch --days 1
python pipeline.py parse --limit 20
```

### データ確認
```bash
python cli.py
# → 4 (データ表示)
# → 7203 (トヨタの証券コード入力（out_dic.csvにトヨタは含まれていない;;）)
```

## ファイル構成

```
├── config.py              # 設定ファイル
├── out_dic.csv           # 対象企業リスト
├── cli.py                # 対話型インターフェース
├── pipeline.py           # データ取得・解析
├── db_setup.py          # DB初期化
├── xbrl_parser.py       # XBRL解析エンジン
└── test_output.py       # データ表示機能
```

## データベース構成

- **edinet_documents** - 書類メタデータ
- **financial_data** - 財務データ（生データ）
- **financial_data_validated** - 検証済みデータ（推定値含む）

## トラブルシューティング

### よくあるエラー

1. **`psycopg2.OperationalError`**
   - PostgreSQLが起動していることを確認
   - config.pyの接続情報を確認

2. **`FileNotFoundError: out_dic.csv`**
   - out_dic.csvが同じディレクトリにあることを確認

3. **データが取得できない**
   - EDINET APIキーが正しいことを確認
   - 対象企業が最近書類を提出しているか確認

### ログの確認
```bash
# 詳細ログを出力
python pipeline.py fetch --days 1 2>&1 | tee fetch.log
```

## 注意事項

- EDINET APIは1秒に1リクエストの制限があります
- 大量のデータ取得には時間がかかります（100件で約10分）
- データ検証モードでは信頼度の低いデータは除外されます