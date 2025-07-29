# AI企業分析システム

IT業界志望の情報系学生を対象とした包括的企業分析プラットフォームです。財務データ、機械学習予測、AI分析を統合し、学生の就職活動における企業選択を客観的データで支援します。

## 📋 目次

- [システム概要](#システム概要)
- [主要機能](#主要機能)
- [技術スタック](#技術スタック)
- [データベースの構築](#データベースの構築)
- [セットアップ](#セットアップ)
- [使用方法](#使用方法)
- [API設定](#api設定)
- [プロジェクト構成](#プロジェクト構成)
- [機械学習手法](#機械学習手法)
- [開発・テスト](#開発テスト)

## 🎯 システム概要

### 目的
- 情報系学生の就職活動支援
- 客観的データに基づく企業選択の意思決定支援
- 機械学習による企業成長性の定量的評価

### 対象ユーザー
- IT業界志望の情報系学生
- 企業分析に興味のある学生・研究者

### データベース規模
- **企業数**: 593社
- **レコード数**: 4,115件
- **期間**: 2015-2025年（11年間）
- **完全な時系列データを持つ企業**: 116社（2015-2024年の10年間）

## 🚀 主要機能

### 1. 企業検索・一覧表示
- キーワードによる企業検索
- 企業一覧とページネーション
- 財務指標による絞り込み

### 2. 企業詳細分析
- **財務データ表示**: 売上高、営業利益、純利益、総資産、純資産、従業員数
- **財務指標計算**: ROE、ROA、自己資本比率
- **時系列グラフ**: Chart.jsによるインタラクティブなグラフ表示

### 3. 機械学習による予測分析
- **成長性予測**: ARIMAモデルによる3年間の財務予測
- **3シナリオ予測**: 楽観・現状・悲観の3つのシナリオ
- **不確実性の定量化**: 予測区間の表示

### 4. ポジショニング分析
- **クラスタリング**: HDBSCANによる企業分類
- **次元削減**: UMAPによる2次元可視化
- **競合他社の特定**: 類似企業の発見

### 5. AI生成型分析
- **Google Gemini統合**: 企業の詳細分析と解釈
- **Web情報統合**: Tavily APIによる最新企業情報の取得
- **学生向けカスタマイズ**: IT業界志望学生に特化した分析

### 6. ユーザー管理
- ユーザー登録・ログイン機能
- プロファイル管理
- 個別設定の保存

## 🛠 技術スタック

### バックエンド
- **Django 5.1.2**: Webフレームワーク
- **PostgreSQL**: データベース
- **Python 3.8+**: プログラミング言語

### 機械学習・データ分析
- **pandas 2.1.3**: データ分析・操作
- **numpy 1.26.3**: 数値計算
- **scikit-learn 1.4.2**: 機械学習ライブラリ
- **umap-learn 0.5.3**: UMAP次元削減
- **hdbscan 0.8.33**: 密度ベースクラスタリング
- **statsmodels 0.14.2**: ARIMAモデルなど統計モデリング

### 可視化
- **matplotlib 3.8.4**: グラフ描画
- **japanize-matplotlib 1.1.3**: 日本語フォント対応
- **Chart.js**: フロントエンドでのインタラクティブグラフ

### AI・API統合
- **google-generativeai 0.8.5**: Gemini API統合
- **tavily-python 0.7.8**: Web検索API統合

### フロントエンド
- **HTML/CSS/JavaScript**: 基本的なWeb技術
- **jQuery**: JavaScript ライブラリ
- **Chart.js**: グラフ描画ライブラリ

## 💾 データベースの構築

このシステムで使用する財務データは、付属の**EDINET-DATAFETCHER**を使用してEDINET API（金融庁の電子開示システム）から取得・構築します。

### EDINET-DATAFETCHERとは

EDINET-DATAFETCHERは、上場企業の有価証券報告書から財務データを自動取得・解析するPythonアプリケーションです。

#### 主な機能
- **EDINET API連携**: 金融庁のEDINET APIから書類メタデータを取得
- **XBRL解析**: 有価証券報告書のXBRLファイルから財務指標を抽出
- **データ検証**: 不整合データの検出・修正
- **PostgreSQL統合**: 解析結果をデータベースに格納

#### データ取得フロー
1. **書類メタデータ取得**: 指定期間の有価証券報告書一覧を取得
2. **XBRLダウンロード**: 個別の書類ファイルをダウンロード
3. **財務データ解析**: XBRL から売上高、利益、資産等を抽出
4. **データベース保存**: PostgreSQL に保存（生データ・検証済みデータ）

### EDINET-DATAFETCHERのセットアップ

#### 1. EDINET API キーの取得
1. [EDINET API](https://disclosure.edinet-fsa.go.jp/E01EW/BLMainController.jsp?TID=3)にアクセス
2. 利用者登録を行い、APIキーを取得

#### 2. 設定ファイルの作成
`EDINET-DATAFATCHER/config.py`を作成：
```python
# EDINET APIキー
EDINET_API_KEY = "YOUR_API_KEY_HERE"

# データベース設定
DB_CONFIG = {
    "host": "localhost",
    "database": "your_db_name",
    "user": "your_db_user", 
    "password": "your_db_password",
    "port": 5432
}
```

#### 3. 対象企業リストの準備
`EDINET-DATAFATCHER/out_dic.csv`に証券コードとEDINETコードを記載：
```csv
証券コード,EDINETコード
7203,E02144
6758,E01706
9984,E04588
```

#### 4. データベース構築の実行

**対話型CLI（推奨）:**
```bash
cd EDINET-DATAFATCHER
python cli.py
```

メニューから順次実行：
1. **データベースのセットアップ** - テーブル作成
2. **書類メタデータの取得** - 企業の提出書類一覧を取得（例：過去30日分）
3. **XBRLデータの解析** - 財務データを抽出・保存（例：100件）
4. **データ品質統計の確認** - 収集状況を確認

**コマンドライン:**
```bash
cd EDINET-DATAFATCHER

# データベース初期化
python db_setup.py

# 過去30日分の書類を取得
python pipeline.py fetch --days 30

# 100件の書類を解析
python pipeline.py parse --limit 100

# データ確認
python test_output.py
```

#### 5. データベース構成

構築されるテーブル：
- **edinet_documents**: 書類メタデータ（提出日、企業名、書類種別等）
- **financial_data**: 財務データ生データ（売上高、利益、資産等）
- **financial_data_validated**: 検証済み財務データ（推定値・補完値含む）

### データ更新の運用

#### 日次更新（推奨）
```bash
cd EDINET-DATAFATCHER
python pipeline.py fetch --days 1    # 前日分の書類取得
python pipeline.py parse --limit 20  # 新規書類の解析
```

#### 注意事項
- EDINET APIは1秒に1リクエストの制限があります
- 大量データ取得には時間がかかります（100件で約10分）
- データ検証モードでは信頼度の低いデータは除外されます

## 🔧 セットアップ

### 1. 前提条件
- Python 3.8以上
- PostgreSQL
- Git

### 2. リポジトリのクローン
```bash
git clone <repository-url>
cd ai_agent
```

### 3. 仮想環境の作成・有効化
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 5. 環境変数の設定
`.env`ファイルを作成し、以下の環境変数を設定：

```env
# Django設定
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=True

# データベース設定
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432

# AI API設定
GEMINI_API_KEY=your-gemini-api-key
TAVILY_API_KEY=your-tavily-api-key

# AI分析デバッグモード
AI_DEBUG_MODE=True
```

### 6. データベースのセットアップ
```bash
python manage.py migrate
python manage.py createsuperuser  # 管理者ユーザー作成
```

### 7. 開発サーバーの起動
```bash
python manage.py runserver
```

アプリケーションは `http://localhost:8000` でアクセス可能です。

## 📖 使用方法

### 基本的な使用フロー

1. **ユーザー登録・ログイン**
   - 新規ユーザー登録または既存アカウントでログイン

2. **企業検索**
   - ホーム画面でキーワード検索
   - 企業名またはEDINETコードで検索可能

3. **企業詳細分析**
   - 企業を選択して詳細ページにアクセス
   - 財務データ、予測分析、ポジショニング分析を確認

4. **AI分析の活用**
   - 自動生成されるAI分析レポートを参考に企業を評価
   - 競合他社との比較分析

## 🔑 API設定

### Google Gemini API
1. [Google AI Studio](https://aistudio.google.com/)でAPIキーを取得
2. `.env`ファイルに`GEMINI_API_KEY`を設定

### Tavily Search API
1. [Tavily](https://tavily.com/)でアカウント作成
2. APIキーを取得し、`.env`ファイルに`TAVILY_API_KEY`を設定

## 📁 プロジェクト構成

```
ai_agent/
├── config/                     # Django設定
│   ├── settings.py             # メイン設定ファイル
│   ├── urls.py                 # ルートURLconf
│   ├── wsgi.py                 # WSGI設定
│   └── asgi.py                 # ASGI設定
├── core/                       # メインアプリケーション
│   ├── src/                    # 機械学習・分析モジュール
│   │   ├── ai_analysis.py      # AI分析機能
│   │   ├── ml_analytics.py     # 機械学習・予測分析
│   │   └── financial_utils.py  # 財務指標計算
│   ├── models.py               # データモデル定義
│   ├── views.py                # ビュー関数
│   ├── urls.py                 # URLルーティング
│   ├── forms.py                # フォーム定義
│   ├── admin.py                # 管理画面設定
│   ├── templates/              # HTMLテンプレート
│   │   ├── financial/          # 金融分析テンプレート
│   │   └── registration/       # ユーザー認証テンプレート
│   ├── static/                 # 静的ファイル
│   │   └── core/
│   │       ├── css/            # スタイルシート
│   │       └── js/             # JavaScript
│   ├── migrations/             # データベースマイグレーション
│   └── templatetags/           # カスタムテンプレートタグ
├── EDINET-DATAFATCHER/         # 財務データ取得システム
│   ├── cli.py                  # 対話型CLI
│   ├── pipeline.py             # データ取得・解析パイプライン
│   ├── db_setup.py             # データベース初期化
│   ├── xbrl_parser.py          # XBRL解析エンジン
│   ├── config.py               # EDINET API設定
│   ├── out_dic.csv             # 対象企業リスト
│   └── README.md               # EDINET-DATAFETCHER詳細説明
├── arima_cache/                # ARIMA予測結果キャッシュ
├── clustering_cache/           # クラスタリング結果キャッシュ
├── tests/                      # テストファイル
│   └── ml_evaluation/          # 機械学習評価テスト
├── requirements.txt            # Python依存関係
├── manage.py                   # Django管理コマンド
└── README.md                   # このファイル
```

## 🤖 機械学習手法

### 1. 成長性予測（ARIMAモデル）
- **手法**: ARIMA時系列分析
- **予測期間**: 3年間
- **シナリオ**: 楽観・現状・悲観の3パターン
- **評価指標**: MAE、RMSE、MAPE

### 2. ポジショニング分析
- **次元削減**: UMAP（n_components=2）
- **クラスタリング**: HDBSCAN（密度ベース）
- **特徴量**: 純資産、総資産、純利益、研究開発費、従業員数
- **前処理**: StandardScaler標準化

### 3. AI分析統合
- **LLM**: Google Gemini Pro
- **Web検索**: Tavily API
- **カスタマイズ**: IT業界志望学生向け分析

## 🧪 開発・テスト

### テストの実行
```bash
# Django単体テスト
python manage.py test

# 機械学習評価テスト
python -m pytest tests/ml_evaluation/

# 特定のテスト実行
python test_evaluation.py
```

### パフォーマンス評価
```bash
# 機械学習手法の性能評価
python test_performance_output.py
```

### デバッグモード
環境変数 `AI_DEBUG_MODE=True` を設定することで、AI分析の詳細ログが出力されます。

### キャッシュ管理
- ARIMAモデルの予測結果は`arima_cache/`にキャッシュされます
- クラスタリング結果は`clustering_cache/`にキャッシュされます
- キャッシュクリア: 各ディレクトリ内のファイルを削除

## 📊 主要データモデル

### FinancialData
- 企業の財務データを格納
- フィールド: 企業名、EDINETコード、売上高、利益、資産等

### FinancialDataValidated
- 検証済み財務データ
- 機械学習分析で使用

### UserProfile
- ユーザープロファイル情報
- キャリア志向、興味分野等

## 📄 ライセンス

このプロジェクトは教育目的で作成されています。

## 使用について

このシステムは情報系学生の学習・就職活動支援を目的として開発されています。実際の投資判断や企業評価に使用する際は、他の情報源と併せて総合的に判断してください。
