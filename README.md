# AI企業分析システム

情報系学生向けの就活支援Webアプリケーション

## 機能

- 企業財務データの表示・分析
- AI予測分析（機械学習モデル）
- クラスタリング分析（企業ポジショニング）
- **AI企業分析（Gemini + Tavily Web Search統合）**

## セットアップ

### 1. 環境変数設定

`.env`ファイルを作成し、以下のAPIキーを設定してください：

```bash
# .env ファイル
GOOGLE_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 2. APIキー取得方法

#### Google Gemini API
1. [Google AI Studio](https://makersuite.google.com/app/apikey)にアクセス
2. APIキーを生成
3. `.env`ファイルの`GOOGLE_API_KEY`に設定

#### Tavily API
1. [Tavily](https://tavily.com/)でアカウント作成
2. APIキーを取得
3. `.env`ファイルの`TAVILY_API_KEY`に設定

### 3. 仮想環境作成（推奨）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows
```

### 4. 依存関係インストール

```bash
pip install -r requirements.txt
```

### 5. データベース設定

```bash
python manage.py migrate
```

### 6. サーバー起動

```bash
python manage.py runserver
```

## 重要なセキュリティ情報

- `.env`ファイルは**絶対にgitにコミットしない**
- APIキーを直接コードに書かない
- 本番環境では環境変数で設定する

## 技術スタック

- **Backend**: Django, Python
- **AI/ML**: scikit-learn, Google Gemini API
- **Web Search**: Tavily API
- **Database**: PostgreSQL
- **Frontend**: HTML, CSS, JavaScript