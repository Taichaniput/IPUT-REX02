# financial/views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Max
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .models import FinancialData, FinancialDataValidated, UserProfile
from .forms import UserRegistrationForm, UserProfileForm
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')
import platform
import os
import unicodedata
import re
import hashlib
from datetime import datetime, timedelta
from django.core.cache import cache
from django.conf import settings
from matplotlib.font_manager import FontProperties

# Google Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# 外部データ取得用
import requests
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# 日本語フォント設定
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'MS Gothic'
    FONT_PATH = None
else:
    FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'ipaexg.ttf')
    plt.rcParams['font.family'] = 'IPAPGothic'
plt.rcParams['axes.unicode_minus'] = False

# Gemini API初期化
if GEMINI_AVAILABLE and settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)


def get_company_additional_info(company_name):
    """企業の追加情報を外部ソース（Tavily Web Search）から取得"""
    additional_info = {}

    if not TAVILY_AVAILABLE:
        additional_info['web_search_summary'] = "Tavilyライブラリがインストールされていません。"
        return additional_info

    if not settings.TAVILY_API_KEY:
        additional_info['web_search_summary'] = "Tavily APIキーが設定されていません。"
        return additional_info

    try:
        # Tavilyクライアントを初期化
        tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)
        
        # 検索クエリを工夫して、多角的な情報を要求
        search_query = f"{company_name}の事業内容、強みと弱み、市場でのポジショニング、最近の重要なニュースについて包括的に調査し、要約してください。"
        
        # TavilyでWeb検索を実行
        # search_depth="advanced"でより詳細な検索を行う
        response = tavily.search(
            query=search_query,
            search_depth="advanced",
            max_results=5  # 5つの検索結果を元に要約
        )
        
        # 結果を結合して一つのサマリーにする
        summary = "\n".join([f"- {obj['content']}" for obj in response['results']])
        additional_info['web_search_summary'] = summary if summary else "ウェブ検索で関連情報が見つかりませんでした。"

    except Exception as e:
        print(f"Tavily search error: {e}")
        additional_info['web_search_summary'] = "ウェブ検索中にエラーが発生しました。"
    
    return additional_info

def generate_ai_company_analysis(company_name, financial_data, prediction_results, edinet_code):
    """Gemini APIを使用して企業分析を生成"""
    
    if not GEMINI_AVAILABLE:
        return "Google Generative AIライブラリがインストールされていません。"
    
    if not settings.GEMINI_API_KEY:
        return "Google Gemini APIキーが設定されていません。環境変数GOOGLE_API_KEYを設定してください。"
    
    try:
        # キャッシュキーの生成
        cache_key = f"ai_analysis_{edinet_code}_{hash(str(prediction_results))}"
        cached_analysis = cache.get(cache_key)
        
        if cached_analysis:
            return cached_analysis
        
        # 企業の追加情報を取得
        additional_info = get_company_additional_info(company_name)
        
        # 財務データの整理
        latest_data = financial_data[0]['data'] if financial_data else None
        financial_summary = ""
        
        if latest_data:
            financial_summary = f"""
            売上高: {latest_data.net_sales // 100000000 if latest_data.net_sales else 0}億円
            営業利益: {latest_data.operating_income // 100000000 if latest_data.operating_income else 0}億円
            純利益: {latest_data.net_income // 100000000 if latest_data.net_income else 0}億円
            総資産: {latest_data.total_assets // 100000000 if latest_data.total_assets else 0}億円
            """
        
        # 予測結果の整理
        prediction_summary = ""
        if prediction_results:
            for metric, result in prediction_results.items():
                if 'ml_universal' in result.get('predictions', {}):
                    ml_pred = result['predictions']['ml_universal']
                    prediction_summary += f"""
                    {result['label']}のAI予測:
                    - 年平均成長率: {ml_pred['growth_rate']:.1f}%
                    - 予測信頼度: {ml_pred['confidence']}
                    """
        
        # プロンプトの作成
        prompt = f"""
        【企業名】{company_name}
        
        【Web検索による企業情報サマリー】
        {additional_info.get('web_search_summary', '情報なし')}
        
        【最新財務データ】{financial_summary}
        
        【AI予測分析】{prediction_summary}
        
        【分析要件】
        1. 企業の事業内容と強みを簡潔に説明（Web検索結果を最重視）
        2. 財務状況と成長性の客観的分析（財務データとAI予測を元に）
        3. 情報系学生にとっての魅力と入社後のキャリア展望
        4. 注意点やリスクがあれば客観的に言及
        
        【注意事項】
        - 提示された情報を基に、客観的で正確な分析をしてください。
        - 過度に楽観的な表現は避けてください。
        - 就活生が意思決定に役立つ具体的な情報を提供してください。
        - 全体で400-500文字程度でまとめてください。
        
        分析レポート：
        """
        
        # Gemini APIに送信（フリー制限内でより利用しやすいモデル）
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        analysis_result = response.text
        
        # キャッシュに保存（24時間）
        cache.set(cache_key, analysis_result, timeout=settings.AI_ANALYSIS_CACHE_TIMEOUT)
        
        return analysis_result
        
    except Exception as e:
        print(f"AI analysis generation error: {e}")
        return "AI分析の生成中にエラーが発生しました。しばらく後に再度お試しください。"


def home(request):
    """企業検索画面"""
    keyword = request.GET.get('keyword', '').strip()
    companies = []
    
    if keyword:
        companies = FinancialData.objects.filter(
            Q(edinet_code__icontains=keyword) |
            Q(filer_name__icontains=keyword)
        ).values(
            'edinet_code', 'filer_name'
        ).distinct().order_by('filer_name')[:50]
    
    return render(request, 'financial/home.html', {
        'companies': companies,
        'keyword': keyword,
    })


def company_detail(request, edinet_code):
    """企業の詳細画面（財務データ・予測・クラスタリング統合）"""
    
    # 基本的な財務データ取得
    financial_data = FinancialData.objects.filter(
        edinet_code=edinet_code
    ).select_related('document').order_by('-fiscal_year')
    
    if not financial_data.exists():
        return render(request, 'financial/company_detail.html', {
            'error': 'この企業のデータが見つかりません。',
            'edinet_code': edinet_code
        })
    
    company_name = financial_data.first().filer_name
    
    # 1. 財務指標を計算
    data_with_indicators = []
    for fd in financial_data:
        indicators = calculate_financial_indicators(fd)
        data_with_indicators.append({
            'data': fd,
            'indicators': indicators
        })
    
    # 2. 予測分析（複数モデル）- 直接読み込み
    prediction_results = {}
    if request.user.is_authenticated and len(financial_data) >= 3:
        try:
            prediction_results = perform_predictions(financial_data)
        except Exception as e:
            print(f"Prediction error: {e}")
    
    # 3. クラスタリング分析 - 直接読み込み
    cluster_info = None
    if request.user.is_authenticated:
        try:
            cluster_info = get_company_cluster_info(edinet_code)
        except Exception as e:
            print(f"Clustering error: {e}")
    
    # 4. AI企業分析（ログインユーザーの場合のみ）
    ai_analysis = None
    if request.user.is_authenticated and settings.AI_ANALYSIS_ENABLED:
        ai_analysis = generate_ai_company_analysis(
            company_name, 
            data_with_indicators, 
            prediction_results, 
            edinet_code
        )
    
    return render(request, 'financial/company_detail.html', {
        'company_name': company_name,
        'edinet_code': edinet_code,
        'financial_data': data_with_indicators,
        'prediction_results': prediction_results if request.user.is_authenticated else {},
        'cluster_info': cluster_info if request.user.is_authenticated else None,
        'ai_analysis': ai_analysis,
        'show_login_prompt': not request.user.is_authenticated,
    })


def perform_predictions(financial_data):
    """複数の予測モデルを実行"""
    results = {}
    
    # データ準備
    df = pd.DataFrame([{
        'fiscal_year': fd.fiscal_year,
        'net_sales': fd.net_sales,
        'operating_income': fd.operating_income,
        'net_income': fd.net_income,
        'total_assets': fd.total_assets
    } for fd in financial_data if fd.fiscal_year])
    
    df = df.sort_values('fiscal_year')
    
    # 各指標について予測
    metrics = ['net_sales', 'net_income']
    
    for metric in metrics:
        if metric not in df.columns or df[metric].isna().all():
            continue
            
        # 有効なデータのみ使用
        valid_data = df.dropna(subset=[metric])
        if len(valid_data) < 3:
            continue
        
        years = valid_data['fiscal_year'].values
        values = valid_data[metric].values / 100000000  # 億円単位
        
        # 予測実行
        predictions = predict_multiple_models(years, values)
        
        # グラフ生成
        chart = create_prediction_chart(
            years, values, predictions,
            f"{get_metric_label(metric)}の予測",
            get_metric_label(metric) + "（億円）"
        )
        
        results[metric] = {
            'predictions': predictions,
            'chart': chart,
            'label': get_metric_label(metric)
        }
    
    return results


def predict_multiple_models(years, values):
    """汎用モデルによる予測"""
    predictions = {}
    
    # 将来の年度
    last_year = years[-1]
    future_years = np.array([last_year + i for i in range(1, 4)])
    
    if len(values) >= 2:
        # 汎用モデルによる予測
        ml_predictions = predict_with_universal_model(years, values)
        
        if ml_predictions:
            predictions['ml_universal'] = {
                'name': '汎用MLモデル',
                'values': ml_predictions['predicted_values'],
                'years': future_years,
                'growth_rate': ml_predictions['avg_growth_rate'],
                'confidence': ml_predictions.get('confidence', 'N/A')
            }
        
        # 従来の成長率モデルも残す（比較用）
        legacy_prediction = predict_legacy_growth_model(years, values)
        if legacy_prediction:
            predictions['growth'] = legacy_prediction
    
    return predictions


def predict_with_universal_model(years, values):
    """全企業データを使った汎用モデルによる予測"""
    try:
        # 全企業のデータを取得してモデル学習
        training_data = prepare_universal_training_data()
        
        if len(training_data) < 10:  # 最低データ数チェック
            return None
        
        # 特徴量作成
        features = create_features_from_timeseries(years, values)
        if not features:
            return None
        
        # モデル学習（全企業データ）
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        X_train = np.array([row['features'] for row in training_data])
        y_train = np.array([row['target_growth'] for row in training_data])
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # ランダムフォレストで学習
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train)
        
        # 対象企業の予測
        company_features = np.array([features]).reshape(1, -1)
        company_features_scaled = scaler.transform(company_features)
        
        predicted_growth_rate = model.predict(company_features_scaled)[0]
        
        # 現実的な制約を適用
        predicted_growth_rate = apply_realistic_constraints(predicted_growth_rate, values)
        
        # 将来値計算
        growth_multiplier = 1 + predicted_growth_rate
        predicted_values = [values[-1] * (growth_multiplier ** i) for i in range(1, 4)]
        
        return {
            'predicted_values': predicted_values,
            'avg_growth_rate': predicted_growth_rate * 100,
            'confidence': f"{model.score(X_train_scaled, y_train):.2f}"
        }
        
    except Exception as e:
        print(f"Universal model prediction error: {e}")
        return None


def prepare_universal_training_data():
    """全企業データから学習用データセットを準備"""
    training_data = []
    
    try:
        # 全企業の財務データを取得
        all_companies = FinancialData.objects.values(
            'edinet_code'
        ).distinct()
        
        for company in all_companies:
            edinet_code = company['edinet_code']
            
            # 各企業の時系列データを取得
            company_data = FinancialData.objects.filter(
                edinet_code=edinet_code
            ).order_by('fiscal_year').values(
                'fiscal_year', 'net_sales', 'net_income', 'total_assets', 'net_assets'
            )
            
            company_list = list(company_data)
            if len(company_list) < 4:  # 最低4年分のデータが必要
                continue
            
            # 各年について翌年の成長率を目的変数とする
            for i in range(len(company_list) - 3):
                current_data = company_list[i:i+3]  # 3年分
                next_year_data = company_list[i+3]
                
                # 特徴量作成
                years = [row['fiscal_year'] for row in current_data]
                sales_values = [row['net_sales'] or 0 for row in current_data]
                
                if any(v <= 0 for v in sales_values):  # 無効なデータはスキップ
                    continue
                
                features = create_features_from_timeseries(years, sales_values)
                if not features:
                    continue
                
                # 目的変数（翌年の成長率）
                current_sales = sales_values[-1]
                next_sales = next_year_data['net_sales'] or 0
                
                if current_sales > 0 and next_sales > 0:
                    target_growth = (next_sales / current_sales) - 1
                    
                    # 異常値を除外（-50% ~ +200%の範囲）
                    if -0.5 <= target_growth <= 2.0:
                        training_data.append({
                            'features': features,
                            'target_growth': target_growth
                        })
        
    except Exception as e:
        print(f"Training data preparation error: {e}")
    
    return training_data


def create_features_from_timeseries(years, values):
    """時系列データから特徴量を作成"""
    if len(years) < 2 or len(values) < 2:
        return None
    
    try:
        features = []
        
        # 1. 現在の値（対数変換で正規化）
        current_value = np.log(max(values[-1], 1))
        features.append(current_value)
        
        # 2. 前年比成長率
        if len(values) >= 2 and values[-2] > 0:
            yoy_growth = (values[-1] / values[-2]) - 1
            features.append(yoy_growth)
        else:
            features.append(0)
        
        # 3. 3年平均成長率
        if len(values) >= 3:
            growth_rates = []
            for i in range(1, min(4, len(values))):
                if values[-i-1] > 0:
                    rate = (values[-1] / values[-i-1]) ** (1/i) - 1
                    growth_rates.append(rate)
            avg_growth = np.mean(growth_rates) if growth_rates else 0
            features.append(avg_growth)
        else:
            features.append(0)
        
        # 4. 変動性（標準偏差）
        if len(values) >= 3:
            volatility = np.std(values) / np.mean(values)
            features.append(volatility)
        else:
            features.append(0)
        
        # 5. トレンド（線形回帰の傾き）
        if len(years) >= 3:
            X = np.array(years).reshape(-1, 1)
            y = np.array(values)
            from sklearn.linear_model import LinearRegression
            trend_model = LinearRegression()
            trend_model.fit(X, y)
            trend_slope = trend_model.coef_[0] / np.mean(values)  # 正規化
            features.append(trend_slope)
        else:
            features.append(0)
        
        return features
        
    except Exception as e:
        print(f"Feature creation error: {e}")
        return None


def apply_realistic_constraints(predicted_growth, historical_values):
    """現実的な制約を適用"""
    # 業界平均的な成長率の範囲に制限
    # 情報通信業の過去実績を参考に設定
    min_growth = -0.3  # -30%
    max_growth = 0.5   # +50%
    
    # 企業の過去実績も考慮
    if len(historical_values) >= 2:
        past_volatility = np.std(historical_values) / np.mean(historical_values)
        # 変動性が高い企業はより広い範囲を許可
        max_growth = min(max_growth + past_volatility, 1.0)
        min_growth = max(min_growth - past_volatility, -0.5)
    
    return np.clip(predicted_growth, min_growth, max_growth)


def predict_legacy_growth_model(years, values):
    """従来の成長率モデル（比較用）"""
    future_years = np.array([years[-1] + i for i in range(1, 4)])
    
    # 複数年の成長率を計算して平均を取る
    growth_rates = []
    for i in range(1, min(4, len(values))):
        if values[-i-1] > 0:
            annual_growth = (values[-1] / values[-i-1]) ** (1 / i) - 1
            growth_rates.append(annual_growth)
    
    if growth_rates:
        avg_growth_rate = np.median(growth_rates) + 1
        pred_values = [values[-1] * (avg_growth_rate ** i) for i in range(1, 4)]
        
        return {
            'name': '従来モデル',
            'values': pred_values,
            'years': future_years,
            'growth_rate': (avg_growth_rate - 1) * 100
        }
    
    return None


def create_prediction_chart(actual_years, actual_values, predictions, title, ylabel):
    """予測結果のグラフを生成（汎用MLモデル対応）"""
    # フォント設定
    font_prop = FontProperties(fname=FONT_PATH)
    
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 実績データ
    plt.plot(actual_years, actual_values, 'o-', color='blue', 
             linewidth=3, markersize=10, label='実績', zorder=3)
    
    # 実績値のラベル
    for x, y in zip(actual_years[-3:], actual_values[-3:]):
        plt.text(x, y + 0.02 * abs(y), f'{y:.1f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                fontproperties=font_prop)
    
    # 汎用MLモデルの予測（メイン）
    if 'ml_universal' in predictions:
        pred_data = predictions['ml_universal']
        plt.plot(pred_data['years'], pred_data['values'], 
                '-', color='green', linewidth=3, 
                marker='s', markersize=10,
                label=f"{pred_data['name']} (年率{pred_data['growth_rate']:.1f}%, 信頼度:{pred_data['confidence']})",
                zorder=2)
        
        # 予測値のラベル
        for x, y in zip(pred_data['years'], pred_data['values']):
            plt.text(x, y + 0.02 * abs(y), f'{y:.1f}', 
                    ha='center', va='bottom', fontsize=9, color='green', 
                    fontweight='bold', fontproperties=font_prop)
    
    # 従来モデルの予測（比較用）
    if 'growth' in predictions:
        pred_data = predictions['growth']
        plt.plot(pred_data['years'], pred_data['values'], 
                '--', color='red', linewidth=2, alpha=0.7,
                marker='^', markersize=8,
                label=f"{pred_data['name']} (年率{pred_data['growth_rate']:.1f}%)",
                zorder=1)
        
        # 予測値のラベル（小さめ）
        for x, y in zip(pred_data['years'], pred_data['values']):
            plt.text(x, y - 0.03 * abs(y), f'{y:.1f}', 
                    ha='center', va='top', fontsize=8, color='red', 
                    alpha=0.7, fontproperties=font_prop)
    
    plt.title(title, fontsize=16, fontweight='bold', fontproperties=font_prop)
    plt.xlabel('年度', fontsize=12, fontproperties=font_prop)
    plt.ylabel(ylabel, fontsize=12, fontproperties=font_prop)
    plt.legend(loc='best', prop=font_prop, fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 零線を追加（負成長が分かりやすくなる）
    if any(v < 0 for prediction in predictions.values() for v in prediction.get('values', [])):
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # 軸の数値フォント設定
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def get_company_cluster_info(edinet_code):
    """企業のクラスタ情報を取得（PCA使用）"""
    try:
        from sklearn.decomposition import PCA
        from django.db.models import Max, OuterRef, Subquery
        
        # 各企業の最新年度を取得するサブクエリ
        latest_year_subquery = FinancialData.objects.filter(
            edinet_code=OuterRef('edinet_code')
        ).values('edinet_code').annotate(
            max_year=Max('fiscal_year')
        ).values('max_year')[:1]
        
        # 各企業の最新データを取得
        latest_data = FinancialData.objects.filter(
            fiscal_year=Subquery(latest_year_subquery)
        )
        
        # 特徴量の定義
        FEATURES = ['net_assets', 'total_assets', 'net_income', 'r_and_d_expenses', 'number_of_employees']
        
        # データフレームに変換
        df = pd.DataFrame.from_records(
            latest_data.values('edinet_code', 'filer_name', 'fiscal_year', *FEATURES)
        )
        df.set_index('edinet_code', inplace=True)
        
        # データ前処理
        df_filled = df.dropna(subset=FEATURES, how='all')
        df_filled[FEATURES] = df_filled[FEATURES].fillna(df_filled[FEATURES].median())
    
        if len(df_filled) < 3 or edinet_code not in df_filled.index:
            return None
    
        # 外れ値除去（対象企業を含む場合のみ実行）
        if edinet_code in df_filled.index:
            q = df_filled[FEATURES].quantile(0.95)
            mask = (df_filled[FEATURES] <= q).all(axis=1)
            # 対象企業が除外される場合は外れ値除去をスキップ
            if edinet_code not in df_filled[mask].index:
                print(f"Warning: Outlier removal would exclude target company {edinet_code}, skipping")
            else:
                df_filled = df_filled[mask]
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_filled[FEATURES])
        
        # PCAで次元削減（2次元）
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        # クラスタリング
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        df_filled['cluster'] = labels
        
        # 対象企業の情報
        company_cluster = df_filled.loc[edinet_code, 'cluster']
        company_year = df_filled.loc[edinet_code, 'fiscal_year']
        
        # 同じクラスタの企業（年度情報付き）
        same_cluster_df = df_filled[df_filled['cluster'] == company_cluster]
        same_cluster_companies = []
        for idx in same_cluster_df.index:
            if idx != edinet_code:
                same_cluster_companies.append({
                    'code': idx,
                    'name': same_cluster_df.loc[idx, 'filer_name'],
                    'year': same_cluster_df.loc[idx, 'fiscal_year']
                })
        same_cluster_companies = same_cluster_companies[:5]  # 上位5社
        
        # クラスタの特徴（平均値）
        cluster_means = df_filled[df_filled['cluster'] == company_cluster][FEATURES].mean()
        overall_means = df_filled[FEATURES].mean()
        
        # 特徴的な指標を特定
        relative_strengths = (cluster_means / overall_means - 1) * 100
        top_features = relative_strengths.abs().nlargest(3).index.tolist()
        
        # PCAの解釈（主成分の意味を理解）
        pca_interpretation = interpret_pca_components(pca, FEATURES)
        
        # グラフ生成（PCA空間でプロット）
        chart = create_cluster_pca_chart(
            X_pca, labels, df_filled.index, df_filled['fiscal_year'].to_dict(),
            edinet_code, pca_interpretation, pca.explained_variance_ratio_
        )
        
        return {
            'cluster_id': int(company_cluster),
            'total_clusters': 3,
            'same_cluster_companies': same_cluster_companies,
            'cluster_characteristics': {
                feat: {
                    'value': cluster_means[feat],
                    'relative': relative_strengths[feat]
                } for feat in top_features
            },
            'chart': chart,
            'company_year': company_year,
            'pca_interpretation': pca_interpretation
        }
        
    except Exception as e:
        print(f"Cluster analysis error: {e}")
        return None


def interpret_pca_components(pca, features):
    """PCA主成分の解釈"""
    components = pca.components_
    feature_labels = [get_feature_label(f) for f in features]
    
    interpretations = []
    for i, component in enumerate(components):
        # 各主成分で重要な特徴量を特定
        abs_component = np.abs(component)
        top_indices = np.argsort(abs_component)[-3:][::-1]  # 上位3つ
        
        interpretation = {
            'component': i + 1,
            'variance_ratio': pca.explained_variance_ratio_[i] * 100,
            'top_features': []
        }
        
        for idx in top_indices:
            interpretation['top_features'].append({
                'name': feature_labels[idx],
                'weight': component[idx],
                'abs_weight': abs_component[idx]
            })
        
        # 主成分の意味を推定
        if i == 0:
            interpretation['meaning'] = '企業規模'  # 通常、第1主成分は規模を表す
        elif i == 1:
            interpretation['meaning'] = '収益性・効率性'  # 第2主成分は収益性など
        
        interpretations.append(interpretation)
    
    return interpretations


def create_cluster_pca_chart(X_pca, labels, index, year_dict, target_code, pca_interpretation, variance_ratio):
    """PCA空間でのクラスタマップ（年度情報付き）"""
    # フォント設定
    font_prop = FontProperties(fname=FONT_PATH)
    
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 色定義
    colors = {0: 'lightcoral', 1: 'lightgreen', 2: 'lightblue'}
    
    # データフレーム作成
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=index)
    pca_df['cluster'] = labels
    pca_df['year'] = pd.Series(year_dict)
    
    # 全企業をプロット
    for cluster_id in [0, 1, 2]:
        cluster_data = pca_df[pca_df['cluster'] == cluster_id]
        plt.scatter(
            cluster_data['PC1'], 
            cluster_data['PC2'],
            c=colors[cluster_id], 
            label=f'クラスタ{cluster_id}',
            alpha=0.6, 
            s=60,
            edgecolors='white',
            linewidth=0.5
        )
    
    # 対象企業をハイライト
    if target_code in pca_df.index:
        target_data = pca_df.loc[target_code]
        plt.scatter(
            target_data['PC1'], 
            target_data['PC2'],
            c='red', 
            s=300, 
            marker='*',
            edgecolors='black',
            linewidth=2,
            label='当該企業',
            zorder=5
        )
        
        # 企業名と年度をアノテーション
        plt.annotate(
            f"{target_code}\n({target_data['year']}年)",
            xy=(target_data['PC1'], target_data['PC2']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            fontproperties=font_prop,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )
    
    # 軸ラベル（主成分の意味を含む）
    pc1_meaning = pca_interpretation[0]['meaning']
    pc2_meaning = pca_interpretation[1]['meaning']
    plt.xlabel(f'第1主成分 ({pc1_meaning}) - 寄与率{variance_ratio[0]*100:.1f}%', 
               fontsize=12, fontproperties=font_prop)
    plt.ylabel(f'第2主成分 ({pc2_meaning}) - 寄与率{variance_ratio[1]*100:.1f}%', 
               fontsize=12, fontproperties=font_prop)
    
    plt.title('企業クラスタマップ（各企業の最新データ）', 
              fontsize=16, fontweight='bold', fontproperties=font_prop)
    plt.legend(loc='best', prop=font_prop)
    plt.grid(True, alpha=0.3)
    
    # 原点を通る線を追加
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 軸の数値フォント設定
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def calculate_financial_indicators(financial_data):
    """財務指標を計算"""
    indicators = {}
    
    def safe_divide(a, b):
        if b and b != 0:
            return a / b
        return None
    
    indicators['roe'] = safe_divide(financial_data.net_income, financial_data.net_assets)
    indicators['roa'] = safe_divide(financial_data.net_income, financial_data.total_assets)
    indicators['operating_margin'] = safe_divide(financial_data.operating_income, financial_data.net_sales)
    indicators['asset_turnover'] = safe_divide(financial_data.net_sales, financial_data.total_assets)
    indicators['equity_ratio'] = safe_divide(financial_data.net_assets, financial_data.total_assets)
    
    return indicators


def get_metric_label(metric):
    """指標名の日本語ラベル"""
    labels = {
        'net_sales': '売上高',
        'operating_income': '営業利益', 
        'net_income': '純利益',
        'total_assets': '総資産'
    }
    return labels.get(metric, metric)


def get_feature_label(feature):
    """特徴量の日本語ラベル"""
    labels = {
        'net_assets': '純資産',
        'total_assets': '総資産',
        'net_income': '純利益',
        'r_and_d_expenses': '研究開発費',
        'number_of_employees': '従業員数'
    }
    return labels.get(feature, feature)


# 認証関連ビュー
def register(request):
    """ユーザー登録"""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'アカウントが作成されました。プロフィール情報を入力してください。')
            return redirect('financial:profile')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'registration/register.html', {'form': form})


@login_required
def profile(request):
    """プロフィール表示・編集"""
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    is_first_time = created or not any([profile.student_id, profile.university, profile.department])
    
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'プロフィールが更新されました。Growth Compassをお楽しみください！')
            return redirect('financial:home')
    else:
        form = UserProfileForm(instance=profile)
    
    return render(request, 'registration/profile.html', {
        'form': form,
        'profile': profile,
        'is_first_time': is_first_time
    })


def logout_view(request):
    """ログアウト処理"""
    logout(request)
    messages.success(request, 'ログアウトしました。')
    return redirect('financial:home')


@require_http_methods(["GET"])
def get_predictions_ajax(request, edinet_code):
    """予測分析のAJAX取得"""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    try:
        # 財務データ取得
        financial_data = FinancialData.objects.filter(
            edinet_code=edinet_code
        ).select_related('document').order_by('-fiscal_year')
        
        if not financial_data.exists() or len(financial_data) < 3:
            return JsonResponse({'error': '予測に必要なデータが不足しています'}, status=400)
        
        # 予測分析実行
        prediction_results = perform_predictions(financial_data)
        
        # レスポンス用にデータを整理
        response_data = {}
        for metric, result in prediction_results.items():
            response_data[metric] = {
                'label': result['label'],
                'chart': result['chart'],
                'predictions': result['predictions']
            }
        
        return JsonResponse({'predictions': response_data})
        
    except Exception as e:
        return JsonResponse({'error': f'予測分析エラー: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def get_clustering_ajax(request, edinet_code):
    """クラスタリング分析のAJAX取得"""
    # 一時的に認証チェックを無効化してテスト
    # if not request.user.is_authenticated:
    #     return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    try:
        cluster_info = get_company_cluster_info(edinet_code)
        
        if not cluster_info:
            return JsonResponse({'error': 'クラスタリング分析に必要なデータが不足しています'}, status=400)
        
        return JsonResponse({'cluster_info': cluster_info})
        
    except Exception as e:
        return JsonResponse({'error': f'クラスタリング分析エラー: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def get_ai_analysis_ajax(request, edinet_code):
    """AI企業分析のAJAX取得"""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    if not settings.AI_ANALYSIS_ENABLED:
        return JsonResponse({'error': 'AI分析機能が無効です'}, status=400)
    
    try:
        # 基本的な財務データとメタデータ取得
        financial_data = FinancialData.objects.filter(
            edinet_code=edinet_code
        ).select_related('document').order_by('-fiscal_year')
        
        if not financial_data.exists():
            return JsonResponse({'error': '企業データが見つかりません'}, status=404)
        
        company_name = financial_data.first().filer_name
        
        # 財務指標を計算
        data_with_indicators = []
        for fd in financial_data:
            indicators = calculate_financial_indicators(fd)
            data_with_indicators.append({
                'data': fd,
                'indicators': indicators
            })
        
        # 予測結果を取得（必要に応じて）
        prediction_results = {}
        if len(financial_data) >= 3:
            prediction_results = perform_predictions(financial_data)
        
        # AI分析生成
        ai_analysis = generate_ai_company_analysis(
            company_name, 
            data_with_indicators, 
            prediction_results, 
            edinet_code
        )
        
        return JsonResponse({'ai_analysis': ai_analysis})
        
    except Exception as e:
        return JsonResponse({'error': f'AI分析エラー: {str(e)}'}, status=500)