# financial/views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Max
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
from matplotlib.font_manager import FontProperties


# Tavily availability check
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


def get_company_additional_info(company_name):
    """企業の追加情報を外部ソース（Tavily Web Search）から取得"""
    from django.conf import settings
    
    additional_info = {}

    if not TAVILY_AVAILABLE:
        additional_info['web_search_summary'] = "Tavilyライブラリがインストールされていません。"
        return additional_info

    if not settings.TAVILY_API_KEY:
        additional_info['web_search_summary'] = "Tavily APIキーが設定されていません。"
        return additional_info

    try:
        from tavily import TavilyClient
        # Tavilyクライアントを初期化
        tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)
        
        # 検索クエリを工夫して、多角的な情報を要求
        search_query = f"{company_name}の事業内容、強みと弱み、市場でのポジショニング、最近の重要なニュースについて包括的に調査し、要約してください。"
        
        # TavilyでWeb検索を実行
        response = tavily.search(
            query=search_query,
            search_depth="advanced",
            max_results=5
        )
        
        # 結果を結合して一つのサマリーにする
        summary = "\n".join([f"- {obj['content']}" for obj in response['results']])
        additional_info['web_search_summary'] = summary if summary else "ウェブ検索で関連情報が見つかりませんでした。"

    except Exception as e:
        print(f"Tavily search error: {e}")
        additional_info['web_search_summary'] = "ウェブ検索中にエラーが発生しました。"
    
    return additional_info


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
    
    # 2. 予測分析（複数モデル）
    prediction_results = {}
    if len(financial_data) >= 3:
        prediction_results = perform_predictions(financial_data)
    
    # 3. クラスタリング分析
    cluster_info = get_company_cluster_info(edinet_code)
    
    # 4. AI分析（ログインユーザーのみ）
    ai_analysis = {}
    if request.user.is_authenticated:
        print(f"Starting AI analysis for {company_name}...")
        print(f"Prediction results available: {bool(prediction_results)}")
        print(f"Cluster info available: {bool(cluster_info)}")
        ai_analysis = generate_comprehensive_ai_analysis(
            company_name, edinet_code, financial_data, 
            prediction_results, cluster_info
        )
        print(f"AI analysis completed with keys: {ai_analysis.keys()}")
    
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
    
    # データ準備 - financial_dataは[{'data': FinancialData, 'indicators': dict}, ...]形式
    df_data = []
    for item in financial_data:
        if isinstance(item, dict) and 'data' in item:
            fd = item['data']
        else:
            fd = item
            
        if fd.fiscal_year:
            df_data.append({
                'fiscal_year': fd.fiscal_year,
                'net_sales': fd.net_sales,
                'operating_income': fd.operating_income,
                'net_income': fd.net_income,
                'total_assets': fd.total_assets
            })
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        return results
    
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
        
        # 3シナリオ予測実行
        scenarios = predict_three_scenarios(years, values)
        
        # グラフ生成
        chart = create_scenario_chart(
            years, values, scenarios,
            f"{get_metric_label(metric)}の3シナリオ予測",
            get_metric_label(metric) + "（億円）"
        )
        
        results[metric] = {
            'predictions': {
                'scenarios': scenarios
            },
            'chart': chart,
            'label': get_metric_label(metric)
        }
    
    return results


def predict_three_scenarios(years, values):
    """楽観・現状・悲観の3シナリオで予測"""
    if len(values) < 2:
        return None
    
    # 成長率を計算
    growth_rates = []
    for i in range(1, len(values)):
        if values[i-1] > 0:
            growth_rate = (values[i] / values[i-1]) - 1
            growth_rates.append(growth_rate)
    
    if not growth_rates:
        return None
    
    # 基本成長率と変動率
    base_growth = np.median(growth_rates)
    growth_volatility = np.std(growth_rates) if len(growth_rates) > 1 else 0.1
    
    # 3シナリオの成長率
    optimistic_growth = base_growth + growth_volatility  # 楽観的
    current_growth = base_growth  # 現状維持
    pessimistic_growth = base_growth - growth_volatility  # 悲観的
    
    # 将来3年の予測値を計算
    last_value = values[-1]
    future_years = [years[-1] + i for i in range(1, 4)]
    
    scenarios = {
        'optimistic': {
            'growth_rate': optimistic_growth * 100,
            'years': future_years,
            'values': [last_value * ((1 + optimistic_growth) ** i) for i in range(1, 4)]
        },
        'current': {
            'growth_rate': current_growth * 100,
            'years': future_years,
            'values': [last_value * ((1 + current_growth) ** i) for i in range(1, 4)]
        },
        'pessimistic': {
            'growth_rate': pessimistic_growth * 100,
            'years': future_years,
            'values': [last_value * ((1 + pessimistic_growth) ** i) for i in range(1, 4)]
        }
    }
    
    return scenarios


def create_scenario_chart(actual_years, actual_values, scenarios, title, ylabel):
    """3シナリオ予測結果のグラフを生成"""
    font_prop = FontProperties(fname=FONT_PATH)
    
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 実績データ
    plt.plot(actual_years, actual_values, 'o-', color='blue', 
             linewidth=3, markersize=10, label='実績', zorder=3)
    
    # 各シナリオの予測線
    colors = {'optimistic': '#28a745', 'current': '#17a2b8', 'pessimistic': '#ffc107'}
    scenario_names = {'optimistic': '楽観', 'current': '現状', 'pessimistic': '悲観'}
    
    for scenario_name, scenario_data in scenarios.items():
        if scenario_data:
            plt.plot(scenario_data['years'], scenario_data['values'], 
                    '-', color=colors[scenario_name], linewidth=3, 
                    marker='s', markersize=8,
                    label=f"{scenario_names[scenario_name]}（年率{scenario_data['growth_rate']:.1f}%）",
                    zorder=2)
    
    plt.title(title, fontsize=16, fontweight='bold', fontproperties=font_prop)
    plt.xlabel('年度', fontsize=12, fontproperties=font_prop)
    plt.ylabel(ylabel, fontsize=12, fontproperties=font_prop)
    plt.legend(loc='best', prop=font_prop, fontsize=11)
    plt.grid(True, alpha=0.3)
    
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


def generate_comprehensive_ai_analysis(company_name, edinet_code, financial_data, prediction_results, cluster_info):
    """Gemini APIを使用して包括的な企業分析を生成"""
    from django.conf import settings
    
    if not settings.GEMINI_API_KEY:
        return create_fallback_analysis(company_name, financial_data, prediction_results, cluster_info)
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Tavilyで追加情報を取得
        additional_info = get_company_additional_info(company_name)
        
        # 財務データを整理
        financial_summary = prepare_financial_summary(financial_data)
        
        # 予測データを整理
        prediction_summary = prepare_prediction_summary(prediction_results)
        
        # クラスタデータを整理
        cluster_summary = prepare_cluster_summary(cluster_info)
        
        # Geminiプロンプトを構築
        prompt = build_comprehensive_analysis_prompt(
            company_name, financial_summary, prediction_summary, 
            cluster_summary, additional_info
        )
        
        # Gemini APIに送信
        response = model.generate_content(prompt)
        
        # レスポンスが正常に生成されているかチェック
        if not response or not response.text:
            return {"error": "Gemini APIからの応答がありません"}
        
        print(f"Gemini response received: {len(response.text)} characters")
        
        # レスポンスを構造化
        structured_analysis = parse_structured_analysis(response.text)
        
        return structured_analysis
        
    except Exception as e:
        print(f"AI analysis generation error: {e}")
        return create_fallback_analysis(company_name, financial_data, prediction_results, cluster_info)


def create_fallback_analysis(company_name, financial_data, prediction_results, cluster_info):
    """API利用できない場合のフォールバック分析"""
    analysis = {}
    
    # 基本的な財務分析
    if financial_data:
        # financial_dataは辞書のリスト形式 [{'data': FinancialData, 'indicators': dict}, ...]
        if isinstance(financial_data[0], dict) and 'data' in financial_data[0]:
            latest = financial_data[0]['data']
        else:
            # 直接FinancialDataオブジェクトの場合
            latest = financial_data[0]
        
        net_sales = (latest.net_sales or 0) / 100000000
        analysis['FINANCIAL_ANALYSIS'] = f"{company_name}は{latest.fiscal_year}年に売上高{net_sales:.1f}億円を記録。情報通信業界における企業として位置づけられています。"
    else:
        analysis['FINANCIAL_ANALYSIS'] = f"{company_name}の詳細な財務分析を表示するためには、API接続が必要です。"
    
    # シナリオ分析のフォールバック
    analysis['GROWTH_SCENARIOS'] = {
        'optimistic': "市場拡大と技術革新により成長が期待されます。",
        'current': "現在のトレンドが継続すると予想されます。",
        'pessimistic': "市場環境の変化により成長に課題が生じる可能性があります。"
    }
    
    analysis['PROFIT_SCENARIOS'] = {
        'optimistic': "効率化により収益性向上が期待されます。",
        'current': "現在の収益構造が維持される見通しです。",
        'pessimistic': "競争激化により収益性に圧力がかかる可能性があります。"
    }
    
    # ポジショニング分析
    if cluster_info:
        analysis['POSITIONING_ANALYSIS'] = f"クラスタ{cluster_info['cluster_id']}に分類され、業界内での特定のポジションを占めています。"
    else:
        analysis['POSITIONING_ANALYSIS'] = "業界内でのポジショニング分析には追加データが必要です。"
    
    # 総括
    analysis['SUMMARY'] = f"{company_name}は情報系学生にとって技術的な成長機会を提供する可能性がある企業です。詳細な分析にはAI機能をご利用ください。"
    
    return analysis


def prepare_financial_summary(financial_data):
    """財務データを要約"""
    if not financial_data:
        return "財務データなし"
    
    summary = []
    for item in financial_data[:3]:  # 最新3年分
        # itemが辞書の場合とFinancialDataオブジェクトの場合を処理
        if isinstance(item, dict) and 'data' in item:
            fd = item['data']
        else:
            fd = item
            
        net_sales = fd.net_sales or 0
        net_income = fd.net_income or 0
        summary.append(f"{fd.fiscal_year}年: 売上{net_sales/100000000:.1f}億円, 純利益{net_income/100000000:.1f}億円")
    
    return "\n".join(summary)


def prepare_prediction_summary(prediction_results):
    """予測データを要約"""
    if not prediction_results:
        return "予測データなし"
    
    summary = []
    for metric, result in prediction_results.items():
        if 'scenarios' in result.get('predictions', {}):
            scenarios = result['predictions']['scenarios']
            summary.append(f"{result['label']}: 楽観{scenarios['optimistic']['growth_rate']:.1f}%, 現状{scenarios['current']['growth_rate']:.1f}%, 悲観{scenarios['pessimistic']['growth_rate']:.1f}%")
    
    return "\n".join(summary)


def prepare_cluster_summary(cluster_info):
    """クラスタデータを要約"""
    if not cluster_info:
        return "クラスタデータなし"
    
    summary = f"当該企業はクラスタ{cluster_info['cluster_id']}/{cluster_info['total_clusters']}に分類\n"
    
    # クラスタの特徴
    if 'cluster_characteristics' in cluster_info:
        summary += "クラスタの特徴:\n"
        for feat, data in cluster_info['cluster_characteristics'].items():
            feat_label = get_feature_label(feat)
            summary += f"- {feat_label}: 全体平均比{data['relative']:.1f}%\n"
    
    # 同じクラスタの企業
    if 'same_cluster_companies' in cluster_info:
        companies = [comp['name'] for comp in cluster_info['same_cluster_companies'][:5]]
        summary += f"\n同クラスタの類似企業: {', '.join(companies)}\n"
    
    # PCA解釈情報
    if 'pca_interpretation' in cluster_info:
        summary += "\n主成分分析による解釈:\n"
        for comp in cluster_info['pca_interpretation']:
            summary += f"- 第{comp['component']}主成分({comp['meaning']}): 寄与率{comp['variance_ratio']:.1f}%\n"
    
    return summary


def build_comprehensive_analysis_prompt(company_name, financial_summary, prediction_summary, cluster_summary, additional_info):
    """包括的分析用のプロンプトを構築"""
    prompt = f"""
あなたは情報系学生の就活支援を専門とする企業分析アナリストです。以下の企業について、構造化された分析を行ってください。

## 分析対象企業
{company_name}

## 財務データ
{financial_summary}

## 成長予測（3シナリオ）
{prediction_summary}

## 業界ポジショニング・クラスタリング分析
{cluster_summary}

## 外部情報（Tavily検索結果）
{additional_info.get('web_search_summary', 'なし')}

## 指示
上記データを統合して情報系学生向けの就活分析を行ってください。必ず以下の形式で出力してください。

[FINANCIAL_ANALYSIS]
財務データに基づく企業の特徴と健全性の簡潔な分析（100-150文字程度）
[/FINANCIAL_ANALYSIS]

[GROWTH_SCENARIOS_OPTIMISTIC]
楽観シナリオの詳細な説明（市場拡大、新技術導入成功、デジタル変革などの要因）
[/GROWTH_SCENARIOS_OPTIMISTIC]

[GROWTH_SCENARIOS_CURRENT]
現状シナリオの詳細な説明（現在のトレンド継続、安定成長）
[/GROWTH_SCENARIOS_CURRENT]

[GROWTH_SCENARIOS_PESSIMISTIC]
悲観シナリオの詳細な説明（市場縮小、競合激化、技術的遅れなどのリスク）
[/GROWTH_SCENARIOS_PESSIMISTIC]

[PROFIT_SCENARIOS_OPTIMISTIC]
収益性楽観シナリオの説明（効率化、高付加価値サービス拡大）
[/PROFIT_SCENARIOS_OPTIMISTIC]

[PROFIT_SCENARIOS_CURRENT]
収益性現状シナリオの説明（現在の収益構造継続）
[/PROFIT_SCENARIOS_CURRENT]

[PROFIT_SCENARIOS_PESSIMISTIC]
収益性悲観シナリオの説明（コスト増、価格競争激化）
[/PROFIT_SCENARIOS_PESSIMISTIC]

[POSITIONING_ANALYSIS]
クラスタデータと同業他社比較に基づく業界内ポジショニング分析。企業の競合優位性、市場での立ち位置、技術力について詳述
[/POSITIONING_ANALYSIS]

[SUMMARY]
情報系学生向けの就活への示唆。この企業のキャリア価値、技術的成長機会、業界トレンド、エンジニアとしてのキャリアパスについて総括
[/SUMMARY]

各セクションは具体的で実用的な内容にしてください。
"""
    return prompt


def parse_structured_analysis(response_text):
    """構造化された分析レスポンスを解析"""
    sections = {}
    
    try:
        # セクションを抽出
        sections['FINANCIAL_ANALYSIS'] = extract_section(response_text, 'FINANCIAL_ANALYSIS')
        
        sections['GROWTH_SCENARIOS'] = {
            'optimistic': extract_section(response_text, 'GROWTH_SCENARIOS_OPTIMISTIC'),
            'current': extract_section(response_text, 'GROWTH_SCENARIOS_CURRENT'),
            'pessimistic': extract_section(response_text, 'GROWTH_SCENARIOS_PESSIMISTIC')
        }
        
        sections['PROFIT_SCENARIOS'] = {
            'optimistic': extract_section(response_text, 'PROFIT_SCENARIOS_OPTIMISTIC'),
            'current': extract_section(response_text, 'PROFIT_SCENARIOS_CURRENT'),
            'pessimistic': extract_section(response_text, 'PROFIT_SCENARIOS_PESSIMISTIC')
        }
        
        sections['POSITIONING_ANALYSIS'] = extract_section(response_text, 'POSITIONING_ANALYSIS')
        sections['SUMMARY'] = extract_section(response_text, 'SUMMARY')
        
    except Exception as e:
        print(f"Response parsing error: {e}")
        sections = {"error": "レスポンス解析中にエラーが発生しました"}
    
    return sections


def extract_section(text, section_name):
    """テキストから指定されたセクションを抽出"""
    start_tag = f"[{section_name}]"
    end_tag = f"[/{section_name}]"
    
    start_index = text.find(start_tag)
    if start_index == -1:
        return "分析中..."
    
    start_index += len(start_tag)
    end_index = text.find(end_tag, start_index)
    
    if end_index == -1:
        return text[start_index:].strip()
    
    return text[start_index:end_index].strip()


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