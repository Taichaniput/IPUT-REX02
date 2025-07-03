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

# 分離したモジュールのインポート（src/パッケージから）
from .src import (
    calculate_financial_indicators,
    perform_predictions, 
    get_company_cluster_info,
    generate_comprehensive_ai_analysis
)

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
    
    # 4. ページを先に表示（AI分析は非同期で後から実行）
    return render(request, 'financial/company_detail.html', {
        'company_name': company_name,
        'edinet_code': edinet_code,
        'financial_data': data_with_indicators,
        'prediction_results': prediction_results if request.user.is_authenticated else {},
        'cluster_info': cluster_info if request.user.is_authenticated else None,
        'ai_analysis': {},  # 空の辞書で初期化、後でAJAXで取得
        'show_login_prompt': not request.user.is_authenticated,
    })


def ai_analysis_ajax(request, edinet_code):
    """AI分析をAJAXで実行するエンドポイント"""
    from django.http import JsonResponse
    
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    try:
        # 基本データを再取得
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
        
        # 予測分析
        prediction_results = {}
        if len(financial_data) >= 3:
            prediction_results = perform_predictions(financial_data)
        
        # クラスタリング分析
        cluster_info = get_company_cluster_info(edinet_code)
        
        # AI分析実行
        print(f"Starting AI analysis for {company_name}...")
        ai_analysis = generate_comprehensive_ai_analysis(
            company_name, edinet_code, data_with_indicators, 
            prediction_results, cluster_info
        )
        print(f"AI analysis completed with keys: {ai_analysis.keys()}")
        
        return JsonResponse({
            'success': True,
            'ai_analysis': ai_analysis
        })
        
    except Exception as e:
        print(f"AI analysis AJAX error: {e}")
        return JsonResponse({
            'error': f'分析中にエラーが発生しました: {str(e)}'
        }, status=500)


















































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