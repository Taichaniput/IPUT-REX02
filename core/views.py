from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Max
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.cache import cache
from .models import FinancialData, FinancialDataValidated, UserProfile
from .forms import UserRegistrationForm, UserProfileForm

from .src import prediction_service, clustering_service, positioning_service, ai_analysis_service

import asyncio
import json
import numpy as np


def convert_numpy_types(obj):
    """
    NumPy型をPython標準型に変換してJSON serializable にする
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def home(request):
    keyword = request.GET.get('keyword', '').strip()
    companies = []
    
    if keyword:
        lower_keyword = keyword.lower()
        companies = FinancialData.objects.filter(
            Q(edinet_code__icontains=lower_keyword) |
            Q(filer_name__icontains=lower_keyword)
        ).values(
            'edinet_code', 'filer_name'
        ).distinct().order_by('filer_name')[:50]
    
    return render(request, 'financial/home.html', {
        'companies': companies,
        'keyword': keyword,
    })


def company_detail(request, edinet_code):
    financial_data = FinancialData.objects.filter(
        edinet_code=edinet_code
    ).select_related('document').order_by('-fiscal_year')
    
    if not financial_data.exists():
        return render(request, 'financial/company_detail.html', {
            'error': 'この企業のデータが見つかりません。しかし、EDINETコードは有効です。',
            'edinet_code': edinet_code
        })
    
    company_name = financial_data.first().filer_name
    
    data_with_indicators = []
    for fd in financial_data:
        indicators = calculate_financial_indicators(fd)
        data_with_indicators.append({
            'data': fd,
            'indicators': indicators
        })
    
    prediction_results = {}
    cluster_info = None
    ai_analysis = {}
    positioning_info = None
    
    return render(request, 'financial/company_detail.html', {
        'company_name': company_name,
        'edinet_code': edinet_code,
        'financial_data': data_with_indicators,
        'prediction_results': prediction_results,
        'cluster_info': cluster_info,
        'ai_analysis': ai_analysis,
        'positioning_info': positioning_info,
        'show_login_prompt': not request.user.is_authenticated,
    })


def ai_analysis_ajax(request, edinet_code):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    try:
        financial_data = FinancialData.objects.filter(
            edinet_code=edinet_code
        ).select_related('document').order_by('-fiscal_year')
        
        if not financial_data.exists():
            return JsonResponse({'error': '企業データが見つかりません'}, status=404)
        
        company_name = financial_data.first().filer_name
        
        data_with_indicators = []
        for fd in financial_data:
            indicators = calculate_financial_indicators(fd)
            data_with_indicators.append({
                'data': fd,
                'indicators': indicators
            })
        
        prediction_results = prediction_service.analyze_financial_predictions(data_with_indicators)
        cluster_info = asyncio.run(clustering_service.get_company_cluster_info(edinet_code))
        positioning_info = asyncio.run(positioning_service.get_company_positioning_analysis(edinet_code))
        
        ai_analysis = ai_analysis_service.generate_comprehensive_analysis(
            company_name, edinet_code, data_with_indicators, 
            prediction_results, cluster_info, positioning_info
        )
        
        if 'error' in ai_analysis:
            return JsonResponse({
                'success': False,
                'error': ai_analysis['error']
            }, status=400)
        
        # NumPy型をPython標準型に変換
        safe_prediction_results = convert_numpy_types(prediction_results) if prediction_results else None
        safe_cluster_info = convert_numpy_types(cluster_info) if cluster_info else None
        safe_positioning_info = convert_numpy_types(positioning_info) if positioning_info else None
        safe_ai_analysis = convert_numpy_types(ai_analysis) if ai_analysis else {}
        
        return JsonResponse({
            'success': True,
            'ai_analysis': {
                **safe_ai_analysis,  # AI分析のテキスト結果
                'prediction_results': safe_prediction_results,  # チャートデータを追加
                'cluster_info': safe_cluster_info,  # クラスタリング情報を追加
                'positioning_info': safe_positioning_info  # ポジショニング情報を追加
            }
        })
        
    except Exception as e:
        print(f"AI analysis AJAX error: {e}")
        return JsonResponse({
            'error': f'分析中にエラーが発生しました: {str(e)}'
        }, status=500)


def register(request):
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
    logout(request)
    messages.success(request, 'ログアウトしました。')
    return redirect('financial:home')


@require_http_methods(["GET"])
def get_predictions_ajax(request, edinet_code):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    try:
        financial_data = FinancialData.objects.filter(
            edinet_code=edinet_code
        ).select_related('document').order_by('-fiscal_year')
        
        if not financial_data.exists() or len(financial_data) < 3:
            return JsonResponse({'error': '予測に必要なデータが不足しています'}, status=400)
        
        prediction_results = prediction_service.analyze_financial_predictions(financial_data)
        
        response_data = {}
        for metric, result in prediction_results.items():
            response_data[metric] = {
                'label': result['label'],
                'chart': result['chart'],
                'predictions': result['predictions']
            }
        
        return JsonResponse({'predictions': convert_numpy_types(response_data)})
        
    except Exception as e:
        return JsonResponse({'error': f'予測分析エラー: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def get_clustering_ajax(request, edinet_code):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    try:
        cluster_info = asyncio.run(clustering_service.get_company_cluster_info(edinet_code))
        
        if not cluster_info:
            return JsonResponse({'error': 'クラスタリング分析に必要なデータが不足しています'}, status=400)
        
        return JsonResponse({'cluster_info': convert_numpy_types(cluster_info)})
        
    except Exception as e:
        return JsonResponse({'error': f'クラスタリング分析エラー: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def get_ai_analysis_ajax(request, edinet_code):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    if not settings.AI_ANALYSIS_ENABLED:
        return JsonResponse({'error': 'AI分析機能が無効です'}, status=400)
    
    try:
        financial_data = FinancialData.objects.filter(
            edinet_code=edinet_code
        ).select_related('document').order_by('-fiscal_year')
        
        if not financial_data.exists():
            return JsonResponse({'error': '企業データが見つかりません'}, status=404)
        
        company_name = financial_data.first().filer_name
        
        data_with_indicators = []
        for fd in financial_data:
            indicators = calculate_financial_indicators(fd)
            data_with_indicators.append({
                'data': fd,
                'indicators': indicators
            })
        
        prediction_results = prediction_service.analyze_financial_predictions(data_with_indicators)
        
        ai_analysis = ai_analysis_service.generate_comprehensive_analysis(
            company_name, 
            edinet_code, 
            data_with_indicators, 
            prediction_results, 
            None, # cluster_info is fetched separately
            None  # positioning_info is fetched separately
        )
        
        return JsonResponse({'ai_analysis': convert_numpy_types(ai_analysis)})
        
    except Exception as e:
        return JsonResponse({'error': f'AI分析エラー: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def get_company_overview_ajax(request, edinet_code):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    try:
        financial_data = FinancialData.objects.filter(
            edinet_code=edinet_code
        ).select_related('document').order_by('-fiscal_year')
        
        if not financial_data.exists():
            return JsonResponse({'error': '企業データが見つかりません'}, status=404)
        
        company_name = financial_data.first().filer_name
        
        company_overview = ai_analysis_service.generate_company_overview_analysis(
            company_name, 
            edinet_code
        )
        
        return JsonResponse({'company_overview': convert_numpy_types(company_overview)})
        
    except Exception as e:
        return JsonResponse({'error': f'企業概要取得エラー: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def get_scenario_analysis_ajax(request, edinet_code, chart_type):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    if not settings.AI_ANALYSIS_ENABLED:
        return JsonResponse({'error': 'AI分析機能が無効です'}, status=400)
    
    if chart_type not in ['sales', 'profit']:
        return JsonResponse({'error': '無効なチャートタイプです'}, status=400)
    
    try:
        cache_key = f"scenario_analysis_{edinet_code}_{chart_type}"
        cached_analysis = cache.get(cache_key)
        
        if cached_analysis:
            return JsonResponse({'scenario_analysis': convert_numpy_types(cached_analysis)})
        
        financial_data = FinancialData.objects.filter(
            edinet_code=edinet_code
        ).select_related('document').order_by('-fiscal_year')
        
        if not financial_data.exists():
            return JsonResponse({'error': '企業データが見つかりません'}, status=404)
        
        company_name = financial_data.first().filer_name
        
        prediction_results = prediction_service.analyze_financial_predictions(financial_data)
        
        scenario_analysis = ai_analysis_service.generate_scenario_analysis(
            company_name, 
            edinet_code, 
            prediction_results,
            chart_type
        )
        
        cache.set(cache_key, scenario_analysis, timeout=43200)
        
        return JsonResponse({'scenario_analysis': convert_numpy_types(scenario_analysis)})
        
    except Exception as e:
        return JsonResponse({'error': f'シナリオ分析エラー: {str(e)}'}, status=500)


@require_http_methods(["GET"])
def scenario_analysis_ajax(request, edinet_code, chart_type):
    return get_scenario_analysis_ajax(request, edinet_code, chart_type)


@require_http_methods(["GET"])
def get_positioning_analysis_ajax(request, edinet_code):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'ログインが必要です'}, status=401)
    
    try:
        print(f"Starting positioning analysis AJAX for {edinet_code}")
        
        positioning_info = asyncio.run(positioning_service.get_company_positioning_analysis(edinet_code))
        
        if not positioning_info:
            print(f"No positioning info returned for {edinet_code}")
            return JsonResponse({'error': '二軸分析に必要なデータが不足しています'}, status=400)
        
        print(f"Positioning analysis successful for {edinet_code}")
        return JsonResponse({'positioning_analysis': convert_numpy_types(positioning_info)})
        
    except Exception as e:
        print(f"Positioning analysis AJAX error for {edinet_code}: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f'二軸分析エラー: {str(e)}'}, status=500)


def calculate_financial_indicators(financial_data_instance):
    roe = None
    roa = None
    equity_ratio = None

    if financial_data_instance.net_income is not None and financial_data_instance.net_assets is not None and financial_data_instance.net_assets != 0:
        roe = financial_data_instance.net_income / financial_data_instance.net_assets

    if financial_data_instance.net_income is not None and financial_data_instance.total_assets is not None and financial_data_instance.total_assets != 0:
        roa = financial_data_instance.net_income / financial_data_instance.total_assets

    if financial_data_instance.net_assets is not None and financial_data_instance.total_assets is not None and financial_data_instance.total_assets != 0:
        equity_ratio = financial_data_instance.net_assets / financial_data_instance.total_assets

    return {
        'roe': roe,
        'roa': roa,
        'equity_ratio': equity_ratio,
    }