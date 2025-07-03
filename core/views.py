# financial/views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Max
from .models import FinancialData, FinancialDataValidated, UserProfile
from .forms import UserRegistrationForm, UserProfileForm

# 分離したモジュールのインポート（src/パッケージから）
from .src import (
    calculate_financial_indicators,
    perform_predictions, 
    get_company_cluster_info,
    generate_comprehensive_ai_analysis
)




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