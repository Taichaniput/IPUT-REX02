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


# 日本語フォント設定
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'MS Gothic'
    FONT_PATH = None
else:
    FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'ipaexg.ttf')
    plt.rcParams['font.family'] = 'IPAPGothic'
plt.rcParams['axes.unicode_minus'] = False


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
    
    return render(request, 'financial/company_detail.html', {
        'company_name': company_name,
        'edinet_code': edinet_code,
        'financial_data': data_with_indicators,
        'prediction_results': prediction_results if request.user.is_authenticated else {},
        'cluster_info': cluster_info if request.user.is_authenticated else None,
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
    """成長率モデルによる予測"""
    predictions = {}
    
    # 将来の年度
    last_year = years[-1]
    future_years = np.array([last_year + i for i in range(1, 4)])
    
    if len(values) >= 2:
        # 複数年の成長率を計算して平均を取る（より安定的な予測）
        growth_rates = []
        for i in range(1, min(4, len(values))):  # 直近3年分まで
            if values[-i-1] > 0:  # ゼロ除算を避ける
                annual_growth = (values[-1] / values[-i-1]) ** (1 / i) - 1
                growth_rates.append(annual_growth)
        
        if growth_rates:
            # 成長率の中央値を使用（外れ値の影響を減らす）
            avg_growth_rate = np.median(growth_rates) + 1
            
            # 将来予測
            pred_values = [values[-1] * (avg_growth_rate ** i) for i in range(1, 4)]
            
            predictions['growth'] = {
                'name': '成長率モデル',
                'values': pred_values,
                'years': future_years,
                'growth_rate': (avg_growth_rate - 1) * 100  # パーセント表示用
            }
    
    return predictions


def create_prediction_chart(actual_years, actual_values, predictions, title, ylabel):
    """予測結果のグラフを生成（成長率モデル）"""
    # フォント設定
    font_prop = FontProperties(fname=FONT_PATH)
    
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 実績データ
    plt.plot(actual_years, actual_values, 'o-', color='blue', 
             linewidth=2, markersize=8, label='実績')
    
    # 実績値のラベル
    for x, y in zip(actual_years[-3:], actual_values[-3:]):
        plt.text(x, y + 0.02 * abs(y), f'{y:.1f}', 
                ha='center', va='bottom', fontsize=9, fontproperties=font_prop)
    
    # 成長率モデルの予測
    if 'growth' in predictions:
        pred_data = predictions['growth']
        plt.plot(pred_data['years'], pred_data['values'], 
                '--', color='red', linewidth=2, 
                marker='^', markersize=8,
                label=f"{pred_data['name']} (年率{pred_data['growth_rate']:.1f}%)")
        
        # 予測値のラベル
        for x, y in zip(pred_data['years'], pred_data['values']):
            plt.text(x, y + 0.02 * abs(y), f'{y:.1f}', 
                    ha='center', va='bottom', fontsize=8, color='red', fontproperties=font_prop)
    
    plt.title(title, fontsize=14, fontweight='bold', fontproperties=font_prop)
    plt.xlabel('年度', fontsize=12, fontproperties=font_prop)
    plt.ylabel(ylabel, fontsize=12, fontproperties=font_prop)
    plt.legend(loc='best', prop=font_prop)
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