# financial/views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Max, Count
from django.db import models
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
import pickle
import hashlib
from datetime import datetime, timedelta
from django.core.cache import cache
from matplotlib.font_manager import FontProperties


# 日本語フォント設定
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'MS Gothic'
    FONT_PATH = None
else:
    FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'ipaexg.ttf')
    plt.rcParams['font.family'] = 'IPAPGothic'
plt.rcParams['axes.unicode_minus'] = False


def normalize_search_string(text):
    """検索文字列を正規化（全角半角、大文字小文字、カタカナひらがな対応）"""
    if not text:
        return text
    
    # 全角英数字を半角に変換
    text = unicodedata.normalize('NFKC', text)
    
    # 大文字小文字を統一（小文字に）
    text = text.lower()
    
    # カタカナをひらがなに変換
    text = re.sub(r'[\u30A1-\u30F6]', lambda x: chr(ord(x.group()) - 0x60), text)
    
    # スペース、ハイフン、ピリオドなどを削除
    text = re.sub(r'[\s\-\.\u3000]', '', text)
    
    return text


def create_flexible_search_conditions(keyword):
    """柔軟な検索条件を作成"""
    conditions = Q()
    
    # 元のキーワードでの検索
    conditions |= Q(edinet_code__icontains=keyword) | Q(filer_name__icontains=keyword)
    
    # 正規化されたキーワードでの検索
    normalized_keyword = normalize_search_string(keyword)
    if normalized_keyword != keyword:
        conditions |= Q(edinet_code__icontains=normalized_keyword) | Q(filer_name__icontains=normalized_keyword)
    
    # 英数字の全角半角変換
    if re.search(r'[A-Za-z0-9]', keyword):
        # 半角→全角
        fullwidth_keyword = keyword.translate(str.maketrans(
            '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        ))
        conditions |= Q(edinet_code__icontains=fullwidth_keyword) | Q(filer_name__icontains=fullwidth_keyword)
    
    if re.search(r'[０-９Ａ-Ｚａ-ｚ]', keyword):
        # 全角→半角
        halfwidth_keyword = keyword.translate(str.maketrans(
            '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
            '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        ))
        conditions |= Q(edinet_code__icontains=halfwidth_keyword) | Q(filer_name__icontains=halfwidth_keyword)
    
    # 大文字小文字バリエーション
    upper_keyword = keyword.upper()
    lower_keyword = keyword.lower()
    if upper_keyword != keyword:
        conditions |= Q(edinet_code__icontains=upper_keyword) | Q(filer_name__icontains=upper_keyword)
    if lower_keyword != keyword:
        conditions |= Q(edinet_code__icontains=lower_keyword) | Q(filer_name__icontains=lower_keyword)
    
    # カタカナ・ひらがな変換
    if re.search(r'[\u3040-\u309F]', keyword):  # ひらがなが含まれる場合
        # ひらがな→カタカナ
        katakana_keyword = re.sub(r'[\u3040-\u309F]', lambda x: chr(ord(x.group()) + 0x60), keyword)
        conditions |= Q(filer_name__icontains=katakana_keyword)
    
    if re.search(r'[\u30A0-\u30FF]', keyword):  # カタカナが含まれる場合
        # カタカナ→ひらがな
        hiragana_keyword = re.sub(r'[\u30A1-\u30F6]', lambda x: chr(ord(x.group()) - 0x60), keyword)
        conditions |= Q(filer_name__icontains=hiragana_keyword)
    
    return conditions


def home(request):
    """企業検索画面（柔軟な検索対応）"""
    keyword = request.GET.get('keyword', '').strip()
    companies = []
    
    if keyword:
        # 柔軟な検索条件を作成
        search_conditions = create_flexible_search_conditions(keyword)
        
        companies = FinancialData.objects.filter(
            search_conditions
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
    """汎用モデルによる予測"""
    predictions = {}
    
    # 将来の年度
    last_year = years[-1]
    future_years = np.array([last_year + i for i in range(1, 4)])
    
    if len(values) >= 2:
        # 汎用MLモデルによる予測のみ
        ml_predictions = predict_with_universal_model(years, values)
        
        if ml_predictions:
            predictions['ml_universal'] = {
                'name': 'AI成長予測モデル',
                'values': ml_predictions['predicted_values'],
                'years': future_years,
                'growth_rate': ml_predictions['avg_growth_rate'],
                'confidence': ml_predictions.get('confidence', 'N/A'),
                'model_type': 'ml'
            }
    
    return predictions


def get_cached_model():
    """キャッシュされたモデルを取得または新規作成"""
    cache_key = "universal_prediction_model"
    cached_data = cache.get(cache_key)
    
    # キャッシュの有効期間（1時間）
    if cached_data and cached_data.get('timestamp'):
        cache_time = cached_data['timestamp']
        if datetime.now() - cache_time < timedelta(hours=1):
            return cached_data['model'], cached_data['scaler']
    
    # キャッシュが無効または存在しない場合、新規作成
    training_data = prepare_universal_training_data()
    
    if len(training_data) < 10:
        return None, None
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    X_train = np.array([row['features'] for row in training_data])
    y_train = np.array([row['target_growth'] for row in training_data])
    sample_weights = np.array([row.get('sample_weight', 1.0) for row in training_data])
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # ランダムフォレストで学習（より保守的かつ高速な設定）
    model = RandomForestRegressor(
        n_estimators=30,   # さらに削減
        random_state=42, 
        min_samples_split=15,
        min_samples_leaf=8,
        max_depth=6,       # 浅めに設定
        max_features='sqrt',
        n_jobs=2           # 並列処理
    )
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # キャッシュに保存
    cache_data = {
        'model': model,
        'scaler': scaler,
        'timestamp': datetime.now()
    }
    cache.set(cache_key, cache_data, timeout=3600)  # 1時間キャッシュ
    
    return model, scaler


def predict_with_universal_model(years, values):
    """全企業データを使った汎用モデルによる予測（高速化版）"""
    try:
        # キャッシュされたモデルを取得
        model, scaler = get_cached_model()
        
        if model is None or scaler is None:
            return None
        
        # 特徴量作成
        features = create_features_from_timeseries(years, values)
        if not features:
            return None
        
        # 予測実行
        company_features = np.array([features]).reshape(1, -1)
        company_features_scaled = scaler.transform(company_features)
        
        predicted_growth_rate = model.predict(company_features_scaled)[0]
        
        # 現実的な制約を適用
        predicted_growth_rate = apply_realistic_constraints(predicted_growth_rate, values)
        
        # 将来値計算
        growth_multiplier = 1 + predicted_growth_rate
        predicted_values = [values[-1] * (growth_multiplier ** i) for i in range(1, 4)]
        
        # 信頼度は簡易計算（キャッシュのため）
        confidence = 0.75 if abs(predicted_growth_rate) < 0.1 else 0.65
        
        return {
            'predicted_values': predicted_values,
            'avg_growth_rate': predicted_growth_rate * 100,
            'confidence': f"{confidence:.2f}"
        }
        
    except Exception as e:
        print(f"Universal model prediction error: {e}")
        return None


def prepare_universal_training_data():
    """全企業データから学習用データセットを準備（高速化版）"""
    # キャッシュから取得を試行
    cache_key = "training_data_cache"
    cached_training_data = cache.get(cache_key)
    
    if cached_training_data:
        return cached_training_data
    
    training_data = []
    
    try:
        # 高速化：最新の財務データのみに限定（過去5年分）
        recent_years = [2019, 2020, 2021, 2022, 2023]
        
        # 効率的なクエリ：必要な企業のみ選択
        valid_companies = FinancialData.objects.filter(
            fiscal_year__in=recent_years,
            net_sales__gte=1000000000  # 10億円以上の企業のみ
        ).values('edinet_code').annotate(
            data_count=models.Count('fiscal_year')
        ).filter(data_count__gte=4).values_list('edinet_code', flat=True)
        
        # サンプル数制限（最大200社、高速化のため）
        sampled_companies = list(valid_companies)[:200]
        
        for edinet_code in sampled_companies:
            # 各企業の時系列データを取得
            company_data = FinancialData.objects.filter(
                edinet_code=edinet_code,
                fiscal_year__in=recent_years
            ).order_by('fiscal_year').values(
                'fiscal_year', 'net_sales'
            )
            
            company_list = list(company_data)
            if len(company_list) < 4:
                continue
            
            # 最大3パターンのみ作成（高速化）
            max_patterns = min(3, len(company_list) - 3)
            for i in range(max_patterns):
                current_data = company_list[i:i+3]
                next_year_data = company_list[i+3]
                
                years = [row['fiscal_year'] for row in current_data]
                sales_values = [row['net_sales'] or 0 for row in current_data]
                
                if any(v <= 0 for v in sales_values):
                    continue
                
                features = create_features_from_timeseries(years, sales_values)
                if not features:
                    continue
                
                current_sales = sales_values[-1]
                next_sales = next_year_data['net_sales'] or 0
                
                if current_sales > 0 and next_sales > 0:
                    target_growth = (next_sales / current_sales) - 1
                    
                    # 厳格な異常値除外
                    if -0.3 <= target_growth <= 0.5:  # さらに厳しく
                        weight = 2.5 if target_growth < 0 else 1.0  # 負成長をより重視
                        training_data.append({
                            'features': features,
                            'target_growth': target_growth,
                            'sample_weight': weight
                        })
        
        # キャッシュに保存（30分）
        cache.set(cache_key, training_data, timeout=1800)
        
    except Exception as e:
        print(f"Training data preparation error: {e}")
    
    return training_data


def create_features_from_timeseries(years, values):
    """時系列データから特徴量を作成（改善版）"""
    if len(years) < 2 or len(values) < 2:
        return None
    
    try:
        features = []
        
        # 1. 企業規模（対数変換で正規化）
        current_value = np.log(max(values[-1], 1))
        features.append(current_value)
        
        # 2. 前年比成長率
        if len(values) >= 2 and values[-2] > 0:
            yoy_growth = (values[-1] / values[-2]) - 1
            features.append(yoy_growth)
        else:
            features.append(0)
        
        # 3. 2年平均成長率（より安定）
        if len(values) >= 3:
            growth_rates = []
            for i in range(1, min(3, len(values))):  # 2年分のみ
                if values[-i-1] > 0:
                    rate = (values[-1] / values[-i-1]) ** (1/i) - 1
                    growth_rates.append(rate)
            avg_growth = np.mean(growth_rates) if growth_rates else 0
            features.append(avg_growth)
        else:
            features.append(0)
        
        # 4. 成長率の安定性（変動係数）
        if len(values) >= 3:
            growth_rates = []
            for i in range(1, len(values)):
                if values[i-1] > 0:
                    rate = (values[i] / values[i-1]) - 1
                    growth_rates.append(rate)
            
            if growth_rates:
                growth_std = np.std(growth_rates)
                growth_mean = np.mean(growth_rates)
                stability = growth_std / (abs(growth_mean) + 0.01)  # 正規化
                features.append(min(stability, 5.0))  # 上限設定
            else:
                features.append(0)
        else:
            features.append(0)
        
        # 5. 最近の成長加速度（2次差分）
        if len(values) >= 3:
            recent_acceleration = 0
            if values[-3] > 0 and values[-2] > 0:
                growth_1 = (values[-2] / values[-3]) - 1
                growth_2 = (values[-1] / values[-2]) - 1
                recent_acceleration = growth_2 - growth_1
            features.append(np.clip(recent_acceleration, -1.0, 1.0))
        else:
            features.append(0)
        
        # 6. 経済サイクル調整（年度ベース）
        latest_year = years[-1]
        # 2020年=コロナ影響、2008-2009=リーマン影響を考慮
        cycle_factor = 0
        if latest_year in [2020, 2021]:
            cycle_factor = -0.3  # コロナ影響
        elif latest_year in [2008, 2009]:
            cycle_factor = -0.5  # リーマン影響
        elif latest_year >= 2022:
            cycle_factor = -0.1  # 物価高・金利上昇
        features.append(cycle_factor)
        
        # 7. 企業規模階層（売上高ベース）
        size_tier = 0
        if values[-1] >= 100000000000:  # 1000億円以上
            size_tier = 1
        elif values[-1] >= 50000000000:  # 500億円以上
            size_tier = 0.5
        elif values[-1] >= 10000000000:  # 100億円以上
            size_tier = 0.2
        else:
            size_tier = -0.2  # 小規模企業はより保守的
        features.append(size_tier)
        
        return features
        
    except Exception as e:
        print(f"Feature creation error: {e}")
        return None


def apply_realistic_constraints(predicted_growth, historical_values):
    """現実的な制約を適用（保守的アプローチ）"""
    
    # より保守的な成長率範囲に制限
    min_growth = -0.25  # -25%
    max_growth = 0.15   # +15%（より現実的）
    
    # 企業の過去実績から制約を調整
    if len(historical_values) >= 3:
        # 過去の実際の変動率を計算
        growth_rates = []
        for i in range(1, len(historical_values)):
            if historical_values[i-1] > 0:
                rate = (historical_values[i] / historical_values[i-1]) - 1
                growth_rates.append(rate)
        
        if growth_rates:
            historical_mean = np.mean(growth_rates)
            historical_std = np.std(growth_rates)
            
            # 過去実績に基づく制約（2σ範囲）
            historical_min = historical_mean - 2 * historical_std
            historical_max = historical_mean + 2 * historical_std
            
            # より厳しい制約を採用
            min_growth = max(min_growth, max(historical_min, -0.4))
            max_growth = min(max_growth, min(historical_max, 0.3))
    
    # 悲観的バイアスを追加（現実的予測のため）
    if predicted_growth > 0:
        predicted_growth *= 0.7  # 楽観的予測を30%削減
    else:
        predicted_growth *= 1.2  # 悲観的予測は20%強化
    
    return np.clip(predicted_growth, min_growth, max_growth)




def create_prediction_chart(actual_years, actual_values, predictions, title, ylabel):
    """AI予測結果のグラフを生成"""
    # フォント設定
    font_prop = FontProperties(fname=FONT_PATH)
    
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 実績データ
    plt.plot(actual_years, actual_values, 'o-', color='#2563eb', 
             linewidth=4, markersize=12, label='実績データ', zorder=3)
    
    # 実績値のラベル
    for x, y in zip(actual_years[-3:], actual_values[-3:]):
        plt.text(x, y + 0.03 * abs(y), f'{y:.1f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                fontproperties=font_prop, color='#1e40af')
    
    # AI予測（メイン）
    if 'ml_universal' in predictions:
        pred_data = predictions['ml_universal']
        growth_rate = pred_data['growth_rate']
        confidence = pred_data['confidence']
        
        # 成長率に応じて色を変更
        if growth_rate >= 5:
            color = '#10b981'  # 緑（成長）
        elif growth_rate >= 0:
            color = '#f59e0b'  # 黄（微成長）
        else:
            color = '#ef4444'  # 赤（減少）
        
        plt.plot(pred_data['years'], pred_data['values'], 
                '-', color=color, linewidth=4, 
                marker='s', markersize=12,
                label=f" AI予測 (年率{growth_rate:.1f}%, 信頼度:{confidence})",
                zorder=2)
        
        # 予測値のラベル
        for x, y in zip(pred_data['years'], pred_data['values']):
            plt.text(x, y + 0.03 * abs(y), f'{y:.1f}', 
                    ha='center', va='bottom', fontsize=10, color=color, 
                    fontweight='bold', fontproperties=font_prop)
        
        # 信頼区間の表示（簡易版）
        confidence_value = float(confidence)
        if confidence_value > 0.6:
            upper_values = [v * 1.1 for v in pred_data['values']]
            lower_values = [v * 0.9 for v in pred_data['values']]
            plt.fill_between(pred_data['years'], lower_values, upper_values, 
                           color=color, alpha=0.2, label='予測信頼区間 (±10%)')
    
    plt.title(f"{title}", fontsize=18, fontweight='bold', fontproperties=font_prop, pad=20)
    plt.xlabel('年度', fontsize=14, fontproperties=font_prop)
    plt.ylabel(f"{ylabel}", fontsize=14, fontproperties=font_prop)
    plt.legend(loc='upper left', prop=font_prop, fontsize=12, frameon=True, 
               fancybox=True, shadow=True, bbox_to_anchor=(0.02, 0.98))
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 零線を追加（負成長が分かりやすくなる）
    if any(v < 0 for prediction in predictions.values() for v in prediction.get('values', [])):
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
        plt.text(actual_years[0], 0, '基準線', fontsize=10, fontproperties=font_prop, 
                va='bottom', ha='left', color='black', alpha=0.7)
    
    # 背景をグラデーションに
    ax = plt.gca()
    ax.set_facecolor('#f8fafc')
    
    # 軸の数値フォント設定
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(11)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, facecolor='white', edgecolor='none')
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