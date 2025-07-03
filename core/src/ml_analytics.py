# core/ml_analytics.py

import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')
import platform
import os
from matplotlib.font_manager import FontProperties
from ..models import FinancialData
from django.db.models import Max, OuterRef, Subquery

# 日本語フォント設定
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'MS Gothic'
    FONT_PATH = None
else:
    # src/ディレクトリから見た相対パスでfontsディレクトリにアクセス
    FONT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts', 'ipaexg.ttf')
    plt.rcParams['font.family'] = 'IPAPGothic'
plt.rcParams['axes.unicode_minus'] = False


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


def get_company_cluster_info(edinet_code):
    """企業のクラスタ情報を取得（PCA使用）"""
    try:
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