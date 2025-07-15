# core/ml_analytics.py

import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
import platform
import os
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
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
        
        # ARIMA単体での予測
        scenarios = predict_arima(years, values)
        
        # グラフ生成
        chart = create_chart(
            years, values, scenarios,
            f"{get_label(metric)}の3シナリオ予測",
            get_label(metric) + "（億円）"
        )
        
        results[metric] = {
            'predictions': {
                'scenarios': scenarios
            },
            'chart': chart,
            'label': get_label(metric)
        }
    
    return results




def predict_arima(years, values):
    """ARIMAモデルによる3シナリオ予測"""
    try:
        if len(values) < 5:
            return None
        
        clean_values = [v for v in values if v is not None and v > 0]
        if len(clean_values) < 5:
            return None
        
        # 事前学習モデルの読み込みを試行
        model_key = gen_model_key(clean_values)
        cached_model = load_model(model_key)
        
        if cached_model:
            results = cached_model
        else:
            model = ARIMA(clean_values, order=(1, 1, 1))
            results = model.fit()
            # 新しいモデルを保存
            save_model(model_key, results)
        
        forecast = results.get_forecast(steps=3)
        pred_mean = forecast.predicted_mean
        pred_ci = forecast.conf_int(alpha=0.3)
        
        if isinstance(pred_mean, np.ndarray):
            pred_mean = pd.Series(pred_mean)
        if isinstance(pred_ci, np.ndarray):
            pred_ci = pd.DataFrame(pred_ci)
        
        future_years = [years[-1] + i for i in range(1, 4)]
        
        def calc_growth_rate(current_val, future_val):
            if current_val > 0:
                return ((future_val / current_val) - 1) * 100
            return 0
        
        last_value = clean_values[-1]
        
        scenarios = {
            'optimistic': {
                'growth_rate': calc_growth_rate(last_value, pred_ci.iloc[-1, 1]),
                'years': future_years,
                'values': pred_ci.iloc[:, 1].tolist()
            },
            'current': {
                'growth_rate': calc_growth_rate(last_value, pred_mean.iloc[-1]),
                'years': future_years,
                'values': pred_mean.tolist()
            },
            'pessimistic': {
                'growth_rate': calc_growth_rate(last_value, pred_ci.iloc[-1, 0]),
                'years': future_years,
                'values': pred_ci.iloc[:, 0].tolist()
            }
        }
        
        return scenarios
        
    except Exception as e:
        print(f"ARIMA prediction error: {e}")
        return None


async def predict_metric(metric, years, values):
    """指標の非同期予測"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        scenarios = await loop.run_in_executor(executor, predict_arima, years, values)
        
        if scenarios:
            chart = await loop.run_in_executor(
                executor, 
                create_chart,
                years, values, scenarios,
                f"{get_label(metric)}の3シナリオ予測",
                get_label(metric) + "（億円）"
            )
            
            return {
                'metric': metric,
                'data': {
                    'predictions': {'scenarios': scenarios},
                    'chart': chart,
                    'label': get_label(metric)
                }
            }
    return None


def gen_model_key(values):
    """データからモデルキーを生成"""
    import hashlib
    data_str = ','.join([f"{v:.2f}" for v in values])
    return hashlib.md5(data_str.encode()).hexdigest()[:16]


def load_model(model_key):
    """事前学習済みARIMAモデルの読み込み"""
    try:
        model_path = f"arima_cache/{model_key}.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Model loading error: {e}")
    return None


def save_model(model_key, model):
    """ARIMAモデルの保存"""
    try:
        os.makedirs('arima_cache', exist_ok=True)
        model_path = f"arima_cache/{model_key}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Model saving error: {e}")


def create_chart(actual_years, actual_values, scenarios, title, ylabel):
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


async def get_cluster_info(edinet_code):
    """企業のクラスタ情報を非同期取得"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, _get_cluster_sync, edinet_code)


def _get_cluster_sync(edinet_code):
    """企業のクラスタ情報を取得（同期版）"""
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
        FEATURES = [
            'net_assets', 'total_assets', 'net_income', 'r_and_d_expenses', 'number_of_employees'
        ]
        
        # データフレームに変換
        df = pd.DataFrame.from_records(
            latest_data.values('edinet_code', 'filer_name', 'fiscal_year', *FEATURES)
        )
        df.set_index('edinet_code', inplace=True)
        
        # データ前処理
        df_filled = df.dropna(subset=FEATURES, how='all')
        
        # 欠損値処理
        for feature in FEATURES:
            if feature in df_filled.columns:
                df_filled[feature] = df_filled[feature].fillna(df_filled[feature].median())
        
        # 無限大値の処理
        numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
        df_filled[numeric_columns] = df_filled[numeric_columns].replace([np.inf, -np.inf], np.nan)
        df_filled[numeric_columns] = df_filled[numeric_columns].fillna(df_filled[numeric_columns].median())
        
        if len(df_filled) < 3 or edinet_code not in df_filled.index:
            return None
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_filled[FEATURES])
        
        # UMAPで次元削減（2次元）
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = umap_reducer.fit_transform(X_scaled)
        
        # 密度ベースクラスタリング
        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=8, metric='euclidean')
        labels = clusterer.fit_predict(X_umap)
        
        # ノイズポイントを最近傍クラスタに割り当て
        from sklearn.neighbors import NearestNeighbors
        noise_mask = labels == -1
        if noise_mask.any():
            cluster_mask = labels != -1
            if cluster_mask.any():
                # クラスタポイントでNearestNeighborsを学習
                nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_umap[cluster_mask])
                # ノイズポイントの最近傍クラスタを取得
                _, nearest_indices = nbrs.kneighbors(X_umap[noise_mask])
                # 最近傍点のクラスタラベルを割り当て
                cluster_labels_only = labels[cluster_mask]
                for i, nearest_idx in enumerate(nearest_indices.flatten()):
                    noise_indices = np.where(noise_mask)[0]
                    labels[noise_indices[i]] = cluster_labels_only[nearest_idx]
        
        df_filled['cluster'] = labels
        
        # 対象企業の情報
        company_cluster = df_filled.loc[edinet_code, 'cluster']
        company_year = df_filled.loc[edinet_code, 'fiscal_year']
        
        # 最近傍割り当て後はノイズポイントは存在しないはず
        # 念のためのチェック
        if company_cluster == -1:
            print(f"Warning: Company {edinet_code} still classified as noise after nearest neighbor assignment")
            company_cluster = 0  # デフォルトクラスタに割り当て
        
        # 同じクラスタの企業
        same_cluster_df = df_filled[df_filled['cluster'] == company_cluster]
        same_cluster_companies = []
        for idx in same_cluster_df.index:
            if idx != edinet_code:
                same_cluster_companies.append({
                    'code': idx,
                    'name': same_cluster_df.loc[idx, 'filer_name'],
                    'year': same_cluster_df.loc[idx, 'fiscal_year']
                })
        same_cluster_companies = same_cluster_companies[:5]
        
        # クラスタの特徴
        cluster_means = df_filled[df_filled['cluster'] == company_cluster][FEATURES].mean()
        overall_means = df_filled[FEATURES].mean()
        
        # 特徴的な指標を特定
        relative_strengths = (cluster_means / overall_means - 1) * 100
        top_features = relative_strengths.abs().nlargest(3).index.tolist()
        
        # UMAPの解釈
        umap_interpretation = interpret_umap(FEATURES)
        
        # グラフ生成
        chart = create_cluster_chart(
            X_umap, labels, df_filled.index, df_filled['fiscal_year'].to_dict(),
            edinet_code, umap_interpretation
        )
        
        # クラスタ数を計算
        unique_clusters = set(labels)
        total_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        
        return {
            'cluster_id': int(company_cluster),
            'total_clusters': total_clusters,
            'same_cluster_companies': same_cluster_companies,
            'cluster_characteristics': {
                feat: {
                    'value': cluster_means[feat],
                    'relative': relative_strengths[feat]
                } for feat in top_features
            },
            'chart': chart,
            'company_year': company_year,
            'umap_interpretation': umap_interpretation
        }
        
    except Exception as e:
        print(f"Cluster analysis error: {e}")
        return None

def interpret_umap(features):
    """UMAP + HDBSCAN解釈"""
    feature_labels = [get_feature_label(f) for f in features]
    
    interpretation = {
        'method': 'UMAP + HDBSCAN',
        'description': '非線形次元削減と密度ベースクラスタリング',
        'features_used': [
            {'name': label, 'original': feat} 
            for feat, label in zip(features, feature_labels)
        ],
        'advantages': [
            '企業間の局所的な類似性を保持',
            '非線形な関係性を捉える',
            '任意の形状のクラスタを発見',
            '密度の異なるクラスタに対応',
            'ノイズポイントを特定'
        ],
        'interpretation_notes': [
            '距離が近い企業は財務特性が類似',
            'クラスタは密度に基づいて自動的に形成',
            'ノイズとして分類された企業は独特な特徴を持つ',
            '球状でないクラスタも適切に検出可能'
        ]
    }
    
    return interpretation


def create_cluster_chart(X_umap, labels, index, year_dict, target_code, umap_interpretation):
    """UMAP空間でのクラスタマップ"""
    # フォント設定
    font_prop = FontProperties(fname=FONT_PATH)
    
    plt.figure(figsize=(12, 9))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # データフレーム作成
    umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'], index=index)
    umap_df['cluster'] = labels
    umap_df['year'] = pd.Series(year_dict)
    
    # クラスタを取得
    unique_clusters = sorted(set(labels))
    
    # 基本色設定
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # 全企業をプロット（凡例は主要クラスタのみ）
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = umap_df[umap_df['cluster'] == cluster_id]
        
        color = colors[i % len(colors)]
        # 大きなクラスタ（10点以上）のみ凡例に表示
        if len(cluster_data) >= 10:
            label = f'クラスタ{cluster_id}'
            show_label = True
        else:
            label = None
            show_label = False
        alpha = 0.8
        s = 60
            
        if len(cluster_data) > 0:
            plt.scatter(
                cluster_data['UMAP1'], 
                cluster_data['UMAP2'],
                c=color, 
                label=label if show_label else None,
                alpha=alpha, 
                s=s,
                edgecolors='black',
                linewidth=0.5
            )
    
    # 対象企業をハイライト
    if target_code in umap_df.index:
        target_data = umap_df.loc[target_code]
        plt.scatter(
            target_data['UMAP1'], 
            target_data['UMAP2'],
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
            xy=(target_data['UMAP1'], target_data['UMAP2']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            fontproperties=font_prop,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )
    
    # 軸ラベル
    plt.xlabel('UMAP 次元1', fontproperties=font_prop, fontsize=12)
    plt.ylabel('UMAP 次元2', fontproperties=font_prop, fontsize=12)
    plt.title('企業の財務特性に基づくクラスタリング分析', fontproperties=font_prop, fontsize=14, fontweight='bold')
    
    # 凡例
    plt.legend(prop=font_prop, loc='upper left')
    
    # グリッドスタイル
    plt.grid(True, alpha=0.3)
    
    # クラスタ数の計算
    num_clusters = len([c for c in unique_clusters if c != -1])
    noise_count = len([c for c in labels if c == -1])
    
    # 説明テキスト
    description_text = f"""
使用特徴量: {', '.join([f['name'] for f in umap_interpretation['features_used'][:3]])}等 {len(umap_interpretation['features_used'])}個
次元削減: UMAP
クラスタリング: HDBSCAN + 最近傍割り当て
発見クラスタ数: {num_clusters}個 (全企業をクラスタに割り当て)
    """
    
    plt.figtext(0.02, 0.02, description_text, fontsize=8, fontproperties=font_prop,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Base64エンコード
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')


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


def get_label(metric):
    """指標名の日本語ラベル"""
    labels = {
        'net_sales': '売上高',
        'operating_income': '営業利益', 
        'net_income': '純利益',
        'total_assets': '総資産'
    }
    return labels.get(metric, metric)


def get_feature_label(feature):
    """特徴量の日本語ラベル（拡張版）"""
    labels = {
        # 基本財務データ
        'net_assets': '純資産',
        'total_assets': '総資産',
        'net_sales': '売上高',
        'operating_income': '営業利益',
        'ordinary_income': '経常利益',
        'net_income': '純利益',
        'operating_cash_flow': '営業キャッシュフロー',
        'r_and_d_expenses': '研究開発費',
        'number_of_employees': '従業員数',
        
        # 財務比率
        'roe': 'ROE（自己資本利益率）',
        'roa': 'ROA（総資産利益率）',
        'operating_margin': '営業利益率',
        'equity_ratio': '自己資本比率',
        'rd_intensity': 'R&D集約度',
        'asset_turnover': '総資産回転率',
        'employee_productivity': '従業員1人当たり売上高'
    }
    return labels.get(feature, feature)
