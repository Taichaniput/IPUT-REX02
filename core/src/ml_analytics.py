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


def calculate_growth_stability_scores(edinet_code):
    """企業の成長性・安定性スコアを計算（二軸分析用）"""
    try:
        # 企業の財務データを取得（最低3年分必要）
        financial_data = FinancialData.objects.filter(
            edinet_code=edinet_code
        ).order_by('-fiscal_year')[:5]  # 最新5年分
        
        if len(financial_data) < 3:
            return None
        
        # データフレームに変換
        df_data = []
        for fd in financial_data:
            df_data.append({
                'fiscal_year': fd.fiscal_year,
                'net_sales': fd.net_sales or 0,
                'operating_income': fd.operating_income or 0,
                'net_income': fd.net_income or 0,
                'total_assets': fd.total_assets or 0,
                'net_assets': fd.net_assets or 0,
                'r_and_d_expenses': fd.r_and_d_expenses or 0,
                'number_of_employees': fd.number_of_employees or 0
            })
        
        df = pd.DataFrame(df_data).sort_values('fiscal_year')
        
        # 成長性スコアの計算
        growth_score = calculate_growth_score(df)
        
        # 安定性スコアの計算
        stability_score = calculate_stability_score(df)
        
        # 企業の基本情報
        latest = financial_data[0]
        company_info = {
            'edinet_code': edinet_code,
            'company_name': latest.filer_name,
            'latest_year': latest.fiscal_year,
            'net_sales_billion': (latest.net_sales or 0) / 100000000,
            'employees': latest.number_of_employees or 0
        }
        
        return {
            'growth_score': growth_score,
            'stability_score': stability_score,
            'company_info': company_info,
            'quadrant': determine_quadrant(growth_score, stability_score),
            'detailed_metrics': {
                'sales_growth_rate': calculate_sales_growth_rate(df),
                'employee_growth_rate': calculate_employee_growth_rate(df),
                'rd_intensity': calculate_rd_intensity(df),
                'equity_ratio': calculate_equity_ratio(df),
                'operating_margin_stability': calculate_operating_margin_stability(df),
                'roa_stability': calculate_roa_stability(df)
            }
        }
        
    except Exception as e:
        print(f"Growth stability calculation error for {edinet_code}: {e}")
        return None


def calculate_growth_score(df):
    """成長性スコアを計算（0-100）"""
    scores = []
    
    # 1. 売上高成長率（3年平均）
    sales_growth = calculate_sales_growth_rate(df)
    scores.append(normalize_growth_rate(sales_growth, 0.05, 0.20))  # 5%を中央値、20%を満点
    
    # 2. 従業員数成長率
    employee_growth = calculate_employee_growth_rate(df)
    scores.append(normalize_growth_rate(employee_growth, 0.03, 0.15))  # 3%を中央値、15%を満点
    
    # 3. R&D集約度
    rd_intensity = calculate_rd_intensity(df)
    scores.append(normalize_rd_intensity(rd_intensity, 0.02, 0.10))  # 2%を中央値、10%を満点
    
    # 加重平均（売上成長重視）
    weighted_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)
    return min(100, max(0, weighted_score))


def calculate_stability_score(df):
    """安定性スコアを計算（0-100）"""
    scores = []
    
    # 1. 自己資本比率
    equity_ratio = calculate_equity_ratio(df)
    scores.append(normalize_ratio(equity_ratio, 0.30, 0.70))  # 30%を最低、70%を満点
    
    # 2. 営業利益率の安定性
    margin_stability = calculate_operating_margin_stability(df)
    scores.append(normalize_stability(margin_stability, 0.10, 0.02))  # 変動係数の逆数
    
    # 3. ROAの安定性
    roa_stability = calculate_roa_stability(df)
    scores.append(normalize_stability(roa_stability, 0.10, 0.02))  # 変動係数の逆数
    
    # 加重平均（自己資本比率重視）
    weighted_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)
    return min(100, max(0, weighted_score))


def calculate_sales_growth_rate(df):
    """売上高成長率を計算（3年平均）"""
    if len(df) < 2:
        return 0
    
    growth_rates = []
    for i in range(1, len(df)):
        prev_sales = df.iloc[i-1]['net_sales']
        curr_sales = df.iloc[i]['net_sales']
        if prev_sales > 0:
            growth_rate = (curr_sales - prev_sales) / prev_sales
            growth_rates.append(growth_rate)
    
    return np.mean(growth_rates) if growth_rates else 0


def calculate_employee_growth_rate(df):
    """従業員数成長率を計算"""
    if len(df) < 2:
        return 0
    
    growth_rates = []
    for i in range(1, len(df)):
        prev_emp = df.iloc[i-1]['number_of_employees']
        curr_emp = df.iloc[i]['number_of_employees']
        if prev_emp > 0:
            growth_rate = (curr_emp - prev_emp) / prev_emp
            growth_rates.append(growth_rate)
    
    return np.mean(growth_rates) if growth_rates else 0


def calculate_rd_intensity(df):
    """R&D集約度を計算（R&D費/売上高）"""
    latest = df.iloc[-1]
    if latest['net_sales'] > 0:
        return latest['r_and_d_expenses'] / latest['net_sales']
    return 0


def calculate_equity_ratio(df):
    """自己資本比率を計算"""
    latest = df.iloc[-1]
    if latest['total_assets'] > 0:
        return latest['net_assets'] / latest['total_assets']
    return 0


def calculate_operating_margin_stability(df):
    """営業利益率の安定性を計算（変動係数の逆数）"""
    margins = []
    for _, row in df.iterrows():
        if row['net_sales'] > 0:
            margin = row['operating_income'] / row['net_sales']
            margins.append(margin)
    
    if len(margins) < 2:
        return 0
    
    cv = np.std(margins) / np.mean(margins) if np.mean(margins) != 0 else float('inf')
    return 1 / (1 + cv)  # 変動係数の逆数（安定性指標）


def calculate_roa_stability(df):
    """ROAの安定性を計算（変動係数の逆数）"""
    roas = []
    for _, row in df.iterrows():
        if row['total_assets'] > 0:
            roa = row['net_income'] / row['total_assets']
            roas.append(roa)
    
    if len(roas) < 2:
        return 0
    
    cv = np.std(roas) / np.mean(roas) if np.mean(roas) != 0 else float('inf')
    return 1 / (1 + cv)  # 変動係数の逆数（安定性指標）


def normalize_growth_rate(rate, mid_point, max_point):
    """成長率を0-100に正規化"""
    if rate <= 0:
        return 0
    elif rate <= mid_point:
        return 50 * (rate / mid_point)
    elif rate <= max_point:
        return 50 + 50 * ((rate - mid_point) / (max_point - mid_point))
    else:
        return 100


def normalize_ratio(ratio, min_point, max_point):
    """比率を0-100に正規化"""
    if ratio <= min_point:
        return 0
    elif ratio >= max_point:
        return 100
    else:
        return 100 * ((ratio - min_point) / (max_point - min_point))


def normalize_rd_intensity(intensity, mid_point, max_point):
    """R&D集約度を0-100に正規化"""
    if intensity <= 0:
        return 0
    elif intensity <= mid_point:
        return 50 * (intensity / mid_point)
    elif intensity <= max_point:
        return 50 + 50 * ((intensity - mid_point) / (max_point - mid_point))
    else:
        return 100


def normalize_stability(stability, min_point, max_point):
    """安定性指標を0-100に正規化"""
    if stability <= min_point:
        return 0
    elif stability >= max_point:
        return 100
    else:
        return 100 * ((stability - min_point) / (max_point - min_point))


def determine_quadrant(growth_score, stability_score):
    """成長性・安定性スコアから象限を決定"""
    growth_high = growth_score >= 50
    stability_high = stability_score >= 50
    
    if growth_high and stability_high:
        return "ideal"  # 理想企業
    elif growth_high and not stability_high:
        return "challenge"  # チャレンジ企業
    elif not growth_high and stability_high:
        return "stable"  # 安定企業
    else:
        return "caution"  # 要注意企業


def get_quadrant_info(quadrant):
    """象限の詳細情報を取得"""
    quadrant_details = {
        "ideal": {
            "name": "理想企業",
            "description": "高成長×高安定",
            "color": "#28a745",  # 緑
            "career_advice": "就活生に最適。成長とキャリア安定性を両立できる理想的な企業です。",
            "risk_level": "低",
            "recommendation": "強く推奨"
        },
        "challenge": {
            "name": "チャレンジ企業", 
            "description": "高成長×不安定",
            "color": "#ffc107",  # 黄
            "career_advice": "ハイリスク・ハイリターン。急成長の可能性があるが、安定性にリスクがあります。",
            "risk_level": "中-高",
            "recommendation": "慎重に検討"
        },
        "stable": {
            "name": "安定企業",
            "description": "低成長×高安定", 
            "color": "#17a2b8",  # 青
            "career_advice": "大手・安定志向。着実なキャリア形成が可能ですが、急速な成長は期待できません。",
            "risk_level": "低",
            "recommendation": "安定志向に推奨"
        },
        "caution": {
            "name": "要注意企業",
            "description": "低成長×不安定",
            "color": "#dc3545",  # 赤
            "career_advice": "慎重な検討が必要。成長性・安定性の両面でリスクがあります。",
            "risk_level": "高",
            "recommendation": "推奨しない"
        }
    }
    return quadrant_details.get(quadrant, quadrant_details["caution"])


async def get_positioning_analysis(edinet_code):
    """企業の二軸分析ポジショニングを非同期取得"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, _get_positioning_sync, edinet_code)


def _get_positioning_sync(edinet_code):
    """企業の二軸分析ポジショニングを取得（同期版）"""
    try:
        print(f"Starting positioning analysis for {edinet_code}")
        
        # 対象企業の二軸分析
        print("Step 1: Calculating growth stability scores...")
        company_analysis = calculate_growth_stability_scores(edinet_code)
        if not company_analysis:
            print(f"Failed to calculate growth stability scores for {edinet_code}")
            return None
        print(f"Company analysis completed: {company_analysis['company_info']['company_name']}")
        
        # 業界平均や類似企業の取得（比較分析用）- パフォーマンス向上のため制限
        print("Step 2: Getting reference companies (limited to 50 for performance)...")
        reference_data = get_reference_companies(edinet_code, sample_size=50)
        print(f"Reference data obtained: {len(reference_data.get('sample_companies', []))} companies")
        
        # ポジショニングマップのグラフ生成
        print("Step 3: Creating positioning map...")
        chart = create_positioning_map(company_analysis, reference_data, edinet_code)
        print("Positioning map created successfully")
        
        # 象限の詳細情報
        print("Step 4: Getting quadrant info...")
        quadrant_info = get_quadrant_info(company_analysis['quadrant'])
        
        # レコメンド企業（既存クラスタリングを活用）
        print("Step 5: Getting recommendations...")
        recommendations = get_quadrant_recommendations(company_analysis['quadrant'], edinet_code, limit=5)
        print(f"Found {len(recommendations)} recommendations")
        
        result = {
            'growth_score': company_analysis['growth_score'],
            'stability_score': company_analysis['stability_score'],
            'quadrant': company_analysis['quadrant'],
            'quadrant_info': quadrant_info,
            'company_info': company_analysis['company_info'],
            'detailed_metrics': company_analysis['detailed_metrics'],
            'chart': chart,
            'recommendations': recommendations,
            'industry_comparison': reference_data.get('industry_stats', {}),
            'interpretation': generate_positioning_interpretation(company_analysis, quadrant_info)
        }
        
        print("Positioning analysis completed successfully")
        return result
        
    except Exception as e:
        print(f"Positioning analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_reference_companies(target_edinet_code, sample_size=None):
    """比較用の参考企業データを取得（全企業を使用）"""
    try:
        # 全企業を取得して比較に使用
        all_companies = FinancialData.objects.values('edinet_code').distinct()
        
        # 対象企業以外の全企業から分析データを取得
        sample_companies = []
        total_companies = all_companies.count()
        processed = 0
        
        print(f"Processing {total_companies} companies for reference data...")
        
        for company in all_companies:
            if company['edinet_code'] != target_edinet_code:
                try:
                    analysis = calculate_growth_stability_scores(company['edinet_code'])
                    if analysis:
                        sample_companies.append(analysis)
                except Exception as e:
                    # 個別企業のエラーはスキップして続行
                    print(f"Error processing {company['edinet_code']}: {e}")
                    continue
                
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{total_companies} companies, found {len(sample_companies)} valid analyses")
                
                # sample_sizeが指定されている場合のみ制限（デフォルトは全企業）
                if sample_size and len(sample_companies) >= sample_size:
                    break
        
        print(f"Completed processing. Total valid companies: {len(sample_companies)}")
        
        # 業界統計の計算
        if sample_companies:
            growth_scores = [comp['growth_score'] for comp in sample_companies]
            stability_scores = [comp['stability_score'] for comp in sample_companies]
            
            industry_stats = {
                'avg_growth': np.mean(growth_scores),
                'avg_stability': np.mean(stability_scores),
                'growth_std': np.std(growth_scores),
                'stability_std': np.std(stability_scores),
                'total_companies': len(sample_companies)
            }
        else:
            industry_stats = {}
        
        return {
            'sample_companies': sample_companies,
            'industry_stats': industry_stats
        }
        
    except Exception as e:
        print(f"Reference companies error: {e}")
        return {'sample_companies': [], 'industry_stats': {}}


def create_positioning_map(company_analysis, reference_data, target_edinet_code):
    """二軸分析ポジショニングマップを生成"""
    font_prop = FontProperties(fname=FONT_PATH)
    
    plt.figure(figsize=(14, 12))  # グラフサイズを拡大
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 参考企業をプロット（薄く表示）
    sample_companies = reference_data.get('sample_companies', [])
    quadrant_colors = {
        'ideal': '#28a745',
        'challenge': '#ffc107', 
        'stable': '#17a2b8',
        'caution': '#dc3545'
    }
    
    for comp in sample_companies:
        color = quadrant_colors.get(comp['quadrant'], '#999999')
        plt.scatter(
            comp['growth_score'], 
            comp['stability_score'],
            c=color,
            alpha=0.3,
            s=30,
            edgecolors='white',
            linewidth=0.5
        )
    
    # 象限の背景色を設定
    plt.axhspan(50, 100, 50, 100, alpha=0.1, color='green', label='理想企業エリア')
    plt.axhspan(0, 50, 50, 100, alpha=0.1, color='orange', label='チャレンジエリア')
    plt.axhspan(50, 100, 0, 50, alpha=0.1, color='blue', label='安定企業エリア')
    plt.axhspan(0, 50, 0, 50, alpha=0.1, color='red', label='要注意エリア')
    
    # 中央線を引く
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    plt.axvline(x=50, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # 対象企業をハイライト
    target_color = quadrant_colors[company_analysis['quadrant']]
    target_x = company_analysis['growth_score']
    target_y = company_analysis['stability_score']
    
    plt.scatter(
        target_x,
        target_y,
        c=target_color,
        s=500,  # サイズを少し大きく
        marker='*',
        edgecolors='black',
        linewidth=3,
        label='当該企業',
        zorder=10
    )
    
    # 企業名アノテーションの位置を動的に調整
    company_name = company_analysis['company_info']['company_name']
    company_year = company_analysis['company_info']['latest_year']
    
    # アノテーション位置の調整（見切れを防ぐ）
    annotation_x = target_x
    annotation_y = target_y
    
    # グラフ上部の企業の場合、アノテーションを下に配置
    if target_y > 85:
        xytext = (0, -25)  # 下に配置
        va = 'top'
    elif target_y < 15:
        xytext = (0, 25)   # 上に配置
        va = 'bottom'
    else:
        # 左右の調整
        if target_x > 75:
            xytext = (-25, 10)  # 左に配置
            ha = 'right'
        else:
            xytext = (25, 10)   # 右に配置
            ha = 'left'
        va = 'center'
    
    # デフォルトの水平配置
    if 'ha' not in locals():
        ha = 'center'
    
    plt.annotate(
        f"{company_name}\n({company_year}年)",
        xy=(target_x, target_y),
        xytext=xytext,
        textcoords='offset points',
        fontsize=11,
        fontweight='bold',
        fontproperties=font_prop,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
        ha=ha,
        va=va,
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='black', alpha=0.7)
    )
    
    # 軸ラベルと象限ラベル
    plt.xlabel('成長性スコア', fontproperties=font_prop, fontsize=14, fontweight='bold')
    plt.ylabel('安定性スコア', fontproperties=font_prop, fontsize=14, fontweight='bold')
    plt.title('企業ポジショニングマップ（成長性 × 安定性）', 
              fontproperties=font_prop, fontsize=16, fontweight='bold', pad=20)
    
    # 象限ラベルを追加（位置を微調整）
    plt.text(75, 85, '理想企業\n(高成長×高安定)', ha='center', va='center', 
             fontproperties=font_prop, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    plt.text(25, 85, '安定企業\n(低成長×高安定)', ha='center', va='center',
             fontproperties=font_prop, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.text(75, 15, 'チャレンジ企業\n(高成長×不安定)', ha='center', va='center',
             fontproperties=font_prop, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    plt.text(25, 15, '要注意企業\n(低成長×不安定)', ha='center', va='center',
             fontproperties=font_prop, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    # 軸の範囲と目盛り設定（少し余裕を持たせる）
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.xticks(range(0, 101, 20), fontproperties=font_prop)
    plt.yticks(range(0, 101, 20), fontproperties=font_prop)
    
    # グリッド
    plt.grid(True, alpha=0.3)
    
    # 業界平均の表示（もしデータがあれば）
    industry_stats = reference_data.get('industry_stats', {})
    if industry_stats:
        avg_growth = industry_stats.get('avg_growth', 50)
        avg_stability = industry_stats.get('avg_stability', 50)
        plt.scatter(avg_growth, avg_stability, c='purple', s=200, marker='D',
                   edgecolors='black', linewidth=2, label='業界平均', zorder=5)
    
    # 凡例（位置を調整）
    plt.legend(prop=font_prop, loc='upper left', fontsize=10, framealpha=0.9)
    
    # 説明テキスト（位置を調整）
    description = f"""成長性: 売上成長率、従業員成長率、R&D集約度
安定性: 自己資本比率、営業利益率安定性、ROA安定性
参考企業数: {len(sample_companies)}社（全企業データを使用）"""
    
    plt.figtext(0.02, 0.02, description, fontsize=9, fontproperties=font_prop,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # レイアウト調整（余白を確保）
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.95)
    
    # Base64エンコード
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')


def get_quadrant_recommendations(quadrant, target_edinet_code, limit=5):
    """同じ象限の企業を推薦（既存クラスタリングを活用）"""
    try:
        # 同じ象限の企業を検索
        recommendations = []
        
        # 全企業から同じ象限のものを抽出
        all_companies = FinancialData.objects.values('edinet_code').distinct()
        
        for company in all_companies:
            if company['edinet_code'] != target_edinet_code:
                try:
                    analysis = calculate_growth_stability_scores(company['edinet_code'])
                    if analysis and analysis['quadrant'] == quadrant:
                        recommendations.append({
                            'edinet_code': analysis['company_info']['edinet_code'],
                            'company_name': analysis['company_info']['company_name'],
                            'growth_score': analysis['growth_score'],
                            'stability_score': analysis['stability_score'],
                            'latest_year': analysis['company_info']['latest_year'],
                            'net_sales_billion': analysis['company_info']['net_sales_billion']
                        })
                    
                    if len(recommendations) >= limit:
                        break
                except Exception as e:
                    # 個別企業のエラーはスキップして続行
                    continue
        
        # スコアでソート
        recommendations.sort(key=lambda x: (x['growth_score'] + x['stability_score']), reverse=True)
        
        return recommendations[:limit]
        
    except Exception as e:
        print(f"Recommendations error: {e}")
        return []


def generate_positioning_interpretation(company_analysis, quadrant_info):
    """ポジショニング分析の解釈テキストを生成"""
    growth_score = company_analysis['growth_score']
    stability_score = company_analysis['stability_score']
    company_name = company_analysis['company_info']['company_name']
    quadrant = company_analysis['quadrant']
    
    # スコアレベルの判定
    def score_level(score):
        if score >= 80:
            return "非常に高い"
        elif score >= 60:
            return "高い"
        elif score >= 40:
            return "中程度"
        elif score >= 20:
            return "低い"
        else:
            return "非常に低い"
    
    growth_level = score_level(growth_score)
    stability_level = score_level(stability_score)
    
    interpretation = f"""
    {company_name}の二軸分析結果：
    
    ■ 成長性スコア: {growth_score:.1f}点 ({growth_level})
    　- 売上高成長率: {company_analysis['detailed_metrics']['sales_growth_rate']*100:.1f}%
    　- 従業員数成長率: {company_analysis['detailed_metrics']['employee_growth_rate']*100:.1f}%
    　- R&D集約度: {company_analysis['detailed_metrics']['rd_intensity']*100:.1f}%
    
    ■ 安定性スコア: {stability_score:.1f}点 ({stability_level})
    　- 自己資本比率: {company_analysis['detailed_metrics']['equity_ratio']*100:.1f}%
    　- 営業利益率安定性: {company_analysis['detailed_metrics']['operating_margin_stability']:.2f}
    　- ROA安定性: {company_analysis['detailed_metrics']['roa_stability']:.2f}
    
    ■ 企業分類: {quadrant_info['name']} ({quadrant_info['description']})
    　リスクレベル: {quadrant_info['risk_level']}
    　推奨度: {quadrant_info['recommendation']}
    
    ■ キャリアアドバイス:
    　{quadrant_info['career_advice']}
    """
    
    return interpretation.strip()
