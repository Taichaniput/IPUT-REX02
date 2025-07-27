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
import hashlib


class PredictionService:
    def __init__(self):
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'MS Gothic'
            self.font_path = None
        else:
            self.font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts', 'ipaexg.ttf')
            plt.rcParams['font.family'] = 'IPAPGothic'
        plt.rcParams['axes.unicode_minus'] = False

    def analyze_financial_predictions(self, financial_data):
        results = {}
        
        df_data = []
        for item in financial_data:
            fd = item['data'] if isinstance(item, dict) and 'data' in item else item
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
        
        metrics = ['net_sales', 'net_income']
        
        for metric in metrics:
            if metric not in df.columns or df[metric].isna().all():
                continue
                
            valid_data = df.dropna(subset=[metric])
            if len(valid_data) < 3:
                continue
            
            years = valid_data['fiscal_year'].values
            values = valid_data[metric].values / 100000000
            
            scenarios = self._predict_arima(years, values)
            
            chart_data = self._generate_chart_data_for_js(
                years, values, scenarios,
                f"{get_label(metric)}の3シナリオ予測",
                get_label(metric) + "（億円）"
            )
            
            results[metric] = {
                'predictions': {
                    'scenarios': scenarios
                },
                'chart_data': chart_data,
                'label': get_label(metric)
            }
        
        return results

    def _predict_arima(self, years, values):
        try:
            if len(values) < 5:
                return None
            
            clean_values = [v for v in values if v is not None and v > 0]
            if len(clean_values) < 5:
                return None
            
            model_key = self._generate_model_key(clean_values)
            cached_model = self._load_model(model_key)
            
            if cached_model:
                results = cached_model
            else:
                model = ARIMA(clean_values, order=(1, 1, 1))
                results = model.fit()
                self._save_model(model_key, results)
            
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

    def _generate_model_key(self, values):
        data_str = ','.join([f"{v:.2f}" for v in values])
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def _load_model(self, model_key):
        try:
            model_path = f"arima_cache/{model_key}.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Model loading error: {e}")
        return None

    def _save_model(self, model_key, model):
        try:
            os.makedirs('arima_cache', exist_ok=True)
            model_path = f"arima_cache/{model_key}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Model saving error: {e}")

    def _generate_chart_data_for_js(self, actual_years, actual_values, scenarios, title, ylabel):
        all_years = list(actual_years)

        datasets = [
            {
                'label': '実績',
                'data': actual_values.tolist(),
                'borderColor': 'blue',
                'backgroundColor': 'blue',
                'fill': False,
                'tension': 0.1
            }
        ]
        
        if scenarios: # scenariosがNoneでない場合のみ処理
            colors = {'optimistic': '#28a745', 'current': '#17a2b8', 'pessimistic': '#ffc107'}
            scenario_names = {'optimistic': '楽観', 'current': '現状', 'pessimistic': '悲観'}
            
            for scenario_name, scenario_data in scenarios.items():
                if scenario_data:
                    datasets.append({
                        'label': f"{scenario_names[scenario_name]}（年率{scenario_data['growth_rate']:.1f}%）",
                        'data': scenario_data['values'],
                        'borderColor': colors[scenario_name],
                        'backgroundColor': colors[scenario_name],
                        'fill': False,
                        'borderDash': [5, 5],
                        'tension': 0.1
                    })
                    # シナリオの年度をall_yearsに追加
                    all_years.extend(scenario_data['years'])
            
            # 全ての年度をソートして重複を削除
            all_years = sorted(list(set(all_years)))

        return {
            'labels': all_years,
            'datasets': datasets,
            'title': title,
            'ylabel': ylabel
        }


class ClusteringService:
    def __init__(self):
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'MS Gothic'
            self.font_path = None
        else:
            self.font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts', 'ipaexg.ttf')
            plt.rcParams['font.family'] = 'IPAPGothic'
        plt.rcParams['axes.unicode_minus'] = False

    async def get_company_cluster_info(self, edinet_code):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self._get_cluster_sync, edinet_code)

    def _get_cluster_sync(self, edinet_code):
        try:
            latest_year_subquery = FinancialData.objects.filter(
                edinet_code=OuterRef('edinet_code')
            ).values('edinet_code').annotate(
                max_year=Max('fiscal_year')
            ).values('max_year')[:1]
            
            latest_data = FinancialData.objects.filter(
                fiscal_year=Subquery(latest_year_subquery)
            )
            
            FEATURES = [
                'net_assets', 'total_assets', 'net_income', 'r_and_d_expenses', 'number_of_employees'
            ]
            
            df = pd.DataFrame.from_records(
                latest_data.values('edinet_code', 'filer_name', 'fiscal_year', *FEATURES)
            )
            df.set_index('edinet_code', inplace=True)
            
            df_filled = df.dropna(subset=FEATURES, how='all')
            
            for feature in FEATURES:
                if feature in df_filled.columns:
                    df_filled[feature] = df_filled[feature].fillna(df_filled[feature].median())
            
            numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
            df_filled[numeric_columns] = df_filled[numeric_columns].replace([np.inf, -np.inf], np.nan)
            df_filled[numeric_columns] = df_filled[numeric_columns].fillna(df_filled[numeric_columns].median())
            
            if len(df_filled) < 3 or edinet_code not in df_filled.index:
                return None
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_filled[FEATURES])
            
            umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            X_umap = umap_reducer.fit_transform(X_scaled)
            
            clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=8, metric='euclidean')
            labels = clusterer.fit_predict(X_umap)
            
            from sklearn.neighbors import NearestNeighbors
            noise_mask = labels == -1
            if noise_mask.any():
                cluster_mask = labels != -1
                if cluster_mask.any():
                    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_umap[cluster_mask])
                    _, nearest_indices = nbrs.kneighbors(X_umap[noise_mask])
                    cluster_labels_only = labels[cluster_mask]
                    for i, nearest_idx in enumerate(nearest_indices.flatten()):
                        noise_indices = np.where(noise_mask)[0]
                        labels[noise_indices[i]] = cluster_labels_only[nearest_idx]
            
            df_filled['cluster'] = labels
            
            company_cluster = df_filled.loc[edinet_code, 'cluster']
            company_year = df_filled.loc[edinet_code, 'fiscal_year']
            
            if company_cluster == -1:
                print(f"Warning: Company {edinet_code} still classified as noise after nearest neighbor assignment")
                company_cluster = 0
            
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
            
            cluster_means = df_filled[df_filled['cluster'] == company_cluster][FEATURES].mean()
            overall_means = df_filled[FEATURES].mean()
            
            relative_strengths = (cluster_means / overall_means - 1) * 100
            top_features = relative_strengths.abs().nlargest(3).index.tolist()
            
            umap_interpretation = self._interpret_umap(FEATURES)
            
            chart_data = self._create_cluster_chart(
                X_umap, labels, df_filled.index, df_filled['fiscal_year'].to_dict(),
                edinet_code, umap_interpretation
            )
            
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
                'chart_data': chart_data,
                'company_year': company_year,
                'umap_interpretation': umap_interpretation
            }
            
        except Exception as e:
            print(f"Cluster analysis error: {e}")
            return None

    def _interpret_umap(self, features):
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

    def _create_cluster_chart(self, X_umap, labels, index, year_dict, target_code, umap_interpretation):
        umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'], index=index)
        umap_df['cluster'] = labels
        umap_df['year'] = pd.Series(year_dict)
        
        unique_clusters = sorted(set(labels))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        datasets = []

        # 各クラスタのデータセット
        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = umap_df[umap_df['cluster'] == cluster_id]
            
            if not cluster_data.empty:
                datasets.append({
                    'label': f'クラスタ{cluster_id}',
                    'data': cluster_data[['UMAP1', 'UMAP2']].values.tolist(),
                    'backgroundColor': colors[i % len(colors)] + '80', # 50% opacity
                    'borderColor': colors[i % len(colors)],
                    'borderWidth': 1,
                    'pointRadius': 5,
                    'pointHoverRadius': 7,
                    'pointStyle': 'circle',
                    'parsing': {
                        'xAxisKey': '0',
                        'yAxisKey': '1'
                    }
                })

        # 対象企業をハイライト
        if target_code in umap_df.index:
            target_data = umap_df.loc[target_code]
            datasets.append({
                'label': '当該企業',
                'data': [{'x': target_data['UMAP1'], 'y': target_data['UMAP2']}],
                'backgroundColor': 'red',
                'borderColor': 'black',
                'borderWidth': 2,
                'pointRadius': 10,
                'pointHoverRadius': 12,
                'pointStyle': 'star',
                'parsing': {
                    'xAxisKey': 'x',
                    'yAxisKey': 'y'
                }
            })
        
        num_clusters = len([c for c in unique_clusters if c != -1])
        
        description_text = f"""
使用特徴量: {', '.join([f['name'] for f in umap_interpretation['features_used'][:3]])}等 {len(umap_interpretation['features_used'])}個
次元削減: UMAP
クラスタリング: HDBSCAN + 最近傍割り当て
発見クラスタ数: {num_clusters}個 (全企業をクラスタに割り当て)
        """

        return {
            'datasets': datasets,
            'title': '企業の財務特性に基づくクラスタリング分析',
            'x_axis_label': 'UMAP 次元1',
            'y_axis_label': 'UMAP 次元2',
            'description': description_text
        }


class PositioningService:
    def __init__(self):
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'MS Gothic'
            self.font_path = None
        else:
            self.font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts', 'ipaexg.ttf')
            plt.rcParams['font.family'] = 'IPAPGothic'
        plt.rcParams['axes.unicode_minus'] = False

    async def get_company_positioning_analysis(self, edinet_code):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self._get_positioning_sync, edinet_code)

    def _get_positioning_sync(self, edinet_code):
        try:
            print(f"Starting positioning analysis for {edinet_code}")
            
            company_analysis = self._calculate_growth_stability_scores(edinet_code)
            if not company_analysis:
                print(f"Failed to calculate growth stability scores for {edinet_code}")
                return {'error': '二軸分析に必要な財務データが不足しています。最低3年分のデータが必要です。'}
            print(f"Company analysis completed: {company_analysis['company_info']['company_name']}")
            
            reference_data = self._get_reference_companies(edinet_code, sample_size=50)
            print(f"Reference data obtained: {len(reference_data.get('sample_companies', []))} companies")
            
            chart = self._create_positioning_map(company_analysis, reference_data, edinet_code)
            print("Positioning map created successfully")
            
            quadrant_info = self._get_quadrant_info(company_analysis['quadrant'])
            
            recommendations = self._get_quadrant_recommendations(company_analysis['quadrant'], edinet_code, limit=5)
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
                'interpretation': self._generate_positioning_interpretation(company_analysis, quadrant_info)
            }
            
            print("Positioning analysis completed successfully")
            return result
            
        except Exception as e:
            print(f"Positioning analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'二軸分析中に予期せぬエラーが発生しました: {str(e)}'}

    def _calculate_growth_stability_scores(self, edinet_code):
        try:
            financial_data = FinancialData.objects.filter(
                edinet_code=edinet_code
            ).order_by('-fiscal_year')[:5]

            print(f"DEBUG: _calculate_growth_stability_scores for {edinet_code} - financial_data count: {len(financial_data)}")

            if len(financial_data) < 3:
                print(f"DEBUG: _calculate_growth_stability_scores for {edinet_code} - Insufficient financial data (less than 3 years).")
                return None

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
            print(f"DEBUG: DataFrame head:\n{df.head()}")

            growth_score = self._calculate_growth_score(df)
            stability_score = self._calculate_stability_score(df)

            print(f"DEBUG: Growth Score: {growth_score}, Stability Score: {stability_score}")

            return {
                'growth_score': growth_score,
                'stability_score': stability_score,
                'company_info': {
                    'edinet_code': edinet_code,
                    'company_name': financial_data[0].filer_name,
                    'latest_year': financial_data[0].fiscal_year,
                    'net_sales_billion': (financial_data[0].net_sales or 0) / 100000000,
                    'employees': financial_data[0].number_of_employees or 0
                },
                'quadrant': self._determine_quadrant(growth_score, stability_score),
                'detailed_metrics': {
                    'sales_growth_rate': self._calculate_sales_growth_rate(df),
                    'employee_growth_rate': self._calculate_employee_growth_rate(df),
                    'rd_intensity': self._calculate_rd_intensity(df),
                    'equity_ratio': self._calculate_equity_ratio(df),
                    'operating_margin_stability': self._calculate_operating_margin_stability(df),
                    'roa_stability': self._calculate_roa_stability(df)
                }
            }

        except Exception as e:
            print(f"ERROR: Growth stability calculation error for {edinet_code}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_growth_score(self, df):
        scores = []
        
        sales_growth = self._calculate_sales_growth_rate(df)
        scores.append(self._normalize_growth_rate(sales_growth, 0.05, 0.20))
        
        employee_growth = self._calculate_employee_growth_rate(df)
        scores.append(self._normalize_growth_rate(employee_growth, 0.03, 0.15))
        
        rd_intensity = self._calculate_rd_intensity(df)
        scores.append(self._normalize_rd_intensity(rd_intensity, 0.02, 0.10))
        
        weighted_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)
        return min(100, max(0, weighted_score))

    def _calculate_stability_score(self, df):
        scores = []
        
        equity_ratio = self._calculate_equity_ratio(df)
        scores.append(self._normalize_ratio(equity_ratio, 0.30, 0.70))
        
        margin_stability = self._calculate_operating_margin_stability(df)
        scores.append(self._normalize_stability(margin_stability, 0.10, 0.02))
        
        roa_stability = self._calculate_roa_stability(df)
        scores.append(self._normalize_stability(roa_stability, 0.10, 0.02))
        
        weighted_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)
        return min(100, max(0, weighted_score))

    def _calculate_sales_growth_rate(self, df):
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

    def _calculate_employee_growth_rate(self, df):
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

    def _calculate_rd_intensity(self, df):
        latest = df.iloc[-1]
        if latest['net_sales'] > 0:
            return latest['r_and_d_expenses'] / latest['net_sales']
        return 0

    def _calculate_equity_ratio(self, df):
        latest = df.iloc[-1]
        if latest['total_assets'] > 0:
            return latest['net_assets'] / latest['total_assets']
        return 0

    def _calculate_operating_margin_stability(self, df):
        margins = []
        for _, row in df.iterrows():
            if row['net_sales'] > 0:
                margin = row['operating_income'] / row['net_sales']
                margins.append(margin)
        
        if len(margins) < 2:
            return 0
        
        cv = np.std(margins) / np.mean(margins) if np.mean(margins) != 0 else float('inf')
        return 1 / (1 + cv)

    def _calculate_roa_stability(self, df):
        roas = []
        for _, row in df.iterrows():
            if row['total_assets'] > 0:
                roa = row['net_income'] / row['total_assets']
                roas.append(roa)
        
        if len(roas) < 2:
            return 0
        
        cv = np.std(roas) / np.mean(roas) if np.mean(roas) != 0 else float('inf')
        return 1 / (1 + cv)

    def _normalize_growth_rate(self, rate, mid_point, max_point):
        if rate <= 0:
            return 0
        elif rate <= mid_point:
            return 50 * (rate / mid_point)
        elif rate <= max_point:
            return 50 + 50 * ((rate - mid_point) / (max_point - mid_point))
        else:
            return 100

    def _normalize_ratio(self, ratio, min_point, max_point):
        if ratio <= min_point:
            return 0
        elif ratio >= max_point:
            return 100
        else:
            return 100 * ((ratio - min_point) / (max_point - min_point))

    def _normalize_rd_intensity(self, intensity, mid_point, max_point):
        if intensity <= 0:
            return 0
        elif intensity <= mid_point:
            return 50 * (intensity / mid_point)
        elif intensity <= max_point:
            return 50 + 50 * ((intensity - mid_point) / (max_point - mid_point))
        else:
            return 100

    def _normalize_stability(self, stability, min_point, max_point):
        if stability <= min_point:
            return 0
        elif stability >= max_point:
            return 100
        else:
            return 100 * ((stability - min_point) / (max_point - min_point))

    def _determine_quadrant(self, growth_score, stability_score):
        growth_high = growth_score >= 50
        stability_high = stability_score >= 50
        
        if growth_high and stability_high:
            return "ideal"
        elif growth_high and not stability_high:
            return "challenge"
        elif not growth_high and stability_high:
            return "stable"
        else:
            return "caution"

    def _get_quadrant_info(self, quadrant):
        quadrant_details = {
            "ideal": {
                "name": "理想企業",
                "description": "高成長×高安定",
                "color": "#28a745",
                "career_advice": "就活生に最適。成長とキャリア安定性を両立できる理想的な企業です。",
                "risk_level": "低",
                "recommendation": "強く推奨"
            },
            "challenge": {
                "name": "チャレンジ企業", 
                "description": "高成長×不安定",
                "color": "#ffc107",
                "career_advice": "ハイリスク・ハイリターン。急成長の可能性があるが、安定性にリスクがあります。",
                "risk_level": "中-高",
                "recommendation": "慎重に検討"
            },
            "stable": {
                "name": "安定企業",
                "description": "低成長×高安定", 
                "color": "#17a2b8",
                "career_advice": "着実なキャリア形成が可能ですが、急速な成長は期待できません。",
                "risk_level": "低",
                "recommendation": "安定志向に推奨"
            },
            "caution": {
                "name": "要注意企業",
                "description": "低成長×不安定",
                "color": "#dc3545",
                "career_advice": "慎重な検討が必要。成長性・安定性の両面でリスクがあります。",
                "risk_level": "高",
                "recommendation": "推奨しない"
            }
        }
        return quadrant_details.get(quadrant, quadrant_details["caution"])

    def _get_reference_companies(self, target_edinet_code, sample_size=None):
        try:
            all_companies = FinancialData.objects.values('edinet_code').distinct()
            
            sample_companies = []
            total_companies = all_companies.count()
            processed = 0
            
            print(f"Processing {total_companies} companies for reference data...")
            
            for company in all_companies:
                if company['edinet_code'] != target_edinet_code:
                    try:
                        analysis = self._calculate_growth_stability_scores(company['edinet_code'])
                        if analysis:
                            sample_companies.append(analysis)
                    except Exception as e:
                        print(f"Error processing {company['edinet_code']}: {e}")
                        continue
                    
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{total_companies} companies, found {len(sample_companies)} valid analyses")
                    
                    if sample_size and len(sample_companies) >= sample_size:
                        break
            
            print(f"Completed processing. Total valid companies: {len(sample_companies)}")
            
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

    def _create_positioning_map(self, company_analysis, reference_data, target_edinet_code):
        sample_companies = reference_data.get('sample_companies', [])
        quadrant_colors = {
            'ideal': '#28a745',
            'challenge': '#ffc107', 
            'stable': '#17a2b8',
            'caution': '#dc3545'
        }

        datasets = []

        # 参考企業データセット
        if sample_companies:
            # 各象限の企業を個別のデータセットとして追加
            quadrant_data = {q: [] for q in quadrant_colors.keys()}
            for comp in sample_companies:
                quadrant_data[comp['quadrant']].append({
                    'x': comp['growth_score'],
                    'y': comp['stability_score'],
                    'name': comp['company_info']['company_name']
                })
            
            for q_name, q_color in quadrant_colors.items():
                if quadrant_data[q_name]:
                    datasets.append({
                        'label': f"{self._get_quadrant_info(q_name)['name']} ({len(quadrant_data[q_name])}社)",
                        'data': quadrant_data[q_name],
                        'backgroundColor': q_color + '4D', # 30% opacity
                        'borderColor': q_color,
                        'borderWidth': 1,
                        'pointRadius': 5,
                        'pointHoverRadius': 7,
                        'pointStyle': 'circle',
                        'parsing': {
                            'xAxisKey': 'x',
                            'yAxisKey': 'y'
                        }
                    })

        # 対象企業データセット
        target_color = quadrant_colors[company_analysis['quadrant']]
        datasets.append({
            'label': company_analysis['company_info']['company_name'],
            'data': [{
                'x': company_analysis['growth_score'],
                'y': company_analysis['stability_score'],
                'name': company_analysis['company_info']['company_name']
            }],
            'backgroundColor': target_color,
            'borderColor': 'black',
            'borderWidth': 2,
            'pointRadius': 10,
            'pointHoverRadius': 12,
            'pointStyle': 'star',
            'parsing': {
                'xAxisKey': 'x',
                'yAxisKey': 'y'
            }
        })

        # 業界平均データセット
        industry_stats = reference_data.get('industry_stats', {})
        if industry_stats:
            datasets.append({
                'label': '業界平均',
                'data': [{
                    'x': industry_stats.get('avg_growth', 50),
                    'y': industry_stats.get('avg_stability', 50),
                    'name': '業界平均'
                }],
                'backgroundColor': 'purple',
                'borderColor': 'black',
                'borderWidth': 2,
                'pointRadius': 8,
                'pointHoverRadius': 10,
                'pointStyle': 'triangle',
                'parsing': {
                    'xAxisKey': 'x',
                    'yAxisKey': 'y'
                }
            })

        return {
            'labels': ['成長性スコア', '安定性スコア'], # Chart.jsでは散布図の場合、labelsは軸のタイトルとして使われる
            'datasets': datasets,
            'title': '企業ポジショニングマップ（成長性 × 安定性）',
            'x_axis_label': '成長性スコア',
            'y_axis_label': '安定性スコア',
            'quadrant_lines': [
                {'axis': 'x', 'value': 50, 'color': 'gray', 'dash': [5, 5]},
                {'axis': 'y', 'value': 50, 'color': 'gray', 'dash': [5, 5]}
            ],
            'quadrant_areas': [
                {'x_min': 50, 'x_max': 100, 'y_min': 50, 'y_max': 100, 'color': 'rgba(0, 128, 0, 0.1)', 'label': '理想企業エリア'},
                {'x_min': 0, 'x_max': 50, 'y_min': 50, 'y_max': 100, 'color': 'rgba(255, 165, 0, 0.1)', 'label': 'チャレンジエリア'},
                {'x_min': 50, 'x_max': 100, 'y_min': 0, 'y_max': 50, 'color': 'rgba(0, 0, 255, 0.1)', 'label': '安定企業エリア'},
                {'x_min': 0, 'x_max': 50, 'y_min': 0, 'y_max': 50, 'color': 'rgba(255, 0, 0, 0.1)', 'label': '要注意エリア'}
            ]
        }

    def _get_quadrant_recommendations(self, quadrant, target_edinet_code, limit=5):
        try:
            recommendations = []
            
            all_companies = FinancialData.objects.values('edinet_code').distinct()
            
            for company in all_companies:
                if company['edinet_code'] != target_edinet_code:
                    try:
                        analysis = self._calculate_growth_stability_scores(company['edinet_code'])
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
                        continue
            
            recommendations.sort(key=lambda x: (x['growth_score'] + x['stability_score']), reverse=True)
            
            return recommendations[:limit]
            
        except Exception as e:
            print(f"Recommendations error: {e}")
            return []

    def _generate_positioning_interpretation(self, company_analysis, quadrant_info):
        growth_score = company_analysis['growth_score']
        stability_score = company_analysis['stability_score']
        company_name = company_analysis['company_info']['company_name']
        quadrant = company_analysis['quadrant']
        
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


def get_label(metric):
    labels = {
        'net_sales': '売上高',
        'operating_income': '営業利益', 
        'net_income': '純利益',
        'total_assets': '総資産'
    }
    return labels.get(metric, metric)


def get_feature_label(feature):
    labels = {
        'net_assets': '純資産',
        'total_assets': '総資産',
        'net_sales': '売上高',
        'operating_income': '営業利益',
        'ordinary_income': '経常利益',
        'net_income': '純利益',
        'operating_cash_flow': '営業キャッシュフロー',
        'r_and_d_expenses': '研究開発費',
        'number_of_employees': '従業員数',
        
        'roe': 'ROE（自己資本利益率）',
        'roa': 'ROA（総資産利益率）',
        'operating_margin': '営業利益率',
        'equity_ratio': '自己資本比率',
        'rd_intensity': 'R&D集約度',
        'asset_turnover': '総資産回転率',
        'employee_productivity': '従業員1人当たり売上高'
    }
    return labels.get(feature, feature)
