import google.generativeai as genai
from django.conf import settings
from tavily import TavilyClient

from .ml_analytics import get_label, get_feature_label, PredictionService, ClusteringService, PositioningService


class AIAnalysisService:
    def __init__(self):
        self.prediction_service = PredictionService()
        self.clustering_service = ClusteringService()
        self.positioning_service = PositioningService()

    def _get_company_additional_info(self, company_name):
        additional_info = {}
        try:
            tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)
            search_query = f"{company_name}の事業内容、技術的特徴、IT・デジタル変革の取り組み、エンジニア採用状況、市場でのポジショニング、成長戦略について包括的に調査し、情報系学生の就活に有用な情報を要約してください。"
            response = tavily.search(
                query=search_query,
                search_depth="advanced",
                max_results=5
            )
            summary = "\n".join([f"- {obj['content']}" for obj in response['results']])
            additional_info['web_search_summary'] = summary if summary else "ウェブ検索で関連情報が見つかりませんでした。"
        except Exception as e:
            print(f"Tavily search error: {e}")
            additional_info['web_search_summary'] = "ウェブ検索中にエラーが発生しました。"
        return additional_info

    def generate_comprehensive_analysis(self, company_name, edinet_code, financial_data, prediction_results, cluster_info, positioning_info=None):
        if getattr(settings, 'AI_DEBUG_MODE', False):
            return self._generate_debug_analysis()
        
        if not settings.GEMINI_API_KEY:
            return self._create_fallback_analysis(company_name, financial_data, prediction_results, cluster_info, positioning_info)
        
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(settings.GEMINI_MODEL)
            
            additional_info = self._get_company_additional_info(company_name)
            financial_summary = self._prepare_financial_summary(financial_data)
            prediction_summary = self._prepare_prediction_summary(prediction_results)
            cluster_summary = self._prepare_cluster_summary(cluster_info)
            positioning_summary = self._prepare_positioning_summary(positioning_info)
            
            prompt = self._build_comprehensive_analysis_prompt(
                company_name, financial_summary, prediction_summary, 
                cluster_summary, additional_info, positioning_summary
            )
            
            response = model.generate_content(prompt)
            
            if not response or not response.text:
                print("ERROR: Gemini APIからの応答がありません")
                return {"error": "Gemini APIからの応答がありません"}
            
            print(f"DEBUG: Raw Gemini comprehensive analysis response: {response.text[:500]}...") # Log first 500 chars
            print(f"Gemini response received: {len(response.text)} characters")
            
            structured_analysis = self._parse_structured_analysis(response.text)
            print(f"DEBUG: Parsed comprehensive analysis sections: {structured_analysis.keys()}")
            
            if 'error' in structured_analysis:
                print(f"ERROR: Analysis parsing failed: {structured_analysis['error']}")
                return structured_analysis
            
            return structured_analysis
            
        except Exception as e:
            print(f"AI analysis generation error: {e}")
            return self._create_fallback_analysis(company_name, financial_data, prediction_results, cluster_info, positioning_info)

    def _generate_debug_analysis(self):
        print("--- [DEBUG] Skipping external API calls and returning dummy data. ---")
        return {
            'FINANCIAL_ANALYSIS': '【デバッグ】財務分析データを表示します。API呼び出しはスキップされました。',
            'COMPANY_OVERVIEW': '【デバッグ】企業概要を表示します。Tavily API呼び出しはスキップされました。',
            'SALES_SCENARIOS': {
                'optimistic': '【デバッグ】売上高（楽観）シナリオです。',
                'current': '【デバッグ】売上高（現状）シナリオです。',
                'pessimistic': '【デバッグ】売上高（悲観）シナリオです。'
            },
            'PROFIT_SCENARIOS': {
                'optimistic': '【デバッグ】純利益（楽観）シナリオです。',
                'current': '【デバッグ】純利益（現状）シナリオです。',
                'pessimistic': '【デバッグ】純利益（悲観）シナリオです。'
            },
            'CHART_SALES_SCENARIOS': {
                'optimistic': '【デバッグ】グラフ用売上高（楽観）',
                'current': '【デバッグ】グラフ用売上高（現状）',
                'pessimistic': '【デバッグ】グラフ用売上高（悲観）'
            },
            'CHART_PROFIT_SCENARIOS': {
                'optimistic': '【デバッグ】グラフ用純利益（楽観）',
                'current': '【デバッグ】グラフ用純利益（現状）',
                'pessimistic': '【デバッグ】グラフ用純利益（悲観）'
            },
            'POSITIONING_ANALYSIS': '【デバッグ】ポジショニング分析を表示します。',
            'SUMMARY': '【デバッグ】総括・キャリア分析を表示します。'
        }

    def _create_fallback_analysis(self, company_name, financial_data, prediction_results, cluster_info, positioning_info=None):
        analysis = {}
        
        if not prediction_results and financial_data:
            edinet_code = self._extract_edinet_code(financial_data)
            if edinet_code:
                print(f"Fallback: generating predictions for {edinet_code}")
                prediction_results = self.prediction_service.analyze_financial_predictions(financial_data)
        
        if not cluster_info and financial_data:
            edinet_code = self._extract_edinet_code(financial_data)
            if edinet_code:
                print(f"Fallback: generating clustering for {edinet_code}")
                cluster_info = self.clustering_service._get_cluster_sync(edinet_code) # Use sync version for fallback

        if financial_data:
            latest = financial_data[0]['data'] if isinstance(financial_data[0], dict) and 'data' in financial_data[0] else financial_data[0]
            net_sales = (latest.net_sales or 0) / 100000000
            financial_text = f"{company_name}は{latest.fiscal_year}年に売上高{net_sales:.1f}億円を記録。"
            if positioning_info:
                quadrant_info = positioning_info.get('quadrant_info', {})
                financial_text += f"二軸分析では{quadrant_info.get('name', '不明')}に分類され、{quadrant_info.get('career_advice', 'AI分析をご利用ください。')}"
            else:
                financial_text += "財務の安定性と成長性を評価するためにはAI分析をご利用ください。"
            analysis['FINANCIAL_ANALYSIS'] = financial_text
        else:
            analysis['FINANCIAL_ANALYSIS'] = f"{company_name}の詳細な財務分析を表示するためには、API接続が必要です。"
        
        analysis['SALES_SCENARIOS'] = {
            'optimistic': "デジタル変革の進展により、AI・IoT・クラウド技術の活用が拡大し、新たな事業機会の創出と市場シェア拡大が期待されます。売上高は大幅な成長を見込むことができます。",
            'current': "現在の事業基盤を維持しながら、安定的な成長を継続する見通しです。技術投資とROIのバランスを保ちつつ、売上高の着実な成長が予想されます。",
            'pessimistic': "競合他社の技術革新により相対的な競争力低下のリスクがあります。デジタル変革の遅れや市場シェア低下により、売上高の成長に制約が生じる可能性があります。"
        }
        
        analysis['PROFIT_SCENARIOS'] = {
            'optimistic': "業務プロセスの自動化・効率化により大幅なコスト削減を実現し、高付加価値サービスの拡大により利益率向上が期待されます。技術革新による新収益源の創出も見込まれます。",
            'current': "現在の収益構造を維持しながら、適度な技術投資により収益性を保持する見通しです。既存事業の安定的な利益確保により、持続的な成長を支えると予想されます。",
            'pessimistic': "人件費や技術投資の増大により収益性に圧力がかかる可能性があります。価格競争の激化や市場シェア低下により、利益率の低下が懸念されます。"
        }
        
        analysis['CHART_SALES_SCENARIOS'] = {
            'optimistic': "デジタル変革の進展により、AI・IoT・クラウド技術の活用拡大で新事業機会創出と市場シェア拡大が期待されます。",
            'current': "現在の事業基盤を維持しながら、技術投資とROIのバランスを保ち、売上高の着実な成長が予想されます。",
            'pessimistic': "競合他社の技術革新により相対的な競争力低下のリスクがあり、市場シェア低下により売上成長に制約が生じる可能性があります。"
        }
        
        analysis['CHART_PROFIT_SCENARIOS'] = {
            'optimistic': "業務プロセスの自動化・効率化によりコスト削減を実現し、高付加価値サービス拡大により利益率向上が期待されます。",
            'current': "現在の収益構造を維持しながら、適度な技術投資により収益性を保持し、持続的な成長を支える見通しです。",
            'pessimistic': "人件費や技術投資の増大により収益性に圧力がかかり、価格競争の激化により利益率低下が懸念されます。"
        }
        
        if positioning_info:
            quadrant_info = positioning_info.get('quadrant_info', {})
            growth_score = positioning_info.get('growth_score', 0)
            stability_score = positioning_info.get('stability_score', 0)
            
            analysis['POSITIONING_ANALYSIS'] = f"""
            二軸分析結果: {quadrant_info.get('name', '不明')}（成長性{growth_score:.1f}点、安定性{stability_score:.1f}点）
            
            企業分類: {quadrant_info.get('description', '')}
            推奨度: {quadrant_info.get('recommendation', '')}
            リスクレベル: {quadrant_info.get('risk_level', '')}
            
            キャリアアドバイス: {quadrant_info.get('career_advice', '')}
            
            詳細な技術力評価、デジタル変革への取り組み状況については、AI分析で詳細をご確認ください。
            """
        elif cluster_info:
            analysis['POSITIONING_ANALYSIS'] = f"クラスタ{cluster_info['cluster_id']}に分類され、同業他社との比較において独自のポジションを占めています。技術力とイノベーション力の詳細な評価、デジタル変革への取り組み状況については、AI分析で詳細をご確認ください。"
        else:
            analysis['POSITIONING_ANALYSIS'] = "業界内でのポジショニング分析、競争優位性の評価、技術力の詳細な比較分析には、AI機能による包括的な分析が必要です。"
        
        if positioning_info:
            quadrant_info = positioning_info.get('quadrant_info', {})
            career_summary = f"{company_name}は{quadrant_info.get('name', '')}として分類され、{quadrant_info.get('career_advice', '')}"
        else:
            career_summary = f"{company_name}は情報系学生にとって技術的な成長機会を提供する可能性が高い企業です。"
        
        analysis['SUMMARY'] = f"{career_summary} エンジニアとしてのキャリアパス、スキル習得環境、長期的なキャリア展望についての詳細な分析は、AI機能をご利用ください。"
        
        analysis['COMPANY_OVERVIEW'] = f"{company_name}の事業内容、技術的特徴、競争優位性についての詳細な分析は、AI機能をご利用ください。"
        
        return analysis

    def _extract_edinet_code(self, financial_data):
        if not financial_data:
            return None
            
        if isinstance(financial_data[0], dict) and 'data' in financial_data[0]:
            return financial_data[0]['data'].edinet_code
        elif hasattr(financial_data[0], 'edinet_code'):
            return financial_data[0].edinet_code
        
        return None

    def _prepare_financial_summary(self, financial_data):
        if not financial_data:
            return "財務データなし"
        
        summary = []
        for item in financial_data[:3]:
            fd = item['data'] if isinstance(item, dict) and 'data' in item else item
                
            net_sales = fd.net_sales or 0
            net_income = fd.net_income or 0
            summary.append(f"{fd.fiscal_year}年: 売上{net_sales/100000000:.1f}億円, 純利益{net_income/100000000:.1f}億円")
        
        return "\n".join(summary)

    def _prepare_prediction_summary(self, prediction_results):
        if not prediction_results:
            return "予測データなし"
        
        summary = []
        for metric, result in prediction_results.items():
            if 'scenarios' in result.get('predictions', {}):
                scenarios = result['predictions']['scenarios']
                summary.append(f"{result['label']}: 楽観{scenarios['optimistic']['growth_rate']:.1f}%, 現状{scenarios['current']['growth_rate']:.1f}%, 悲観{scenarios['pessimistic']['growth_rate']:.1f}%")
        
        return "\n".join(summary)

    def _prepare_cluster_summary(self, cluster_info):
        if not cluster_info:
            return "クラスタデータなし"
        
        summary = f"当該企業はクラスタ{cluster_info['cluster_id']}/{cluster_info['total_clusters']}に分類\n"
        
        if 'cluster_characteristics' in cluster_info:
            summary += "クラスタの特徴:\n"
            for feat, data in cluster_info['cluster_characteristics'].items():
                feat_label = get_feature_label(feat)
                summary += f"- {feat_label}: 全体平均比{data['relative']:.1f}%\n"
        
        if 'same_cluster_companies' in cluster_info:
            companies = [comp['name'] for comp in cluster_info['same_cluster_companies'][:5]]
            summary += f"\n同クラスタの類似企業: {', '.join(companies)}\n"
        
        if 'umap_interpretation' in cluster_info:
            umap_info = cluster_info['umap_interpretation']
            summary += f"\n{umap_info['method']}による次元削減解釈:\n"
            summary += f"- {umap_info['description']}\n"
            for advantage in umap_info['advantages']:
                summary += f"- {advantage}\n"
        elif 'pca_interpretation' in cluster_info:
            summary += "\n主成分分析による解釈:\n"
            for comp in cluster_info['pca_interpretation']:
                summary += f"- 第{comp['component']}主成分({comp['meaning']}): 寄与率{comp['variance_ratio']:.1f}%\n"
        
        return summary

    def _prepare_positioning_summary(self, positioning_info):
        if not positioning_info:
            return "二軸分析データなし"
        
        growth_score = positioning_info.get('growth_score', 0)
        stability_score = positioning_info.get('stability_score', 0)
        quadrant_info = positioning_info.get('quadrant_info', {})
        detailed_metrics = positioning_info.get('detailed_metrics', {})
        
        summary = f"""二軸分析（成長性×安定性）による企業ポジショニング結果:

■ スコア
- 成長性スコア: {growth_score:.1f}/100点
- 安定性スコア: {stability_score:.1f}/100点

■ 企業分類
- 象限: {quadrant_info.get('name', '不明')} ({quadrant_info.get('description', '')})
- 推奨度: {quadrant_info.get('recommendation', '')}
- リスクレベル: {quadrant_info.get('risk_level', '')}

■ 詳細指標
- 売上高成長率: {detailed_metrics.get('sales_growth_rate', 0)*100:.1f}%
- 従業員数成長率: {detailed_metrics.get('employee_growth_rate', 0)*100:.1f}%
- R&D集約度: {detailed_metrics.get('rd_intensity', 0)*100:.1f}%
- 自己資本比率: {detailed_metrics.get('equity_ratio', 0)*100:.1f}%
- 営業利益率安定性: {detailed_metrics.get('operating_margin_stability', 0):.2f}
- ROA安定性: {detailed_metrics.get('roa_stability', 0):.2f}

■ キャリアアドバイス
{quadrant_info.get('career_advice', '')}

■ 同象限の推薦企業"""
        
        recommendations = positioning_info.get('recommendations', [])
        if recommendations:
            summary += "\n"
            for i, rec in enumerate(recommendations[:3], 1):
                summary += f"{i}. {rec.get('company_name', '')} (成長性{rec.get('growth_score', 0):.1f}点、安定性{rec.get('stability_score', 0):.1f}点)\n"
        else:
            summary += "\n（同象限の企業データなし）"
        
        return summary

    def _build_comprehensive_analysis_prompt(self, company_name, financial_summary, prediction_summary, cluster_summary, additional_info, positioning_summary=None):
        positioning_section = ""
        if positioning_summary and positioning_summary != "二軸分析データなし":
            positioning_section = f"""
## 企業ポジショニング分析（二軸分析: 成長性×安定性）
{positioning_summary}
"""
        
        prompt = f"""
あなたは情報系学生の就活支援を専門とする企業分析アナリストです。以下の企業について、情報系学生のキャリア形成の観点から構造化された分析を行ってください。

## 分析対象企業
{company_name}

## 財務データ
{financial_summary}

## 成長予測（ARIMA時系列モデル3シナリオ）
{prediction_summary}

## 業界ポジショニング・クラスタリング分析（UMAPベース）
{cluster_summary}
{positioning_section}
## 外部情報（検索結果）
{additional_info.get('web_search_summary', 'なし')}

## 指示
上記データを統合して情報系学生向けの就活分析を行ってください。**特に二軸分析による企業分類（成長性×安定性）を重視し、就活生の企業選択判断に直結する価値ある分析を提供してください。**

必ず以下の形式で出力してください。

[FINANCIAL_ANALYSIS]
財務データに基づく企業の特徴と健全性の簡潔な分析。二軸分析の結果も含めて企業の財務安全性と成長性を評価（100-150文字程度）
[/FINANCIAL_ANALYSIS]

[COMPANY_OVERVIEW]
Tavily検索結果に基づく企業の簡潔な説明。事業内容、主要サービス、業界での位置づけ、技術的特徴、競争優位性を含む（200-250文字程度）
[/COMPANY_OVERVIEW]

[SALES_SCENARIOS_OPTIMISTIC]
売上高楽観シナリオの詳細分析（150-200文字程度）：
・市場拡大要因（DX推進、新技術導入、規制緩和など）
・技術革新による事業機会（AI、IoT、クラウドなど）
・デジタル変革による競争優位性向上
・売上高成長のドライバー
[/SALES_SCENARIOS_OPTIMISTIC]

[SALES_SCENARIOS_CURRENT]
売上高現状シナリオの詳細分析（150-200文字程度）：
・現在のトレンド継続による安定成長
・既存事業の着実な拡大
・技術投資とROIのバランス
・売上高の安定的成長要因
[/SALES_SCENARIOS_CURRENT]

[SALES_SCENARIOS_PESSIMISTIC]
売上高悲観シナリオの詳細分析（150-200文字程度）：
・市場縮小・競合激化のリスク
・技術的遅れによる競争力低下
・デジタル変革の遅れ
・売上高減少のリスク要因
[/SALES_SCENARIOS_PESSIMISTIC]

[PROFIT_SCENARIOS_OPTIMISTIC]
純利益楽観シナリオの詳細分析（150-200文字程度）：
・業務効率化とコスト削減効果
・高付加価値サービス・製品の拡大
・技術革新による新収益源の創出
・デジタル化による利益率向上
[/PROFIT_SCENARIOS_OPTIMISTIC]

[PROFIT_SCENARIOS_CURRENT]
純利益現状シナリオの詳細分析（150-200文字程度）：
・現在の収益構造の継続
・既存事業の安定的な利益確保
・適度な技術投資による収益性維持
・市場での競争力維持
[/PROFIT_SCENARIOS_CURRENT]

[PROFIT_SCENARIOS_PESSIMISTIC]
純利益悲観シナリオの詳細分析（150-200文字程度）：
・コスト増加圧力（人件費、システム投資など）
・価格競争激化による利益率低下
・技術投資負担の増大
・市場シェア低下による収益性悪化
[/PROFIT_SCENARIOS_PESSIMISTIC]

[POSITIONING_ANALYSIS]
業界内ポジショニング分析（250-300文字程度）：
・**二軸分析による企業分類の意味と就活生へのインパクト**
・クラスタデータに基づく同業他社比較
・技術力・イノベーション力の評価
・市場での競争優位性と差別化要因
・デジタル変革への取り組み状況
・情報系人材の活用・育成環境
・業界トレンドへの適応力
・**同象限企業との比較による相対的優位性**
[/POSITIONING_ANALYSIS]

[SUMMARY]
情報系学生向けキャリア総括（300-350文字程度）：
・**二軸分析結果に基づく企業選択の推奨度と理由**
・この企業でのキャリア形成価値
・技術的成長機会とスキル習得環境
・エンジニアとしてのキャリアパス
・業界での将来性と安定性
・**リスクレベルに応じた就活戦略のアドバイス**
・長期的なキャリア展望
・**同象限の企業群との比較優位性**
[/SUMMARY]

各セクションは具体的で実用的な内容にし、特に情報系学生の視点から技術的成長機会、キャリア形成、業界トレンドを重視した分析を行ってください。

## 分析時の重要な視点（二軸分析統合版）
- **成長性×安定性の組み合わせが就活生のキャリア戦略に与える影響**
- **リスク許容度に応じた企業選択指針の提供**
- プログラミング言語、開発環境、技術スタックの言及
- エンジニアの成長環境（研修制度、勉強会、技術コミュニティ）
- 将来性の高い技術分野（AI、IoT、クラウド、DX）への取り組み
- 情報系学生が活躍できる職種・部門の具体的な説明
- 業界内での技術力・イノベーション力の客観的評価
- 長期的なキャリア形成の可能性（昇進、転職市場価値）
- **同象限企業群の中での相対的ポジション評価**
- **個人のキャリア志向（安定志向/成長志向/チャレンジ志向）との適合性**
"""
        return prompt

    def _parse_structured_analysis(self, response_text):
        sections = {}
        
        try:
            sections['FINANCIAL_ANALYSIS'] = self._extract_section(response_text, 'FINANCIAL_ANALYSIS')
            sections['COMPANY_OVERVIEW'] = self._extract_section(response_text, 'COMPANY_OVERVIEW')
            
            sections['SALES_SCENARIOS'] = {
                'optimistic': self._extract_section(response_text, 'SALES_SCENARIOS_OPTIMISTIC'),
                'current': self._extract_section(response_text, 'SALES_SCENARIOS_CURRENT'),
                'pessimistic': self._extract_section(response_text, 'SALES_SCENARIOS_PESSIMISTIC')
            }
            
            sections['PROFIT_SCENARIOS'] = {
                'optimistic': self._extract_section(response_text, 'PROFIT_SCENARIOS_OPTIMISTIC'),
                'current': self._extract_section(response_text, 'PROFIT_SCENARIOS_CURRENT'),
                'pessimistic': self._extract_section(response_text, 'PROFIT_SCENARIOS_PESSIMISTIC')
            }
            
            sections['CHART_SALES_SCENARIOS'] = {
                'optimistic': self._extract_section(response_text, 'CHART_SALES_OPTIMISTIC'),
                'current': self._extract_section(response_text, 'CHART_SALES_CURRENT'),
                'pessimistic': self._extract_section(response_text, 'CHART_SALES_PESSIMISTIC')
            }
            
            sections['CHART_PROFIT_SCENARIOS'] = {
                'optimistic': self._extract_section(response_text, 'CHART_PROFIT_OPTIMISTIC'),
                'current': self._extract_section(response_text, 'CHART_PROFIT_CURRENT'),
                'pessimistic': self._extract_section(response_text, 'CHART_PROFIT_PESSIMISTIC')
            }
            
            sections['POSITIONING_ANALYSIS'] = self._extract_section(response_text, 'POSITIONING_ANALYSIS')
            sections['SUMMARY'] = self._extract_section(response_text, 'SUMMARY')
            
        except Exception as e:
            print(f"Response parsing error: {e}")
            sections = {"error": "レスポンス解析中にエラーが発生しました"}
        
        print('DEBUG: Parsed sections content:', {k: v[:100] + '...' if isinstance(v, str) and len(v) > 100 else v for k, v in sections.items()})
        return sections

    def _extract_section(self, text, section_name):
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

    def generate_scenario_analysis(self, company_name, edinet_code, prediction_results, chart_type='sales'):
        if not settings.GEMINI_API_KEY:
            return self._create_fallback_scenario_analysis(company_name, chart_type)
        
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(settings.GEMINI_MODEL)
            
            chart_data = self._prepare_chart_data(prediction_results, chart_type)
            prompt = self._build_scenario_analysis_prompt(company_name, chart_data, chart_type)
            response = model.generate_content(prompt)
            
            if not response or not response.text:
                print("ERROR: Gemini APIからの応答がありません")
                return self._create_fallback_scenario_analysis(company_name, chart_type)
            
            print(f"DEBUG: Raw Gemini scenario analysis response ({chart_type}): {response.text[:500]}...")
            structured_analysis = self._parse_scenario_analysis(response.text)
            print(f"DEBUG: Parsed scenario analysis ({chart_type}): {structured_analysis.keys()}")
            
            return structured_analysis
            
        except Exception as e:
            print(f"Scenario analysis generation error: {e}")
            return self._create_fallback_scenario_analysis(company_name, chart_type)

    def _create_fallback_scenario_analysis(self, company_name, chart_type):
        chart_label = "売上高" if chart_type == 'sales' else "純利益"
        
        scenarios = {
            'optimistic': f"デジタル変革の進展により、AI・IoT・クラウド技術の活用が拡大し、新たな事業機会の創出と{chart_label}の大幅な成長が期待されます。情報系人材の積極採用により技術力向上が見込まれ、高付加価値サービスの展開で収益性も向上する可能性があります。",
            'current': f"現在の事業基盤を維持しながら、安定的な{chart_label}成長を継続する見通しです。技術投資とROIのバランスを保ちつつ、市場での競争力を維持していくと予想されます。既存の技術スタックを活用した着実な成長が見込まれます。",
            'pessimistic': f"競合他社の技術革新により相対的な競争力低下のリスクがあり、{chart_label}の成長に制約が生じる可能性があります。デジタル変革の遅れや優秀な人材確保の困難により、市場シェアの縮小や収益性の悪化が懸念されます。"
        }
        
        return scenarios

    def _prepare_chart_data(self, prediction_results, chart_type):
        if not prediction_results:
            return "予測データなし"
        
        target_metric = 'net_sales' if chart_type == 'sales' else 'net_income'
        
        if target_metric in prediction_results:
            result = prediction_results[target_metric]
            if 'scenarios' in result.get('predictions', {}):
                scenarios = result['predictions']['scenarios']
                return f"""
                楽観シナリオ: 年平均成長率 {scenarios['optimistic']['growth_rate']:.1f}%
                現状シナリオ: 年平均成長率 {scenarios['current']['growth_rate']:.1f}%
                悲観シナリオ: 年平均成長率 {scenarios['pessimistic']['growth_rate']:.1f}%
                """
        
        return "該当する予測データが見つかりません"

    def _build_scenario_analysis_prompt(self, company_name, chart_data, chart_type):
        chart_label = "売上高" if chart_type == 'sales' else "純利益"
        
        prompt = f"""
あなたは情報系学生の就活支援を専門とする企業分析アナリストです。以下の企業について、ARIMA時系列モデルによる{chart_label}予測グラフに基づく3シナリオ分析を行ってください。

## 分析対象企業
{company_name}

## {chart_label}予測データ（ARIMAモデル単体予測）
{chart_data}

## 予測手法について
- **ARIMA時系列モデル**: 企業の過去データから時系列パターンを学習し、将来の予測値と信頼区間を算出
- **パラメータ**: order=(1,1,1)固定による安定した予測
- **3シナリオ構成**: 楽観(信頼区間上限)、現状(予測中央値)、悲観(信頼区間下限)

## 指示
上記のARIMAベース{chart_label}予測データを基に、情報系学生のキャリア形成の観点から3つのシナリオ分析を行ってください。必ず以下の形式で出力してください。

[OPTIMISTIC_SCENARIO]
楽観シナリオ（ARIMA信頼区間上限、150文字程度）：
・ARIMAモデルが予測する最良のケースでの{chart_label}成長
・時系列パターンから見た上振れ要因（DX推進、新技術導入、規制緩和など）
・技術革新による事業機会の最大化（AI、IoT、クラウド活用）
・情報系人材の積極採用と技術力向上による競争優位性確立
[/OPTIMISTIC_SCENARIO]

[CURRENT_SCENARIO]
現状シナリオ（ARIMA予測中央値、150文字程度）：
・ARIMAモデルによる最も可能性の高い{chart_label}成長予測
・過去の時系列パターンに基づく安定的な事業拡大
・既存技術と新技術のバランスの良い投資継続
・市場ポジション維持と漸進的な技術力向上
[/CURRENT_SCENARIO]

[PESSIMISTIC_SCENARIO]
悲観シナリオ（ARIMA信頼区間下限、150文字程度）：
・ARIMAモデルが予測するリスクシナリオでの{chart_label}成長
・時系列分析から見た下振れ要因（市場競合激化、技術変化対応遅れ）
・デジタル変革の遅れによる競争力低下
・人材確保困難と技術投資不足による成長制約
[/PESSIMISTIC_SCENARIO]

各シナリオは、ARIMAモデルの予測に基づく具体的で実用的な内容にし、特に情報系学生の視点から技術的成長機会、キャリア形成、業界トレンドを重視した分析を行ってください。

## 分析時の重要な視点（ARIMAベース予測考慮）
- **時系列分析の優位性**: 統計的に信頼性の高い予測による成長性評価
- **信頼区間の解釈**: 70%信頼区間による予測の不確実性を考慮したリスク評価
- **パラメータ安定性**: order=(1,1,1)固定による一貫した予測基準
- **技術投資トレンド**: ARIMAが捉える企業の研究開発費・設備投資の時系列パターン
- **将来技術への対応**: 予測データから読み取る次世代技術（AI、IoT、クラウド）投資動向
- 情報系学生が活躍できる職種・部門の具体的な説明
- 業界内での技術力・イノベーション力の客観的評価
- 長期的なキャリア形成の可能性（昇進、転職市場価値）
"""
        return prompt

    def _parse_scenario_analysis(self, response_text):
        scenarios = {}
        
        try:
            scenarios['optimistic'] = self._extract_section(response_text, 'OPTIMISTIC_SCENARIO')
            scenarios['current'] = self._extract_section(response_text, 'CURRENT_SCENARIO')
            scenarios['pessimistic'] = self._extract_section(response_text, 'PESSIMISTIC_SCENARIO')
            
        except Exception as e:
            print(f"Scenario analysis parsing error: {e}")
            scenarios = {"error": "シナリオ分析の解析中にエラーが発生しました"}
        
        return scenarios

    def generate_company_overview_analysis(self, company_name, edinet_code):
        if not settings.GEMINI_API_KEY:
            return "Google Gemini APIキーが設定されていません。環境変数GOOGLE_API_KEYを設定してください。"
        
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            additional_info = self._get_company_additional_info(company_name)
            
            prompt = f"""
            【企業名】{company_name}
            
            【Web検索による企業情報】
            {additional_info.get('web_search_summary', '情報なし')}
            
            【要求事項】
            上記の企業について、財務タブの「企業概要」セクションに表示する簡潔な企業紹介文を作成してください。
            
            【内容に含めるべき要素】
            1. 企業の主要事業内容
            2. 業界での位置づけ・特徴
            3. 強みや競争優位性
            4. 最新の動向や注目点
            
            【注意事項】
            - Web検索結果を最重視し、事実に基づいた情報のみを記載
            - 情報系学生にとって分かりやすい表現を使用
            - 200-250文字程度で簡潔にまとめる
            - 過度に宣伝的な表現は避ける
            
            企業概要：
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"Company overview AI analysis error: {e}")
            return "企業概要の生成中にエラーが発生しました。しばらく後に再度お試しください。"
