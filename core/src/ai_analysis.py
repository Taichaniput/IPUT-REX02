# core/ai_analysis.py

def get_company_additional_info(company_name):
    """企業の追加情報を外部ソース（Tavily Web Search）から取得"""
    from django.conf import settings
    
    additional_info = {}

    # Tavily availability check
    try:
        from tavily import TavilyClient
        TAVILY_AVAILABLE = True
    except ImportError:
        TAVILY_AVAILABLE = False

    if not TAVILY_AVAILABLE:
        additional_info['web_search_summary'] = "Tavilyライブラリがインストールされていません。"
        return additional_info

    if not settings.TAVILY_API_KEY:
        additional_info['web_search_summary'] = "Tavily APIキーが設定されていません。"
        return additional_info

    try:
        from tavily import TavilyClient
        # Tavilyクライアントを初期化
        tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)
        
        # 包括的AI分析用のプロンプト（詳細分析用）
        search_query = f"{company_name}の事業内容、技術的特徴、IT・デジタル変革の取り組み、エンジニア採用状況、市場でのポジショニング、成長戦略について包括的に調査し、情報系学生の就活に有用な情報を要約してください。"
        
        # TavilyでWeb検索を実行
        response = tavily.search(
            query=search_query,
            search_depth="advanced",
            max_results=5
        )
        
        # 結果を結合して一つのサマリーにする
        summary = "\n".join([f"- {obj['content']}" for obj in response['results']])
        additional_info['web_search_summary'] = summary if summary else "ウェブ検索で関連情報が見つかりませんでした。"

    except Exception as e:
        print(f"Tavily search error: {e}")
        additional_info['web_search_summary'] = "ウェブ検索中にエラーが発生しました。"
    
    return additional_info


def generate_comprehensive_ai_analysis(company_name, edinet_code, financial_data, prediction_results, cluster_info):
    """Gemini APIを使用して包括的な企業分析を生成"""
    from django.conf import settings
    
    if not settings.GEMINI_API_KEY:
        return create_fallback_analysis(company_name, financial_data, prediction_results, cluster_info)
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Tavilyで追加情報を取得
        additional_info = get_company_additional_info(company_name)
        
        # 財務データを整理
        financial_summary = prepare_financial_summary(financial_data)
        
        # 予測データを整理
        prediction_summary = prepare_prediction_summary(prediction_results)
        
        # クラスタデータを整理
        cluster_summary = prepare_cluster_summary(cluster_info)
        
        # Geminiプロンプトを構築
        prompt = build_comprehensive_analysis_prompt(
            company_name, financial_summary, prediction_summary, 
            cluster_summary, additional_info
        )
        
        # Gemini APIに送信
        response = model.generate_content(prompt)
        
        # レスポンスが正常に生成されているかチェック
        if not response or not response.text:
            print("ERROR: Gemini APIからの応答がありません")
            return {"error": "Gemini APIからの応答がありません"}
        
        print(f"Gemini response received: {len(response.text)} characters")
        
        # レスポンスを構造化
        structured_analysis = parse_structured_analysis(response.text)
        
        # 構造化された分析にエラーがある場合の処理
        if 'error' in structured_analysis:
            print(f"ERROR: Analysis parsing failed: {structured_analysis['error']}")
            return structured_analysis
        
        return structured_analysis
        
    except Exception as e:
        print(f"AI analysis generation error: {e}")
        return create_fallback_analysis(company_name, financial_data, prediction_results, cluster_info)


def create_fallback_analysis(company_name, financial_data, prediction_results, cluster_info):
    """API利用できない場合のフォールバック分析"""
    analysis = {}
    
    # 基本的な財務分析
    if financial_data:
        # financial_dataは辞書のリスト形式 [{'data': FinancialData, 'indicators': dict}, ...]
        if isinstance(financial_data[0], dict) and 'data' in financial_data[0]:
            latest = financial_data[0]['data']
        else:
            # 直接FinancialDataオブジェクトの場合
            latest = financial_data[0]
        
        net_sales = (latest.net_sales or 0) / 100000000
        analysis['FINANCIAL_ANALYSIS'] = f"{company_name}は{latest.fiscal_year}年に売上高{net_sales:.1f}億円を記録。財務の安定性と成長性を評価するためにはAI分析をご利用ください。"
    else:
        analysis['FINANCIAL_ANALYSIS'] = f"{company_name}の詳細な財務分析を表示するためには、API接続が必要です。"
    
    # 売上高シナリオ分析のフォールバック
    analysis['SALES_SCENARIOS'] = {
        'optimistic': "デジタル変革の進展により、AI・IoT・クラウド技術の活用が拡大し、新たな事業機会の創出と市場シェア拡大が期待されます。売上高は大幅な成長を見込むことができます。",
        'current': "現在の事業基盤を維持しながら、安定的な成長を継続する見通しです。技術投資とROIのバランスを保ちつつ、売上高の着実な成長が予想されます。",
        'pessimistic': "競合他社の技術革新により相対的な競争力低下のリスクがあります。デジタル変革の遅れや市場シェア低下により、売上高の成長に制約が生じる可能性があります。"
    }
    
    # 収益性シナリオ分析のフォールバック
    analysis['PROFIT_SCENARIOS'] = {
        'optimistic': "業務プロセスの自動化・効率化により大幅なコスト削減を実現し、高付加価値サービスの拡大により利益率向上が期待されます。技術革新による新収益源の創出も見込まれます。",
        'current': "現在の収益構造を維持しながら、適度な技術投資により収益性を保持する見通しです。既存事業の安定的な利益確保により、持続的な成長を支えると予想されます。",
        'pessimistic': "人件費や技術投資の増大により収益性に圧力がかかる可能性があります。価格競争の激化や市場シェア低下により、利益率の低下が懸念されます。"
    }
    
    # 3シナリオ予測分析のフォールバック（売上高・純利益グラフ用）
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
    
    # ポジショニング分析
    if cluster_info:
        analysis['POSITIONING_ANALYSIS'] = f"クラスタ{cluster_info['cluster_id']}に分類され、同業他社との比較において独自のポジションを占めています。技術力とイノベーション力の詳細な評価、デジタル変革への取り組み状況については、AI分析で詳細をご確認ください。"
    else:
        analysis['POSITIONING_ANALYSIS'] = "業界内でのポジショニング分析、競争優位性の評価、技術力の詳細な比較分析には、AI機能による包括的な分析が必要です。"
    
    # 総括・キャリア分析
    analysis['SUMMARY'] = f"{company_name}は情報系学生にとって技術的な成長機会を提供する可能性が高い企業です。エンジニアとしてのキャリアパス、スキル習得環境、長期的なキャリア展望についての詳細な分析は、AI機能をご利用ください。"
    
    # 企業概要（フォールバック）
    analysis['COMPANY_OVERVIEW'] = f"{company_name}の事業内容、技術的特徴、競争優位性についての詳細な分析は、AI機能をご利用ください。"
    
    return analysis


def prepare_financial_summary(financial_data):
    """財務データを要約"""
    if not financial_data:
        return "財務データなし"
    
    summary = []
    for item in financial_data[:3]:  # 最新3年分
        # itemが辞書の場合とFinancialDataオブジェクトの場合を処理
        if isinstance(item, dict) and 'data' in item:
            fd = item['data']
        else:
            fd = item
            
        net_sales = fd.net_sales or 0
        net_income = fd.net_income or 0
        summary.append(f"{fd.fiscal_year}年: 売上{net_sales/100000000:.1f}億円, 純利益{net_income/100000000:.1f}億円")
    
    return "\n".join(summary)


def prepare_prediction_summary(prediction_results):
    """予測データを要約"""
    if not prediction_results:
        return "予測データなし"
    
    summary = []
    for metric, result in prediction_results.items():
        if 'scenarios' in result.get('predictions', {}):
            scenarios = result['predictions']['scenarios']
            summary.append(f"{result['label']}: 楽観{scenarios['optimistic']['growth_rate']:.1f}%, 現状{scenarios['current']['growth_rate']:.1f}%, 悲観{scenarios['pessimistic']['growth_rate']:.1f}%")
    
    return "\n".join(summary)


def prepare_cluster_summary(cluster_info):
    """クラスタデータを要約"""
    if not cluster_info:
        return "クラスタデータなし"
    
    summary = f"当該企業はクラスタ{cluster_info['cluster_id']}/{cluster_info['total_clusters']}に分類\n"
    
    # クラスタの特徴
    if 'cluster_characteristics' in cluster_info:
        summary += "クラスタの特徴:\n"
        for feat, data in cluster_info['cluster_characteristics'].items():
            feat_label = get_feature_label(feat)
            summary += f"- {feat_label}: 全体平均比{data['relative']:.1f}%\n"
    
    # 同じクラスタの企業
    if 'same_cluster_companies' in cluster_info:
        companies = [comp['name'] for comp in cluster_info['same_cluster_companies'][:5]]
        summary += f"\n同クラスタの類似企業: {', '.join(companies)}\n"
    
    # PCA解釈情報
    if 'pca_interpretation' in cluster_info:
        summary += "\n主成分分析による解釈:\n"
        for comp in cluster_info['pca_interpretation']:
            summary += f"- 第{comp['component']}主成分({comp['meaning']}): 寄与率{comp['variance_ratio']:.1f}%\n"
    
    return summary


def get_feature_label(feature):
    """特徴量の日本語ラベル"""
    # ml_analytics.pyのget_feature_labelと統合するため、そちらから import
    from .ml_analytics import get_feature_label as ml_get_feature_label
    return ml_get_feature_label(feature)


def build_comprehensive_analysis_prompt(company_name, financial_summary, prediction_summary, cluster_summary, additional_info):
    """包括的分析用のプロンプトを構築"""
    prompt = f"""
あなたは情報系学生の就活支援を専門とする企業分析アナリストです。以下の企業について、情報系学生のキャリア形成の観点から構造化された分析を行ってください。

## 分析対象企業
{company_name}

## 財務データ
{financial_summary}

## 成長予測（3シナリオ）
{prediction_summary}

## 業界ポジショニング・クラスタリング分析
{cluster_summary}

## 外部情報（検索結果）
{additional_info.get('web_search_summary', 'なし')}

## 指示
上記データを統合して情報系学生向けの就活分析を行ってください。必ず以下の形式で出力してください。

[FINANCIAL_ANALYSIS]
財務データに基づく企業の特徴と健全性の簡潔な分析（100-150文字程度）
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
業界内ポジショニング分析（200-250文字程度）：
・クラスタデータに基づく同業他社比較
・技術力・イノベーション力の評価
・市場での競争優位性と差別化要因
・デジタル変革への取り組み状況
・情報系人材の活用・育成環境
・業界トレンドへの適応力
[/POSITIONING_ANALYSIS]

[SUMMARY]
情報系学生向けキャリア総括（250-300文字程度）：
・この企業でのキャリア形成価値
・技術的成長機会とスキル習得環境
・エンジニアとしてのキャリアパス
・業界での将来性と安定性
・情報系学生に推奨する理由・注意点
・長期的なキャリア展望
[/SUMMARY]

各セクションは具体的で実用的な内容にし、特に情報系学生の視点から技術的成長機会、キャリア形成、業界トレンドを重視した分析を行ってください。

## 分析時の重要な視点
- プログラミング言語、開発環境、技術スタックの言及
- エンジニアの成長環境（研修制度、勉強会、技術コミュニティ）
- 将来性の高い技術分野（AI、IoT、クラウド、DX）への取り組み
- 情報系学生が活躍できる職種・部門の具体的な説明
- 業界内での技術力・イノベーション力の客観的評価
- 長期的なキャリア形成の可能性（昇進、転職市場価値）
"""
    return prompt


def parse_structured_analysis(response_text):
    """構造化された分析レスポンスを解析"""
    sections = {}
    
    try:
        # セクションを抽出
        sections['FINANCIAL_ANALYSIS'] = extract_section(response_text, 'FINANCIAL_ANALYSIS')
        sections['COMPANY_OVERVIEW'] = extract_section(response_text, 'COMPANY_OVERVIEW')
        
        sections['SALES_SCENARIOS'] = {
            'optimistic': extract_section(response_text, 'SALES_SCENARIOS_OPTIMISTIC'),
            'current': extract_section(response_text, 'SALES_SCENARIOS_CURRENT'),
            'pessimistic': extract_section(response_text, 'SALES_SCENARIOS_PESSIMISTIC')
        }
        
        sections['PROFIT_SCENARIOS'] = {
            'optimistic': extract_section(response_text, 'PROFIT_SCENARIOS_OPTIMISTIC'),
            'current': extract_section(response_text, 'PROFIT_SCENARIOS_CURRENT'),
            'pessimistic': extract_section(response_text, 'PROFIT_SCENARIOS_PESSIMISTIC')
        }
        
        # 3シナリオ予測分析用（グラフ用）
        sections['CHART_SALES_SCENARIOS'] = {
            'optimistic': extract_section(response_text, 'CHART_SALES_OPTIMISTIC'),
            'current': extract_section(response_text, 'CHART_SALES_CURRENT'),
            'pessimistic': extract_section(response_text, 'CHART_SALES_PESSIMISTIC')
        }
        
        sections['CHART_PROFIT_SCENARIOS'] = {
            'optimistic': extract_section(response_text, 'CHART_PROFIT_OPTIMISTIC'),
            'current': extract_section(response_text, 'CHART_PROFIT_CURRENT'),
            'pessimistic': extract_section(response_text, 'CHART_PROFIT_PESSIMISTIC')
        }
        
        sections['POSITIONING_ANALYSIS'] = extract_section(response_text, 'POSITIONING_ANALYSIS')
        sections['SUMMARY'] = extract_section(response_text, 'SUMMARY')
        
    except Exception as e:
        print(f"Response parsing error: {e}")
        sections = {"error": "レスポンス解析中にエラーが発生しました"}
    
    return sections


def extract_section(text, section_name):
    """テキストから指定されたセクションを抽出"""
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


def generate_scenario_analysis(company_name, edinet_code, prediction_results, chart_type='sales'):
    """3シナリオ分析を生成（売上高・純利益予測グラフ用）"""
    from django.conf import settings
    
    if not settings.GEMINI_API_KEY:
        return create_fallback_scenario_analysis(company_name, chart_type)
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # 予測データを整理
        chart_data = prepare_chart_data(prediction_results, chart_type)
        
        # Geminiプロンプトを構築
        prompt = build_scenario_analysis_prompt(company_name, chart_data, chart_type)
        
        # Gemini APIに送信
        response = model.generate_content(prompt)
        
        # レスポンスが正常に生成されているかチェック
        if not response or not response.text:
            print("ERROR: Gemini APIからの応答がありません")
            return create_fallback_scenario_analysis(company_name, chart_type)
        
        # レスポンスを構造化
        structured_analysis = parse_scenario_analysis(response.text)
        
        return structured_analysis
        
    except Exception as e:
        print(f"Scenario analysis generation error: {e}")
        return create_fallback_scenario_analysis(company_name, chart_type)


def create_fallback_scenario_analysis(company_name, chart_type):
    """グラフ用シナリオ分析のフォールバック"""
    if chart_type == 'sales':
        return {
            'optimistic': "デジタル変革の進展により、AI・IoT・クラウド技術の活用拡大で新事業機会創出と市場シェア拡大が期待されます。",
            'current': "現在の事業基盤を維持しながら、技術投資とROIのバランスを保ち、売上高の着実な成長が予想されます。",
            'pessimistic': "競合他社の技術革新により相対的な競争力低下のリスクがあり、市場シェア低下により売上成長に制約が生じる可能性があります。"
        }
    elif chart_type == 'profit':
        return {
            'optimistic': "業務プロセスの自動化・効率化によりコスト削減を実現し、高付加価値サービス拡大により利益率向上が期待されます。",
            'current': "現在の収益構造を維持しながら、適度な技術投資により収益性を保持し、持続的な成長を支える見通しです。",
            'pessimistic': "人件費や技術投資の増大により収益性に圧力がかかり、価格競争の激化により利益率低下が懸念されます。"
        }
    else:
        return {
            'optimistic': "詳細な分析はAI機能をご利用ください。",
            'current': "詳細な分析はAI機能をご利用ください。",
            'pessimistic': "詳細な分析はAI機能をご利用ください。"
        }


def prepare_chart_data(prediction_results, chart_type):
    """グラフ用データを整理"""
    if not prediction_results:
        return "予測データが不足しています。"
    
    if chart_type == 'sales' and 'net_sales' in prediction_results:
        return f"売上高予測データ: {prediction_results['net_sales']}"
    elif chart_type == 'profit' and 'net_income' in prediction_results:
        return f"純利益予測データ: {prediction_results['net_income']}"
    else:
        return "予測データが不足しています。"


def build_scenario_analysis_prompt(company_name, chart_data, chart_type):
    """シナリオ分析用プロンプト構築"""
    metric_name = "売上高" if chart_type == 'sales' else "純利益"
    
    prompt = f"""
{company_name}の{metric_name}予測グラフに基づいて、以下の3つのシナリオ分析を行ってください。

予測データ:
{chart_data}

各シナリオについて、情報系学生の視点から技術的要因を重視した分析を行い、
120-150文字程度で簡潔に分析してください。

[OPTIMISTIC]
楽観シナリオ分析（技術革新・市場拡大要因を重視）
[/OPTIMISTIC]

[CURRENT]
現状維持シナリオ分析（現在の事業基盤・技術投資を重視）
[/CURRENT]

[PESSIMISTIC]
悲観シナリオ分析（競合・技術的リスクを重視）
[/PESSIMISTIC]

各シナリオは具体的で実用的な内容にし、特に情報系学生の視点から技術的成長機会、
業界トレンドを重視した分析を行ってください。
"""
    return prompt


def parse_scenario_analysis(response_text):
    """シナリオ分析レスポンスを解析"""
    scenarios = {}
    
    try:
        scenarios['optimistic'] = extract_section(response_text, 'OPTIMISTIC')
        scenarios['current'] = extract_section(response_text, 'CURRENT')
        scenarios['pessimistic'] = extract_section(response_text, 'PESSIMISTIC')
        
    except Exception as e:
        print(f"Scenario parsing error: {e}")
        scenarios = {
            'optimistic': "分析中...",
            'current': "分析中...",
            'pessimistic': "分析中..."
        }
    
    return scenarios


def prepare_chart_data(prediction_results, chart_type):
    """予測結果からチャート用データを準備"""
    if not prediction_results:
        return "予測データなし"
    
    # chart_typeに応じて適切なメトリクスを選択
    if chart_type == 'sales':
        target_metric = 'net_sales'
    elif chart_type == 'profit':
        target_metric = 'net_income'
    else:
        target_metric = 'net_sales'
    
    # 該当するメトリクスのデータを取得
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


def build_scenario_analysis_prompt(company_name, chart_data, chart_type):
    """3シナリオ分析用のプロンプトを構築"""
    chart_label = "売上高" if chart_type == 'sales' else "純利益"
    
    prompt = f"""
あなたは情報系学生の就活支援を専門とする企業分析アナリストです。以下の企業について、{chart_label}予測グラフに基づく3シナリオ分析を行ってください。

## 分析対象企業
{company_name}

## {chart_label}予測データ
{chart_data}

## 指示
上記の{chart_label}予測データを基に、情報系学生のキャリア形成の観点から3つのシナリオ分析を行ってください。必ず以下の形式で出力してください。

[OPTIMISTIC_SCENARIO]
楽観シナリオ（150-200文字程度）：
・{chart_label}成長の主要要因（DX推進、新技術導入、規制緩和など）
・技術革新による事業機会（AI、IoT、クラウドなど）
・デジタル変革による競争優位性向上
・情報系人材の採用拡大と成長への寄与
[/OPTIMISTIC_SCENARIO]

[CURRENT_SCENARIO]
現状シナリオ（150-200文字程度）：
・現在のトレンド継続による安定成長
・既存事業の着実な拡大
・技術投資とROIのバランス
・市場での安定的なポジション維持
[/CURRENT_SCENARIO]

[PESSIMISTIC_SCENARIO]
悲観シナリオ（150-200文字程度）：
・市場縮小・競合激化のリスク
・技術的遅れによる競争力低下
・デジタル変革の遅れ
・人材確保困難による成長制約
[/PESSIMISTIC_SCENARIO]

各シナリオは具体的で実用的な内容にし、特に情報系学生の視点から技術的成長機会、キャリア形成、業界トレンドを重視した分析を行ってください。

## 分析時の重要な視点
- プログラミング言語、開発環境、技術スタックの言及
- エンジニアの成長環境（研修制度、勉強会、技術コミュニティ）
- 将来性の高い技術分野（AI、IoT、クラウド、DX）への取り組み
- 情報系学生が活躍できる職種・部門の具体的な説明
- 業界内での技術力・イノベーション力の客観的評価
- 長期的なキャリア形成の可能性（昇進、転職市場価値）
"""
    return prompt


def parse_scenario_analysis(response_text):
    """3シナリオ分析レスポンスを解析"""
    scenarios = {}
    
    try:
        scenarios['optimistic'] = extract_section(response_text, 'OPTIMISTIC_SCENARIO')
        scenarios['current'] = extract_section(response_text, 'CURRENT_SCENARIO')
        scenarios['pessimistic'] = extract_section(response_text, 'PESSIMISTIC_SCENARIO')
        
    except Exception as e:
        print(f"Scenario analysis parsing error: {e}")
        scenarios = {"error": "シナリオ分析の解析中にエラーが発生しました"}
    
    return scenarios


def create_fallback_scenario_analysis(company_name, chart_type):
    """API利用できない場合の3シナリオ分析フォールバック"""
    chart_label = "売上高" if chart_type == 'sales' else "純利益"
    
    scenarios = {
        'optimistic': f"デジタル変革の進展により、AI・IoT・クラウド技術の活用が拡大し、新たな事業機会の創出と{chart_label}の大幅な成長が期待されます。情報系人材の積極採用により技術力向上が見込まれ、高付加価値サービスの展開で収益性も向上する可能性があります。",
        'current': f"現在の事業基盤を維持しながら、安定的な{chart_label}成長を継続する見通しです。技術投資とROIのバランスを保ちつつ、市場での競争力を維持していくと予想されます。既存の技術スタックを活用した着実な成長が見込まれます。",
        'pessimistic': f"競合他社の技術革新により相対的な競争力低下のリスクがあり、{chart_label}の成長に制約が生じる可能性があります。デジタル変革の遅れや優秀な人材確保の困難により、市場シェアの縮小や収益性の悪化が懸念されます。"
    }
    
    return scenarios