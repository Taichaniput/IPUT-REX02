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
        
        # 検索クエリを工夫して、多角的な情報を要求
        search_query = f"{company_name}の事業内容、強みと弱み、市場でのポジショニング、最近の重要なニュースについて包括的に調査し、要約してください。"
        
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
            return {"error": "Gemini APIからの応答がありません"}
        
        print(f"Gemini response received: {len(response.text)} characters")
        
        # レスポンスを構造化
        structured_analysis = parse_structured_analysis(response.text)
        
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
        analysis['FINANCIAL_ANALYSIS'] = f"{company_name}は{latest.fiscal_year}年に売上高{net_sales:.1f}億円を記録。情報通信業界における企業として位置づけられています。"
    else:
        analysis['FINANCIAL_ANALYSIS'] = f"{company_name}の詳細な財務分析を表示するためには、API接続が必要です。"
    
    # シナリオ分析のフォールバック
    analysis['GROWTH_SCENARIOS'] = {
        'optimistic': "市場拡大と技術革新により成長が期待されます。",
        'current': "現在のトレンドが継続すると予想されます。",
        'pessimistic': "市場環境の変化により成長に課題が生じる可能性があります。"
    }
    
    analysis['PROFIT_SCENARIOS'] = {
        'optimistic': "効率化により収益性向上が期待されます。",
        'current': "現在の収益構造が維持される見通しです。",
        'pessimistic': "競争激化により収益性に圧力がかかる可能性があります。"
    }
    
    # ポジショニング分析
    if cluster_info:
        analysis['POSITIONING_ANALYSIS'] = f"クラスタ{cluster_info['cluster_id']}に分類され、業界内での特定のポジションを占めています。"
    else:
        analysis['POSITIONING_ANALYSIS'] = "業界内でのポジショニング分析には追加データが必要です。"
    
    # 総括
    analysis['SUMMARY'] = f"{company_name}は情報系学生にとって技術的な成長機会を提供する可能性がある企業です。詳細な分析にはAI機能をご利用ください。"
    
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
あなたは情報系学生の就活支援を専門とする企業分析アナリストです。以下の企業について、構造化された分析を行ってください。

## 分析対象企業
{company_name}

## 財務データ
{financial_summary}

## 成長予測（3シナリオ）
{prediction_summary}

## 業界ポジショニング・クラスタリング分析
{cluster_summary}

## 外部情報（Tavily検索結果）
{additional_info.get('web_search_summary', 'なし')}

## 指示
上記データを統合して情報系学生向けの就活分析を行ってください。必ず以下の形式で出力してください。

[FINANCIAL_ANALYSIS]
財務データに基づく企業の特徴と健全性の簡潔な分析（100-150文字程度）
[/FINANCIAL_ANALYSIS]

[COMPANY_OVERVIEW]
Tavily検索結果に基づく企業の簡潔な説明。事業内容、主要サービス、業界での位置づけを含む（200-250文字程度）
[/COMPANY_OVERVIEW]

[GROWTH_SCENARIOS_OPTIMISTIC]
楽観シナリオの詳細な説明（市場拡大、新技術導入成功、デジタル変革などの要因）
[/GROWTH_SCENARIOS_OPTIMISTIC]

[GROWTH_SCENARIOS_CURRENT]
現状シナリオの詳細な説明（現在のトレンド継続、安定成長）
[/GROWTH_SCENARIOS_CURRENT]

[GROWTH_SCENARIOS_PESSIMISTIC]
悲観シナリオの詳細な説明（市場縮小、競合激化、技術的遅れなどのリスク）
[/GROWTH_SCENARIOS_PESSIMISTIC]

[PROFIT_SCENARIOS_OPTIMISTIC]
収益性楽観シナリオの説明（効率化、高付加価値サービス拡大）
[/PROFIT_SCENARIOS_OPTIMISTIC]

[PROFIT_SCENARIOS_CURRENT]
収益性現状シナリオの説明（現在の収益構造継続）
[/PROFIT_SCENARIOS_CURRENT]

[PROFIT_SCENARIOS_PESSIMISTIC]
収益性悲観シナリオの説明（コスト増、価格競争激化）
[/PROFIT_SCENARIOS_PESSIMISTIC]

[POSITIONING_ANALYSIS]
クラスタデータと同業他社比較に基づく業界内ポジショニング分析。企業の競合優位性、市場での立ち位置、技術力について詳述
[/POSITIONING_ANALYSIS]

[SUMMARY]
情報系学生向けの就活への示唆。この企業のキャリア価値、技術的成長機会、業界トレンド、エンジニアとしてのキャリアパスについて総括
[/SUMMARY]

各セクションは具体的で実用的な内容にしてください。
"""
    return prompt


def parse_structured_analysis(response_text):
    """構造化された分析レスポンスを解析"""
    sections = {}
    
    try:
        # セクションを抽出
        sections['FINANCIAL_ANALYSIS'] = extract_section(response_text, 'FINANCIAL_ANALYSIS')
        sections['COMPANY_OVERVIEW'] = extract_section(response_text, 'COMPANY_OVERVIEW')
        
        sections['GROWTH_SCENARIOS'] = {
            'optimistic': extract_section(response_text, 'GROWTH_SCENARIOS_OPTIMISTIC'),
            'current': extract_section(response_text, 'GROWTH_SCENARIOS_CURRENT'),
            'pessimistic': extract_section(response_text, 'GROWTH_SCENARIOS_PESSIMISTIC')
        }
        
        sections['PROFIT_SCENARIOS'] = {
            'optimistic': extract_section(response_text, 'PROFIT_SCENARIOS_OPTIMISTIC'),
            'current': extract_section(response_text, 'PROFIT_SCENARIOS_CURRENT'),
            'pessimistic': extract_section(response_text, 'PROFIT_SCENARIOS_PESSIMISTIC')
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