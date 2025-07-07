// Company Detail Page JavaScript

/**
 * タブ切り替え機能
 * @param {string} tabName - 表示するタブの名前 ('financial-data' または 'ai-analysis')
 */
function showTab(tabName) {
    // すべてのタブコンテンツを非表示
    const contents = document.getElementsByClassName('tab-content');
    for (let content of contents) {
        content.classList.remove('active');
    }
    
    // すべてのタブボタンを非アクティブ化
    const buttons = document.getElementsByClassName('tab-button');
    for (let button of buttons) {
        button.classList.remove('active');
    }
    
    // 選択されたタブを表示
    const targetTab = document.getElementById(tabName);
    if (targetTab) {
        targetTab.classList.add('active');
    }
    
    // 対応するボタンをアクティブ化
    const activeButton = Array.from(buttons).find(
        button => button.textContent.includes(
            tabName === 'financial-data' ? '財務データ' : 'AI企業分析'
        )
    );
    if (activeButton) {
        activeButton.classList.add('active');
    }
    
    // AI分析タブが選択された場合の処理
    if (tabName === 'ai-analysis') {
        // 分析結果がまだ読み込まれていない場合のみAJAXで取得
        if (!window.aiAnalysisLoaded && !window.aiAnalysisLoading) {
            loadAIAnalysis();
        }
    }
}

/**
 * AI分析を非同期で読み込む
 */
async function loadAIAnalysis() {
    // すでに読み込み中または完了している場合は何もしない
    if (window.aiAnalysisLoading || window.aiAnalysisLoaded) {
        return;
    }
    
    window.aiAnalysisLoading = true;
    
    // EDINETコードを取得
    const pathParts = window.location.pathname.split('/');
    const edinetCode = pathParts[pathParts.indexOf('company') + 1];
    
    if (!edinetCode) {
        console.error('EDINETコードが見つかりません');
        return;
    }
    
    try {
        // ローディング状態を表示
        showAIAnalysisLoading();
        
        // AJAX リクエスト
        console.log(`Fetching AI analysis for edinet code: ${edinetCode}`);
        const response = await fetch(`/api/ai-analysis/${edinetCode}/`, {
            method: 'GET',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/json',
            }
        });
        
        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Response data:', data);
        
        if (response.ok && data.success) {
            // AI分析結果を表示
            displayAIAnalysis(data.ai_analysis);
            window.aiAnalysisLoaded = true;
        } else {
            throw new Error(data.error || '分析に失敗しました');
        }
        
    } catch (error) {
        console.error('AI分析エラー:', error);
        showAIAnalysisError(error.message);
    } finally {
        window.aiAnalysisLoading = false;
    }
}

/**
 * AI分析のローディング状態を表示
 */
function showAIAnalysisLoading() {
    const analysisContainer = document.querySelector('#ai-analysis');
    if (!analysisContainer) return;
    
    // ローディング用の既存コンテンツを探す
    let loadingContent = analysisContainer.querySelector('.ai-loading-content');
    
    if (!loadingContent) {
        // ローディングコンテンツを作成
        loadingContent = document.createElement('div');
        loadingContent.className = 'ai-loading-content';
        loadingContent.innerHTML = `
            <div class="loading-section">
                <div class="loading-spinner-large"></div>
                <h3>AI分析を実行中...</h3>
                <p>企業の財務データ、市場ポジション、将来予測を総合的に分析しています。</p>
                <div class="loading-steps">
                    <div class="loading-step active">📊 財務データ解析</div>
                    <div class="loading-step">🔍 市場調査</div>
                    <div class="loading-step">🤖 AI予測モデル実行</div>
                    <div class="loading-step">📝 レポート生成</div>
                </div>
            </div>
        `;
        
        // 既存のコンテンツを隠して、ローディングを表示
        const existingContent = analysisContainer.querySelectorAll(':not(.ai-loading-content)');
        existingContent.forEach(el => el.style.display = 'none');
        
        analysisContainer.appendChild(loadingContent);
    }
    
    // ローディングステップアニメーション
    animateLoadingSteps();
}

/**
 * ローディングステップのアニメーション
 */
function animateLoadingSteps() {
    const steps = document.querySelectorAll('.loading-step');
    let currentStep = 0;
    
    const interval = setInterval(() => {
        // 前のステップを完了状態に
        if (currentStep > 0) {
            steps[currentStep - 1].classList.remove('active');
            steps[currentStep - 1].classList.add('completed');
        }
        
        // 現在のステップをアクティブに
        if (currentStep < steps.length) {
            steps[currentStep].classList.add('active');
            currentStep++;
        } else {
            clearInterval(interval);
        }
    }, 1500);
    
    // 最大15秒でタイムアウト
    setTimeout(() => {
        clearInterval(interval);
    }, 15000);
}

/**
 * AI分析結果を表示
 */
function displayAIAnalysis(aiAnalysis) {
    const analysisContainer = document.querySelector('#ai-analysis');
    if (!analysisContainer) return;
    
    // ローディングコンテンツを削除
    const loadingContent = analysisContainer.querySelector('.ai-loading-content');
    if (loadingContent) {
        loadingContent.remove();
    }
    
    // 既存のコンテンツを表示
    const existingContent = analysisContainer.querySelectorAll(':not(.ai-loading-content)');
    existingContent.forEach(el => el.style.display = 'block');
    
    // AI分析結果をDOMに反映
    updateAnalysisContent(aiAnalysis);
    
    // 3シナリオ分析を各チャートに対してロード
    loadChartScenarioAnalysis();
    
    // 成功メッセージを表示
    showNotification('AI分析が完了しました！', 'success');
}

/**
 * AI分析結果をDOMに更新
 */
function updateAnalysisContent(aiAnalysis) {
    // 各セクションを更新
    updateScenarioAnalysis(aiAnalysis);
    updatePositioningAnalysis(aiAnalysis);
    updateSummaryAnalysis(aiAnalysis);
    updateCompanyOverview(aiAnalysis);
}

/**
 * シナリオ分析を更新
 */
function updateScenarioAnalysis(aiAnalysis) {
    // 売上高シナリオ分析の更新
    if (aiAnalysis.SALES_SCENARIOS) {
        const salesScenarios = document.querySelectorAll('.sales-scenarios');
        salesScenarios.forEach(container => {
            updateElement(container.querySelector('.scenario.optimistic .scenario-explanation'), aiAnalysis.SALES_SCENARIOS.optimistic);
            updateElement(container.querySelector('.scenario.current .scenario-explanation'), aiAnalysis.SALES_SCENARIOS.current);
            updateElement(container.querySelector('.scenario.pessimistic .scenario-explanation'), aiAnalysis.SALES_SCENARIOS.pessimistic);
        });
    }
    
    // 純利益シナリオ分析の更新
    if (aiAnalysis.PROFIT_SCENARIOS) {
        const profitScenarios = document.querySelectorAll('.profit-scenarios');
        profitScenarios.forEach(container => {
            updateElement(container.querySelector('.scenario.optimistic .scenario-explanation'), aiAnalysis.PROFIT_SCENARIOS.optimistic);
            updateElement(container.querySelector('.scenario.current .scenario-explanation'), aiAnalysis.PROFIT_SCENARIOS.current);
            updateElement(container.querySelector('.scenario.pessimistic .scenario-explanation'), aiAnalysis.PROFIT_SCENARIOS.pessimistic);
        });
    }
    
    // 旧形式の成長シナリオにも対応（後方互換性）
    if (aiAnalysis.GROWTH_SCENARIOS && !aiAnalysis.SALES_SCENARIOS) {
        const salesScenarios = document.querySelectorAll('.sales-scenarios');
        salesScenarios.forEach(container => {
            updateElement(container.querySelector('.scenario.optimistic .scenario-explanation'), aiAnalysis.GROWTH_SCENARIOS.optimistic);
            updateElement(container.querySelector('.scenario.current .scenario-explanation'), aiAnalysis.GROWTH_SCENARIOS.current);
            updateElement(container.querySelector('.scenario.pessimistic .scenario-explanation'), aiAnalysis.GROWTH_SCENARIOS.pessimistic);
        });
    }
}

/**
 * ポジショニング分析を更新
 */
function updatePositioningAnalysis(aiAnalysis) {
    if (aiAnalysis.POSITIONING_ANALYSIS) {
        updateElement('.positioning-explanation', aiAnalysis.POSITIONING_ANALYSIS);
    }
}

/**
 * 総括分析を更新
 */
function updateSummaryAnalysis(aiAnalysis) {
    if (aiAnalysis.SUMMARY) {
        updateElement('.summary-content', aiAnalysis.SUMMARY);
    }
}

/**
 * 企業概要を更新（財務表タブ用）
 */
function updateCompanyOverview(aiAnalysis) {
    if (aiAnalysis.COMPANY_OVERVIEW) {
        const overviewContent = document.querySelector('.company-overview-content');
        if (overviewContent) {
            overviewContent.innerHTML = `<p>${aiAnalysis.COMPANY_OVERVIEW}</p>`;
            overviewContent.classList.add('fade-in-content');
        }
    }
}

/**
 * 要素のテキストを更新
 */
function updateElement(selector, content) {
    let element;
    if (typeof selector === 'string') {
        element = document.querySelector(selector);
    } else {
        element = selector;
    }
    
    if (element && content) {
        element.textContent = content;
        element.classList.add('fade-in-content');
    }
}

/**
 * AI分析エラーを表示
 */
function showAIAnalysisError(message) {
    const analysisContainer = document.querySelector('#ai-analysis');
    if (!analysisContainer) return;
    
    // ローディングコンテンツを削除
    const loadingContent = analysisContainer.querySelector('.ai-loading-content');
    if (loadingContent) {
        loadingContent.remove();
    }
    
    // エラーメッセージを表示
    const errorDiv = document.createElement('div');
    errorDiv.className = 'ai-analysis-error';
    errorDiv.innerHTML = `
        <div class="error-content">
            <h3>🚫 AI分析でエラーが発生しました</h3>
            <p>${message}</p>
            <button onclick="retryAIAnalysis()" class="btn">再試行</button>
        </div>
    `;
    
    analysisContainer.appendChild(errorDiv);
    
    showNotification('AI分析でエラーが発生しました', 'error');
}

/**
 * AI分析を再試行
 */
function retryAIAnalysis() {
    window.aiAnalysisLoaded = false;
    window.aiAnalysisLoading = false;
    
    // エラーコンテンツを削除
    const errorContent = document.querySelector('.ai-analysis-error');
    if (errorContent) {
        errorContent.remove();
    }
    
    // 再実行
    loadAIAnalysis();
}

/**
 * 通知メッセージを表示
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // 3秒後に削除
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// ページ読み込み時の初期化
document.addEventListener('DOMContentLoaded', function() {
    // デフォルトで財務データタブを表示
    showTab('financial-data');
    
    // タブボタンのクリックイベントを設定
    const tabButtons = document.getElementsByClassName('tab-button');
    for (let button of tabButtons) {
        button.addEventListener('click', function() {
            const tabName = this.textContent.includes('財務データ') ? 'financial-data' : 'ai-analysis';
            showTab(tabName);
        });
    }
    
    // 認証済みユーザーの場合、ページロード後に自動でAI分析を開始
    // 少し遅延を入れてページレンダリングを優先
    if (document.querySelector('.login-required-section') === null) {
        setTimeout(() => {
            if (!window.aiAnalysisLoaded && !window.aiAnalysisLoading) {
                loadAIAnalysis();
            }
            // 企業概要セクションも自動でロード
            if (!window.companyOverviewLoaded && !window.companyOverviewLoading) {
                loadCompanyOverview();
            }
        }, 500); // 500ms後に開始
    }
});

/**
 * 企業概要セクションをAJAXで読み込む
 */
async function loadCompanyOverview() {
    // すでに読み込み中または完了している場合は何もしない
    if (window.companyOverviewLoading || window.companyOverviewLoaded) {
        return;
    }
    
    window.companyOverviewLoading = true;
    
    // EDINETコードを取得
    const pathParts = window.location.pathname.split('/');
    const edinetCode = pathParts[pathParts.indexOf('company') + 1];
    
    if (!edinetCode) {
        console.error('EDINETコードが見つかりません');
        return;
    }
    
    try {
        console.log(`Fetching company overview for edinet code: ${edinetCode}`);
        const response = await fetch(`/api/company-overview/${edinetCode}/`, {
            method: 'GET',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (response.ok && data.company_overview) {
            // 企業概要を表示
            displayCompanyOverview(data.company_overview);
            window.companyOverviewLoaded = true;
        } else {
            throw new Error(data.error || '企業概要の取得に失敗しました');
        }
        
    } catch (error) {
        console.error('企業概要エラー:', error);
        showCompanyOverviewError(error.message);
    } finally {
        window.companyOverviewLoading = false;
    }
}

/**
 * 企業概要を表示
 */
function displayCompanyOverview(companyOverview) {
    const overviewContent = document.querySelector('.company-overview-content');
    if (overviewContent) {
        overviewContent.innerHTML = `<p>${companyOverview}</p>`;
        overviewContent.classList.add('fade-in-content');
    }
}

/**
 * 企業概要エラーを表示
 */
function showCompanyOverviewError(message) {
    const overviewContent = document.querySelector('.company-overview-content');
    if (overviewContent) {
        overviewContent.innerHTML = `
            <div class="error-content">
                <p>企業概要の取得でエラーが発生しました: ${message}</p>
                <button onclick="retryCompanyOverview()" class="btn btn-small">再試行</button>
            </div>
        `;
    }
}

/**
 * 企業概要を再試行
 */
function retryCompanyOverview() {
    window.companyOverviewLoaded = false;
    window.companyOverviewLoading = false;
    loadCompanyOverview();
}

/**
 * スムーズスクロール機能（必要に応じて）
 * @param {string} elementId - スクロール先の要素ID
 */
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

/**
 * 各チャートに対して3シナリオ分析をロード
 */
function loadChartScenarioAnalysis() {
    console.log('Loading chart scenario analysis...');
    const chartAnalysisSections = document.querySelectorAll('.chart-ai-analysis');
    console.log(`Found ${chartAnalysisSections.length} chart analysis sections`);
    
    chartAnalysisSections.forEach((section, index) => {
        const chartType = section.getAttribute('data-chart-type');
        const metric = section.getAttribute('data-metric');
        console.log(`Section ${index}: chart-type = ${chartType}, metric = ${metric}`);
        console.log(`Section ${index} element:`, section);
        
        if (chartType) {
            console.log(`Loading scenario analysis for chart type: ${chartType}`);
            loadScenarioAnalysisInternal(chartType, section);
        } else {
            console.warn(`Section ${index} has no data-chart-type attribute`);
            console.warn(`Section ${index} HTML:`, section.outerHTML);
        }
    });
}

/**
 * 新しい3シナリオ分析読み込み関数（推奨）
 * @param {string} edinetCode - EDINETコード
 * @param {string} chartType - チャートタイプ ('sales' または 'profit')
 * @returns {Promise<Object>} - シナリオ分析結果
 */
async function loadScenarioAnalysis(edinetCode, chartType) {
    try {
        console.log(`Fetching scenario analysis for ${chartType} chart: ${edinetCode}`);
        
        const response = await fetch(`/api/scenario-analysis/${edinetCode}/${chartType}/`, {
            method: 'GET',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (response.ok && data.scenario_analysis) {
            return data.scenario_analysis;
        } else {
            throw new Error(data.error || 'シナリオ分析の取得に失敗しました');
        }
        
    } catch (error) {
        console.error('シナリオ分析エラー:', error);
        throw error;
    }
}

/**
 * 3シナリオ分析をAJAXで取得（内部使用）
 * @param {string} chartType - チャートタイプ ('sales' または 'profit')
 * @param {HTMLElement} targetSection - 更新対象のセクション
 */
async function loadScenarioAnalysisInternal(chartType, targetSection) {
    console.log(`=== loadScenarioAnalysisInternal START ===`);
    console.log(`chartType: ${chartType}`);
    console.log(`targetSection:`, targetSection);
    console.log(`targetSection.dataset.loaded: ${targetSection.dataset.loaded}`);
    
    // すでに読み込み済みの場合はスキップ
    if (targetSection.dataset.loaded === 'true') {
        console.log(`Chart ${chartType} already loaded, skipping...`);
        return;
    }
    
    // EDINETコードを取得
    const pathParts = window.location.pathname.split('/');
    const edinetCode = pathParts[pathParts.indexOf('company') + 1];
    
    if (!edinetCode) {
        console.error('EDINETコードが見つかりません');
        return;
    }
    
    console.log(`Starting scenario analysis fetch for ${chartType}, EDINET: ${edinetCode}`);
    console.log(`API URL: /api/scenario-analysis/${edinetCode}/${chartType}/`);
    
    try {
        const response = await fetch(`/api/scenario-analysis/${edinetCode}/${chartType}/`, {
            method: 'GET',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/json',
            }
        });
        
        console.log(`Response status for ${chartType}: ${response.status}`);
        console.log(`Response headers:`, response.headers);
        
        const data = await response.json();
        console.log(`Response data for ${chartType}:`, data);
        
        if (response.ok && data.scenario_analysis) {
            console.log(`Successfully received scenario analysis for ${chartType}`);
            console.log(`Scenario analysis data:`, data.scenario_analysis);
            // シナリオ分析を表示
            updateChartScenarioAnalysis(targetSection, data.scenario_analysis);
            targetSection.dataset.loaded = 'true';
            console.log(`Successfully updated chart ${chartType}`);
        } else {
            throw new Error(data.error || 'シナリオ分析の取得に失敗しました');
        }
        
    } catch (error) {
        console.error(`シナリオ分析エラー (${chartType}):`, error);
        showChartScenarioError(targetSection, error.message);
    }
    
    console.log(`=== loadScenarioAnalysisInternal END ===`);
}

/**
 * チャートシナリオ分析を更新
 * @param {HTMLElement} targetSection - 更新対象のセクション
 * @param {Object} scenarioAnalysis - シナリオ分析データ
 */
function updateChartScenarioAnalysis(targetSection, scenarioAnalysis) {
    console.log(`=== updateChartScenarioAnalysis START ===`);
    console.log(`targetSection:`, targetSection);
    console.log(`scenarioAnalysis:`, scenarioAnalysis);
    
    const scenarios = targetSection.querySelectorAll('.scenario');
    console.log(`Found ${scenarios.length} scenario elements`);
    
    scenarios.forEach((scenario, index) => {
        const scenarioType = scenario.classList.contains('optimistic') ? 'optimistic' :
                           scenario.classList.contains('current') ? 'current' :
                           scenario.classList.contains('pessimistic') ? 'pessimistic' : null;
        
        console.log(`Scenario ${index}: type = ${scenarioType}`);
        console.log(`Scenario ${index} element:`, scenario);
        
        if (scenarioType && scenarioAnalysis[scenarioType]) {
            const explanationElement = scenario.querySelector('.scenario-explanation');
            console.log(`Scenario ${index} explanation element:`, explanationElement);
            
            if (explanationElement) {
                console.log(`Updating scenario ${scenarioType} with text: ${scenarioAnalysis[scenarioType]}`);
                explanationElement.textContent = scenarioAnalysis[scenarioType];
                explanationElement.classList.add('fade-in-content');
                console.log(`Successfully updated scenario ${scenarioType}`);
            } else {
                console.warn(`No explanation element found for scenario ${scenarioType}`);
            }
        } else {
            console.warn(`No data found for scenario type ${scenarioType}`);
        }
    });
    
    console.log(`=== updateChartScenarioAnalysis END ===`);
}

/**
 * チャートシナリオエラーを表示
 * @param {HTMLElement} targetSection - 更新対象のセクション
 * @param {string} message - エラーメッセージ
 */
function showChartScenarioError(targetSection, message) {
    const scenarios = targetSection.querySelectorAll('.scenario .scenario-explanation');
    scenarios.forEach(explanation => {
        explanation.textContent = `エラー: ${message}`;
        explanation.classList.add('error-text');
    });
}