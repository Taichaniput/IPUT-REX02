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
        const response = await fetch(`/api/ai-analysis/${edinetCode}/`, {
            method: 'GET',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
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
    // 成長率分析の更新（growth-scenariosクラス内のみ）
    if (aiAnalysis.GROWTH_SCENARIOS) {
        const growthScenarios = document.querySelectorAll('.growth-scenarios');
        growthScenarios.forEach(container => {
            updateElement(container.querySelector('.scenario.optimistic .scenario-explanation'), aiAnalysis.GROWTH_SCENARIOS.optimistic);
            updateElement(container.querySelector('.scenario.current .scenario-explanation'), aiAnalysis.GROWTH_SCENARIOS.current);
            updateElement(container.querySelector('.scenario.pessimistic .scenario-explanation'), aiAnalysis.GROWTH_SCENARIOS.pessimistic);
        });
    }
    
    // 純利益分析の更新（profit-scenariosクラス内のみ）
    if (aiAnalysis.PROFIT_SCENARIOS) {
        const profitScenarios = document.querySelector('.profit-scenarios');
        if (profitScenarios) {
            updateElement(profitScenarios.querySelector('.scenario.optimistic .scenario-explanation'), aiAnalysis.PROFIT_SCENARIOS.optimistic);
            updateElement(profitScenarios.querySelector('.scenario.current .scenario-explanation'), aiAnalysis.PROFIT_SCENARIOS.current);
            updateElement(profitScenarios.querySelector('.scenario.pessimistic .scenario-explanation'), aiAnalysis.PROFIT_SCENARIOS.pessimistic);
        }
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
        }, 500); // 500ms後に開始
    }
});

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