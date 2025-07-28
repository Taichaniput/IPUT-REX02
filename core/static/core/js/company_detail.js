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
        console.log('DEBUG: AI analysis tab selected');
        console.log('DEBUG: aiAnalysisLoaded:', window.aiAnalysisLoaded);
        console.log('DEBUG: aiAnalysisLoading:', window.aiAnalysisLoading);
        
        // 分析結果がまだ読み込まれていない場合のみAJAXで取得
        if (!window.aiAnalysisLoaded && !window.aiAnalysisLoading) {
            console.log('DEBUG: Starting AI analysis load...');
            loadAIAnalysis();
        } else if (window.aiAnalysisLoaded) {
            console.log('DEBUG: AI analysis already loaded, checking charts...');
            // 既に読み込み済みだが、チャートが表示されていない可能性がある場合の再描画
            setTimeout(async () => {
                await renderChartsWithSynchronization();
            }, 100);
        }
        // チャートシナリオ分析もここでロード
        loadChartScenarioAnalysis();
    }
}

/**
 * AI分析を非同期で読み込む
 */
async function loadAIAnalysis() {
    if (window.aiAnalysisLoading || window.aiAnalysisLoaded) {
        return;
    }
    
    window.aiAnalysisLoading = true;
    
    const pathParts = window.location.pathname.split('/');
    const edinetCode = pathParts[pathParts.indexOf('company') + 1];
    
    if (!edinetCode) {
        console.error('EDINETコードが見つかりません');
        return;
    }
    
    try {
        showAIAnalysisLoading();
        
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
        console.log('=== FRONTEND AI ANALYSIS DEBUG START ===');
        console.log('Response data (full AI analysis):', data);
        
        if (data.warnings) {
            console.warn('Server warnings:', data.warnings);
        }
        
        if (response.ok && data.success) {
            // データの詳細検証
            console.log('Data structure validation:');
            console.log('- Has ai_analysis:', !!data.ai_analysis);
            console.log('- Has prediction_results:', !!data.ai_analysis?.prediction_results);
            console.log('- Has cluster_info:', !!data.ai_analysis?.cluster_info);
            console.log('- Has positioning_info:', !!data.ai_analysis?.positioning_info);
            
            if (data.ai_analysis?.prediction_results) {
                console.log('Prediction results keys:', Object.keys(data.ai_analysis.prediction_results));
                for (const [key, result] of Object.entries(data.ai_analysis.prediction_results)) {
                    console.log(`- ${key}: has chart_data=${!!result.chart_data}, has predictions=${!!result.predictions}`);
                }
            }
            
            if (data.ai_analysis?.cluster_info) {
                console.log('Cluster info keys:', Object.keys(data.ai_analysis.cluster_info));
                console.log('- cluster_id:', data.ai_analysis.cluster_info.cluster_id);
                console.log('- has chart_data:', !!data.ai_analysis.cluster_info.chart_data);
            }
            
            if (data.ai_analysis?.positioning_info) {
                console.log('Positioning info keys:', Object.keys(data.ai_analysis.positioning_info));
                console.log('- growth_score:', data.ai_analysis.positioning_info.growth_score);
                console.log('- stability_score:', data.ai_analysis.positioning_info.stability_score);
                console.log('- has chart:', !!data.ai_analysis.positioning_info.chart);
            }
            
            // グローバル変数に格納
            window.predictionResults = data.ai_analysis.prediction_results; 
            window.clusterInfo = data.ai_analysis.cluster_info;
            window.positioningInfo = data.ai_analysis.positioning_info;
            
            console.log('Global variables set:');
            console.log('- window.predictionResults:', !!window.predictionResults);
            console.log('- window.clusterInfo:', !!window.clusterInfo);
            console.log('- window.positioningInfo:', !!window.positioningInfo);
            
            displayAIAnalysis(data.ai_analysis);
            window.aiAnalysisLoaded = true;
            console.log('=== FRONTEND AI ANALYSIS DEBUG END ===');
        } else {
            console.error('Analysis failed:', data.error);
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
    
    const loadingContent = analysisContainer.querySelector('.ai-loading-content');
    if (loadingContent) {
        loadingContent.remove();
    }
    
    const existingContent = analysisContainer.querySelectorAll(':not(.ai-loading-content)');
    existingContent.forEach(el => el.style.display = 'block');
    
    updateAnalysisContent(aiAnalysis);
    
    // データの整合性チェックと同期化された描画
    setTimeout(async () => {
        await renderChartsWithSynchronization();
    }, 100); // DOMが完全に更新されるまで少し待つ
    
    showNotification('AI分析が完了しました！', 'success');
}

/**
 * チャートの同期化された描画処理
 */
async function renderChartsWithSynchronization() {
    console.log('DEBUG: Starting synchronized chart rendering');
    console.log('DEBUG: Available data:', {
        predictionResults: !!window.predictionResults,
        clusterInfo: !!window.clusterInfo,
        positioningInfo: !!window.positioningInfo
    });

    // 1. 予測チャートの描画（優先度高）
    try {
        if (window.predictionResults) {
            console.log('DEBUG: Rendering prediction charts...');
            
            // 売上高予測チャート
            if (window.predictionResults.net_sales?.chart_data) {
                console.log('DEBUG: Rendering sales chart...');
                const salesCanvas = document.getElementById('sales-chart');
                if (salesCanvas) {
                    await renderChart('sales-chart', window.predictionResults.net_sales.chart_data);
                } else {
                    console.log('DEBUG: Sales chart canvas not found, creating dynamic chart');
                    // canvas要素が存在しない場合は動的に作成
                    createAndRenderChart('sales', window.predictionResults.net_sales);
                }
            } else {
                console.warn('WARNING: Sales chart data not available');
                showNoDataMessage('sales', '売上高予測データが不足しています');
            }
            
            // 純利益予測チャート
            if (window.predictionResults.net_income?.chart_data) {
                console.log('DEBUG: Rendering profit chart...');
                const profitCanvas = document.getElementById('profit-chart');
                if (profitCanvas) {
                    await renderChart('profit-chart', window.predictionResults.net_income.chart_data);
                } else {
                    console.log('DEBUG: Profit chart canvas not found, creating dynamic chart');
                    // canvas要素が存在しない場合は動的に作成
                    createAndRenderChart('profit', window.predictionResults.net_income);
                }
            } else {
                console.warn('WARNING: Profit chart data not available');
                showNoDataMessage('profit', '純利益予測データが不足しています');
            }
        } else {
            console.warn('WARNING: No prediction results available for chart rendering');
            // 予測結果がない場合の代替表示
            showNoDataMessage('sales', '予測分析データが利用できません（データ不足）');
            showNoDataMessage('profit', '予測分析データが利用できません（データ不足）');
        }
    } catch (error) {
        console.error('ERROR: Failed to render prediction charts:', error);
        showNoDataMessage('sales', `予測チャート描画エラー: ${error.message}`);
        showNoDataMessage('profit', `予測チャート描画エラー: ${error.message}`);
    }

    // 2. クラスタリングチャートの描画
    try {
        if (window.clusterInfo?.chart_data) {
            console.log('DEBUG: Rendering clustering chart...');
            console.log('DEBUG: Clustering chart data structure:', window.clusterInfo.chart_data);
            const clusteringCanvas = document.getElementById('clustering-chart');
            if (clusteringCanvas) {
                renderClusteringChart('clustering-chart', window.clusterInfo.chart_data);
            } else {
                console.log('DEBUG: Clustering chart canvas not found');
                showChartError('clustering-chart', 'クラスタリングチャート要素が見つかりません');
            }
        } else {
            console.warn('WARNING: No clustering info available for chart rendering');
            console.log('DEBUG: clusterInfo structure:', window.clusterInfo);
            showChartError('clustering-chart', 'クラスタリング分析データが利用できません（データ不足）');
        }
    } catch (error) {
        console.error('ERROR: Failed to render clustering chart:', error);
        showChartError('clustering-chart', `クラスタリングチャート描画エラー: ${error.message}`);
    }

    // 3. ポジショニングチャートの描画
    try {
        console.log('DEBUG: Checking positioning chart data...');
        console.log('DEBUG: window.positioningInfo:', window.positioningInfo);
        
        // ポジショニングデータの構造を確認
        const hasChartData = window.positioningInfo?.chart || window.positioningInfo?.datasets;
        console.log('DEBUG: hasChartData:', hasChartData);
        
        if (hasChartData) {
            console.log('DEBUG: Rendering positioning chart...');
            const positioningCanvas = document.getElementById('positioning-chart');
            if (positioningCanvas) {
                // chart プロパティがある場合はそれを使用、無い場合は全体を使用
                const chartData = window.positioningInfo.chart || window.positioningInfo;
                console.log('DEBUG: Using chart data:', chartData);
                renderPositioningChart('positioning-chart', chartData);
            } else {
                console.warn('WARNING: Positioning chart canvas not found');
                showChartError('positioning-chart', 'ポジショニングチャートの表示要素が見つかりません');
            }
        } else {
            console.warn('WARNING: No positioning chart data available');
            console.log('DEBUG: positioningInfo keys:', window.positioningInfo ? Object.keys(window.positioningInfo) : 'null');
            
            // Note: ポジショニング分析は updatePositioningAnalysis で別途処理される
            console.log('DEBUG: Positioning analysis will be handled by updatePositioningAnalysis');
        }
    } catch (error) {
        console.error('ERROR: Failed to render positioning chart:', error);
        showChartError('positioning-chart', `ポジショニングチャート描画エラー: ${error.message}`);
    }

    // 4. シナリオ分析の読み込み（チャート描画後）
    setTimeout(() => {
        try {
            console.log('DEBUG: Loading scenario analysis...');
            // canvas要素の存在確認後にシナリオ分析をロード
            const salesCanvas = document.getElementById('sales-chart');
            const profitCanvas = document.getElementById('profit-chart');
            console.log('DEBUG: Canvas elements found - sales:', !!salesCanvas, 'profit:', !!profitCanvas);
            
            loadChartScenarioAnalysis();
        } catch (error) {
            console.error('ERROR: Failed to load scenario analysis:', error);
        }
    }, 500); // チャート描画と動的要素作成の完了を待つ

    console.log('DEBUG: Synchronized chart rendering completed');
}

/**
 * 統合テスト関数 - チャート機能の動作確認
 */
function runChartIntegrationTest() {
    console.log('=== CHART INTEGRATION TEST START ===');
    
    const testResults = {
        chartLibrary: false,
        canvasElements: {},
        dataAvailability: {},
        renderingAttempts: {}
    };
    
    // 1. Chart.js ライブラリのテスト
    console.log('1. Testing Chart.js library availability...');
    testResults.chartLibrary = typeof Chart !== 'undefined';
    console.log(`   Chart.js available: ${testResults.chartLibrary}`);
    
    // 2. Canvas要素の存在確認
    console.log('2. Testing canvas elements...');
    const canvasIds = ['sales-chart', 'profit-chart', 'clustering-chart', 'positioning-chart'];
    canvasIds.forEach(canvasId => {
        const element = document.getElementById(canvasId);
        testResults.canvasElements[canvasId] = {
            exists: !!element,
            visible: element ? element.getBoundingClientRect().width > 0 : false,
            inDOM: element ? document.contains(element) : false
        };
        console.log(`   ${canvasId}:`, testResults.canvasElements[canvasId]);
    });
    
    // 3. データ可用性の確認
    console.log('3. Testing data availability...');
    testResults.dataAvailability = {
        predictionResults: {
            exists: !!window.predictionResults,
            netSales: !!(window.predictionResults?.net_sales?.chart_data),
            netIncome: !!(window.predictionResults?.net_income?.chart_data)
        },
        clusterInfo: {
            exists: !!window.clusterInfo,
            chartData: !!(window.clusterInfo?.chart_data)
        },
        positioningInfo: {
            exists: !!window.positioningInfo,
            chartData: !!(window.positioningInfo?.chart_data)
        }
    };
    console.log('   Data availability:', testResults.dataAvailability);
    
    // 4. 描画機能のテスト
    console.log('4. Testing chart rendering functions...');
    if (testResults.chartLibrary) {
        // 売上高チャートのテスト
        if (testResults.canvasElements['sales-chart'].exists && testResults.dataAvailability.predictionResults.netSales) {
            console.log('   Testing sales chart rendering...');
            const chartData = window.predictionResults.net_sales.chart_data;
            console.log('DEBUG: Sales chart data structure:', {
                hasLabels: !!chartData?.labels,
                labelsLength: chartData?.labels?.length,
                hasDatasets: !!chartData?.datasets,
                datasetsLength: chartData?.datasets?.length,
                type: chartData?.type,
                title: chartData?.title
            });
            
            // データ構造の検証
            if (chartData && chartData.labels && chartData.datasets && chartData.labels.length > 0 && chartData.datasets.length > 0) {
                try {
                    renderChart('sales-chart', chartData);
                    testResults.renderingAttempts.salesChart = 'success';
                    console.log('   ✓ Sales chart rendering succeeded');
                } catch (error) {
                    testResults.renderingAttempts.salesChart = `error: ${error.message}`;
                    console.error('   ✗ Sales chart rendering failed:', error);
                }
            } else {
                testResults.renderingAttempts.salesChart = 'skipped: invalid data structure';
                console.log('   ⚠ Sales chart rendering skipped due to invalid data structure');
            }
        }
        
        // 利益チャートのテスト
        if (testResults.canvasElements['profit-chart'].exists && testResults.dataAvailability.predictionResults.netIncome) {
            console.log('   Testing profit chart rendering...');
            const chartData = window.predictionResults.net_income.chart_data;
            console.log('DEBUG: Profit chart data structure:', {
                hasLabels: !!chartData?.labels,
                labelsLength: chartData?.labels?.length,
                hasDatasets: !!chartData?.datasets,
                datasetsLength: chartData?.datasets?.length,
                type: chartData?.type,
                title: chartData?.title
            });
            
            // データ構造の検証
            if (chartData && chartData.labels && chartData.datasets && chartData.labels.length > 0 && chartData.datasets.length > 0) {
                try {
                    renderChart('profit-chart', chartData);
                    testResults.renderingAttempts.profitChart = 'success';
                    console.log('   ✓ Profit chart rendering succeeded');
                } catch (error) {
                    testResults.renderingAttempts.profitChart = `error: ${error.message}`;
                    console.error('   ✗ Profit chart rendering failed:', error);
                }
            } else {
                testResults.renderingAttempts.profitChart = 'skipped: invalid data structure';
                console.log('   ⚠ Profit chart rendering skipped due to invalid data structure');
            }
        }
        
        // クラスタリングチャートのテスト
        if (testResults.canvasElements['clustering-chart'].exists && testResults.dataAvailability.clusterInfo.chartData) {
            console.log('   Testing clustering chart rendering...');
            const chartData = window.clusterInfo.chart_data;
            console.log('DEBUG: Clustering chart data structure:', {
                hasDatasets: !!chartData?.datasets,
                datasetsLength: chartData?.datasets?.length,
                type: chartData?.type,
                title: chartData?.title
            });
            
            // データ構造の検証（散布図なのでlabelsは不要）
            if (chartData && chartData.datasets && chartData.datasets.length > 0) {
                try {
                    renderClusteringChart('clustering-chart', chartData);
                    testResults.renderingAttempts.clusteringChart = 'success';
                    console.log('   ✓ Clustering chart rendering succeeded');
                } catch (error) {
                    testResults.renderingAttempts.clusteringChart = `error: ${error.message}`;
                    console.error('   ✗ Clustering chart rendering failed:', error);
                }
            } else {
                testResults.renderingAttempts.clusteringChart = 'skipped: invalid data structure';
                console.log('   ⚠ Clustering chart rendering skipped due to invalid data structure');
            }
        }
        
        // ポジショニングチャートのテスト
        if (testResults.canvasElements['positioning-chart'].exists && testResults.dataAvailability.positioningInfo.chartData) {
            console.log('   Testing positioning chart rendering...');
            const chartData = window.positioningInfo.chart_data;
            console.log('DEBUG: Positioning chart data structure:', {
                hasDatasets: !!chartData?.datasets,
                datasetsLength: chartData?.datasets?.length,
                type: chartData?.type,
                title: chartData?.title
            });
            
            // データ構造の検証（散布図なのでlabelsは不要）
            if (chartData && chartData.datasets && chartData.datasets.length > 0) {
                try {
                    renderPositioningChart('positioning-chart', chartData);
                    testResults.renderingAttempts.positioningChart = 'success';
                    console.log('   ✓ Positioning chart rendering succeeded');
                } catch (error) {
                    testResults.renderingAttempts.positioningChart = `error: ${error.message}`;
                    console.error('   ✗ Positioning chart rendering failed:', error);
                }
            } else {
                testResults.renderingAttempts.positioningChart = 'skipped: invalid data structure';
                console.log('   ⚠ Positioning chart rendering skipped due to invalid data structure');
            }
        }
    }
    
    // 5. テスト結果のサマリー
    console.log('5. Test Summary:');
    const totalTests = Object.keys(testResults.renderingAttempts).length;
    const successfulTests = Object.values(testResults.renderingAttempts).filter(result => result === 'success').length;
    console.log(`   Charts tested: ${totalTests}`);
    console.log(`   Successful: ${successfulTests}`);
    console.log(`   Failed: ${totalTests - successfulTests}`);
    
    if (totalTests === 0) {
        console.warn('   ⚠️ No charts could be tested (missing data or canvas elements)');
    } else if (successfulTests === totalTests) {
        console.log('   ✅ All chart tests passed!');
    } else {
        console.error('   ❌ Some chart tests failed');
    }
    
    console.log('=== CHART INTEGRATION TEST END ===');
    
    return testResults;
}

/**
 * デバッグ情報の包括的出力
 */
function generateDebugReport() {
    console.log('=== COMPREHENSIVE DEBUG REPORT ===');
    
    const report = {
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        viewport: {
            width: window.innerWidth,
            height: window.innerHeight
        },
        environment: {
            chartJsLoaded: typeof Chart !== 'undefined',
            aiAnalysisLoaded: !!window.aiAnalysisLoaded,
            currentTab: document.querySelector('.tab-button.active')?.getAttribute('data-tab') || 'unknown'
        },
        dataStatus: {
            predictionResults: window.predictionResults ? Object.keys(window.predictionResults) : null,
            clusterInfo: !!window.clusterInfo,
            positioningInfo: !!window.positioningInfo
        },
        domElements: {},
        chartInstances: {}
    };
    
    // DOM要素の状態確認
    const importantElements = [
        'sales-chart', 'profit-chart', 'clustering-chart', 'positioning-chart',
        'ai-analysis', 'financial-data'
    ];
    
    importantElements.forEach(elementId => {
        const element = document.getElementById(elementId);
        report.domElements[elementId] = {
            exists: !!element,
            visible: element ? getComputedStyle(element).display !== 'none' : false,
            dimensions: element ? element.getBoundingClientRect() : null,
            hasChart: element ? !!element.chart : false
        };
    });
    
    // チャートインスタンスの状態確認
    ['sales-chart', 'profit-chart', 'clustering-chart', 'positioning-chart'].forEach(canvasId => {
        const canvas = document.getElementById(canvasId);
        if (canvas && canvas.chart) {
            report.chartInstances[canvasId] = {
                type: canvas.chart.config.type,
                datasetCount: canvas.chart.data.datasets ? canvas.chart.data.datasets.length : 0,
                hasData: !!(canvas.chart.data.labels && canvas.chart.data.labels.length > 0)
            };
        }
    });
    
    console.log('Debug Report:', report);
    
    // ローカルストレージに保存（デバッグ用）
    try {
        localStorage.setItem('chartDebugReport', JSON.stringify(report));
        console.log('Debug report saved to localStorage as "chartDebugReport"');
    } catch (error) {
        console.warn('Could not save debug report to localStorage:', error);
    }
    
    return report;
}

// ページ読み込み完了時の自動テスト実行
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM content loaded, scheduling integration test...');
    
    // AI分析が完了した後にテストを実行
    const checkAndTest = () => {
        if (window.aiAnalysisLoaded) {
            setTimeout(() => {
                runChartIntegrationTest();
                generateDebugReport();
            }, 1000); // AI分析完了から1秒後にテスト実行
        } else {
            setTimeout(checkAndTest, 500); // 0.5秒後に再チェック
        }
    };
    
    setTimeout(checkAndTest, 2000); // 初期ロードから2秒後に開始
});

// Primary clustering chart function - removed duplicate

async function renderChart(canvasId, chartData) {
    try {
        console.log(`DEBUG: Starting chart render for ID: ${canvasId}`);
        console.log('DEBUG: CRITICAL - Canvas ID being used:', canvasId);
        console.log('DEBUG: Chart data structure:', {
            hasLabels: !!chartData?.labels,
            labelsLength: chartData?.labels?.length,
            hasDatasets: !!chartData?.datasets,
            datasetsLength: chartData?.datasets?.length,
            title: chartData?.title,
            ylabel: chartData?.ylabel
        });
        
        // Chart.js ライブラリの確認
        if (typeof Chart === 'undefined') {
            console.error('ERROR: Chart.js library is not loaded');
            showChartError(canvasId, 'Chart.js ライブラリが読み込まれていません');
            return;
        }
        
        // Canvas要素の存在確認
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`ERROR: Canvas element with ID ${canvasId} not found`);
            console.error('ERROR: Available canvas elements:', Array.from(document.querySelectorAll('canvas')).map(c => c.id));
            console.error('ERROR: All elements with IDs containing "chart":', Array.from(document.querySelectorAll('[id*="chart"]')).map(e => e.id));
            throw new Error(`Canvas element with ID "${canvasId}" not found`);
        }
        
        // Canvas要素の表示状態確認
        const canvasRect = ctx.getBoundingClientRect();
        console.log(`DEBUG: Canvas ${canvasId} dimensions:`, {
            width: canvasRect.width,
            height: canvasRect.height,
            visible: canvasRect.width > 0 && canvasRect.height > 0
        });
        
        // データ検証
        if (!chartData) {
            console.error(`ERROR: No chart data provided for ${canvasId}`);
            showChartError(canvasId, 'チャートデータがありません');
            return;
        }
        
        // 散布図の場合は専用関数に転送
        if (chartData.type === 'scatter' || canvasId.includes('positioning') || canvasId.includes('cluster')) {
            console.log(`DEBUG: Detected scatter chart for ${canvasId}, redirecting to appropriate function`);
            if (canvasId.includes('positioning')) {
                return renderPositioningChart(canvasId, chartData);
            } else if (canvasId.includes('cluster')) {
                return renderClusteringChart(canvasId, chartData);
            }
        }
        
        // Line chart用データ検証と補完（散布図以外）
        if (!chartData.labels || !Array.isArray(chartData.labels) || chartData.labels.length === 0) {
            console.warn(`WARNING: Invalid labels for line chart ${canvasId}, using default labels`);
            chartData.labels = ['2022', '2023', '2024'];
        }
        
        if (!chartData.datasets || !Array.isArray(chartData.datasets) || chartData.datasets.length === 0) {
            console.warn(`WARNING: Invalid datasets for chart ${canvasId}, using default dataset`);
            chartData.datasets = [{
                label: 'データなし',
                data: [0, 0, 0],
                borderColor: '#ccc',
                backgroundColor: 'rgba(204, 204, 204, 0.1)',
                borderWidth: 2
            }];
        }
        
        // 既存チャートの破棄
        const existingChart = Chart.getChart(ctx);
        if (existingChart) {
            existingChart.destroy();
        }
        
        // Chart.js設定の検証
        const chartConfig = {
            type: chartData.type || 'line',
            data: {
                labels: chartData.labels,
                datasets: chartData.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                aspectRatio: 2,
                plugins: {
                    title: {
                        display: true,
                        text: chartData.title || 'チャート',
                        font: {
                            size: 18,
                            weight: 'bold'
                        },
                        padding: 20
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'category',
                        title: {
                            display: true,
                            text: '年度',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: chartData.ylabel || '値',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        };
        
        console.log(`DEBUG: Chart config for ${canvasId}:`, chartConfig);
        
        // Chart.js初期化（再試行機能付き）
        let chart = null;
        let retryCount = 0;
        const maxRetries = 3;
        
        while (retryCount < maxRetries && !chart) {
            try {
                console.log(`DEBUG: Chart ${canvasId} creation attempt ${retryCount + 1}/${maxRetries}`);
                
                // Canvas要素の初期化
                ctx.width = ctx.parentElement.clientWidth || 800;
                ctx.height = ctx.parentElement.clientHeight || 400;
                ctx.style.width = '100%';
                ctx.style.height = '100%';
                
                chart = new Chart(ctx, {
                    ...chartConfig,
                    options: {
                        ...chartConfig.options,
                        animation: {
                            duration: retryCount > 0 ? 0 : 1000 // 再試行時はアニメーション無効
                        }
                    }
                });
                
                ctx.chart = chart;
                console.log(`SUCCESS: Chart ${canvasId} rendered successfully on attempt ${retryCount + 1}`);
                break;
                
            } catch (createError) {
                retryCount++;
                console.error(`ERROR: Chart ${canvasId} creation failed on attempt ${retryCount}:`, createError);
                
                if (retryCount < maxRetries) {
                    console.log(`DEBUG: Retrying chart ${canvasId} creation in 100ms...`);
                    await new Promise(resolve => setTimeout(resolve, 100));
                } else {
                    console.error(`ERROR: All chart ${canvasId} creation attempts failed`);
                    throw createError;
                }
            }
        }
        
        if (!chart) {
            throw new Error(`Chart ${canvasId} creation failed after all retry attempts`);
        }
        
        // チャート初期化後のサイズ確認
        setTimeout(() => {
            const finalRect = ctx.getBoundingClientRect();
            console.log(`DEBUG: Final chart ${canvasId} dimensions:`, {
                width: finalRect.width,
                height: finalRect.height,
                chartExists: !!ctx.chart
            });
        }, 100);
        
    } catch (error) {
        console.error(`ERROR: Failed to render chart ${canvasId}:`, error);
        console.error('ERROR Stack:', error.stack);
        showChartError(canvasId, `チャート描画エラー: ${error.message}`);
    }
}

// Duplicate showChartError function removed - using primary version below

/**
 * 従業員数推移チャートを描画
 */
function renderEmployeeChart(employeeDataJson) {
    console.log('DEBUG: Rendering employee chart...');
    const employeeData = JSON.parse(employeeDataJson);
    const canvas = document.getElementById('employeeChart');

    if (!canvas) {
        console.error('Canvas element with ID "employeeChart" not found.');
        return;
    }

    const ctx = canvas.getContext('2d');

    // 既存のChart.jsインスタンスがあれば破棄
    if (window.employeeChartInstance) {
        window.employeeChartInstance.destroy();
    }

    const years = employeeData.map(item => item.year);
    const employees = employeeData.map(item => item.employees);

    window.employeeChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: years,
            datasets: [{
                label: '従業員数',
                data: employees,
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '従業員数推移',
                    font: {
                        size: 18,
                        weight: 'bold'
                    },
                    padding: 20
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '年度',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '従業員数',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        callback: function(value) {
                            return value.toLocaleString(); // 数値をカンマ区切りで表示
                        }
                    }
                }
            }
        }
    });
    console.log('DEBUG: Employee chart rendered successfully.');
}

/**
 * クラスタリングチャートを描画
 */
function renderClusteringChart(canvasId, chartData) {
    console.log(`DEBUG: Rendering clustering chart for canvas ID: ${canvasId}`);
    console.log(`DEBUG: Clustering chart data:`, chartData);
    
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas element with ID "${canvasId}" not found`);
        showChartError(canvasId, `チャート要素 ${canvasId} が見つかりません`);
        return;
    }
    
    // データ構造の検証と補完
    if (!chartData || !chartData.datasets || !Array.isArray(chartData.datasets) || chartData.datasets.length === 0) {
        console.warn('WARNING: Invalid clustering chart data, using default dataset');
        chartData = {
            datasets: [{
                label: 'クラスタ1',
                data: [{x: 0, y: 0}],
                backgroundColor: 'white',
                borderColor: '#28a745',
                borderWidth: 2,
                pointRadius: 8
            }],
            title: '企業の財務特性に基づくクラスタリング分析',
            type: 'scatter'
        };
    }
    
    // 既存のChart.jsインスタンスがあれば破棄
    if (window.chartInstances && window.chartInstances[canvasId]) {
        console.log(`DEBUG: Destroying existing clustering chart instance for ${canvasId}`);
        window.chartInstances[canvasId].destroy();
    }
    
    if (!window.chartInstances) {
        window.chartInstances = {};
    }
    
    const ctx = canvas.getContext('2d');
    console.log(`DEBUG: Canvas context acquired for ${canvasId}:`, ctx);
    
    try {
        const config = {
            type: 'scatter',
            data: {
                datasets: chartData.datasets || []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: chartData.x_axis_label || 'X軸'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: chartData.y_axis_label || 'Y軸'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: chartData.title || 'クラスタリング分析'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        };
        
        const chart = new Chart(ctx, config);
        window.chartInstances[canvasId] = chart;
        
        console.log(`DEBUG: Successfully created clustering chart for ${canvasId}`);
        console.log(`DEBUG: Chart instance:`, chart);
        console.log(`DEBUG: Chart datasets:`, chart.data.datasets.length);
        return chart;
        
    } catch (error) {
        console.error(`ERROR: Failed to create clustering chart for ${canvasId}:`, error);
        throw error;
    }
}

/**
 * ポジショニングチャートを描画
 */
function renderPositioningChart(canvasId, chartData) {
    console.log(`DEBUG: Rendering positioning chart for canvas ID: ${canvasId}`);
    console.log(`DEBUG: Positioning chart data:`, chartData);
    
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas element with ID "${canvasId}" not found`);
        showChartError(canvasId, `チャート要素 ${canvasId} が見つかりません`);
        return;
    }
    
    // データ構造の検証と補完
    if (!chartData || !chartData.datasets || !Array.isArray(chartData.datasets) || chartData.datasets.length === 0) {
        console.warn('WARNING: Invalid positioning chart data, using default dataset');
        chartData = {
            datasets: [{
                label: '企業ポジション',
                data: [{x: 50, y: 50}],
                backgroundColor: 'white',
                borderColor: '#007bff',
                borderWidth: 2,
                pointRadius: 8
            }],
            title: '企業ポジショニングマップ（成長性 × 安定性）',
            type: 'scatter'
        };
    }
    
    // 既存のChart.jsインスタンスがあれば破棄
    if (window.chartInstances && window.chartInstances[canvasId]) {
        console.log(`DEBUG: Destroying existing positioning chart instance for ${canvasId}`);
        window.chartInstances[canvasId].destroy();
    }
    
    if (!window.chartInstances) {
        window.chartInstances = {};
    }
    
    const ctx = canvas.getContext('2d');
    console.log(`DEBUG: Canvas context acquired for positioning ${canvasId}:`, ctx);
    
    try {
        // 象限背景色とライン描画用プラグイン
        const quadrantPlugin = {
            id: 'quadrantPlugin',
            afterDraw: (chart) => {
                const ctx = chart.ctx;
                const chartArea = chart.chartArea;
                
                if (!chartArea) {
                    console.log('DEBUG: Chart area not available, skipping quadrant plugin');
                    return;
                }
                
                const {left, top, width, height} = chartArea;
                console.log('DEBUG: Drawing quadrants with area:', {left, top, width, height});
                
                // 象限背景色の描画
                const quadrants = [
                    {x: left + width/2, y: top, w: width/2, h: height/2, color: 'rgba(40, 167, 69, 0.1)'}, // 理想（右上）
                    {x: left, y: top, w: width/2, h: height/2, color: 'rgba(255, 193, 7, 0.1)'}, // チャレンジ（左上）
                    {x: left + width/2, y: top + height/2, w: width/2, h: height/2, color: 'rgba(23, 162, 184, 0.1)'}, // 安定（右下）
                    {x: left, y: top + height/2, w: width/2, h: height/2, color: 'rgba(220, 53, 69, 0.1)'} // 要注意（左下）
                ];
                
                quadrants.forEach(quad => {
                    ctx.fillStyle = quad.color;
                    ctx.fillRect(quad.x, quad.y, quad.w, quad.h);
                });
                
                // 50%基準線の描画
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 5]);
                
                // 縦線（成長性50%）
                ctx.beginPath();
                ctx.moveTo(left + width/2, top);
                ctx.lineTo(left + width/2, top + height);
                ctx.stroke();
                
                // 横線（安定性50%）
                ctx.beginPath();
                ctx.moveTo(left, top + height/2);
                ctx.lineTo(left + width, top + height/2);
                ctx.stroke();
                
                ctx.setLineDash([]);
                
                // 象限ラベル
                ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
                ctx.font = '12px sans-serif';
                ctx.textAlign = 'center';
                
                const labelOffset = 15;
                ctx.fillText('理想企業', left + width*0.75, top + labelOffset);
                ctx.fillText('チャレンジ企業', left + width*0.25, top + labelOffset);
                ctx.fillText('安定企業', left + width*0.75, top + height - labelOffset);
                ctx.fillText('要注意企業', left + width*0.25, top + height - labelOffset);
            }
        };

        // データセットの形式を確認・修正
        const processedDatasets = (chartData.datasets || []).map(dataset => {
            console.log('DEBUG: Processing dataset:', dataset.label, 'data points:', dataset.data.length);
            console.log('DEBUG: Sample data point:', dataset.data[0]);
            
            return {
                ...dataset,
                // Chart.jsの散布図では、data はx,yオブジェクトの配列である必要がある
                // 元の名前プロパティを保持
                data: dataset.data.map(point => ({
                    x: point.x,
                    y: point.y,
                    name: point.name // 企業名を保持
                })),
                // parsingを削除（Chart.jsが自動でx,yを認識）
                parsing: false
            };
        });
        
        console.log('DEBUG: Processed datasets:', processedDatasets.length);
        console.log('DEBUG: First processed dataset sample:', processedDatasets[0]);

        console.log('DEBUG: Creating Chart.js config...');
        
        const config = {
            type: 'scatter',
            data: {
                datasets: processedDatasets
            },
            plugins: [quadrantPlugin],
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: chartData.x_axis_label || '成長性スコア',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            stepSize: 25
                        }
                    },
                    y: {
                        type: 'linear',
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: chartData.y_axis_label || '安定性スコア',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            stepSize: 25
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: chartData.title || 'ポジショニング分析（成長性 × 安定性）',
                        font: {
                            size: 18,
                            weight: 'bold'
                        },
                        padding: 20
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            padding: 15,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const name = context.raw.name || '';
                                return `${name} (${label}): 成長性 ${context.raw.x}点, 安定性 ${context.raw.y}点`;
                            }
                        }
                    }
                }
            }
        };
        
        console.log('DEBUG: About to create Chart.js instance...');
        console.log('DEBUG: Final config:', JSON.stringify(config, null, 2));
        
        const chart = new Chart(ctx, config);
        window.chartInstances[canvasId] = chart;
        
        console.log(`DEBUG: Successfully created positioning chart for ${canvasId}`);
        console.log(`DEBUG: Chart instance:`, chart);
        console.log(`DEBUG: Chart datasets:`, chart.data.datasets.length);
        
        // Chart初期化の最終確認
        setTimeout(() => {
            if (chart.isInitialized) {
                console.log('DEBUG: Chart is fully initialized');
            } else {
                console.warn('WARNING: Chart initialization may be incomplete');
            }
        }, 100);
        
        return chart;
        
    } catch (error) {
        console.error(`ERROR: Failed to create positioning chart for ${canvasId}:`, error);
        throw error;
    }
}

/**
 * チャートエラーを表示
 */
function showChartError(canvasId, errorMessage) {
    console.log(`DEBUG: Showing chart error for ${canvasId}: ${errorMessage}`);
    
    const canvas = document.getElementById(canvasId);
    if (canvas) {
        const container = canvas.parentElement;
        if (container) {
            container.innerHTML = `
                <div class="chart-error">
                    <div class="error-icon">⚠️</div>
                    <p class="error-message">${errorMessage}</p>
                    <p class="error-help">ページを再読み込みするか、しばらく待ってから再試行してください。</p>
                </div>
            `;
        }
    }
}

/**
 * 動的にチャートを作成して描画する
 */
function createAndRenderChart(chartType, predictionData) {
    console.log(`DEBUG: Creating dynamic chart for ${chartType}`);
    console.log(`DEBUG: Prediction data:`, predictionData);
    
    // 予測分析セクションを探す（predictions-contentの中に挿入）
    const analysisSection = document.querySelector('.predictions-content');
    if (!analysisSection) {
        console.error(`ERROR: Could not find predictions content section for ${chartType}`);
        console.error(`DEBUG: Available containers:`, Array.from(document.querySelectorAll('[class*="prediction"], [class*="analysis"]')).map(el => el.className));
        return;
    }
    
    console.log(`DEBUG: Found analysis section for ${chartType}:`, analysisSection);
    
    // チャートコンテナを作成
    const chartContainer = document.createElement('div');
    chartContainer.className = 'prediction-item dynamic-chart';
    chartContainer.innerHTML = `
        <h4>${predictionData.label}の予測</h4>
        <div class="chart-container">
            <canvas id="${chartType}-chart" class="prediction-chart" data-chart-metric="${chartType === 'sales' ? 'net_sales' : 'net_income'}"></canvas>
        </div>
        <div class="chart-ai-analysis" data-chart-type="${chartType}" data-metric="${chartType === 'sales' ? 'net_sales' : 'net_income'}">
            <h5>🤖 AI分析</h5>
            <div class="${chartType === 'sales' ? 'sales' : 'profit'}-scenarios">
                <div class="scenario optimistic">
                    <h6>楽観シナリオ</h6>
                    <p class="scenario-explanation">分析中...</p>
                </div>
                <div class="scenario current">
                    <h6>現状シナリオ</h6>
                    <p class="scenario-explanation">分析中...</p>
                </div>
                <div class="scenario pessimistic">
                    <h6>悲観シナリオ</h6>
                    <p class="scenario-explanation">分析中...</p>
                </div>
            </div>
        </div>
    `;
    
    // 既存の"no-data"メッセージがあれば削除
    const existingNoData = analysisSection.querySelector('.no-data');
    if (existingNoData) {
        existingNoData.remove();
    }
    
    // チャートコンテナを追加
    analysisSection.appendChild(chartContainer);
    console.log(`DEBUG: Successfully added chart container for ${chartType} to DOM`);
    console.log(`DEBUG: Chart container HTML:`, chartContainer.outerHTML);
    
    // チャートを描画
    setTimeout(async () => {
        try {
            await renderChart(`${chartType}-chart`, predictionData.chart_data);
            console.log(`DEBUG: Successfully rendered dynamic ${chartType} chart`);
        } catch (error) {
            console.error(`ERROR: Failed to render dynamic ${chartType} chart:`, error);
            showChartError(`${chartType}-chart`, `チャート描画エラー: ${error.message}`);
        }
    }, 100);
}

/**
 * データがない場合のメッセージを表示
 */
function showNoDataMessage(chartType, message) {
    console.log(`DEBUG: Showing no-data message for ${chartType}: ${message}`);
    
    // 予測分析セクションを探す（predictions-contentの中に挿入）
    const analysisSection = document.querySelector('.predictions-content');
    if (!analysisSection) {
        console.error(`ERROR: Could not find predictions content section for ${chartType}`);
        return;
    }
    
    // 既存のno-dataメッセージを探す
    let noDataContainer = analysisSection.querySelector('.no-data');
    
    if (!noDataContainer) {
        // no-dataコンテナが存在しない場合は作成
        noDataContainer = document.createElement('div');
        noDataContainer.className = 'no-data';
        analysisSection.appendChild(noDataContainer);
    }
    
    // メッセージを追加
    const messageElement = document.createElement('div');
    messageElement.className = 'no-data-item';
    messageElement.innerHTML = `
        <div class="no-data-icon">📊</div>
        <p class="no-data-message">${message}</p>
        <p class="no-data-type">${chartType === 'sales' ? '売上高予測' : '純利益予測'}</p>
    `;
    
    noDataContainer.appendChild(messageElement);
}

/**
 * AI分析結果をDOMに更新
 */
function updateAnalysisContent(aiAnalysis) {
    console.log('DEBUG: Updating AI analysis content.');
    console.log('DEBUG: AI Analysis object:', aiAnalysis);
    updateScenarioAnalysis(aiAnalysis);
    updatePositioningAnalysis(aiAnalysis);
    updateClusteringAnalysis(aiAnalysis);
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
    console.log('DEBUG: updatePositioningAnalysis called');
    console.log('DEBUG: aiAnalysis keys:', Object.keys(aiAnalysis));
    console.log('DEBUG: Positioning info exists:', !!aiAnalysis.positioning_info);
    console.log('DEBUG: Positioning info:', aiAnalysis.positioning_info);
    
    if (aiAnalysis.POSITIONING_ANALYSIS) {
        console.log('DEBUG: Updating POSITIONING_ANALYSIS text');
        updateElement('.positioning-explanation', aiAnalysis.POSITIONING_ANALYSIS);
    }
    
    // ポジショニングデータが利用可能な場合は表示
    if (aiAnalysis.positioning_info) {
        console.log('DEBUG: Calling displayPositioningAnalysis...');
        console.log('DEBUG: positioning_info has chart:', !!aiAnalysis.positioning_info.chart);
        console.log('DEBUG: positioning_info chart datasets:', aiAnalysis.positioning_info.chart?.datasets?.length);
        displayPositioningAnalysis(aiAnalysis.positioning_info);
    } else {
        console.warn('WARNING: No positioning_info in aiAnalysis');
        console.log('DEBUG: aiAnalysis structure:', aiAnalysis);
    }
}

/**
 * クラスタリング分析を更新
 */
function updateClusteringAnalysis(aiAnalysis) {
    console.log('DEBUG: Updating clustering analysis...');
    console.log('DEBUG: Cluster info:', aiAnalysis.cluster_info);
    
    const clusteringLoading = document.querySelector('.clustering-loading');
    const clusteringContent = document.querySelector('.clustering-content');
    const clusteringError = document.querySelector('.clustering-error');
    
    if (!clusteringLoading || !clusteringContent || !clusteringError) {
        console.warn('WARNING: Clustering display elements not found');
        return;
    }
    
    if (aiAnalysis.cluster_info && aiAnalysis.cluster_info.cluster_id !== undefined) {
        console.log('DEBUG: Displaying clustering results...');
        
        // Hide loading and error, show content
        clusteringLoading.style.display = 'none';
        clusteringError.style.display = 'none';
        clusteringContent.style.display = 'block';
        
        // Update cluster information
        const clusterTitle = clusteringContent.querySelector('.cluster-title');
        const clusterYear = clusteringContent.querySelector('.cluster-year');
        const similarCompaniesList = clusteringContent.querySelector('.similar-companies-list');
        const characteristicsContainer = clusteringContent.querySelector('.characteristics-container');
        
        if (clusterTitle) {
            clusterTitle.textContent = `クラスタ${aiAnalysis.cluster_info.cluster_id} (全${aiAnalysis.cluster_info.total_clusters}クラスタ中)`;
        }
        
        if (clusterYear) {
            clusterYear.textContent = `データ年度: ${aiAnalysis.cluster_info.company_year}年`;
        }
        
        // Update similar companies
        if (similarCompaniesList && aiAnalysis.cluster_info.same_cluster_companies) {
            similarCompaniesList.innerHTML = '';
            aiAnalysis.cluster_info.same_cluster_companies.forEach(company => {
                const li = document.createElement('li');
                li.textContent = `${company.name} (${company.year}年)`;
                similarCompaniesList.appendChild(li);
            });
        }
        
        // Update cluster characteristics
        if (characteristicsContainer && aiAnalysis.cluster_info.cluster_characteristics) {
            characteristicsContainer.innerHTML = '';
            Object.entries(aiAnalysis.cluster_info.cluster_characteristics).forEach(([feature, data]) => {
                const div = document.createElement('div');
                div.className = 'characteristic-item';
                div.innerHTML = `
                    <span class="characteristic-label">${feature}:</span>
                    <span class="characteristic-value">${data.relative > 0 ? '+' : ''}${data.relative.toFixed(1)}%</span>
                `;
                characteristicsContainer.appendChild(div);
            });
        }
        
        console.log('DEBUG: Clustering content updated successfully');
    } else {
        console.log('DEBUG: No clustering data available, showing error');
        
        // Hide loading and content, show error
        clusteringLoading.style.display = 'none';
        clusteringContent.style.display = 'none';
        clusteringError.style.display = 'block';
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

// 初期チャート表示関数
function initializePlaceholderCharts() {
    const chartIds = ['sales-chart', 'profit-chart', 'clustering-chart', 'positioning-chart'];
    
    chartIds.forEach(canvasId => {
        const canvas = document.getElementById(canvasId);
        if (canvas) {
            // データ読み込み中のプレースホルダーチャートを表示
            const placeholderData = {
                labels: ['読み込み中...'],
                datasets: [{
                    label: 'データを読み込んでいます',
                    data: [0],
                    borderColor: '#ddd',
                    backgroundColor: 'rgba(221, 221, 221, 0.1)',
                    borderWidth: 2
                }],
                title: 'データを読み込み中...'
            };
            
            // 散布図の場合
            if (canvasId.includes('cluster') || canvasId.includes('positioning')) {
                placeholderData.datasets = [{
                    label: 'データ読み込み中',
                    data: [{x: 50, y: 50}],
                    backgroundColor: 'white',
                    borderColor: '#ddd',
                    borderWidth: 2,
                    pointRadius: 8
                }];
                placeholderData.type = 'scatter';
                delete placeholderData.labels;
            }
            
            try {
                if (canvasId.includes('cluster')) {
                    renderClusteringChart(canvasId, placeholderData);
                } else if (canvasId.includes('positioning')) {
                    renderPositioningChart(canvasId, placeholderData);
                } else {
                    renderChart(canvasId, placeholderData);
                }
            } catch (error) {
                console.warn(`Failed to initialize placeholder chart for ${canvasId}:`, error);
            }
        }
    });
}

// ページ読み込み時の初期化
document.addEventListener('DOMContentLoaded', function() {
    // Chart.jsの読み込み確認
    console.log('DEBUG: Chart.js loaded:', typeof Chart !== 'undefined');
    if (typeof Chart === 'undefined') {
        console.error('ERROR: Chart.js is not loaded!');
        // ユーザーに表示
        const chartContainers = document.querySelectorAll('.chart-container');
        chartContainers.forEach(container => {
            container.innerHTML = '<div class="chart-error">Chart.jsライブラリが読み込まれていません。ページを再読み込みしてください。</div>';
        });
    } else {
        // プレースホルダーチャートを初期化
        initializePlaceholderCharts();
        console.log('DEBUG: Chart.js version:', Chart.version);
        console.log('DEBUG: Chart.js registry available:', !!Chart.registry);
        console.log('DEBUG: Chart.js plugins available:', !!Chart.registry?.plugins);
    }
    
    // デフォルトで財務データタブを表示
    showTab('financial-data');

    // 従業員数チャートの描画
    const employeeDataElement = document.getElementById('employee-data');
    if (employeeDataElement) {
        const employeeDataJson = employeeDataElement.textContent;
        if (employeeDataJson) {
            renderEmployeeChart(employeeDataJson);
        }
    }
    
    // タブボタンのクリックイベントを設定
    const tabButtons = document.getElementsByClassName('tab-button');
    for (let button of tabButtons) {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            if (tabName) {
                showTab(tabName);
            }
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
            // 二軸分析（ポジショニング）も自動でロード
            if (!window.positioningAnalysisLoaded && !window.positioningAnalysisLoading) {
                loadPositioningAnalysis();
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
    console.log('DEBUG: Loading chart scenario analysis...');
    
    // 実行状態の管理 - 重複実行を防ぐ
    if (window.scenarioAnalysisState?.isLoading) {
        console.log('DEBUG: Scenario analysis already in progress, skipping duplicate call');
        return;
    }
    
    if (window.scenarioAnalysisState?.isCompleted) {
        console.log('DEBUG: Scenario analysis already completed, skipping duplicate call');
        return;
    }
    
    // 初期化状態
    if (!window.scenarioAnalysisState) {
        window.scenarioAnalysisState = {
            isLoading: false,
            isCompleted: false,
            processedCharts: new Set()
        };
    }
    
    // 実行開始
    window.scenarioAnalysisState.isLoading = true;
    
    try {
        // チャート描画が完了しているかチェック
        if (!window.predictionResults) {
            console.warn('WARNING: No prediction results available for scenario analysis');
            window.scenarioAnalysisState.isLoading = false;
            return;
        }
        
        const chartAnalysisSections = document.querySelectorAll('.chart-ai-analysis');
        console.log(`DEBUG: Found ${chartAnalysisSections.length} chart analysis sections`);
        
        let processedCount = 0;
        
        chartAnalysisSections.forEach((section, index) => {
            const chartType = section.getAttribute('data-chart-type');
            const metric = section.getAttribute('data-metric');
            console.log(`DEBUG: Section ${index}: chart-type = ${chartType}, metric = ${metric}`);
            
            // 既に処理済みのチャートをスキップ
            if (window.scenarioAnalysisState.processedCharts.has(chartType)) {
                console.log(`DEBUG: Chart type ${chartType} already processed, skipping`);
                return;
            }
            
            if (chartType && window.predictionResults && window.predictionResults[metric]) {
                const chartData = window.predictionResults[metric].chart_data;
                if (chartData) {
                    // 英語のIDを使用（HTMLのcanvas要素と一致）
                    const canvasId = chartType === 'sales' ? 'sales-chart' : 'profit-chart';
                    console.log(`DEBUG: Processing chart with ID: ${canvasId}`);
                    
                    const canvas = document.getElementById(canvasId);
                    if (canvas) {
                        // 既存のチャートインスタンスがない場合のみ描画
                        if (!window.chartInstances || !window.chartInstances[canvasId]) {
                            console.log(`DEBUG: Rendering chart ${canvasId} for first time`);
                            renderChart(canvasId, chartData);
                        } else {
                            console.log(`DEBUG: Chart ${canvasId} already exists, skipping render`);
                        }
                        
                        // 処理済みマークを追加
                        window.scenarioAnalysisState.processedCharts.add(chartType);
                        processedCount++;
                    } else {
                        console.warn(`DEBUG: Canvas element ${canvasId} not found, will retry later`);
                    }
                }
                
                console.log(`DEBUG: Loading scenario analysis for chart type: ${chartType}`);
                loadScenarioAnalysisInternal(chartType, section);
            } else {
                console.warn(`DEBUG: Section ${index} missing data - chartType: ${chartType}, hasMetricData: ${!!(window.predictionResults && window.predictionResults[metric])}`);
            }
        });
        
        console.log(`DEBUG: Processed ${processedCount} charts in this run`);
        
        // 完了状態の設定
        if (processedCount > 0 || chartAnalysisSections.length === 0) {
            window.scenarioAnalysisState.isCompleted = true;
            console.log('DEBUG: Scenario analysis completed successfully');
        }
        
    } catch (error) {
        console.error('ERROR: Chart scenario analysis failed:', error);
    } finally {
        window.scenarioAnalysisState.isLoading = false;
    }
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

/**
 * 二軸分析（ポジショニング分析）をAJAXで読み込む
 */
async function loadPositioningAnalysis() {
    // すでに読み込み中または完了している場合は何もしない
    if (window.positioningAnalysisLoading || window.positioningAnalysisLoaded) {
        return;
    }
    
    window.positioningAnalysisLoading = true;
    
    // EDINETコードを取得
    const pathParts = window.location.pathname.split('/');
    const edinetCode = pathParts[pathParts.indexOf('company') + 1];
    
    if (!edinetCode) {
        console.error('EDINETコードが見つかりません');
        return;
    }
    
    try {
        console.log(`Fetching positioning analysis for edinet code: ${edinetCode}`);
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 2分でタイムアウト
        
        const response = await fetch(`/api/company/${edinetCode}/positioning/`, {
            method: 'GET',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/json',
            },
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        console.log(`Response status: ${response.status}, statusText: ${response.statusText}`);
        
        const data = await response.json();
        console.log('Positioning analysis response:', data);
        
        if (response.ok && data.positioning_analysis) {
            // 二軸分析結果を表示
            displayPositioningAnalysis(data.positioning_analysis);
            window.positioningAnalysisLoaded = true;
        } else {
            console.error('Server returned error:', data);
            throw new Error(data.error || `サーバーエラー (${response.status}): 二軸分析の取得に失敗しました`);
        }
        
    } catch (error) {
        console.error('二軸分析エラー:', error);
        
        let errorMessage = error.message;
        if (error.name === 'AbortError') {
            errorMessage = 'リクエストがタイムアウトしました。データ処理に時間がかかっています。しばらく待ってから再試行してください。';
        }
        
        showPositioningAnalysisError(errorMessage);
    } finally {
        window.positioningAnalysisLoading = false;
    }
}

/**
 * 二軸分析結果を表示
 */
function displayPositioningAnalysis(positioningData) {
    console.log('=== DISPLAY POSITIONING ANALYSIS START ===');
    console.log('DEBUG: positioningData:', positioningData);
    console.log('DEBUG: positioningData keys:', Object.keys(positioningData));
    console.log('DEBUG: chart property exists:', !!positioningData.chart);
    if (positioningData.chart) {
        console.log('DEBUG: chart datasets:', positioningData.chart.datasets?.length);
        console.log('DEBUG: chart structure:', {
            title: positioningData.chart.title,
            x_axis_label: positioningData.chart.x_axis_label,
            y_axis_label: positioningData.chart.y_axis_label,
            datasetsCount: positioningData.chart.datasets?.length
        });
    }
    
    const loadingContainer = document.querySelector('.positioning-loading');
    if (loadingContainer) {
        loadingContainer.style.display = 'none';
        loadingContainer.style.visibility = 'hidden';
    }
    
    const loadingSpinner = document.querySelector('.loading-spinner');
    if (loadingSpinner) {
        loadingSpinner.style.display = 'none';
        loadingSpinner.style.visibility = 'hidden';
    }
    
    const initialExplanationElement = document.querySelector('.positioning-explanation');
    if (initialExplanationElement) {
        initialExplanationElement.textContent = '二軸分析が完了しました。以下の結果をご確認ください。';
        initialExplanationElement.classList.add('fade-in-content');
    }
    
    const resultsContainer = document.querySelector('.positioning-results');
    if (resultsContainer) {
        resultsContainer.style.display = 'block';
        
        const quadrantInfo = positioningData.quadrant_info || {};
        const quadrantName = document.querySelector('.quadrant-name');
        const quadrantDescription = document.querySelector('.quadrant-description');
        
        if (quadrantName) quadrantName.textContent = quadrantInfo.name || '';
        if (quadrantDescription) quadrantDescription.textContent = quadrantInfo.description || '';
        
        const quadrantBadge = document.querySelector('.quadrant-badge');
        if (quadrantBadge && quadrantInfo.color) {
            quadrantBadge.style.backgroundColor = quadrantInfo.color;
            quadrantBadge.style.color = '#fff';
        }
        
        const growthScoreElement = document.querySelector('.growth-score');
        const stabilityScoreElement = document.querySelector('.stability-score');
        
        if (growthScoreElement) {
            growthScoreElement.textContent = `${positioningData.growth_score?.toFixed(1) || 0}点`;
        }
        if (stabilityScoreElement) {
            stabilityScoreElement.textContent = `${positioningData.stability_score?.toFixed(1) || 0}点`;
        }
        
        // ポジショニングマップをChart.jsで描画
        console.log('DEBUG: Checking positioning chart data...');
        console.log('DEBUG: positioningData.chart:', positioningData.chart);
        console.log('DEBUG: positioningData full structure:', positioningData);
        
        // Canvas要素の存在確認
        const positioningCanvas = document.getElementById('positioning-chart');
        console.log('DEBUG: Positioning canvas element:', positioningCanvas);
        console.log('DEBUG: Canvas parent:', positioningCanvas?.parentElement);
        
        if (positioningData.chart && positioningData.chart.datasets) {
            console.log('DEBUG: Rendering positioning chart with datasets count:', positioningData.chart.datasets.length);
            console.log('DEBUG: First dataset sample:', positioningData.chart.datasets[0]);
            
            if (!positioningCanvas) {
                console.error('ERROR: positioning-chart canvas element not found!');
                console.log('DEBUG: Available canvas elements:', Array.from(document.querySelectorAll('canvas')).map(c => c.id));
                showChartError('positioning-chart', 'Canvas要素が見つかりません');
                return;
            }
            
            try {
                const result = renderPositioningChart('positioning-chart', positioningData.chart);
                console.log('DEBUG: Positioning chart rendered successfully, result:', result);
            } catch (error) {
                console.error('ERROR: Failed to render positioning chart:', error);
                console.error('ERROR: Error stack:', error.stack);
                showChartError('positioning-chart', `チャート描画エラー: ${error.message}`);
            }
        } else {
            console.warn('WARNING: No chart data available for positioning analysis');
            console.log('DEBUG: Available positioning data keys:', Object.keys(positioningData));
            showChartError('positioning-chart', 'ポジショニングチャートデータが不足しています');
        }
        
        const adviceElement = document.querySelector('.advice-text');
        if (adviceElement && quadrantInfo.career_advice) {
            adviceElement.textContent = quadrantInfo.career_advice;
        }
        
        const recommendationsList = document.querySelector('.recommendations-list');
        if (recommendationsList && positioningData.recommendations) {
            recommendationsList.innerHTML = '';
            positioningData.recommendations.forEach(rec => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <a href="/company/${rec.edinet_code}/">${rec.company_name}</a>
                    <span class="company-scores">
                        (成長性: ${rec.growth_score?.toFixed(1) || 0}点, 
                         安定性: ${rec.stability_score?.toFixed(1) || 0}点)
                    </span>
                `;
                recommendationsList.appendChild(li);
            });
        }
        
        const detailedMetrics = positioningData.detailed_metrics || {};
        
        const salesGrowthElement = document.querySelector('.sales-growth');
        if (salesGrowthElement) {
            salesGrowthElement.textContent = `${(detailedMetrics.sales_growth_rate * 100)?.toFixed(1) || 0}%`;
        }
        
        const employeeGrowthElement = document.querySelector('.employee-growth');
        if (employeeGrowthElement) {
            employeeGrowthElement.textContent = `${(detailedMetrics.employee_growth_rate * 100)?.toFixed(1) || 0}%`;
        }
        
        const rdIntensityElement = document.querySelector('.rd-intensity');
        if (rdIntensityElement) {
            rdIntensityElement.textContent = `${(detailedMetrics.rd_intensity * 100)?.toFixed(1) || 0}%`;
        }
        
        const equityRatioElement = document.querySelector('.equity-ratio');
        if (equityRatioElement) {
            equityRatioElement.textContent = `${(detailedMetrics.equity_ratio * 100)?.toFixed(1) || 0}%`;
        }
        
        resultsContainer.classList.add('fade-in-content');
    }
    
    const explanationElement = document.querySelector('.positioning-explanation');
    if (explanationElement && positioningData.interpretation) {
        explanationElement.innerHTML = positioningData.interpretation.replace(/\n/g, '<br>');
        explanationElement.classList.add('fade-in-content');
    }
}

// This function is replaced by the improved renderPositioningChart function above

/**
 * 二軸分析エラーを表示
 */
function showPositioningAnalysisError(message) {
    console.error('Positioning analysis error:', message);
    
    // ローディング表示を完全に隠す
    const loadingContainer = document.querySelector('.positioning-loading');
    if (loadingContainer) {
        loadingContainer.style.display = 'none';
        loadingContainer.style.visibility = 'hidden';
    }
    
    // ローディングスピナーも個別に隠す
    const loadingSpinner = document.querySelector('.loading-spinner');
    if (loadingSpinner) {
        loadingSpinner.style.display = 'none';
        loadingSpinner.style.visibility = 'hidden';
    }
    
    // エラーメッセージを表示
    const errorExplanationElement = document.querySelector('.positioning-explanation');
    if (errorExplanationElement) {
        errorExplanationElement.innerHTML = `
            <div class="error-content">
                <p>🚫 二軸分析でエラーが発生しました: ${message}</p>
                <button onclick="retryPositioningAnalysis()" class="btn btn-small">再試行</button>
            </div>
        `;
        errorExplanationElement.classList.add('error-text');
    }
}

/**
 * 二軸分析を再試行
 */
function retryPositioningAnalysis() {
    window.positioningAnalysisLoaded = false;
    window.positioningAnalysisLoading = false;
    
    // エラー表示をクリア
    const retryExplanationElement = document.querySelector('.positioning-explanation');
    if (retryExplanationElement) {
        retryExplanationElement.textContent = '企業の成長性と安定性を分析中...';
        retryExplanationElement.classList.remove('error-text');
    }
    
    // ローディング表示を再表示
    const loadingContainer = document.querySelector('.positioning-loading');
    if (loadingContainer) {
        loadingContainer.style.display = 'block';
    }
    
    // 結果表示エリアを隠す
    const resultsContainer = document.querySelector('.positioning-results');
    if (resultsContainer) {
        resultsContainer.style.display = 'none';
    }
    
    // 再実行
    loadPositioningAnalysis();
}

// グローバルスコープに関数を公開（後方互換性のため）
window.showTab = showTab;
window.retryAIAnalysis = retryAIAnalysis;
window.retryCompanyOverview = retryCompanyOverview;
window.retryPositioningAnalysis = retryPositioningAnalysis;