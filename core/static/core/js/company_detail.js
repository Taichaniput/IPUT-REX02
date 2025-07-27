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
        console.log('Response data (full AI analysis):', data); // Full AI analysis object
        
        if (response.ok && data.success) {
            window.predictionResults = data.ai_analysis.prediction_results; 
            window.clusterInfo = data.ai_analysis.cluster_info;
            window.positioningInfo = data.ai_analysis.positioning_info;
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
                    console.warn('WARNING: Sales chart canvas not found');
                }
            }
            
            // 純利益予測チャート
            if (window.predictionResults.net_income?.chart_data) {
                console.log('DEBUG: Rendering profit chart...');
                const profitCanvas = document.getElementById('profit-chart');
                if (profitCanvas) {
                    await renderChart('profit-chart', window.predictionResults.net_income.chart_data);
                } else {
                    console.warn('WARNING: Profit chart canvas not found');
                }
            }
        } else {
            console.warn('WARNING: No prediction results available for chart rendering');
        }
    } catch (error) {
        console.error('ERROR: Failed to render prediction charts:', error);
    }

    // 2. クラスタリングチャートの描画
    try {
        if (window.clusterInfo?.chart_data) {
            console.log('DEBUG: Rendering clustering chart...');
            const clusteringCanvas = document.getElementById('clustering-chart');
            if (clusteringCanvas) {
                renderClusteringChart('clustering-chart', window.clusterInfo.chart_data);
            } else {
                console.warn('WARNING: Clustering chart canvas not found');
            }
        } else {
            console.warn('WARNING: No clustering info available for chart rendering');
        }
    } catch (error) {
        console.error('ERROR: Failed to render clustering chart:', error);
    }

    // 3. ポジショニングチャートの描画
    try {
        if (window.positioningInfo?.chart_data) {
            console.log('DEBUG: Rendering positioning chart...');
            const positioningCanvas = document.getElementById('positioning-chart');
            if (positioningCanvas) {
                renderPositioningChart('positioning-chart', window.positioningInfo.chart_data);
            } else {
                console.warn('WARNING: Positioning chart canvas not found');
            }
        } else {
            console.warn('WARNING: No positioning info available for chart rendering');
        }
    } catch (error) {
        console.error('ERROR: Failed to render positioning chart:', error);
    }

    // 4. シナリオ分析の読み込み（チャート描画後）
    setTimeout(() => {
        try {
            console.log('DEBUG: Loading scenario analysis...');
            loadChartScenarioAnalysis();
        } catch (error) {
            console.error('ERROR: Failed to load scenario analysis:', error);
        }
    }, 200); // チャート描画の完了を待つ

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
            try {
                renderChart('sales-chart', window.predictionResults.net_sales.chart_data);
                testResults.renderingAttempts.salesChart = 'success';
                console.log('   ✓ Sales chart rendering succeeded');
            } catch (error) {
                testResults.renderingAttempts.salesChart = `error: ${error.message}`;
                console.error('   ✗ Sales chart rendering failed:', error);
            }
        }
        
        // 利益チャートのテスト
        if (testResults.canvasElements['profit-chart'].exists && testResults.dataAvailability.predictionResults.netIncome) {
            console.log('   Testing profit chart rendering...');
            try {
                renderChart('profit-chart', window.predictionResults.net_income.chart_data);
                testResults.renderingAttempts.profitChart = 'success';
                console.log('   ✓ Profit chart rendering succeeded');
            } catch (error) {
                testResults.renderingAttempts.profitChart = `error: ${error.message}`;
                console.error('   ✗ Profit chart rendering failed:', error);
            }
        }
        
        // クラスタリングチャートのテスト
        if (testResults.canvasElements['clustering-chart'].exists && testResults.dataAvailability.clusterInfo.chartData) {
            console.log('   Testing clustering chart rendering...');
            try {
                renderClusteringChart('clustering-chart', window.clusterInfo.chart_data);
                testResults.renderingAttempts.clusteringChart = 'success';
                console.log('   ✓ Clustering chart rendering succeeded');
            } catch (error) {
                testResults.renderingAttempts.clusteringChart = `error: ${error.message}`;
                console.error('   ✗ Clustering chart rendering failed:', error);
            }
        }
        
        // ポジショニングチャートのテスト
        if (testResults.canvasElements['positioning-chart'].exists && testResults.dataAvailability.positioningInfo.chartData) {
            console.log('   Testing positioning chart rendering...');
            try {
                renderPositioningChart('positioning-chart', window.positioningInfo.chart_data);
                testResults.renderingAttempts.positioningChart = 'success';
                console.log('   ✓ Positioning chart rendering succeeded');
            } catch (error) {
                testResults.renderingAttempts.positioningChart = `error: ${error.message}`;
                console.error('   ✗ Positioning chart rendering failed:', error);
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

function renderClusteringChart(canvasId, chartData) {
    try {
        console.log(`DEBUG: Starting clustering chart render for ID: ${canvasId}`);
        console.log('DEBUG: Clustering chart data:', chartData);
        
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
            showChartError(canvasId, `チャート要素 ${canvasId} が見つかりません`);
            return;
        }

        // データ構造の検証
        if (!chartData) {
            console.error('ERROR: Chart data is null or undefined');
            showChartError(canvasId, 'チャートデータが見つかりません');
            return;
        }

        if (!chartData.datasets || !Array.isArray(chartData.datasets)) {
            console.error('ERROR: Invalid datasets in chart data');
            showChartError(canvasId, '無効なデータセット構造です');
            return;
        }

        console.log('DEBUG: Clustering chart validation passed, creating chart...');

        // 既存のチャートを破棄
        if (ctx.chart) {
            ctx.chart.destroy();
        }

        ctx.chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: chartData.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: chartData.title || 'クラスタリング分析',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const name = context.raw.name || '';
                                return `${name} (${label}): UMAP1 ${context.raw.x}, UMAP2 ${context.raw.y}`;
                            }
                        }
                    },
                    legend: {
                        display: true,
                        position: 'bottom'
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: chartData.x_axis_label || 'UMAP 1'
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: chartData.y_axis_label || 'UMAP 2'
                        }
                    }
                }
            }
        });

        console.log('DEBUG: Clustering chart created successfully:', ctx.chart);

        // 説明文の更新
        const descriptionElement = document.querySelector('.clustering-analysis-content .description-text');
        if (descriptionElement && chartData.description) {
            descriptionElement.innerHTML = chartData.description.replace(/\n/g, '<br>');
        }

    } catch (error) {
        console.error('ERROR: Clustering chart rendering failed:', error);
        showChartError(canvasId, 'クラスタリングチャートの表示に失敗しました: ' + error.message);
    }
}

async function renderChart(canvasId, chartData) {
    try {
        console.log(`DEBUG: Starting chart render for ID: ${canvasId}`);
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
            showChartError(canvasId, `チャート要素 ${canvasId} が見つかりません`);
            return;
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
        
        if (!chartData.labels || !Array.isArray(chartData.labels) || chartData.labels.length === 0) {
            console.error(`ERROR: Invalid labels for chart ${canvasId}:`, chartData.labels);
            showChartError(canvasId, 'チャートラベルが無効です');
            return;
        }
        
        if (!chartData.datasets || !Array.isArray(chartData.datasets) || chartData.datasets.length === 0) {
            console.error(`ERROR: Invalid datasets for chart ${canvasId}:`, chartData.datasets);
            showChartError(canvasId, 'チャートデータセットが無効です');
            return;
        }
        
        // 既存チャートの破棄
        if (ctx.chart) {
            console.log(`DEBUG: Destroying existing chart for ${canvasId}`);
            ctx.chart.destroy();
        }
        
        // Chart.js設定の検証
        const chartConfig = {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: chartData.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: chartData.title || 'チャート',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: '年度'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: chartData.ylabel || '値'
                        }
                    }
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
                
                // Canvas要素のリセット
                if (retryCount > 0) {
                    ctx.width = ctx.offsetWidth;
                    ctx.height = ctx.offsetHeight;
                }
                
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

function showChartError(canvasId, message) {
    console.error(`Chart error for ${canvasId}: ${message}`);
    
    const ctx = document.getElementById(canvasId);
    if (ctx) {
        const container = ctx.parentElement;
        if (container) {
            // Canvas要素を隠す
            ctx.style.display = 'none';
            
            // エラー表示要素を作成
            const errorDiv = document.createElement('div');
            errorDiv.className = 'chart-error-container';
            errorDiv.innerHTML = `
                <div class="chart-error-icon">⚠️</div>
                <p class="chart-error-message">${message}</p>
                <p class="chart-error-message" style="font-size: 12px; margin-top: 10px; opacity: 0.7;">Chart ID: ${canvasId}</p>
            `;
            
            // 既存のエラー表示を削除
            const existingError = container.querySelector('.chart-error-container');
            if (existingError) {
                existingError.remove();
            }
            
            // 新しいエラー表示を追加
            container.style.position = 'relative';
            container.appendChild(errorDiv);
        }
    }
}

/**
 * AI分析結果をDOMに更新
 */
function updateAnalysisContent(aiAnalysis) {
    console.log('DEBUG: Updating AI analysis content.');
    console.log('DEBUG: AI Analysis object:', aiAnalysis);
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
    console.log('Loading chart scenario analysis...');
    const chartAnalysisSections = document.querySelectorAll('.chart-ai-analysis');
    console.log(`Found ${chartAnalysisSections.length} chart analysis sections`);
    
    chartAnalysisSections.forEach((section, index) => {
        const chartType = section.getAttribute('data-chart-type');
        const metric = section.getAttribute('data-metric');
        console.log(`Section ${index}: chart-type = ${chartType}, metric = ${metric}`);
        console.log(`Section ${index} element:`, section);
        
        if (chartType && window.predictionResults && window.predictionResults[metric]) {
            const chartData = window.predictionResults[metric].chart_data;
            if (chartData) {
                const canvasId = `${window.predictionResults[metric].label}_chart`;
                renderChart(canvasId, chartData);
            }
            console.log(`Loading scenario analysis for chart type: ${chartType}`);
            loadScenarioAnalysisInternal(chartType, section);
        } else {
            console.warn(`Section ${index} has no data-chart-type attribute or prediction results are missing.`);
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
    console.log('Displaying positioning analysis:', positioningData);
    
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
        if (positioningData.chart) {
            renderPositioningChart('positioning-chart', positioningData.chart);
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

function renderPositioningChart(canvasId, chartData) {
    try {
        console.log(`DEBUG: Starting positioning chart render for ID: ${canvasId}`);
        console.log('DEBUG: Positioning chart data:', chartData);
        
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
            showChartError(canvasId, `チャート要素 ${canvasId} が見つかりません`);
            return;
        }

        // データ構造の検証
        if (!chartData) {
            console.error('ERROR: Chart data is null or undefined');
            showChartError(canvasId, 'チャートデータが見つかりません');
            return;
        }

        if (!chartData.datasets || !Array.isArray(chartData.datasets)) {
            console.error('ERROR: Invalid datasets in chart data');
            showChartError(canvasId, '無効なデータセット構造です');
            return;
        }

        console.log('DEBUG: Positioning chart validation passed, creating chart...');

        // 既存のチャートを破棄
        if (ctx.chart) {
            ctx.chart.destroy();
        }

        ctx.chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: chartData.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: chartData.title || '二軸分析（成長性 × 安定性）',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const name = context.raw.name || '';
                                return `${name} (${label}): 成長性 ${context.raw.x}点, 安定性 ${context.raw.y}点`;
                            }
                        }
                    },
                    annotation: {
                        annotations: {
                            lineX: {
                                type: 'line',
                                xMin: 50, xMax: 50,
                                borderColor: 'gray',
                                borderWidth: 1,
                                borderDash: [5, 5]
                            },
                            lineY: {
                                type: 'line',
                                yMin: 50, yMax: 50,
                                borderColor: 'gray',
                                borderWidth: 1,
                                borderDash: [5, 5]
                            },
                            // 象限の背景色
                            quadrant1: {
                                type: 'box',
                                xMin: 50, xMax: 100, yMin: 50, yMax: 100,
                                backgroundColor: 'rgba(0, 128, 0, 0.1)',
                                borderColor: 'rgba(0, 0, 0, 0)'
                            },
                            quadrant2: {
                                type: 'box',
                                xMin: 0, xMax: 50, yMin: 50, yMax: 100,
                                backgroundColor: 'rgba(255, 165, 0, 0.1)',
                                borderColor: 'rgba(0, 0, 0, 0)'
                            },
                            quadrant3: {
                                type: 'box',
                                xMin: 50, xMax: 100, yMin: 0, yMax: 50,
                                backgroundColor: 'rgba(0, 0, 255, 0.1)',
                                borderColor: 'rgba(0, 0, 0, 0)'
                            },
                            quadrant4: {
                                type: 'box',
                                xMin: 0, xMax: 50, yMin: 0, yMax: 50,
                                backgroundColor: 'rgba(255, 0, 0, 0.1)',
                                borderColor: 'rgba(0, 0, 0, 0)'
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: chartData.x_axis_label || '成長性'
                        },
                        min: 0,
                        max: 100
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: chartData.y_axis_label || '安定性'
                        },
                        min: 0,
                        max: 100
                    }
                }
            }
        });

        console.log('DEBUG: Positioning chart created successfully:', ctx.chart);

    } catch (error) {
        console.error('ERROR: Positioning chart rendering failed:', error);
        showChartError(canvasId, 'ポジショニングチャートの表示に失敗しました: ' + error.message);
    }
}

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