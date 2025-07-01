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