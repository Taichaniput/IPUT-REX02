(function() {
// Home Page JavaScript

/**
 * 検索フォームの処理
 */
function initializeSearchForm() {
    const searchForm = document.querySelector('.search-form');
    const searchInput = document.querySelector('input[name="keyword"]');
    
    if (searchForm && searchInput) {
        // 検索フォームの送信時の処理
        searchForm.addEventListener('submit', function(e) {
            const keyword = searchInput.value.trim();
            if (!keyword) {
                e.preventDefault();
                showMessage('検索キーワードを入力してください。', 'error');
                searchInput.focus();
                return false;
            }
            
            // ローディング表示
            showLoadingState();
        });
        
        // 検索入力フィールドのキーボードイベント
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchForm.dispatchEvent(new Event('submit'));
            }
        });
        
        // 検索入力フィールドのフォーカス時の処理
        searchInput.addEventListener('focus', function() {
            this.select();
        });
    }
}

/**
 * メッセージ表示機能
 * @param {string} message - 表示するメッセージ
 * @param {string} type - メッセージタイプ ('success', 'error', 'info')
 */
function showMessage(message, type = 'info') {
    // 既存のメッセージを削除
    const existingMessage = document.querySelector('.dynamic-message');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    // 新しいメッセージを作成
    const messageDiv = document.createElement('div');
    messageDiv.className = `alert alert-${type} dynamic-message`;
    messageDiv.textContent = message;
    
    // メッセージを挿入
    const contentDiv = document.querySelector('[data-content="main"]') || document.body;
    contentDiv.insertBefore(messageDiv, contentDiv.firstChild);
    
    // 3秒後に自動削除
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.remove();
        }
    }, 3000);
}

/**
 * ローディング状態を表示
 */
function showLoadingState() {
    const submitButton = document.querySelector('.search-form button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="loading-spinner-inline"></span> 検索中...';
        
        // ページ全体にローディングオーバーレイを表示
        showPageLoadingOverlay('企業データを検索中...');
        
        // 10秒後にボタンを元に戻す（タイムアウト対策）
        setTimeout(() => {
            submitButton.disabled = false;
            submitButton.textContent = '検索';
            hidePageLoadingOverlay();
        }, 10000);
    }
}

/**
 * 企業リンクのホバー効果を強化
 */
function enhanceCompanyLinks() {
    const companyLinks = document.querySelectorAll('.company-link');
    
    companyLinks.forEach(link => {
        link.addEventListener('mouseenter', function() {
            this.style.transform = 'translateX(5px)';
            this.style.transition = 'transform 0.2s ease';
        });
        
        link.addEventListener('mouseleave', function() {
            this.style.transform = 'translateX(0)';
        });
        
        // クリック時のローディング表示
        link.addEventListener('click', function(e) {
            showCompanyDetailLoading(this);
            showMessage('企業詳細を読み込んでいます...', 'info');
        });
    });
}

/**
 * CTAボタンのアニメーション効果
 */
function enhanceCTAButtons() {
    const ctaButtons = document.querySelectorAll('.cta-button, .login-btn');
    
    ctaButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            // リップル効果
            const ripple = document.createElement('span');
            ripple.classList.add('ripple');
            this.appendChild(ripple);
            
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
}

/**
 * 企業詳細ページへのナビゲーション時のローディング表示
 * @param {HTMLElement} linkElement - クリックされたリンク要素
 */
function showCompanyDetailLoading(linkElement) {
    // クリックされたリンクをローディング状態に
    const originalContent = linkElement.innerHTML;
    linkElement.style.opacity = '0.6';
    linkElement.style.pointerEvents = 'none';
    
    // ローディングアイコンを追加
    const companyInfo = linkElement.querySelector('.company-info');
    if (companyInfo) {
        const loadingIcon = document.createElement('span');
        loadingIcon.className = 'loading-spinner-inline';
        loadingIcon.style.marginLeft = '10px';
        companyInfo.appendChild(loadingIcon);
    }
    
    // ページ全体にローディングオーバーレイを表示
    showPageLoadingOverlay('企業詳細ページを読み込み中...');
}

/**
 * ページ全体のローディングオーバーレイを表示
 * @param {string} message - 表示するメッセージ
 */
function showPageLoadingOverlay(message = '読み込み中...') {
    // 既存のオーバーレイを削除
    hidePageLoadingOverlay();
    
    const overlay = document.createElement('div');
    overlay.id = 'page-loading-overlay';
    overlay.className = 'page-loading-overlay';
    overlay.innerHTML = `
        <div class="loading-content">
            <div class="loading-spinner-large"></div>
            <h3>${message}</h3>
            <p>しばらくお待ちください...</p>
        </div>
    `;
    
    document.body.appendChild(overlay);
    
    // フェードイン効果
    setTimeout(() => {
        overlay.classList.add('show');
    }, 10);
}

/**
 * ページ全体のローディングオーバーレイを隠す
 */
function hidePageLoadingOverlay() {
    const overlay = document.getElementById('page-loading-overlay');
    if (overlay) {
        overlay.classList.remove('show');
        setTimeout(() => {
            overlay.remove();
        }, 300);
    }
}

/**
 * スムーズスクロール機能
 * @param {string} targetId - スクロール先の要素ID
 */
function smoothScrollTo(targetId) {
    const element = document.getElementById(targetId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

/**
 * 検索キーワードのハイライト
 */
function highlightSearchKeyword() {
    const urlParams = new URLSearchParams(window.location.search);
    const keyword = urlParams.get('keyword');
    
    if (keyword) {
        const companyNames = document.querySelectorAll('.company-name');
        companyNames.forEach(nameElement => {
            const text = nameElement.textContent;
            const highlightedText = text.replace(
                new RegExp(`(${keyword})`, 'gi'),
                '<mark>$1</mark>'
            );
            nameElement.innerHTML = highlightedText;
        });
    }
}

// ページ読み込み時の初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeSearchForm();
    enhanceCompanyLinks();
    enhanceCTAButtons();
    highlightSearchKeyword();
    
    // 検索フィールドにフォーカス
    const searchInput = document.querySelector('input[name="keyword"]');
    if (searchInput && !searchInput.value) {
        searchInput.focus();
    }
});

// リップル効果と追加のローディングCSS（動的に追加）
const dynamicCSS = `
.ripple {
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: scale(0);
    animation: ripple-animation 0.6s linear;
    pointer-events: none;
}

@keyframes ripple-animation {
    to {
        transform: scale(2);
        opacity: 0;
    }
}

mark {
    background-color: #fff3cd;
    padding: 2px 4px;
    border-radius: 3px;
}

/* ローディングスピナー */
.loading-spinner-inline {
    width: 16px;
    height: 16px;
    border: 2px solid #f3f3f3;
    border-top: 2px solid #007bff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    display: inline-block;
    vertical-align: middle;
}

.loading-spinner-large {
    width: 60px;
    height: 60px;
    border: 6px solid #f3f3f3;
    border-top: 6px solid #007bff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* ページローディングオーバーレイ */
.page-loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    z-index: 9999;
    display: flex;
    justify-content: center;
    align-items: center;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.page-loading-overlay.show {
    opacity: 1;
}

.loading-content {
    text-align: center;
    color: white;
    background: rgba(255, 255, 255, 0.1);
    padding: 40px;
    border-radius: 12px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.loading-content h3 {
    margin: 0 0 10px 0;
    font-size: 1.4rem;
    font-weight: 600;
}

.loading-content p {
    margin: 0;
    opacity: 0.8;
    font-size: 1rem;
}
`;

// CSSを動的に追加
const style = document.createElement('style');
style.textContent = dynamicCSS;
document.head.appendChild(style);

// ページ離脱時の処理
window.addEventListener('beforeunload', function() {
    hidePageLoadingOverlay();
});

// ページが完全に読み込まれた時の処理
window.addEventListener('load', function() {
    hidePageLoadingOverlay();
});
})();