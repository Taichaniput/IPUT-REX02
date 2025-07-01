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
        submitButton.textContent = '検索中...';
        
        // 3秒後にボタンを元に戻す（タイムアウト対策）
        setTimeout(() => {
            submitButton.disabled = false;
            submitButton.textContent = '検索';
        }, 3000);
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
        
        // クリック時のフィードバック
        link.addEventListener('click', function() {
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

// リップル効果用のCSS（動的に追加）
const rippleCSS = `
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
`;

// CSSを動的に追加
const style = document.createElement('style');
style.textContent = rippleCSS;
document.head.appendChild(style);