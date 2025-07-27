(function() {
// Base Template JavaScript

/**
 * ナビゲーションバーの機能強化
 */
function enhanceNavigation() {
    const navbar = document.querySelector('nav');
    const navLinks = document.querySelectorAll('.nav-link');
    
    if (navbar) {
        // スクロール時のナビゲーションバー効果
        let lastScrollTop = 0;
        window.addEventListener('scroll', function() {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            
            if (scrollTop > lastScrollTop && scrollTop > 100) {
                // 下にスクロール - ナビバーを隠す
                navbar.style.transform = 'translateY(-100%)';
            } else {
                // 上にスクロール - ナビバーを表示
                navbar.style.transform = 'translateY(0)';
            }
            lastScrollTop = scrollTop;
        });
        
        // スムーズなトランジション
        navbar.style.transition = 'transform 0.3s ease-in-out';
    }
    
    // ナビゲーションリンクのアクティブ状態
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            // クリック時のフィードバック
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });
}

/**
 * ページローディング状態の管理
 */
function initializeLoadingStates() {
    // フォーム送信時のローディング表示
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitButton = this.querySelector('button[type="submit"], input[type="submit"]');
            if (submitButton) {
                const originalText = submitButton.textContent || submitButton.value;
                submitButton.disabled = true;
                submitButton.textContent = '処理中...';
                
                // 5秒後にタイムアウト
                setTimeout(() => {
                    submitButton.disabled = false;
                    submitButton.textContent = originalText;
                }, 5000);
            }
        });
    });
    
    // リンククリック時のローディング表示
    const navigationLinks = document.querySelectorAll('a[href*="/company/"]');
    navigationLinks.forEach(link => {
        link.addEventListener('click', function() {
            showGlobalLoading();
        });
    });
}

/**
 * グローバルローディング表示
 */
function showGlobalLoading() {
    // ローディングオーバーレイを作成
    const loadingOverlay = document.createElement('div');
    loadingOverlay.id = 'global-loading';
    loadingOverlay.innerHTML = `
        <div class="loading-spinner"></div>
        <div class="loading-text">読み込み中...</div>
    `;
    document.body.appendChild(loadingOverlay);
    
    // 10秒後に自動削除（タイムアウト対策）
    setTimeout(() => {
        const overlay = document.getElementById('global-loading');
        if (overlay) {
            overlay.remove();
        }
    }, 10000);
}

/**
 * アクセシビリティ機能の強化
 */
function enhanceAccessibility() {
    // フォーカス可能な要素のアウトライン強化
    const focusableElements = document.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    focusableElements.forEach(element => {
        element.addEventListener('focus', function() {
            this.classList.add('keyboard-focused');
        });
        
        element.addEventListener('blur', function() {
            this.classList.remove('keyboard-focused');
        });
        
        element.addEventListener('mousedown', function() {
            this.classList.remove('keyboard-focused');
        });
    });
    
    // キーボードナビゲーション
    document.addEventListener('keydown', function(e) {
        // Escキーでモーダルやオーバーレイを閉じる
        if (e.key === 'Escape') {
            const overlay = document.getElementById('global-loading');
            if (overlay) {
                overlay.remove();
            }
        }
        
        // Alt + Homeでホームページに移動
        if (e.altKey && e.key === 'Home') {
            e.preventDefault();
            window.location.href = '/';
        }
    });
}

/**
 * エラーハンドリング機能
 */
function initializeErrorHandling() {
    // グローバルエラーハンドラ
    window.addEventListener('error', function(e) {
        console.error('JavaScript Error:', e.error);
        // 本番環境では詳細なエラー情報を隠す
    });
    
    // Unhandled Promise Rejection
    window.addEventListener('unhandledrejection', function(e) {
        console.error('Unhandled Promise Rejection:', e.reason);
    });
}

/**
 * パフォーマンス監視
 */
function initializePerformanceMonitoring() {
    // ページロード時間の測定
    window.addEventListener('load', function() {
        const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
        if (loadTime > 3000) {
            console.warn('Page load time is slow:', loadTime + 'ms');
        }
    });
}

/**
 * ユーティリティ関数
 */
const Utils = {
    /**
     * 要素が表示されているかチェック
     */
    isElementVisible: function(element) {
        const rect = element.getBoundingClientRect();
        return rect.top >= 0 && rect.bottom <= window.innerHeight;
    },
    
    /**
     * デバウンス関数
     */
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    /**
     * 要素のスムーズスクロール
     */
    smoothScrollTo: function(element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
};

// ページ読み込み時の初期化
document.addEventListener('DOMContentLoaded', function() {
    enhanceNavigation();
    initializeLoadingStates();
    enhanceAccessibility();
    initializeErrorHandling();
    initializePerformanceMonitoring();
    
    // メインコンテナにフェードインアニメーション
    const mainContainer = document.querySelector('.main-container');
    if (mainContainer) {
        mainContainer.classList.add('fade-in');
    }
});

// グローバルオブジェクトとして公開
window.GrowthCompass = {
    Utils,
    showGlobalLoading,
    enhanceNavigation,
    enhanceAccessibility
};

// ローディングスピナーのCSS（動的に追加）
const loadingCSS = `
#global-loading {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.9);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 10000;
    backdrop-filter: blur(2px);
}

.loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #007bff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

.loading-text {
    margin-top: 15px;
    color: #666;
    font-size: 16px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.keyboard-focused {
    outline: 3px solid #007bff !important;
    outline-offset: 2px !important;
}
`;

// CSSを動的に追加
const style = document.createElement('style');
style.textContent = loadingCSS;
document.head.appendChild(style);
})();